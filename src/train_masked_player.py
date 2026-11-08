"""Run champion-count and player-mask ablations on the fixed holdout."""
import argparse
from collections import defaultdict
from datetime import datetime
from statistics import mean, stdev

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .dataset import IGNORE_INDEX, compute_class_weights
from .evaluate import evaluate
from .experiment import SEEDS
from .experiment_extended import read_json
from .masking import EVAL_MASK_SEED, MASK_COUNT
from .metrics import MetricAccumulator
from .model import masked_multitask_loss
from .model_masked import MASKED_MODELS
from .model_masked_player import PLAYER_MODELS, masked_batch
from .preprocess import fingerprint, write_csv, write_json
from .train_extended import (BATCH_SIZE, GRAD_CLIP, LR, MAX_EPOCHS, MODEL_OPTIONS, PATIENCE,
                             WEIGHT_DECAY, experiment_config, prepare_extended_split, publish_epoch, save_epoch)
from .train_masked import OUTPUT as CHAMPION_OUTPUT, seed_dir, task_scores, validation_result
from .train_masked_pretrain import OUTPUT as PRETRAIN_OUTPUT
from .utils import ROOT, seed_everything, select_device


OUTPUT = ROOT / 'runs/masked_player'
CONDITIONS = {'B': (3, 0), 'C': (2, 1)}
PLAYER_SEED_OFFSET = 1_000_000


def run_pass(model, loader, device, condition, mask_seed, joint, optimizer=None, weights=None):
    training = optimizer is not None
    model.train(training)
    tasks = MetricAccumulator() if joint or not training else None
    champion_count, player_count = CONDITIONS[condition]
    champion_rng = torch.Generator().manual_seed(mask_seed)
    player_rng = torch.Generator().manual_seed(mask_seed + PLAYER_SEED_OFFSET) if player_count else None
    stats = {'champion': [0.0, 0, 0, 0]}
    if player_count:
        stats['player'] = [0.0, 0, 0, 0]
    objectives, multitask_objectives = [], []
    with torch.set_grad_enabled(training):
        for batch in loader:
            if training:
                optimizer.zero_grad(set_to_none=True)
            multitask = None
            if tasks is not None:
                labels = batch['labels'].to(device)
                ordinary = model(batch['player_ids'].to(device), batch['champion_ids'].to(device))
                tasks.update(ordinary, labels)
                if training:
                    multitask = masked_multitask_loss(ordinary, labels, weights)
            logits, targets = masked_batch(model, batch, device, champion_count, champion_rng, player_rng)
            losses, count = [], 0
            for kind, scores in logits.items():
                valid = targets[kind] != IGNORE_INDEX
                truth = targets[kind][valid]
                if not truth.numel():
                    continue
                scores = scores[valid]
                ce = F.cross_entropy(scores, truth, reduction='sum')
                losses.append(ce)
                count += truth.numel()
                matches = scores.detach().topk(min(5, scores.shape[-1] - 1)).indices == truth[:, None]
                stats[kind][0] += ce.detach().item()
                stats[kind][1] += matches[:, 0].sum().item()
                stats[kind][2] += matches.any(dim=1).sum().item()
                stats[kind][3] += truth.numel()
            if training:
                if not count:
                    raise ValueError('No eligible masked targets in the training batch')
                total = sum(losses) / count
                if multitask is not None:
                    total = total + multitask
                    multitask_objectives.append(multitask.detach().item())
                if not torch.isfinite(total):
                    raise ValueError('Non-finite optimization loss')
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP, error_if_nonfinite=True)
                optimizer.step()
                objectives.append(total.detach().item())

    valid_total = sum(values[3] for values in stats.values())
    if not valid_total:
        raise ValueError('No eligible reconstruction targets in this split')
    report = tasks.compute() if tasks is not None else {}
    report['masked_loss'] = sum(values[0] for values in stats.values()) / valid_total
    for kind, (loss, top1, top5, valid) in stats.items():
        report[f'masked_{kind}'] = dict(loss=loss / valid if valid else None,
                                        top1_accuracy=top1 / valid if valid else None,
                                        top5_accuracy=top5 / valid if valid else None, valid_samples=valid)
    if training:
        report['optimization_loss'] = mean(objectives)
        if joint:
            report['multitask_optimization_loss'] = mean(multitask_objectives) if multitask_objectives else None
    return report


def history_metrics(report):
    values = {key: report[key] for key in ('masked_loss', 'optimization_loss', 'multitask_optimization_loss')
              if key in report}
    if 'tasks' in report:
        values.update(loss=report['loss'], **task_scores(report))
    for kind in ('masked_champion', 'masked_player'):
        if kind in report:
            values.update({f'{kind}_{key}': value for key, value in report[kind].items()})
    return values


def run_config(variant, seed, split, device, condition, stage):
    champion_count, player_count = CONDITIONS[condition]
    return {
        **experiment_config(variant, seed, split, device),
        'condition': condition, 'stage': stage, 'champion_masks': champion_count, 'player_masks': player_count,
        'eval_mask_seed': EVAL_MASK_SEED, 'player_mask_seed_offset': PLAYER_SEED_OFFSET,
        'masked_lambda': 1.0 if stage == 'joint' else None, 'masked_reduction': 'mean_over_valid_tokens',
        'selection_metric': ('validation_winner_macro_f1' if stage == 'joint' else 'validation_masked_loss'),
    }


def run_seed(output, variant, seed, train, validation, split, device, condition, stage):
    config = run_config(variant, seed, split, device, condition, stage)
    if (output / 'metrics.json').is_file():
        saved = read_json(output / 'config.json')
        if any(saved[key] != value for key, value in config.items() if key != 'device'):
            raise ValueError(f'Completed run settings differ: {output}')
        print(f'Already complete: {output}', flush=True)
        return

    joint = stage == 'joint'
    seed_everything(seed)
    generator = torch.Generator().manual_seed(seed)
    model_class, layers = MASKED_MODELS[variant]
    if config['player_masks']:
        model_class = PLAYER_MODELS[variant]
    model = model_class(player_vocab_size=len(train.player_vocab), champion_vocab_size=len(train.champion_vocab),
                        num_layers=layers, **MODEL_OPTIONS).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    history = []
    if output.exists():
        checkpoint = torch.load(output / 'last.pt', map_location='cpu', weights_only=True)
        state = checkpoint['run_config']
        if state['config'] != config:
            raise ValueError(f'Resume settings or data differ: {output}')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        history = state['history']
        if history:
            publish_epoch(output, history)
        generator.set_state(state['rng']['shuffle'])
        torch.set_rng_state(state['rng']['cpu'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state(state['rng']['device'], device)
        elif device.type == 'mps':
            torch.mps.set_rng_state(state['rng']['device'])
        del checkpoint
    else:
        output.mkdir(parents=True)
        write_json(output / 'config.json', {**config, 'model': model.config})
        save_epoch(output, model, optimizer, config, train, history, generator, device)

    weights = {task: weight.to(device) for task, weight in compute_class_weights(train.labels).items()} if joint else None
    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, generator=generator)
    eval_train = DataLoader(train, batch_size=BATCH_SIZE, shuffle=False)
    eval_val = DataLoader(validation, batch_size=BATCH_SIZE, shuffle=False)
    best_epoch = history[-1]['best_epoch'] if history else 0
    best_score = history[best_epoch - 1]['selection_score'] if best_epoch else -float('inf')
    for epoch in range(len(history) + 1, MAX_EPOCHS + 1):
        if epoch - 1 - best_epoch >= PATIENCE:
            break
        online = run_pass(model, train_loader, device, condition, seed * MAX_EPOCHS + epoch,
                          joint, optimizer, weights)
        train_eval = task_scores(evaluate(model, eval_train, device)) if joint else {}
        val = run_pass(model, eval_val, device, condition, EVAL_MASK_SEED, joint)
        score = val['tasks']['winner_side']['macro_f1'] if joint else -val['masked_loss']
        if score > best_score:
            best_score, best_epoch = score, epoch
        history.append({
            'epoch': epoch, 'best_epoch': best_epoch, 'selection_score': score,
            'lr': optimizer.param_groups[0]['lr'],
            **{f'train_{key}': value for key, value in history_metrics(online).items()},
            **{f'train_eval_{key}': value for key, value in train_eval.items()},
            **{f'validation_{key}': value for key, value in history_metrics(val).items()},
        })
        save_epoch(output, model, optimizer, config, train, history, generator, device)
        print(f'{condition} | {variant} | seed {seed} | {stage} | epoch {epoch} | '
              f'validation masked loss {val["masked_loss"]:.4f} | best {best_epoch}', flush=True)

    path = output / 'best.pt'
    best = torch.load(path, map_location='cpu', weights_only=True)
    model.load_state_dict(best['model_state_dict'])
    write_json(output / 'metrics.json', {
        'best_epoch': best_epoch, 'last_epoch': history[-1]['epoch'], 'checkpoint': fingerprint(path),
        'winner_head_trained': joint,
        'training': run_pass(model, eval_train, device, condition, EVAL_MASK_SEED, joint),
        'validation': run_pass(model, eval_val, device, condition, EVAL_MASK_SEED, joint),
    })
    print(f'Output: {output}', flush=True)


def comparison_row(report, condition, variant, seed, stage):
    val = report['validation']
    winner = val.get('tasks', {}).get('winner_side', {})
    row = dict(condition=condition, model=variant, seed=seed, stage=stage, best_epoch=report['best_epoch'],
               winner_head_trained=stage == 'joint', winner_accuracy=winner.get('accuracy'),
               winner_macro_f1=winner.get('macro_f1'))
    for kind in ('masked_champion', 'masked_player'):
        stats = val.get(kind, {})
        for metric in ('loss', 'top1_accuracy', 'top5_accuracy', 'valid_samples'):
            row[f'{kind}_{metric}'] = stats.get(metric)
    return row


def compare(variant, seeds, conditions, stage, split):
    rows = []
    for seed in seeds:
        if stage == 'joint':
            original = validation_result(CHAMPION_OUTPUT / 'lambda_1/validation', variant, seed, split, 1.0)
        else:
            run = seed_dir(PRETRAIN_OUTPUT, variant, seed) / 'pretrain'
            config = read_json(run / 'config.json')
            expected = experiment_config(variant, seed, split, config['device'])
            expected.update(stage='pretrain', mask_count=MASK_COUNT, eval_mask_seed=EVAL_MASK_SEED,
                            selection_metric='validation_masked_champion_loss')
            if any(config[key] != value for key, value in expected.items()):
                raise ValueError(f'Incompatible existing pretraining run: {run}')
            original = read_json(run / 'metrics.json')
        rows.append(comparison_row(original, 'A', variant, seed, stage))
        for condition in conditions:
            run = seed_dir(OUTPUT / stage / condition, variant, seed)
            config = read_json(run / 'config.json')
            expected = run_config(variant, seed, split, config['device'], condition, stage)
            if any(config[key] != value for key, value in expected.items()):
                raise ValueError(f'Incompatible ablation run: {run}')
            rows.append(comparison_row(read_json(run / 'metrics.json'), condition, variant, seed, stage))
    scores = defaultdict(list)
    for row in rows:
        for key, value in row.items():
            if key == 'best_epoch' or key.startswith(('winner_', 'masked_')) and key != 'winner_head_trained':
                if value is not None:
                    scores[row['condition'], key].append(value)
    summary = [dict(condition=condition, model=variant, stage=stage, metric=metric, seeds=len(values),
                    mean=mean(values), std_ddof1=stdev(values) if len(values) > 1 else None)
               for (condition, metric), values in scores.items()]
    output = OUTPUT / 'comparisons' / f'{stage}_{variant}_{datetime.now():%Y%m%d_%H%M%S_%f}'
    output.mkdir(parents=True)
    write_csv(output / 'by_seed.csv', rows, tuple(rows[0]))
    write_csv(output / 'summary.csv', summary, tuple(summary[0]))
    print(f'Output: {output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('joint', 'pretrain', 'compare'))
    parser.add_argument('--mode', choices=('joint', 'pretrain'), default='joint', help='Stage to compare')
    parser.add_argument('--model', choices=tuple(MASKED_MODELS), default='6l_256d')
    parser.add_argument('--conditions', nargs='+', choices=tuple(CONDITIONS), default=list(CONDITIONS))
    parser.add_argument('--seeds', nargs='+', type=int, choices=SEEDS)
    parser.add_argument('--device', choices=('auto', 'cuda', 'mps', 'cpu'), default='auto')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.threads <= 0:
        parser.error('Use positive --threads')
    seeds = tuple(dict.fromkeys(args.seeds or (SEEDS if args.stage == 'compare' else (137,))))
    conditions = tuple(dict.fromkeys(args.conditions))
    try:
        torch.set_num_threads(args.threads)
        train, validation, split = prepare_extended_split()
        if args.stage == 'compare':
            compare(args.model, seeds, conditions, args.mode, split)
            return
        device = select_device(args.device)
        for condition in conditions:
            for seed in seeds:
                output = seed_dir(OUTPUT / args.stage / condition, args.model, seed)
                run_seed(output, args.model, seed, train, validation, split, device, condition, args.stage)
    except (KeyError, ValueError, OSError) as exc:
        parser.exit(1, f'Masked Player experiment error: {exc}\n')


if __name__ == '__main__':
    main()
