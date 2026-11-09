"""Run Condition D on the existing Regular Season holdout."""
import argparse
from datetime import datetime
from itertools import combinations
from statistics import mean

import torch
from filelock import FileLock, Timeout
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .dataset import compute_class_weights
from .experiment import SEEDS
from .experiment_extended import read_json
from .masking import EVAL_MASK_SEED, UNK_ID
from .metrics import MetricAccumulator
from .model import masked_multitask_loss
from .model_masked import MASKED_MODELS
from .model_masked_player import PLAYER_MODELS
from .preprocess import fingerprint, write_json
from .summary_condition_d import COMBINATIONS, OUTPUT, write_summary
from .train import train_epoch
from .train_extended import (BATCH_SIZE, GRAD_CLIP, LR, MAX_EPOCHS, MODEL_OPTIONS, PATIENCE,
                             WEIGHT_DECAY, experiment_config, prepare_extended_split, publish_epoch, save_epoch)
from .train_masked import seed_dir
from .train_masked_player import history_metrics
from .utils import seed_everything, select_device


SLOTS_PER_KIND = 10
MASK_COUNT = 3
MASK_WEIGHT = 1.0
MASK_TRIPLES = torch.tensor(list(combinations(range(2 * SLOTS_PER_KIND), MASK_COUNT)))


def sample_positions(size, generator):
    # Each unordered triple has exactly one row, including triples containing OOV slots.
    return MASK_TRIPLES[torch.randint(len(MASK_TRIPLES), (size,), generator=generator)]


def masked_batch(model, batch, positions, device):
    original = torch.cat((batch['player_ids'], batch['champion_ids']), dim=1)
    targets = original.gather(1, positions)
    is_player = positions < SLOTS_PER_KIND
    replacements = torch.where(is_player, model.player_mask_id, model.mask_id)
    players, champions = original.scatter(1, positions, replacements).split(SLOTS_PER_KIND, dim=1)

    # The existing head API takes a rectangular set of indices per type. Fill unused
    # entries with a safe index and exclude those entries before loss or scoring.
    player_positions = positions.masked_fill(~is_player, 0)
    champion_positions = (positions - SLOTS_PER_KIND).masked_fill(is_player, 0)
    scores = model.forward_masked_both(players.to(device), champions.to(device),
                                       player_positions.to(device), champion_positions.to(device))
    return scores, targets.to(device), is_player.to(device)


def reconstruction_loss(scores, targets, is_player, stats):
    losses = []
    count = 0
    for kind, slots in (('player', is_player), ('champion', ~is_player)):
        # A held-out identity outside the train vocabulary has no reconstruction class.
        valid = slots & (targets != UNK_ID)
        truth = targets[valid]
        logits = scores[kind][valid]
        size = truth.numel()
        if size == 0:
            losses.append(logits.new_zeros(()))
            continue
        loss = F.cross_entropy(logits, truth, reduction='sum')
        losses.append(loss)
        count += size
        matches = logits.detach().topk(min(5, logits.shape[-1] - 1)).indices == truth[:, None]
        stats[kind][0] += loss.detach().item()
        stats[kind][1] += matches[:, 0].sum().item()
        stats[kind][2] += matches.any(dim=1).sum().item()
        stats[kind][3] += size
    return sum(losses) / max(count, 1)


def reconstruction_metrics(totals):
    loss, top1, top5, count = totals
    return dict(loss=loss / count if count else 0.0, valid_samples=count,
                top1_accuracy=top1 / count if count else None, top5_accuracy=top5 / count if count else None)


def run_pass(model, loader, device, stage, positions, optimizer=None, weights=None):
    training = optimizer is not None
    model.train(training)
    tasks = MetricAccumulator() if stage == 'joint' or not training else None
    stats = {kind: [0.0, 0, 0, 0] for kind in ('player', 'champion')}
    combo_counts = torch.zeros(MASK_COUNT + 1, dtype=torch.long)
    objectives = []
    offset = 0
    with torch.set_grad_enabled(training):
        for batch in loader:
            size = len(batch['player_ids'])
            selected = positions[offset:offset + size]
            offset += size
            combo_counts += torch.bincount((selected < SLOTS_PER_KIND).sum(1), minlength=MASK_COUNT + 1)
            if training:
                optimizer.zero_grad(set_to_none=True)
            multitask = None
            if tasks is not None:
                logits = model(batch['player_ids'].to(device), batch['champion_ids'].to(device))
                labels = batch['labels'].to(device)
                tasks.update(logits, labels)
                if training:
                    multitask = masked_multitask_loss(logits, labels, weights)
            scores, targets, is_player = masked_batch(model, batch, selected, device)
            mask_loss = reconstruction_loss(scores, targets, is_player, stats)
            if training:
                loss = mask_loss if multitask is None else multitask + MASK_WEIGHT * mask_loss
                if not torch.isfinite(loss):
                    raise ValueError('Non-finite optimization loss')
                if loss.requires_grad:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP, error_if_nonfinite=True)
                    optimizer.step()
                objectives.append(loss.detach().item())

    report = tasks.compute() if tasks is not None else {}
    for kind, totals in stats.items():
        report[f'masked_{kind}'] = reconstruction_metrics(totals)
    report['masked_token'] = reconstruction_metrics(tuple(map(sum, zip(*stats.values()))))
    report['masked_loss'] = report['masked_token']['loss']
    report['mask_combinations'] = dict(zip(COMBINATIONS, combo_counts.tolist()))
    if training:
        report['optimization_loss'] = mean(objectives)
    return report


def epoch_metrics(report):
    values = history_metrics(report)
    if 'masked_token' in report:
        values.update({f'masked_token_{key}': value for key, value in report['masked_token'].items()})
        values.update(report['mask_combinations'])
    return values


def run_stage(output, stage, variant, seed, train, validation, validation_positions, split, device,
              pretrained=None):
    selection_metric = ('validation_masked_token_top1_accuracy' if stage == 'pretrain'
                        else 'validation_winner_macro_f1')
    config = dict(experiment_config(variant, seed, split, device), condition='D', stage=stage,
                  selection_metric=selection_metric, mask_count=MASK_COUNT, eval_mask_seed=EVAL_MASK_SEED,
                  mask_sampling='uniform_triplet_over_20_slots', masked_reduction='mean_over_in_vocab_targets',
                  masked_lambda=MASK_WEIGHT if stage == 'joint' else None)
    if pretrained is not None:
        config['pretrained_checkpoint'] = fingerprint(pretrained)
    if (output / 'metrics.json').is_file():
        saved = read_json(output / 'config.json')
        if any(saved.get(key) != value for key, value in config.items() if key != 'device'):
            raise ValueError(f'Completed run settings differ: {output}')
        print(f'Already complete: {output}', flush=True)
        return

    seed_everything(seed)
    generator = torch.Generator().manual_seed(seed)
    _, layers = MASKED_MODELS[variant]
    model = PLAYER_MODELS[variant](player_vocab_size=len(train.player_vocab),
                                  champion_vocab_size=len(train.champion_vocab),
                                  num_layers=layers, **MODEL_OPTIONS).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    history = []
    if (output / 'last.pt').is_file():
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
        if pretrained is not None:
            checkpoint = torch.load(pretrained, map_location='cpu', weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            del checkpoint
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / 'config.json', {**config, 'model': model.config})
        save_epoch(output, model, optimizer, config, train, history, generator, device)

    weights = None if stage == 'pretrain' else {
        task: weight.to(device) for task, weight in compute_class_weights(train.labels).items()
    }
    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, generator=generator)
    val_loader = DataLoader(validation, batch_size=BATCH_SIZE, shuffle=False)
    best_epoch = history[-1]['best_epoch'] if history else 0
    best_score = history[best_epoch - 1]['selection_score'] if best_epoch else -float('inf')
    for epoch in range(len(history) + 1, MAX_EPOCHS + 1):
        if epoch - 1 - best_epoch >= PATIENCE:
            break
        if stage == 'finetune':
            online = train_epoch(model, train_loader, optimizer, device, GRAD_CLIP, weights)
        else:
            mask_rng = torch.Generator().manual_seed(seed * MAX_EPOCHS + epoch)
            online = run_pass(model, train_loader, device, stage, sample_positions(len(train), mask_rng),
                              optimizer, weights)
        val = run_pass(model, val_loader, device, stage, validation_positions)
        score = (val['masked_token']['top1_accuracy'] if stage == 'pretrain'
                 else val['tasks']['winner_side']['macro_f1'])
        if score is None:
            raise ValueError(f'No eligible validation labels for {selection_metric}')
        if score > best_score:
            best_score, best_epoch = score, epoch
        history.append(dict(epoch=epoch, best_epoch=best_epoch, selection_score=score, lr=LR,
                            **{f'train_{key}': value for key, value in epoch_metrics(online).items()},
                            **{f'validation_{key}': value for key, value in epoch_metrics(val).items()}))
        save_epoch(output, model, optimizer, config, train, history, generator, device)
        print(f'D | {variant} | seed {seed} | {stage} | epoch {epoch} | '
              f'{selection_metric} {score:.4f} | best {best_epoch}', flush=True)

    best_path = output / 'best.pt'
    checkpoint = torch.load(best_path, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    del checkpoint
    write_json(output / 'metrics.json', dict(best_epoch=best_epoch, last_epoch=history[-1]['epoch'],
                                           checkpoint=fingerprint(best_path), winner_head_trained=stage != 'pretrain',
                                           validation=run_pass(model, val_loader, device, stage, validation_positions)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', nargs='?', default='all', choices=('all', 'joint', 'pretrain', 'pipeline', 'summary'))
    parser.add_argument('--models', nargs='+', choices=tuple(PLAYER_MODELS), default=list(PLAYER_MODELS))
    parser.add_argument('--seeds', nargs='+', type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument('--device', choices=('auto', 'cuda', 'mps', 'cpu'), default='auto')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.threads <= 0:
        parser.error('Use positive --threads')
    models, seeds = tuple(dict.fromkeys(args.models)), tuple(dict.fromkeys(args.seeds))
    torch.set_num_threads(args.threads)
    train, validation, split = prepare_extended_split()
    if args.mode != 'summary':
        device = select_device(args.device)
        positions = sample_positions(len(validation), torch.Generator().manual_seed(EVAL_MASK_SEED))
        for variant in models:
            for seed in seeds:
                run = seed_dir(OUTPUT, variant, seed)
                settings = (variant, seed, train, validation, positions, split, device)
                run.mkdir(parents=True, exist_ok=True)
                try:
                    with FileLock(run / '.lock', timeout=0):
                        if args.mode in ('all', 'joint'):
                            run_stage(run / 'joint', 'joint', *settings)
                        if args.mode in ('all', 'pretrain', 'pipeline'):
                            run_stage(run / 'pretrain', 'pretrain', *settings)
                        if args.mode in ('all', 'pipeline'):
                            run_stage(run / 'finetune', 'finetune', *settings, pretrained=run / 'pretrain/best.pt')
                except Timeout:
                    parser.exit(1, f'Condition D run already in use: {run}\n')
    if set(models) == set(PLAYER_MODELS) and set(seeds) == set(SEEDS):
        summary = OUTPUT / 'summary.csv'
        with FileLock(summary.with_suffix('.lock')):
            write_summary(summary, PLAYER_MODELS, SEEDS, split)
    else:
        summary = OUTPUT / 'comparisons' / f'{datetime.now():%Y%m%d_%H%M%S_%f}' / 'summary.csv'
        write_summary(summary, models, seeds, split)
    print(f'Summary: {summary}', flush=True)


if __name__ == '__main__':
    main()
