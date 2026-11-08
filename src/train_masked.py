"""Train, refit, or compare the independent Masked Champion experiments."""
import argparse
import math
from statistics import mean, stdev

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .dataset import GameDataset, TARGETS, compute_class_weights, load_vocab, validate_training_inputs
from .evaluate import save_evaluation
from .evaluate_masked import evaluate_masked
from .experiment import SEEDS
from .experiment_extended import read_holdout, read_json
from .masking import EVAL_MASK_SEED, MASK_COUNT, mask_champions
from .metrics import MetricAccumulator, print_metrics
from .model import masked_multitask_loss
from .model_masked import MASKED_MODELS, load_masked_checkpoint
from .preprocess import fingerprint, write_csv, write_json
from .train_extended import (BATCH_SIZE, GRAD_CLIP, LR, MAX_EPOCHS, MODEL_OPTIONS, OUTPUT as BASELINE,
                             PATIENCE, SOURCE, WEIGHT_DECAY, experiment_config, prepare_extended_split,
                             publish_epoch, save_epoch)
from .utils import ROOT, seed_everything, select_device


OUTPUT = ROOT / 'runs/masked_champion'


def train_epoch(model, loader, optimizer, device, weights, coefficient, mask_generator):
    model.train()
    metrics = MetricAccumulator()
    multitask_sum = champion_sum = total_sum = 0.0
    for batch in loader:
        players, champions, labels = (batch[key].to(device) for key in ('player_ids', 'champion_ids', 'labels'))
        optimizer.zero_grad(set_to_none=True)
        logits = model(players, champions)
        multitask = masked_multitask_loss(logits, labels, weights)
        total = multitask
        if coefficient:
            masked, positions, targets = mask_champions(batch['champion_ids'], model.mask_id, mask_generator)
            champion_logits = model.forward_masked(players, masked.to(device), positions.to(device))
            champion_loss = F.cross_entropy(champion_logits.flatten(0, 1), targets.flatten().to(device))
            total = multitask + coefficient * champion_loss
            champion_sum += champion_loss.detach().item()
        if not torch.isfinite(total):
            raise ValueError('Non-finite training loss')
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP, error_if_nonfinite=True)
        optimizer.step()
        metrics.update(logits, labels)
        multitask_sum += multitask.detach().item()
        total_sum += total.detach().item()
    return {**metrics.compute(), 'multitask_loss': multitask_sum / len(loader),
            'masked_champion_loss': champion_sum / len(loader) if coefficient else None,
            'optimization_loss': total_sum / len(loader)}


def task_scores(report):
    return {f'{task}_{metric}': report['tasks'][task][metric]
            for task in TARGETS for metric in ('accuracy', 'macro_f1')}


def run_seed(output, variant, seed, train, evaluation, split, device, coefficient, fixed_epochs=None):
    final = fixed_epochs is not None
    eval_name = 'playoff' if final else 'validation'
    config = {
        **experiment_config(variant, seed, split, device), 'masked_lambda': coefficient,
        'mask_count': MASK_COUNT, 'eval_mask_seed': EVAL_MASK_SEED,
        'max_epochs': fixed_epochs if final else MAX_EPOCHS,
        'patience': None if final else PATIENCE,
        'selection_metric': 'fixed_epoch_from_validation' if final else 'validation_winner_macro_f1',
    }
    if output.exists() and (output / 'metrics.json').exists():
        saved = read_json(output / 'config.json')
        if any(saved[key] != value for key, value in config.items() if key != 'device'):
            raise ValueError(f'Completed run configuration differs: {output}')
        print(f'Already complete: {output}', flush=True)
        return

    seed_everything(seed)
    generator = torch.Generator().manual_seed(seed)
    checkpoint = None
    if output.exists():
        model, checkpoint = load_masked_checkpoint(output / 'last.pt', device)
        if checkpoint['run_config']['config'] != config:
            raise ValueError(f'Resume settings or data differ: {output}')
    else:
        output.mkdir(parents=True)
        model_class, layers = MASKED_MODELS[variant]
        model = model_class(player_vocab_size=len(train.player_vocab), champion_vocab_size=len(train.champion_vocab),
                            num_layers=layers, **MODEL_OPTIONS).to(device)
        write_json(output / 'config.json', {**config, 'model': model.config})

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    history = []
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        state = checkpoint['run_config']
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
        save_epoch(output, model, optimizer, config, train, history, generator, device)

    weights = {task: weight.to(device) for task, weight in compute_class_weights(train.labels).items()}
    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, generator=generator)
    eval_train = DataLoader(train, batch_size=BATCH_SIZE, shuffle=False)
    eval_other = DataLoader(evaluation, batch_size=BATCH_SIZE, shuffle=False)
    best_epoch = history[-1]['best_epoch'] if history else 0
    best_f1 = history[best_epoch - 1]['validation_winner_side_macro_f1'] if best_epoch and not final else -1.0
    for epoch in range(len(history) + 1, config['max_epochs'] + 1):
        if not final and epoch - 1 - best_epoch >= PATIENCE:
            break
        # Epoch-local mask RNG resumes without affecting the dropout or shuffle RNG.
        mask_generator = torch.Generator().manual_seed(seed * MAX_EPOCHS + epoch)
        online = train_epoch(model, train_loader, optimizer, device, weights, coefficient, mask_generator)
        row = {
            'epoch': epoch, 'train_loss': online['loss'], 'train_multitask_loss': online['multitask_loss'],
            'train_masked_champion_loss': online['masked_champion_loss'],
            'optimization_loss': online['optimization_loss'], 'lr': optimizer.param_groups[0]['lr'],
            **{f'train_{key}': value for key, value in task_scores(online).items()},
        }
        if final:
            best_epoch = fixed_epochs if epoch == fixed_epochs else 0
        else:
            # Match the baseline's unmasked train eval pass and keep its scores separate.
            train_report = evaluate_masked(model, eval_train, device, evaluate_champions=False)
            validation = evaluate_masked(model, eval_other, device, evaluate_champions=bool(coefficient))
            val_f1 = validation['tasks']['winner_side']['macro_f1']
            if val_f1 > best_f1:
                best_f1, best_epoch = val_f1, epoch
            row.update({f'train_eval_{key}': value for key, value in task_scores(train_report).items()})
            row.update({f'validation_{key}': value for key, value in task_scores(validation).items()})
            if coefficient:
                row.update({f'validation_masked_champion_{key}': value
                            for key, value in validation['masked_champion'].items()})
        row['best_epoch'] = best_epoch
        history.append(row)
        save_epoch(output, model, optimizer, config, train, history, generator, device)
        print(f'{variant} seed {seed} | {eval_name} | epoch {epoch}/{config["max_epochs"]} '
              f'| total loss {online["optimization_loss"]:.4f} | best epoch {best_epoch}', flush=True)

    selected_model, _ = load_masked_checkpoint(output / 'best.pt', device)
    metrics = {
        'best_epoch': best_epoch, 'last_epoch': history[-1]['epoch'],
        'stop_reason': ('patience' if not final and history[-1]['epoch'] - best_epoch >= PATIENCE
                        else 'fixed_epoch' if final else 'max_epochs'),
        'checkpoint': fingerprint(output / 'best.pt'),
        'training': evaluate_masked(selected_model, eval_train, device, evaluate_champions=bool(coefficient)),
        eval_name: evaluate_masked(selected_model, eval_other, device, evaluate_champions=bool(coefficient)),
    }
    if final:
        save_evaluation(metrics['playoff'], output, {
            'epoch': best_epoch, 'checkpoint': metrics['checkpoint'], 'dataset': split['playoff'],
        })
    write_json(output / 'metrics.json', metrics)
    print(f'{variant} seed {seed} | best epoch {best_epoch} | {eval_name.capitalize()}')
    print_metrics(metrics[eval_name], per_class=False)


def seed_dir(root, variant, seed):
    return root / variant / f'seed_{seed}'


def validation_result(holdout, variant, seed, split, coefficient):
    if not coefficient:
        baseline = seed_dir(BASELINE, variant, seed)
        if (baseline / 'metrics.json').is_file():
            return read_holdout(baseline, variant, seed, split)
    run = seed_dir(holdout, variant, seed)
    config = read_json(run / 'config.json')
    expected = experiment_config(variant, seed, split, config['device'])
    if (any(config[key] != value for key, value in expected.items())
            or config['masked_lambda'] != coefficient or config['eval_mask_seed'] != EVAL_MASK_SEED
            or config['mask_count'] != MASK_COUNT):
        raise ValueError(f'Incompatible validation run: {run}')
    report = read_json(run / 'metrics.json')
    if not 1 <= report['best_epoch'] <= report['last_epoch'] <= MAX_EPOCHS:
        raise ValueError(f'Invalid validation epochs: {run}')
    return report


def summarize(path, groups, splits):
    rows = []
    for (condition, variant), reports in groups.items():
        values = {('selection', 'best_epoch'): [report['best_epoch'] for report in reports]}
        for split in splits:
            task_results = [task_scores(report[split]) for report in reports]
            for key in task_results[0]:
                values[split, key] = [result[key] for result in task_results]
            if 'masked_champion' in reports[0][split]:
                for key in ('loss', 'top1_accuracy', 'top5_accuracy'):
                    values[split, f'masked_champion_{key}'] = [r[split]['masked_champion'][key] for r in reports]
        for (split, metric), scores in values.items():
            valid = all(score is not None for score in scores)
            average = mean(scores) if valid else None
            deviation = stdev(scores) if valid and len(scores) > 1 else None
            rows.append(dict(condition=condition, model=variant, split=split, metric=metric, seeds=len(scores),
                             mean=average, std_ddof1=deviation))
            if metric in ('best_epoch', 'winner_side_accuracy', 'winner_side_macro_f1'):
                display = 'N/A' if average is None else f'{average:.6f}'
                if deviation is not None:
                    display += f' ± {deviation:.6f}'
                print(f'{condition} | {variant} | {split} | {metric} | {display}')
    write_csv(path, rows, tuple(rows[0]))
    print(f'Summary: {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('validation', 'final', 'compare'))
    parser.add_argument('--model', nargs='+', choices=tuple(MASKED_MODELS), default=list(MASKED_MODELS))
    parser.add_argument('--seeds', nargs='+', type=int, choices=SEEDS)
    parser.add_argument('--lambda', dest='coefficient', type=float, default=1.0)
    parser.add_argument('--device', choices=('auto', 'cuda', 'mps', 'cpu'), default='auto')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if not math.isfinite(args.coefficient) or args.coefficient < 0 or args.threads <= 0:
        parser.error('Use finite nonnegative --lambda and positive --threads')
    seeds = tuple(dict.fromkeys(args.seeds or ((137,) if args.stage == 'validation' else SEEDS)))
    variants = tuple(dict.fromkeys(args.model))
    coefficient = args.coefficient or 0.0
    root = OUTPUT / f"lambda_{str(coefficient).removesuffix('.0')}"
    holdout = root / 'validation'
    condition = root.name if coefficient else 'baseline'
    try:
        train, validation, split = prepare_extended_split()
        if (len(train), len(validation)) != (3904, 976):
            raise ValueError('Expected the 3,904 / 976 Regular Season split')
        if args.stage == 'compare':
            if not coefficient:
                raise ValueError('Use a positive --lambda for the masked condition in a comparison')
            groups = {}
            for variant in variants:
                groups['baseline', variant] = [
                    validation_result(OUTPUT / 'lambda_0/validation', variant, seed, split, 0.0)
                    for seed in seeds]
                groups[condition, variant] = [
                    validation_result(holdout, variant, seed, split, coefficient) for seed in seeds]
            suffix = '' if variants == tuple(MASKED_MODELS) and seeds == SEEDS else (
                '_' + '_'.join(variants) + '_' + '_'.join(map(str, seeds)))
            summarize(holdout / f'comparison{suffix}.csv', groups, ('training', 'validation'))
            return

        torch.set_num_threads(args.threads)
        device = select_device(args.device)
        stage_dir = root / args.stage
        evaluation = validation
        selected = {}
        if args.stage == 'final':
            for variant in variants:
                for seed in seeds:
                    report = validation_result(holdout, variant, seed, split, coefficient)
                    selected[variant, seed] = report['best_epoch']
            player_vocab = load_vocab(SOURCE.with_name('player_vocab.json'))
            champion_vocab = load_vocab(SOURCE.with_name('champion_vocab.json'))
            train = GameDataset(SOURCE, player_vocab, champion_vocab, 'Regular Season')
            playoff_path = SOURCE.with_name('test.csv')
            evaluation = GameDataset(playoff_path, player_vocab, champion_vocab, 'Playoff')
            validate_training_inputs(train, evaluation)
            if len(train) != 4880:
                raise ValueError('Expected 4,880 Regular Season games')
            split = {'source': split['source'], 'playoff': fingerprint(playoff_path), 'validation_split': split}

        for variant in variants:
            for seed in seeds:
                if args.stage == 'validation' and not coefficient:
                    baseline = seed_dir(BASELINE, variant, seed)
                    if (baseline / 'metrics.json').is_file():
                        read_holdout(baseline, variant, seed, split)
                        print(f'Reusing baseline: {baseline}')
                        continue
                run_seed(seed_dir(stage_dir, variant, seed), variant, seed, train, evaluation, split, device,
                         coefficient, fixed_epochs=selected.get((variant, seed)))
        groups = {}
        for variant in MASKED_MODELS:
            paths = [seed_dir(stage_dir, variant, seed) / 'metrics.json' for seed in SEEDS]
            if args.stage == 'validation' and not coefficient:
                paths = [seed_dir(BASELINE, variant, seed) / 'metrics.json'
                         if (seed_dir(BASELINE, variant, seed) / 'metrics.json').is_file() else path
                         for seed, path in zip(SEEDS, paths)]
            if all(path.exists() for path in paths):
                groups[condition, variant] = (
                    [validation_result(holdout, variant, seed, split, coefficient) for seed in SEEDS]
                    if args.stage == 'validation' else [read_json(path) for path in paths])
        if groups:
            summarize(stage_dir / 'five_seed_summary.csv', groups,
                      ('training', 'playoff' if args.stage == 'final' else 'validation'))
    except (KeyError, ValueError, OSError) as exc:
        parser.exit(1, f'Masked experiment error: {exc}\n')


if __name__ == '__main__':
    main()
