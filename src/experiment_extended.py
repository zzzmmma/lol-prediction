"""Complete the five-seed holdout or refit on the full Regular Season."""
import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev

import torch

from .dataset import TARGETS
from .experiment import SEEDS
from .model_emb16 import MatchTransformerEmb16
from .preprocess import write_csv
from .train import run_training
from .train_extended import (BATCH_SIZE, GRAD_CLIP, LR, MAX_EPOCHS, MODEL_OPTIONS, MODELS,
                             OUTPUT, SOURCE, WEIGHT_DECAY, experiment_config,
                             prepare_extended_split, run_seed)
from .utils import ROOT, prepare_output_dir, select_device


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_holdout(run, variant, seed, split):
    report = read_json(run / 'metrics.json')
    config = read_json(run / 'config.json')
    expected = experiment_config(variant, seed, split, config['device'])
    model_class, layers = MODELS[variant]
    model_options = {**MODEL_OPTIONS, 'num_layers': layers}
    token_dim = 16 if model_class is MatchTransformerEmb16 else 256
    if (any(config[key] != value for key, value in expected.items())
            or any(config['model'][key] != value for key, value in model_options.items())
            or config['model'].get('token_embedding_dim', 256) != token_dim):
        raise ValueError(f'Holdout settings or data split differ: {run}')
    if not 1 <= report['best_epoch'] <= report['last_epoch'] <= MAX_EPOCHS:
        raise ValueError(f'Invalid holdout epochs: {run}')
    if (report['training']['samples'] != split['train_games']
            or report['validation']['samples'] != split['validation_games']):
        raise ValueError(f'Holdout sample counts differ: {run}')
    return report


def save_summary(output, results, splits):
    per_seed = []
    grouped = defaultdict(list)
    for variant in MODELS:
        for seed in SEEDS:
            report = results[variant, seed]
            epochs = {key: report[key] for key in ('best_epoch', 'last_epoch')}
            if 'validation' in splits:
                grouped[variant, 'selection', '', 'best_epoch'].append(epochs['best_epoch'])
            for split in splits:
                for task in TARGETS:
                    values = {metric: report[split]['tasks'][task][metric]
                              for metric in ('accuracy', 'macro_f1')}
                    per_seed.append(dict(model=variant, seed=seed, **epochs, split=split, task=task, **values))
                    for metric, value in values.items():
                        grouped[variant, split, task, metric].append(value)

    summary = []
    for (variant, split, task, metric), values in grouped.items():
        average = deviation = None
        if all(value is not None for value in values):
            average, deviation = mean(values), stdev(values)
        summary.append(dict(model=variant, split=split, task=task, metric=metric,
                            seeds=len(values), mean=average, std_ddof1=deviation))
        score = 'N/A' if average is None else f'{average:.6f} ± {deviation:.6f}'
        print(f'{variant} | {split} | {task or "-"} | {metric} | {score}')
    write_csv(output / 'five_seed_per_seed.csv', per_seed, tuple(per_seed[0]))
    write_csv(output / 'five_seed_summary.csv', summary, tuple(summary[0]))
    print(f'Summary: {output / "five_seed_summary.csv"}')


def complete_holdout(args, train, validation, split, results):
    pending = [(variant, seed) for variant in MODELS for seed in SEEDS
               if (variant, seed) not in results]
    if pending:
        torch.set_num_threads(args.threads)
        device = select_device(args.device)
    for variant, seed in pending:
        run = args.holdout_dir / variant / f'seed_{seed}'
        run_seed(run, variant, seed, train, validation, split, device, resume=True)
        results[variant, seed] = read_holdout(run, variant, seed, split)
    save_summary(args.holdout_dir, results, ('training', 'validation'))


def refit_full_season(args, holdout):
    output = prepare_output_dir(ROOT / 'runs/extended_regular_full' /
                                datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    results = {}
    for variant, (model_class, layers) in MODELS.items():
        for seed in SEEDS:
            epochs = holdout[variant, seed]['best_epoch']
            run = output / variant / f'seed_{seed}'
            print(f'{variant} seed {seed} | full Regular Season | {epochs} epochs | {run}', flush=True)
            _, playoff = run_training(argparse.Namespace(
                train=SOURCE, test=SOURCE.with_name('test.csv'),
                player_vocab=SOURCE.with_name('player_vocab.json'),
                champion_vocab=SOURCE.with_name('champion_vocab.json'),
                output_dir=run, evaluate_training=True, epochs=epochs, batch_size=BATCH_SIZE,
                layers=layers, heads=MODEL_OPTIONS['num_heads'],
                embedding_dim=MODEL_OPTIONS['embedding_dim'],
                token_embedding_dim=16 if model_class is MatchTransformerEmb16 else None,
                ff_dim=MODEL_OPTIONS['feedforward_dim'], dropout=MODEL_OPTIONS['dropout'],
                lr=LR, weight_decay=WEIGHT_DECAY, class_weight=True, grad_clip=GRAD_CLIP,
                seed=seed, threads=args.threads, device=args.device,
                validation_metrics=args.holdout_dir / variant / f'seed_{seed}' / 'metrics.json',
            ))
            results[variant, seed] = {
                'best_epoch': epochs, 'last_epoch': epochs, 'playoff': playoff,
                'training': read_json(run / 'training_metrics.json'),
            }
    save_summary(output, results, ('training', 'playoff'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('validation', 'final'))
    parser.add_argument('--holdout-dir', type=Path, default=OUTPUT)
    parser.add_argument('--device', choices=('auto', 'cuda', 'mps', 'cpu'), default='auto')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    args.holdout_dir = args.holdout_dir.resolve()
    if args.threads <= 0 or args.holdout_dir.is_relative_to((ROOT / 'data').resolve()):
        parser.error('Use positive --threads and a holdout directory outside data/')
    try:
        train, validation, split = prepare_extended_split()
        if len(train) + len(validation) != 4880:
            raise ValueError('Expected 4,880 eligible Regular Season games')
        results = {}
        for variant in MODELS:
            for seed in SEEDS:
                run = args.holdout_dir / variant / f'seed_{seed}'
                # Seed 137 is always required and never sent to the holdout trainer.
                if args.stage == 'final' or seed == 137 or (run / 'metrics.json').exists():
                    results[variant, seed] = read_holdout(run, variant, seed, split)
        if args.stage == 'validation':
            complete_holdout(args, train, validation, split, results)
        else:
            refit_full_season(args, results)
    except (KeyError, ValueError, OSError) as exc:
        parser.exit(1, f'Experiment error: {exc}\n')


if __name__ == '__main__':
    main()
