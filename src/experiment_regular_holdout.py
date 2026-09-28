"""Five-seed 6L/FFN512 regular-season holdout experiment.

Run from the project root: .venv/bin/python -m src.experiment_regular_holdout
Use --check-only to validate the fixed split without training or saving a run.
"""
import argparse
import random
import statistics
import sys
import tempfile
from datetime import datetime
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from src.dataset import (CLASS_WEIGHT_POWERS, IGNORE_INDEX, TASK_CLASSES, TASK_LOSS_WEIGHTS, TARGETS,
                         GameDataset, compute_class_weights, validate_training_inputs)
from src.evaluate import evaluate
from src.experiment import SEEDS
from src.metrics import print_metrics
from src.model import MatchTransformer
from src.preprocess import build_vocab, fingerprint, read_games, write_csv, write_json
from src.train import train_epoch
from src.utils import ROOT, prepare_output_dir, save_checkpoint, seed_everything, select_device


def prepare_split(source, directory, split_seed):
    rows = read_games(source)
    if not rows:
        raise ValueError(f'Empty source: {source}')
    required = {'game_id', 'stage', 'winner_side'}
    if required - rows[0].keys():
        raise ValueError(f'Missing source columns: {sorted(required - rows[0].keys())}')
    regular = [row for row in rows if row['stage'] == 'Regular Season'
               and row['winner_side'] in ('BLUE', 'RED')]
    ids = [row['game_id'] for row in regular]
    if not ids or any(not game_id for game_id in ids) or len(set(ids)) != len(ids):
        raise ValueError('Eligible Regular Season games need unique nonempty game_id values')
    shuffled = sorted(ids)
    random.Random(split_seed).shuffle(shuffled)
    train_count = int(len(shuffled) * 0.8)
    if train_count == 0 or train_count == len(shuffled):
        raise ValueError('Need at least two eligible Regular Season games')
    train_ids = set(shuffled[:train_count])
    train_rows = [row for row in regular if row['game_id'] in train_ids]
    test_rows = [row for row in regular if row['game_id'] not in train_ids]
    directory = Path(directory)
    write_csv(directory / 'train.csv', train_rows, tuple(rows[0]))
    write_csv(directory / 'test.csv', test_rows, tuple(rows[0]))
    player_vocab = build_vocab(train_rows, 'player')
    champion_vocab = build_vocab(train_rows, 'champion')
    write_json(directory / 'player_vocab.json', player_vocab)
    write_json(directory / 'champion_vocab.json', champion_vocab)
    train = GameDataset(directory / 'train.csv', player_vocab, champion_vocab, 'Regular Season')
    test = GameDataset(directory / 'test.csv', player_vocab, champion_vocab, 'Regular Season')
    validate_training_inputs(train, test)
    manifest = {'source': fingerprint(Path(source).resolve()), 'split_seed': split_seed,
                'unit': 'game_id', 'train_fraction': 0.8, 'train_games': len(train),
                'test_games': len(test), 'vocabulary_scope': 'train_only',
                'training_seeds': list(SEEDS)}
    write_json(directory / 'split_manifest.json', manifest)
    return train, test, manifest


def summarize(reports):
    summary = {}
    for task in TARGETS:
        summary[task] = {}
        for split in ('training', 'test'):
            summary[task][split] = {}
            for metric in ('accuracy', 'macro_f1'):
                values = [report[split]['tasks'][task][metric] for report in reports]
                summary[task][split][metric] = (None if any(v is None for v in values) else
                    {'mean': statistics.mean(values), 'std_ddof1': statistics.stdev(values)})
    return summary


def run(args):
    if args.epochs <= 0 or args.batch_size <= 0 or args.threads <= 0:
        raise ValueError('epochs, batch-size and threads must be positive')
    if args.check_only:
        with tempfile.TemporaryDirectory(prefix='regular_holdout_') as temporary:
            train, test, manifest = prepare_split(args.source, temporary, args.split_seed)
            print(f"Split validated: {manifest['train_games']} train / {manifest['test_games']} test games; "
                  f'overlap=0, train-only vocabularies, split seed={args.split_seed}')
            print(f'Targets: {len(TARGETS)}; train valid labels: '
                  f'{int((train.labels != IGNORE_INDEX).sum())}; '
                  f'test valid labels: {int((test.labels != IGNORE_INDEX).sum())}')
        return

    torch.set_num_threads(args.threads)
    device = select_device(args.device)
    output = prepare_output_dir(args.output_dir or ROOT / 'runs' /
                                datetime.now().strftime('regular_holdout_6l_%Y%m%d_%H%M%S_%f'))
    train, test, manifest = prepare_split(args.source, output / 'split', args.split_seed)
    print(f"Regular holdout: {manifest['train_games']} train / {manifest['test_games']} test games; "
          f'split seed={args.split_seed}; device={device}', flush=True)
    weights = {task: weight.to(device) for task, weight in compute_class_weights(train.labels).items()}
    config = {'model': {'num_layers': 6, 'embedding_dim': 256, 'num_heads': 8,
                        'feedforward_dim': 512, 'dropout': 0.1},
              'optimizer': {'name': 'AdamW', 'lr': 3e-4, 'weight_decay': 0.01,
                            'grad_clip': 1.0, 'epochs': args.epochs, 'batch_size': args.batch_size},
              'class_weighting': {'enabled': True, 'powers': CLASS_WEIGHT_POWERS,
                                  'weights': {task: value.cpu().tolist() for task, value in weights.items()}},
              'task_classes': {task: list(classes) for task, classes in TASK_CLASSES.items()},
              'task_loss_weights': TASK_LOSS_WEIGHTS,
              'evaluation': 'Final model in eval mode on full train and held-out Regular Season test; fixed-class macro F1',
              'split_manifest': manifest}
    write_json(output / 'config.json', config)
    reports = []
    for seed in SEEDS:
        seed_everything(seed)
        model = MatchTransformer(len(train.player_vocab), len(train.champion_vocab),
                                 num_layers=6, embedding_dim=256, num_heads=8,
                                 feedforward_dim=512, dropout=0.1).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
        generator = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                                  generator=generator, num_workers=0)
        for epoch in range(1, args.epochs + 1):
            epoch_report = train_epoch(model, train_loader, optimizer, device, 1.0, weights)
            print(f'Seed {seed} epoch {epoch}/{args.epochs}: '
                  f"optimization_loss={epoch_report['optimization_loss']}", flush=True)
        eval_train = DataLoader(train, batch_size=args.batch_size, shuffle=False, num_workers=0)
        eval_test = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=0)
        training_report = evaluate(model, eval_train, device)
        test_report = evaluate(model, eval_test, device)
        run_dir = output / f'seed_{seed}'
        run_dir.mkdir()
        write_json(run_dir / 'training_metrics.json', training_report)
        write_json(run_dir / 'test_metrics.json', test_report)
        save_checkpoint(run_dir / 'last.pt', model, optimizer, args.epochs,
                        train.player_vocab, train.champion_vocab, config)
        reports.append({'seed': seed, 'training': training_report, 'test': test_report})
        print(f'\nSeed {seed} training:', flush=True)
        print_metrics(training_report, per_class=False)
        print(f'Seed {seed} Regular holdout test:', flush=True)
        print_metrics(test_report, per_class=False)

    summary = summarize(reports)
    write_json(output / 'summary.json', {'split': manifest, 'seeds': list(SEEDS), 'tasks': summary})
    def display(value):
        return 'N/A' if value is None else f"{value['mean']:.6f} ± {value['std_ddof1']:.6f}"
    print('\nAcross five training seeds (mean ± standard deviation, ddof=1):')
    print('task | Training Accuracy | Training Macro F1 | Regular holdout Test Accuracy | Regular holdout Test Macro F1')
    for task, values in summary.items():
        print(f"{task} | {display(values['training']['accuracy'])} | "
              f"{display(values['training']['macro_f1'])} | "
              f"{display(values['test']['accuracy'])} | {display(values['test']['macro_f1'])}")
    print(f'Output: {output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'data/splits/train.csv',
                        help='Regular Season source; defaults to the original experiment training games')
    parser.add_argument('--split-seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', choices=('auto', 'cuda', 'mps', 'cpu'), default='auto')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--check-only', action='store_true', help='Validate split and data without training')
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, OSError) as exc:
        parser.exit(1, f'Experiment error: {exc}\n')


if __name__ == '__main__':
    main()
