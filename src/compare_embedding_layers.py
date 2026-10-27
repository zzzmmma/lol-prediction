"""Summarize six existing experiment directories; never train or change source runs."""
import argparse
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from src.dataset import GameDataset, TARGETS, TASK_CLASSES
from src.evaluate import evaluate
from src.experiment import SEEDS
from src.preprocess import fingerprint, write_csv, write_json
from src.utils import ROOT, load_checkpoint, prepare_output_dir, select_device


VARIANTS = ((256, 6), (16, 6), (256, 8), (256, 10), (16, 8), (16, 10))


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def training_report(run, config, output, device):
    path = run / 'training_metrics.json'
    if path.exists():
        return read_json(path)
    # Older 6L runs only saved online training metrics. Evaluate the final
    # checkpoint instead; online metrics include dropout and changing weights.
    dataset_path = Path(config['arguments']['train'])
    source = fingerprint(dataset_path)
    if source['sha256'] != config['sources']['train']['sha256']:
        raise ValueError(f'Training data changed since this run: {dataset_path}')
    model, checkpoint = load_checkpoint(run / 'last.pt', device)
    dataset = GameDataset(dataset_path, checkpoint['player_vocab'],
                          checkpoint['champion_vocab'], 'Regular Season')
    print(f'Evaluating missing final Training metrics: {run}', flush=True)
    report = evaluate(model, DataLoader(dataset, batch_size=config['arguments']['batch_size'],
                                       shuffle=False), device)
    report['metadata'] = {'checkpoint': fingerprint(run / 'last.pt'), 'dataset': source,
                          'epoch': checkpoint['epoch'], 'device': str(device),
                          'evaluation_mode': 'Final model in eval mode over complete training data'}
    write_json(output / 'training_metrics.json', report)
    return report


def summarize(run_dirs, output, device, allow_different_test=False):
    rows, seed_rows = [], []
    reference = None
    test_sources = {}
    for dimensions, layers in VARIANTS:
        label = f'{dimensions}D/{layers}L'
        root = run_dirs[dimensions, layers].resolve()
        reports = {'Training': [], 'Playoff': []}
        for seed in SEEDS:
            run = root / f'seed_{seed}'
            config = read_json(run / 'config.json')
            model, args = config['model'], config['arguments']
            expected = {'embedding_dim': 256, 'num_layers': layers, 'num_heads': 8,
                        'feedforward_dim': 512, 'dropout': 0.1}
            if (any(model.get(k) != v for k, v in expected.items())
                    or model.get('token_embedding_dim', 256) != dimensions or args['seed'] != seed):
                raise ValueError(f'Model/seed configuration mismatch: {run}')
            # Reject mixed splits, task definitions or optimization settings.
            signature = {
                'sources': {k: v['sha256'] for k, v in config['sources'].items()},
                'input_columns': config['input_columns'], 'task_classes': config['task_classes'],
                'task_loss_weights': config['task_loss_weights'],
                'class_weighting': config['class_weighting'],
                'training': {k: args[k] for k in ('epochs', 'batch_size', 'lr', 'weight_decay',
                                                'grad_clip', 'class_weight')},
            }
            test_sources[str(run)] = {'sha256': signature['sources'].pop('test'),
                                      'games': config['dataset_summary']['test']['games']}
            if reference is None:
                reference = signature
            if signature != reference:
                raise ValueError(f'Incomparable training settings or datasets: {run}')
            if (not allow_different_test and len({s['sha256'] for s in test_sources.values()}) > 1):
                raise ValueError('Playoff datasets differ across runs. Use --allow-different-test '
                                 'only to display existing results with a comparison warning.')
            if config['task_classes'] != {t: list(c) for t, c in TASK_CLASSES.items()}:
                raise ValueError(f'Task class definitions differ from current code: {run}')
            reports['Playoff'].append(read_json(run / 'test_metrics.json'))
            reports['Training'].append(training_report(
                run, config, output / f'emb{dimensions}_{layers}l' / f'seed_{seed}', device))
            for split in reports:
                for task in TARGETS:
                    values = reports[split][-1]['tasks'][task]
                    seed_rows.append(dict(model=label, seed=seed, split=split, task=task,
                                          accuracy=values['accuracy'], macro_f1=values['macro_f1'],
                                          source_run=str(run)))
        for split, split_reports in reports.items():
            for task in TARGETS:
                item = dict(model=label, split=split, task=task, seeds=len(SEEDS), source_run=str(root))
                for metric in ('accuracy', 'macro_f1'):
                    values = [r['tasks'][task][metric] for r in split_reports]
                    valid = all(v is not None for v in values)
                    item[f'{metric}_mean'] = statistics.mean(values) if valid else None
                    item[f'{metric}_std'] = statistics.stdev(values) if valid else None
                rows.append(item)
    winners = [r for r in rows if r['task'] == 'winner_side']
    for name, data in (('all_tasks', rows), ('winner_summary', winners), ('per_seed', seed_rows)):
        write_csv(output / f'{name}.csv', data, tuple(data[0]))
    warning = ('WARNING: Playoff datasets differ; these scores are not a controlled model comparison.'
               if len({s['sha256'] for s in test_sources.values()}) > 1 else None)
    write_json(output / 'summary.json', {'seeds': list(SEEDS), 'ddof': 1, 'score_range': [0, 1],
                                         'warning': warning, 'test_sources': test_sources,
                                         'all_tasks': rows, 'winner_summary': winners})
    def formatted(row, metric):
        mean, std = row[f'{metric}_mean'], row[f'{metric}_std']
        return 'N/A' if mean is None else f'{mean:.6f} ± {std:.6f}'
    lines = ['All scores are 0..1; standard deviation uses ddof=1.', '',
             'Winner Prediction comparison',
             'Model | Split | Accuracy mean ± std | Macro F1 mean ± std']
    if warning:
        lines.insert(0, warning)
        for run, source in test_sources.items():
            lines.insert(1, f"Playoff source: {run}: games={source['games']}, sha256={source['sha256']}")
    for row in winners:
        lines.append(f"{row['model']} | {row['split']} | {formatted(row, 'accuracy')} | {formatted(row, 'macro_f1')}")
    lines.extend(['', 'All tasks', 'Model | Split | Task | Accuracy mean ± std | Macro F1 mean ± std'])
    for row in rows:
        lines.append(f"{row['model']} | {row['split']} | {row['task']} | {formatted(row, 'accuracy')} | {formatted(row, 'macro_f1')}")
    text = '\n'.join(lines) + '\n'
    (output / 'summary.txt').write_text(text, encoding='utf-8')
    print(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for dimensions, layers in VARIANTS:
        parser.add_argument(f'--emb{dimensions}-{layers}l', type=Path, required=True,
                            help='Existing experiment root containing all five seed directories')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--device', choices=('auto', 'cuda', 'mps', 'cpu'), default='auto')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--allow-different-test', action='store_true',
                        help='Display differing saved Playoff datasets with an explicit warning')
    args = parser.parse_args()
    if args.threads <= 0:
        parser.error('--threads must be positive')
    torch.set_num_threads(args.threads)
    run_dirs = {(d, l): getattr(args, f'emb{d}_{l}l') for d, l in VARIANTS}
    output = prepare_output_dir(args.output_dir or ROOT / 'runs' / datetime.now().strftime(
        'embedding_layers_comparison_%Y%m%d_%H%M%S_%f'))
    summarize(run_dirs, output, select_device(args.device), args.allow_different_test)
    print(f'Results: {output}')


if __name__ == '__main__':
    main()
