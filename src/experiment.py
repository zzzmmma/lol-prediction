"""Run five seeds through the shared training CLI and summarize final test metrics."""
import json
import statistics
import subprocess
import sys
from datetime import datetime

from src.dataset import TARGETS
from src.metrics import print_metrics
from src.utils import ROOT, prepare_output_dir


SEEDS = (137, 482, 911, 2027, 7643)


def run_experiment(num_layers, dim_feedforward=512, evaluate_training=False, embedding_dim=256):
    prefix = f'layers_{num_layers}'
    if embedding_dim != 256:
        prefix += f'_d{embedding_dim}'
    if dim_feedforward != 512:
        prefix += f'_ffn{dim_feedforward}'
    output = prepare_output_dir(
        ROOT / 'runs' / datetime.now().strftime(f'{prefix}_%Y%m%d_%H%M%S_%f'))
    reports = []
    training_reports = []
    for seed in SEEDS:
        run_dir = output / f'seed_{seed}'
        console_log = output / f'seed_{seed}_console.log'
        command = [sys.executable, '-m', 'src.train', '--num-layers', str(num_layers),
                   '--dim-feedforward', str(dim_feedforward), '--seed', str(seed),
                   '--output-dir', str(run_dir)]
        if embedding_dim != 256:
            command.extend(['--embedding-dim', str(embedding_dim)])
        if evaluate_training:
            command.append('--evaluate-training')
        with console_log.open('w', encoding='utf-8') as stream:
            result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'Training failed for seed {seed}; see {console_log}')
        reports.append(json.loads((run_dir / 'test_metrics.json').read_text(encoding='utf-8')))
        print(f'\nSeed {seed}: final Playoff test evaluation')
        print_metrics(reports[-1], per_class=False)
        if evaluate_training:
            training_report = json.loads((run_dir / 'training_metrics.json').read_text(encoding='utf-8'))
            training_reports.append(training_report)
            print(f'\nSeed {seed}: final training evaluation')
            for task in TARGETS:
                values = training_report['tasks'][task]
                accuracy, macro_f1 = (
                    'N/A' if values[metric] is None else f'{values[metric]:.6f}'
                    for metric in ('accuracy', 'macro_f1'))
                print(f'{task}: Training Accuracy={accuracy} Training Macro F1={macro_f1}')

    # Sample standard deviation across all five seeds (ddof=1); scores are 0..1.
    print('\nPlayoff test results across all five seeds:')
    print('task | Accuracy mean ± std (ddof=1) | Macro F1 mean ± std (ddof=1)')
    for task in TARGETS:
        summaries = []
        for metric in ('accuracy', 'macro_f1'):
            values = [report['tasks'][task][metric] for report in reports]
            summaries.append('N/A' if any(value is None for value in values) else
                             f'{statistics.mean(values):.6f} ± {statistics.stdev(values):.6f}')
        print(f'{task} | {summaries[0]} | {summaries[1]}')

    if evaluate_training:
        print('\nTraining results across all five seeds:')
        print('\ntask | Training Accuracy mean ± std (ddof=1) | Training Macro F1 mean ± std (ddof=1)')
        for task in TARGETS:
            summaries = []
            for metric in ('accuracy', 'macro_f1'):
                values = [report['tasks'][task][metric] for report in training_reports]
                summaries.append('N/A' if any(value is None for value in values) else
                                 f'{statistics.mean(values):.6f} ± {statistics.stdev(values):.6f}')
            print(f'{task} | {summaries[0]} | {summaries[1]}')
