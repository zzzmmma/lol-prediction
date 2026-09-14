"""Run three seeds through the shared training CLI and summarize final test metrics."""
import json
import statistics
import subprocess
import sys
from datetime import datetime

from src.dataset import TARGETS
from src.utils import ROOT, prepare_output_dir


SEEDS = (42, 43, 44)


def run_experiment(num_layers):
    output = prepare_output_dir(
        ROOT / 'runs' / datetime.now().strftime(f'layers_{num_layers}_%Y%m%d_%H%M%S_%f'))
    reports = []
    for seed in SEEDS:
        run_dir = output / f'seed_{seed}'
        console_log = output / f'seed_{seed}_console.log'
        command = [sys.executable, '-m', 'src.train', '--num-layers', str(num_layers),
                   '--dim-feedforward', '512', '--seed', str(seed),
                   '--output-dir', str(run_dir)]
        with console_log.open('w', encoding='utf-8') as stream:
            result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'Training failed for seed {seed}; see {console_log}')
        reports.append(json.loads((run_dir / 'test_metrics.json').read_text(encoding='utf-8')))

    # Sample standard deviation across all three seeds (ddof=1); scores are 0..1.
    print('task | Accuracy mean ± std (ddof=1) | Macro F1 mean ± std (ddof=1)')
    for task in TARGETS:
        summaries = []
        for metric in ('accuracy', 'macro_f1'):
            values = [report['tasks'][task][metric] for report in reports]
            summaries.append('N/A' if any(value is None for value in values) else
                             f'{statistics.mean(values):.6f} ± {statistics.stdev(values):.6f}')
        print(f'{task} | {summaries[0]} | {summaries[1]}')
