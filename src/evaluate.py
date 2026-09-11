"""Evaluate a saved baseline checkpoint on Playoff data without training."""
import argparse
import sys
from datetime import datetime
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from src.dataset import GameDataset
from src.metrics import MetricAccumulator, class_rows, metric_rows, print_metrics
from src.preprocess import fingerprint, write_csv, write_json
from src.utils import ROOT, load_checkpoint, prepare_output_dir, select_device


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    metrics = MetricAccumulator()
    for batch in loader:
        player = batch['player_ids'].to(device)
        champion = batch['champion_ids'].to(device)
        labels = batch['labels'].to(device)
        metrics.update(model(player, champion), labels)
    return metrics.compute()


def save_evaluation(report, output_dir, metadata):
    output_dir = Path(output_dir)
    write_json(output_dir / 'test_metrics.json', {'metadata': metadata, **report})
    tasks = metric_rows(report, metadata['epoch'], 'test')
    classes = class_rows(report)
    write_csv(output_dir / 'test_task_metrics.csv', tasks, tuple(tasks[0]))
    write_csv(output_dir / 'test_class_metrics.csv', classes, tuple(classes[0]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--test', type=Path, default=ROOT / 'data/splits/test.csv')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', choices=('auto', 'cuda', 'mps', 'cpu'), default='auto')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.threads <= 0:
        parser.error('batch-size and threads must be positive')
    torch.set_num_threads(args.threads)
    try:
        device = select_device(args.device)
        model, checkpoint = load_checkpoint(args.checkpoint, device)
        dataset = GameDataset(args.test, checkpoint['player_vocab'], checkpoint['champion_vocab'], 'Playoff')
        output = prepare_output_dir(args.output_dir or ROOT / 'runs' / datetime.now().strftime('evaluation_%Y%m%d_%H%M%S_%f'))
        report = evaluate(model, DataLoader(dataset, batch_size=args.batch_size, shuffle=False), device)
        metadata = {'checkpoint': fingerprint(args.checkpoint.resolve()), 'dataset': fingerprint(args.test.resolve()),
                    'epoch': checkpoint['epoch'], 'device': str(device), 'dataset_summary': dataset.summary(),
                    'macro_f1_definition': 'Unweighted mean over all configured classes; zero_division=0; all-masked task=null'}
        save_evaluation(report, output, metadata)
    except (ValueError, OSError) as exc:
        parser.exit(1, f'Evaluation error: {exc}\n')
    print_metrics(report)
    print(f'Reports saved: {output}')


if __name__ == '__main__':
    main()
