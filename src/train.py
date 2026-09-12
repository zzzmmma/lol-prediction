"""Train the fixed-size Transformer baseline; evaluate Playoff once after training."""
import argparse
import json
import logging
import math
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from src.dataset import GameDataset, TASK_CLASSES, compute_class_weights, load_vocab, validate_training_inputs
from src.evaluate import evaluate, save_evaluation
from src.metrics import MetricAccumulator, metric_rows, print_metrics
from src.model import MatchTransformer, masked_multitask_loss
from src.preprocess import INPUTS, fingerprint, write_csv, write_json
from src.utils import ROOT, prepare_output_dir, save_checkpoint, seed_everything, select_device


def train_epoch(model, loader, optimizer, device, grad_clip, class_weights=None):
    model.train()
    metrics, skipped = MetricAccumulator(), 0
    optimization_losses = []
    for batch in loader:
        player, champion, labels = (batch[k].to(device) for k in ('player_ids', 'champion_ids', 'labels'))
        optimizer.zero_grad(set_to_none=True)
        logits = model(player, champion)
        loss = masked_multitask_loss(logits, labels, class_weights)
        if loss is None:
            skipped += 1
        else:
            if not bool(torch.isfinite(loss)):
                raise ValueError('Non-finite loss; training stopped')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=True)
            optimizer.step()
            optimization_losses.append(loss.detach().item())
        metrics.update(logits, labels)
    report = metrics.compute()
    report['skipped_batches'] = skipped
    report['optimization_loss'] = sum(optimization_losses) / len(optimization_losses) if optimization_losses else None
    return report


def run_training(args):
    if args.epochs <= 0 or args.batch_size <= 0 or args.threads <= 0:
        raise ValueError('epochs, batch-size and threads must be positive')
    if not math.isfinite(args.lr) or args.lr <= 0 or not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError('Use a finite positive learning rate and nonnegative weight decay')
    if not math.isfinite(args.grad_clip) or args.grad_clip <= 0:
        raise ValueError('grad-clip must be finite and positive')
    torch.set_num_threads(args.threads)
    seed_everything(args.seed)
    device = select_device(args.device)
    player_vocab, champion_vocab = load_vocab(args.player_vocab), load_vocab(args.champion_vocab)
    train = GameDataset(args.train, player_vocab, champion_vocab, 'Regular Season')
    test = GameDataset(args.test, player_vocab, champion_vocab, 'Playoff')
    validate_training_inputs(train, test)
    class_weights = ({task: weight.to(device) for task, weight in compute_class_weights(train.labels).items()}
                     if args.class_weight else None)
    model = MatchTransformer(len(player_vocab), len(champion_vocab), num_layers=args.layers,
                             num_heads=args.heads, feedforward_dim=args.ff_dim, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=0)
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=0)
    output = prepare_output_dir(args.output_dir or ROOT / 'runs' / datetime.now().strftime('transformer_%Y%m%d_%H%M%S_%f'))
    config = {
        'arguments': {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'model': model.config, 'parameters': sum(p.numel() for p in model.parameters()),
        'input_columns': list(INPUTS), 'task_classes': {t: list(c) for t, c in TASK_CLASSES.items()},
        'sources': {name: fingerprint(getattr(args, name).resolve()) for name in ('train', 'test', 'player_vocab', 'champion_vocab')},
        'dataset_summary': {'train': train.summary(), 'test': test.summary()},
        'class_weighting': {
            'enabled': args.class_weight,
            'formula': '1 / sqrt(class_count), using valid train labels only; absent classes=0; winner_side unweighted',
            'frequencies': {task: [int((train.labels[:, i] == c).sum()) for c in range(len(classes))]
                            for i, (task, classes) in enumerate(TASK_CLASSES.items())},
            'weights': {task: weight.cpu().tolist() for task, weight in class_weights.items()} if class_weights is not None else None,
        },
        'environment': {'python': platform.python_version(), 'torch': str(torch.__version__),
                        'device': str(device), 'cuda_runtime': torch.version.cuda,
                        'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None},
        'model_selection': 'Final fixed epoch. Test is evaluated once, never used for early stopping or model selection.',
        'loss_definition': 'Mean of per-task CrossEntropy means; weighted means divide by sum of target-class weights when enabled. MISSING/N/A and first_dragon NONE are masked.',
        'reported_loss_definition': 'loss remains unweighted for comparison; optimization_loss is the mean of actual batch objectives (weighted when enabled).',
        'metrics_definition': 'Unweighted Accuracy and fixed-class macro F1 ignore MISSING/N/A and first_dragon NONE. Absent class P/R/F1=0; all-masked task=null. Confusion rows=true, columns=predicted.',
        'train_metrics_definition': 'Online epoch metrics before each batch update, with dropout active; not a final-model evaluation.',
    }
    write_json(output / 'config.json', config)
    write_json(output / 'player_vocab.json', player_vocab)
    write_json(output / 'champion_vocab.json', champion_vocab)
    logger = logging.getLogger(f'baseline.{output.name}')
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(output / 'train.log', encoding='utf-8')]
    for handler in handlers:
        handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(handler)
    history_rows = []
    try:
        logger.info('Device=%s, parameters=%s, train=%s, test=%s', device, config['parameters'], len(train), len(test))
        logger.info('Output: %s', output)
        logger.info('Class weighting: %s', 'enabled (inverse sqrt train frequencies; winner_side unweighted)' if args.class_weight else 'disabled')
        if class_weights is not None:
            logger.info('Class weights in task_classes order: %s', config['class_weighting']['weights'])
        for task, summary in config['dataset_summary']['train']['targets'].items():
            absent = set(TASK_CLASSES[task]) - set(summary['classes'])
            if absent:
                logger.info('Training classes with zero support for %s: %s', task, sorted(absent))
        with (output / 'history.jsonl').open('w', encoding='utf-8') as history:
            for epoch in range(1, args.epochs + 1):
                start = time.perf_counter()
                report = train_epoch(model, train_loader, optimizer, device, args.grad_clip, class_weights)
                seconds = time.perf_counter() - start
                history.write(json.dumps({'epoch': epoch, 'split': 'train', 'seconds': seconds, **report}, allow_nan=False) + '\n')
                history.flush()
                history_rows.extend(metric_rows(report, epoch, 'train'))
                write_csv(output / 'history.csv', history_rows, tuple(history_rows[0]))
                save_checkpoint(output / 'last.pt', model, optimizer, epoch, player_vocab, champion_vocab, config)
                logger.info('Epoch %d/%d: train unweighted loss=%.4f optimization loss=%.4f time=%.1fs skipped=%d',
                            epoch, args.epochs, report['loss'], report['optimization_loss'], seconds, report['skipped_batches'])
                print_metrics(report, logger.info, per_class=False)
            # No test metrics were computed inside the optimization loop.
            report = evaluate(model, test_loader, device)
            history.write(json.dumps({'epoch': args.epochs, 'split': 'test', **report}, allow_nan=False) + '\n')
            history_rows.extend(metric_rows(report, args.epochs, 'test'))
            write_csv(output / 'history.csv', history_rows, tuple(history_rows[0]))
        save_evaluation(report, output, {
            'epoch': args.epochs, 'checkpoint': fingerprint(output / 'last.pt'),
            'dataset': config['sources']['test'], 'dataset_summary': config['dataset_summary']['test'],
            'device': str(device), 'metrics_definition': config['metrics_definition']})
        logger.info('\nFinal Playoff evaluation:')
        print_metrics(report, logger.info)
        logger.info('Checkpoint: %s', output / 'last.pt')
    finally:
        for handler in handlers:
            handler.close()
            logger.removeHandler(handler)
    return output, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, default=ROOT / 'data/splits/train.csv')
    parser.add_argument('--test', type=Path, default=ROOT / 'data/splits/test.csv')
    parser.add_argument('--player-vocab', type=Path, default=ROOT / 'data/splits/player_vocab.json')
    parser.add_argument('--champion-vocab', type=Path, default=ROOT / 'data/splits/champion_vocab.json')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--layers', type=int, choices=(2, 3, 4), default=2)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--ff-dim', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--class-weight', action=argparse.BooleanOptionalAction, default=True,
                        help='Use inverse-sqrt train-frequency weights except winner_side (default on); --no-class-weight disables')
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', choices=('auto', 'cuda', 'mps', 'cpu'), default='auto')
    args = parser.parse_args()
    try:
        run_training(args)
    except (ValueError, OSError) as exc:
        parser.exit(1, f'Training error: {exc}\n')


if __name__ == '__main__':
    main()
