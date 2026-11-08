"""Run the extended Regular Season holdout experiments."""
import argparse
import shutil
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .dataset import CLASS_WEIGHT_POWERS, TASK_LOSS_WEIGHTS, compute_class_weights
from .evaluate import evaluate
from .experiment import SEEDS
from .experiment_regular_holdout import prepare_split
from .metrics import print_metrics
from .model import MatchTransformer
from .model_emb16 import MatchTransformerEmb16
from .preprocess import fingerprint, write_csv, write_json
from .train import train_epoch
from .utils import ROOT, load_checkpoint, save_checkpoint, seed_everything, select_device


MODELS = {'6l_256d': (MatchTransformer, 6), '10l_16d': (MatchTransformerEmb16, 10)}
MODEL_OPTIONS = {'embedding_dim': 256, 'num_heads': 8, 'feedforward_dim': 512, 'dropout': 0.1}
SOURCE = ROOT / 'data/splits/train.csv'
OUTPUT = ROOT / 'runs/extended_regular_holdout'
SPLIT_SEED = 42
MAX_EPOCHS = 100
PATIENCE = 15
BATCH_SIZE = 64
LR = 3e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0


def save_epoch(output, model, optimizer, config, train, history, generator, device):
    rng = {'cpu': torch.get_rng_state(), 'shuffle': generator.get_state()}
    if device.type == 'cuda':
        rng['device'] = torch.cuda.get_rng_state(device)
    elif device.type == 'mps':
        rng['device'] = torch.mps.get_rng_state()
    state = {'config': config, 'history': history, 'rng': rng}
    save_checkpoint(output / 'last.pt', model, optimizer, len(history),
                    train.player_vocab, train.champion_vocab, state)
    if history:
        publish_epoch(output, history)


def publish_epoch(output, history):
    # last.pt is the commit point. Resume can repair a partially published epoch.
    row = history[-1]
    if row['epoch'] == row['best_epoch']:
        temporary = output / 'best.pt.tmp'
        shutil.copyfile(output / 'last.pt', temporary)
        temporary.replace(output / 'best.pt')
    write_csv(output / 'history.csv', history, tuple(row))


def experiment_config(variant, seed, split, device):
    return {
        'variant': variant, 'seed': seed, 'device': str(device), 'split': split,
        'max_epochs': MAX_EPOCHS, 'patience': PATIENCE, 'batch_size': BATCH_SIZE,
        'optimizer': 'AdamW', 'lr': LR, 'weight_decay': WEIGHT_DECAY, 'grad_clip': GRAD_CLIP,
        'selection_metric': 'validation_winner_macro_f1',
        'task_loss_weights': TASK_LOSS_WEIGHTS, 'class_weight_powers': CLASS_WEIGHT_POWERS,
    }


def prepare_extended_split():
    with tempfile.TemporaryDirectory(prefix='extended_holdout_') as directory:
        train, validation, split = prepare_split(SOURCE, directory, SPLIT_SEED)
        split['validation_games'] = split.pop('test_games')
        split['train_sha256'] = fingerprint(train.path)['sha256']
        split['validation_sha256'] = fingerprint(validation.path)['sha256']
    return train, validation, split


def run_seed(output, variant, seed, train, validation, split, device, resume):
    seed_everything(seed)
    config = experiment_config(variant, seed, split, device)
    generator = torch.Generator().manual_seed(seed)
    checkpoint = None
    if resume and output.exists():
        model, checkpoint = load_checkpoint(output / 'last.pt', device)
        if checkpoint['run_config'].get('config') != config:
            raise ValueError(f'Resume settings or data differ from the saved run: {output}')
        if (output / 'metrics.json').exists():
            print(f'{variant} seed {seed} | already complete | {output}', flush=True)
            return
    else:
        output.mkdir(parents=True)
        model_class, layers = MODELS[variant]
        model = model_class(len(train.player_vocab), len(train.champion_vocab),
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
    eval_validation = DataLoader(validation, batch_size=BATCH_SIZE, shuffle=False)

    best_epoch = history[-1]['best_epoch'] if history else 0
    best_f1 = history[best_epoch - 1]['validation_winner_macro_f1'] if best_epoch else -1.0
    for epoch in range(len(history) + 1, MAX_EPOCHS + 1):
        if epoch - 1 - best_epoch >= PATIENCE:
            break
        training = train_epoch(model, train_loader, optimizer, device, GRAD_CLIP, weights)
        train_winner = training['tasks']['winner_side']
        eval_train_winner = evaluate(model, eval_train, device)['tasks']['winner_side']
        val_winner = evaluate(model, eval_validation, device)['tasks']['winner_side']
        val_f1 = val_winner['macro_f1']
        if val_f1 > best_f1:
            best_f1, best_epoch = val_f1, epoch
        history.append({
            'epoch': epoch, 'train_loss': training['loss'],
            'optimization_loss': training['optimization_loss'],
            'train_winner_loss': train_winner['loss'],
            'train_winner_accuracy': train_winner['accuracy'],
            'train_winner_macro_f1': train_winner['macro_f1'],
            'train_eval_winner_accuracy': eval_train_winner['accuracy'],
            'train_eval_winner_macro_f1': eval_train_winner['macro_f1'],
            'validation_winner_accuracy': val_winner['accuracy'],
            'validation_winner_macro_f1': val_f1,
            'lr': optimizer.param_groups[0]['lr'], 'best_epoch': best_epoch,
        })
        save_epoch(output, model, optimizer, config, train, history, generator, device)
        print(f'{variant} seed {seed} | epoch {epoch}/{MAX_EPOCHS} | train loss {training["loss"]:.4f}'
              f' | validation Winner Macro F1 {val_f1:.4f} | best epoch {best_epoch}', flush=True)

    # Use the selected checkpoint for both reports, even when the last epoch differs.
    best_model, best_checkpoint = load_checkpoint(output / 'best.pt', device)
    metrics = {
        'best_epoch': best_checkpoint['epoch'], 'last_epoch': history[-1]['epoch'],
        'stop_reason': ('patience' if history[-1]['epoch'] - best_epoch >= PATIENCE else 'max_epochs'),
        'checkpoint': fingerprint(output / 'best.pt'),
        'training': evaluate(best_model, eval_train, device),
        'validation': evaluate(best_model, eval_validation, device),
    }
    write_json(output / 'metrics.json', metrics)
    print(f'\n{variant} seed {seed} | best epoch {best_epoch} | {metrics["stop_reason"]}')
    for split_name in ('training', 'validation'):
        print(f'{split_name.capitalize()} (model.eval())')
        print_metrics(metrics[split_name], per_class=False)
    print(f'Output: {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', nargs='+', choices=tuple(MODELS), required=True)
    parser.add_argument('--seeds', nargs='+', type=int, choices=SEEDS, default=[137])
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    parser.add_argument('--resume', action='store_true',
                        help='Continue existing seed directories from last.pt and start missing seeds')
    parser.add_argument('--device', choices=('auto', 'cuda', 'mps', 'cpu'), default='auto')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if args.threads <= 0 or output.is_relative_to((ROOT / 'data').resolve()):
        parser.error('Use positive --threads and an output directory outside data/')
    torch.set_num_threads(args.threads)
    try:
        device = select_device(args.device)
        train, validation, split = prepare_extended_split()
        print(f'Split seed {SPLIT_SEED} | Train {len(train)} | Validation {len(validation)}', flush=True)
        for variant in dict.fromkeys(args.model):
            for seed in dict.fromkeys(args.seeds):
                run_seed(output / variant / f'seed_{seed}', variant, seed,
                         train, validation, split, device, args.resume)
    except (ValueError, OSError) as exc:
        parser.exit(1, f'Training error: {exc}\n')


if __name__ == '__main__':
    main()
