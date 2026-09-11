"""Runtime and portable checkpoint helpers for the baseline."""
import random
from pathlib import Path

import torch

from .dataset import TASK_CLASSES, validate_vocab
from .model import MatchTransformer
from .preprocess import INPUTS

ROOT = Path(__file__).resolve().parents[1]


def select_device(name='auto'):
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    if name == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable; install a CUDA PyTorch build or use --device cpu')
    if name == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS unavailable on this machine')
    return torch.device(name)


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def save_checkpoint(path, model, optimizer, epoch, player_vocab, champion_vocab, run_config):
    path = Path(path)
    payload = {'format_version': 1, 'epoch': epoch, 'model_config': model.config,
               'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
               'player_vocab': player_vocab, 'champion_vocab': champion_vocab,
               'task_classes': {t: list(c) for t, c in TASK_CLASSES.items()},
               'input_columns': list(INPUTS), 'run_config': run_config}
    temporary = path.with_suffix('.pt.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('format_version') != 1 or checkpoint['input_columns'] != list(INPUTS):
        raise ValueError('Checkpoint input schema/version mismatch')
    if checkpoint['task_classes'] != {t: list(c) for t, c in TASK_CLASSES.items()}:
        raise ValueError('Checkpoint task class ordering mismatch; legacy 3-class first_dragon checkpoints '
                         'cannot load into the binary model. Keep their reports and create a new binary run.')
    player, champion = validate_vocab(checkpoint['player_vocab']), validate_vocab(checkpoint['champion_vocab'])
    config = checkpoint['model_config']
    if config['player_vocab_size'] != len(player) or config['champion_vocab_size'] != len(champion):
        raise ValueError('Checkpoint vocabulary size mismatch')
    model = MatchTransformer(**config)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model.to(device), checkpoint


def prepare_output_dir(path):
    path = Path(path).resolve()
    if path.is_relative_to((ROOT / 'data').resolve()):
        raise ValueError('Training/evaluation outputs must be outside data/')
    if path.exists() and any(path.iterdir()):
        raise ValueError(f'Output directory is not empty: {path}; choose a new run directory')
    path.mkdir(parents=True, exist_ok=True)
    return path
