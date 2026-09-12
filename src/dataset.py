"""Encode only the 20 roster/champion slots using the saved train vocabularies."""
import json
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .preprocess import INPUTS, TARGETS, TARGET_VALUES, read_games

IGNORE_INDEX = -100
TASK_CLASSES = {task: tuple(c for c in TARGET_VALUES[task] if c != 'N/A') for task in TARGETS}
TASK_CLASSES['first_dragon_side'] = ('BLUE', 'RED')
PLAYER_COLUMNS = INPUTS[:10]
CHAMPION_COLUMNS = INPUTS[10:]


def validate_vocab(vocab):
    if not isinstance(vocab, dict) or vocab.get('<UNK>') != 0:
        raise ValueError('Vocabulary must be an object with <UNK>=0')
    if any(not isinstance(k, str) or not k.strip() for k in vocab):
        raise ValueError('Vocabulary contains an empty/non-string name')
    if any(type(v) is not int for v in vocab.values()) or sorted(vocab.values()) != list(range(len(vocab))):
        raise ValueError('Vocabulary IDs must be unique contiguous integers starting at 0')
    return vocab


def load_vocab(path):
    return validate_vocab(json.loads(Path(path).read_text(encoding='utf-8-sig')))


def encode_label(value, task):
    value = (value or '').strip()
    if value in ('', 'MISSING', 'N/A'):
        return IGNORE_INDEX
    # A real no-capture game is outside the binary task, not a BLUE/RED label.
    if task == 'first_dragon_side' and value == 'NONE':
        return IGNORE_INDEX
    if value not in TASK_CLASSES[task]:
        raise ValueError(f'Invalid {task} class: {value!r}')
    return TASK_CLASSES[task].index(value)


def compute_class_weights(train_labels):
    """Inverse square-root frequency, fitted only on valid training labels.

    Observed class weight = 1 / sqrt(count). winner_side is excluded.
    Absent classes get 0;
    an entirely masked task gets an all-zero vector, without dividing by zero.
    """
    weights = {}
    for i, task in enumerate(TARGETS):
        if task == 'winner_side':
            continue
        labels = train_labels[:, i]
        counts = torch.bincount(labels[labels != IGNORE_INDEX], minlength=len(TASK_CLASSES[task])).float()
        observed = counts > 0
        weight = torch.zeros_like(counts)
        if bool(observed.any()):
            weight[observed] = counts[observed].rsqrt()
        weights[task] = weight
    return weights


class GameDataset(Dataset):
    def __init__(self, path, player_vocab, champion_vocab, expected_stage):
        self.path = Path(path)
        self.player_vocab = validate_vocab(player_vocab)
        self.champion_vocab = validate_vocab(champion_vocab)
        self.rows = read_games(path)
        if not self.rows:
            raise ValueError(f'Empty dataset: {path}')
        required = {'game_id', 'stage', *INPUTS, *TARGETS}
        for row in self.rows:
            if required - row.keys():
                raise ValueError(f'{path}: missing columns {sorted(required - row.keys())}')
            if not row['game_id'] or row['stage'] != expected_stage:
                raise ValueError(f'{path}: expected {expected_stage} with a nonempty game_id, got {row["game_id"]} {row["stage"]}')
            if any(not (row[c] or '').strip() for c in INPUTS):
                raise ValueError(f'{row["game_id"]}: missing player/champion slot')
        self.game_ids = [r['game_id'] for r in self.rows]
        if len(set(self.game_ids)) != len(self.game_ids):
            raise ValueError(f'{path}: duplicate game_id')
        self.player_ids = torch.tensor([[player_vocab.get(r[c], 0) for c in PLAYER_COLUMNS]
                                        for r in self.rows], dtype=torch.long)
        self.champion_ids = torch.tensor([[champion_vocab.get(r[c], 0) for c in CHAMPION_COLUMNS]
                                          for r in self.rows], dtype=torch.long)
        self.labels = torch.tensor([[encode_label(r[t], t) for t in TARGETS] for r in self.rows], dtype=torch.long)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return {'player_ids': self.player_ids[index], 'champion_ids': self.champion_ids[index],
                'labels': self.labels[index]}

    def summary(self):
        return {'games': len(self), 'stage': self.rows[0]['stage'],
                'player_oov_slots': int((self.player_ids == 0).sum()),
                'champion_oov_slots': int((self.champion_ids == 0).sum()),
                'targets': {task: {
                    'valid': int((self.labels[:, i] != IGNORE_INDEX).sum()),
                    'missing': sum((r[task] or '').strip() in ('', 'MISSING') for r in self.rows),
                    'not_applicable': sum((r[task] or '').strip() == 'N/A' for r in self.rows),
                    'excluded_nonbinary': sum(task == 'first_dragon_side' and (r[task] or '').strip() == 'NONE'
                                               for r in self.rows),
                    'classes': dict(Counter((r[task] or '').strip() for r in self.rows
                                             if encode_label(r[task], task) != IGNORE_INDEX)),
                } for i, task in enumerate(TARGETS)}}


def validate_training_inputs(train, test):
    overlap = set(train.game_ids) & set(test.game_ids)
    if overlap:
        raise ValueError(f'Train/test overlap: {len(overlap)} games')
    # Refuse a processed/all-season vocabulary, even if it has no test OOVs.
    for columns, vocab in ((PLAYER_COLUMNS, train.player_vocab), (CHAMPION_COLUMNS, train.champion_vocab)):
        observed = {r[c] for r in train.rows for c in columns}
        if set(vocab) != observed | {'<UNK>'}:
            raise ValueError('Vocabulary must contain exactly the train names plus <UNK>; use data/splits vocabularies')
    if not bool((train.labels != IGNORE_INDEX).any()):
        raise ValueError('Training dataset has no supervised targets')
