"""Compare champion embedding cosine similarity by most frequent training role."""

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.analyze_champion_embedding import DEFAULT_RUN_DIR, PROJECT_ROOT, load_champion_similarities


ROLES = ('top', 'jungle', 'mid', 'adc', 'support')
LABELS = ('TOP', 'JG', 'MID', 'ADC', 'SUP')
GROUPS = tuple(f'{label}-{label}' for label in LABELS) + ('same-role', 'different-role')


def load_champion_roles(train_path, champions):
    """Assign each champion its most frequent role; ROLES order breaks ties."""
    columns = [(f'{side}_champion_{role}', role)
               for side in ('blue', 'red') for role in ROLES]
    counts = defaultdict(Counter)
    with train_path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        missing_columns = {column for column, _ in columns} - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f'Missing champion columns: {sorted(missing_columns)}')
        for row in reader:
            for column, role in columns:
                name = (row[column] or '').strip()
                if name and name != '<UNK>':
                    counts[name][role] += 1

    missing_champions = [name for name in champions if not counts[name]]
    if missing_champions:
        raise ValueError(f'No training role for champions: {missing_champions}')
    return {name: max(ROLES, key=lambda role: counts[name][role]) for name in champions}


def role_pair_means(champions, similarities, roles):
    """Use only the upper triangle so self-pairs and duplicate pairs are excluded."""
    left, right = torch.triu_indices(len(champions), len(champions), offset=1)
    values = similarities[left, right].double()
    role_ids = torch.tensor([ROLES.index(roles[name]) for name in champions])
    same = role_ids[left] == role_ids[right]
    masks = {f'{label}-{label}': same & (role_ids[left] == index)
             for index, label in enumerate(LABELS)}
    masks.update({'same-role': same, 'different-role': ~same})
    return {group: (float(values[mask].mean()), int(mask.sum()))
            for group, mask in masks.items() if bool(mask.any())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=DEFAULT_RUN_DIR,
                        help='Directory containing five seed_*/last.pt checkpoints')
    parser.add_argument('--train', type=Path, default=PROJECT_ROOT / 'data/splits/train.csv',
                        help='Training CSV with champion role columns')
    args = parser.parse_args()

    try:
        checkpoints = sorted(args.run_dir.glob('seed_*/last.pt'),
                             key=lambda path: int(path.parent.name.removeprefix('seed_')))
        if len(checkpoints) != 5:
            raise ValueError(f'Expected five seed checkpoints, found {len(checkpoints)} in {args.run_dir}')

        results = []
        reference_champions = None
        for checkpoint_path in checkpoints:
            champions, similarities, epoch = load_champion_similarities(checkpoint_path)
            if reference_champions is None:
                reference_champions = champions
                roles = load_champion_roles(args.train, champions)
                print(f'Role data: {args.train.resolve()}')
                print('Role assignment: most appearances; ties use TOP/JG/MID/ADC/SUP order')
                print('Champions by role: ' + ', '.join(
                    f'{label}={sum(role == value for role in roles.values())}'
                    for value, label in zip(ROLES, LABELS)))
            elif champions != reference_champions:
                raise ValueError('All five checkpoints must use the same champion vocabulary')

            result = role_pair_means(champions, similarities, roles)
            if set(result) != set(GROUPS):
                raise ValueError('At least one role group has no champion pairs')
            results.append(result)
            print(f'\nSeed {checkpoint_path.parent.name.removeprefix("seed_")} (epoch={epoch}):')
            print('Pair group | Mean cosine similarity | Unique pairs')
            for group in GROUPS:
                score, count = result[group]
                print(f'{group} | {score:.6f} | {count}')

        print('\nAcross 5 seeds (mean ± sample standard deviation, ddof=1):')
        print('Pair group | Mean cosine similarity')
        for group in GROUPS:
            scores = [result[group][0] for result in results]
            print(f'{group} | {mean(scores):.6f} ± {stdev(scores):.6f}')
    except (ValueError, OSError, KeyError, RuntimeError, csv.Error) as exc:
        parser.exit(1, f'Champion role similarity analysis error: {exc}\n')


if __name__ == '__main__':
    main()
