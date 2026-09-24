"""Compare player embedding cosine similarity by role across five 6L / FFN512 seeds."""

import argparse
import csv
import sys
from pathlib import Path
from statistics import mean, stdev

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analyze_player_embedding import (
    load_player_roles,
    load_player_similarities,
    role_similarities,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = PROJECT_ROOT / 'runs/layers_6_20260921_111124_289760'
GROUPS = ('TOP-TOP', 'JG-JG', 'MID-MID', 'ADC-ADC', 'SUP-SUP',
          'same-role', 'different-role')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=DEFAULT_RUN_DIR,
                        help='Directory containing five seed_*/last.pt checkpoints')
    parser.add_argument('--role-data', type=Path, default=PROJECT_ROOT / 'data/splits/train.csv',
                        help='Training CSV containing blue/red player role columns')
    args = parser.parse_args()

    try:
        checkpoints = sorted(args.run_dir.glob('seed_*/last.pt'),
                             key=lambda path: int(path.parent.name.removeprefix('seed_')))
        if len(checkpoints) != 5:
            raise ValueError(f'Expected five seed checkpoints, found {len(checkpoints)} in {args.run_dir}')

        results = []
        reference_players = None
        for checkpoint in checkpoints:
            players, similarities, epoch = load_player_similarities(checkpoint)
            if reference_players is None:
                reference_players = players
                roles = load_player_roles(args.role_data, players)
            elif players != reference_players:
                raise ValueError('All five checkpoints must use the same player vocabulary')

            result = role_similarities(players, similarities, roles)
            results.append(result)
            print(f'\nSeed {checkpoint.parent.name.removeprefix("seed_")} (epoch={epoch}):')
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
        parser.exit(1, f'Role similarity analysis error: {exc}\n')


if __name__ == '__main__':
    main()
