"""Read-only dataset EDA. Uses the existing schema and Python standard library."""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Direct execution (including VS Code Run Python File) starts sys.path in src/.
if __package__ in (None, ''):
    sys.path.insert(0, str(ROOT))

from src.preprocess import (COLUMNS, INPUTS, TARGETS, TARGET_VALUES, fingerprint,
                            read_games, validate, write_json)


def ratio(numerator, denominator):
    """An empty population has an undefined rate, represented as JSON null."""
    return numerator / denominator if denominator else None


def label(value):
    return 'MISSING' if value is None or not value.strip() else value


def target_distribution(rows, target):
    observed = Counter(label(row[target]) for row in rows)
    classes = list(dict.fromkeys((*TARGET_VALUES[target], 'N/A', 'MISSING', *sorted(observed))))
    counts = {name: observed[name] for name in classes}
    valid = [name for name in TARGET_VALUES[target] if name != 'N/A']
    known = sum(observed[name] for name in valid)
    return {
        'total_games': len(rows),
        'known_applicable_games': known,
        'counts': counts,
        'proportions': {name: ratio(n, len(rows)) for name, n in counts.items()},
        'known_applicable_proportions': {name: ratio(observed[name], known) for name in valid},
        'missing_rate': ratio(observed['MISSING'], len(rows)),
        'na_rate': ratio(observed['N/A'], len(rows)),
        'invalid_class_counts': {name: n for name, n in observed.items()
                                 if name not in (*TARGET_VALUES[target], 'MISSING')},
    }


def appearances(rows, kind):
    columns = [column for column in INPUTS if f'_{kind}_' in column]
    counts = Counter(row[c] for row in rows for c in columns if label(row[c]) != 'MISSING')
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def summarize(rows):
    ids = Counter(row['game_id'] for row in rows)
    try:
        validate(rows)
        validation_error = None
    except (ValueError, KeyError, TypeError, OverflowError) as exc:
        # An EDA report should expose anomalies rather than repair or drop them.
        validation_error = str(exc)
    return {
        'total_games': len(rows),
        'games_by_year': dict(sorted(Counter(label(r['year']) for r in rows).items())),
        'games_by_stage': dict(sorted(Counter(label(r['stage']) for r in rows).items())),
        'games_by_year_stage': [dict(year=y, stage=s, games=n) for (y, s), n in sorted(
            Counter((label(r['year']), label(r['stage'])) for r in rows).items())],
        'player_appearances': appearances(rows, 'player'),
        'champion_appearances': appearances(rows, 'champion'),
        'targets': {target: target_distribution(rows, target) for target in TARGETS},
        'checks': {
            'existing_validator_first_error': validation_error,
            'duplicate_game_ids': {key: n for key, n in ids.items() if n > 1},
            'missing_by_column': {c: sum(label(row[c]) == 'MISSING' for row in rows) for c in COLUMNS},
            'duplicate_tokens_within_game': {
                kind: [row['game_id'] for row in rows
                       if any(n > 1 for n in Counter(row[c] for c in INPUTS
                              if f'_{kind}_' in c and label(row[c]) != 'MISSING').values())]
                for kind in ('player', 'champion')},
        },
    }


def compare_stages(rows):
    regular = [row for row in rows if row['stage'] == 'Regular Season']
    playoff = [row for row in rows if row['stage'] == 'Playoff']
    result = {'source': 'games', 'regular_games': len(regular), 'playoff_games': len(playoff), 'targets': {}}
    for target in TARGETS:
        a, b = target_distribution(regular, target), target_distribution(playoff, target)
        differences = {}
        for scope in ('proportions', 'known_applicable_proportions'):
            differences[scope] = {}
            for name in dict.fromkeys((*a[scope], *b[scope])):
                # An unobserved invalid class has zero share in a nonempty group.
                left = a[scope].get(name, 0.0 if regular else None)
                right = b[scope].get(name, 0.0 if playoff else None)
                differences[scope][name] = 100 * (right - left) if left is not None and right is not None else None
        result['targets'][target] = {'regular': a, 'playoff': b,
                                     'playoff_minus_regular_percentage_points': differences}
    return result


def vocabulary_overlap(train_counts, test_counts):
    train, test = set(train_counts), set(test_counts)
    shared, test_only = train & test, test - train
    unseen_slots = sum(test_counts[token] for token in test_only)
    return {
        'train_unique': len(train), 'test_unique': len(test), 'shared_count': len(shared),
        'union_count': len(train | test), 'jaccard_overlap': ratio(len(shared), len(train | test)),
        'test_vocabulary_coverage': ratio(len(shared), len(test)),
        'train_only_count': len(train - test), 'test_only_count': len(test_only),
        'shared_tokens': sorted(shared), 'train_only_tokens': sorted(train - test),
        'test_only_tokens': sorted(test_only),
        'test_only_appearances': {token: test_counts[token] for token in sorted(test_only)},
        'test_oov_slots': unseen_slots,
        'test_oov_slot_rate': ratio(unseen_slots, sum(test_counts.values())),
    }


def analyze(paths):
    datasets, sources = {}, {}
    for name, path in paths.items():
        # Inspect the header even when the CSV has no data rows.
        with Path(path).open(encoding='utf-8-sig', newline='') as stream:
            header = next(csv.reader(stream), [])
        missing = set(COLUMNS) - set(header)
        if missing or len(header) != len(set(header)):
            raise ValueError(f'{path}: missing columns {sorted(missing)} or duplicate column names')
        datasets[name] = read_games(path)
        sources[name] = fingerprint(Path(path).resolve())
    summaries = {name: summarize(rows) for name, rows in datasets.items()}
    ids = {name: {row['game_id'] for row in rows} for name, rows in datasets.items()}
    by_id = {row['game_id']: row for row in datasets['games']}
    return {
        'sources': sources,
        'definitions': {
            'game': 'One CSV row is one game/set; rows are not filtered or deduplicated.',
            'missing': 'Empty/whitespace cells and literal MISSING; N/A and NONE stay distinct.',
            'proportions': 'Class count / all rows in the corresponding dataset or stage; rates are 0..1.',
            'known_applicable_proportions': 'Exclude MISSING, N/A and invalid labels; retain NONE and TIE where valid.',
            'stage_difference': 'Playoff minus Regular Season in percentage points, using games.csv stages, not split filenames.',
            'appearances': 'Counts over the ten player or champion slots. Duplicate tokens are flagged, not removed.',
            'vocabulary': 'Observed names in train/test CSV slots, not vocabulary JSON IDs; no alias merging or UNK insertion.',
            'undefined_rate': 'JSON null / console -- when the denominator is zero.',
        },
        'datasets': summaries,
        'regular_vs_playoff': compare_stages(datasets['games']),
        'vocabulary_overlap': {kind: vocabulary_overlap(summaries['train'][f'{kind}_appearances'],
                                                        summaries['test'][f'{kind}_appearances'])
                               for kind in ('player', 'champion')},
        'split_checks': {
            'train_test_overlap_game_ids': sorted(ids['train'] & ids['test']),
            'split_ids_absent_from_games': {name: sorted(ids[name] - ids['games']) for name in ('train', 'test')},
            'split_rows_differing_from_games': {
                name: [row['game_id'] for row in datasets[name]
                       if row['game_id'] in by_id and row != by_id[row['game_id']]] for name in ('train', 'test')},
        },
    }


def percent(value):
    return '--' if value is None else f'{value * 100:.2f}%'


def print_report(report, top=20):
    print('EDA (read only): proportions include all games; -- means undefined.')
    for name, data in report['datasets'].items():
        print(f'\n[{name}] {data["total_games"]:,} games')
        print('By year:', data['games_by_year'])
        print('By stage:', data['games_by_stage'])
        print('By year/stage:', data['games_by_year_stage'])
        for kind in ('player', 'champion'):
            items = list(data[f'{kind}_appearances'].items())
            shown = items[:top] if top else items
            print(f'{kind} appearances ({len(items)} unique; showing {len(shown)}, full list in JSON):')
            for token, count in shown:
                print(f'  {token}: {count}')
        for target, dist in data['targets'].items():
            print(f'{target}: ' + ', '.join(f'{c}={n} ({percent(dist["proportions"][c])})'
                                           for c, n in dist['counts'].items()))
            print(f'  MISSING rate={percent(dist["missing_rate"])}; N/A rate={percent(dist["na_rate"])}')
        checks = data['checks']
        print('Existing validator:', checks['existing_validator_first_error'] or 'OK')
        print('Duplicate game IDs:', checks['duplicate_game_ids'])
        print('Missing columns (nonzero):', {c: n for c, n in checks['missing_by_column'].items() if n})
        print('Duplicate tokens within games:', checks['duplicate_tokens_within_game'])
    comparison = report['regular_vs_playoff']
    print(f'\n[Regular vs Playoff in games.csv] {comparison["regular_games"]} / {comparison["playoff_games"]} games')
    print('Each entry: Regular% / Playoff% / delta in percentage points')
    for target, detail in comparison['targets'].items():
        for scope in ('proportions', 'known_applicable_proportions'):
            entries = []
            for name, delta in detail['playoff_minus_regular_percentage_points'][scope].items():
                change = '--' if delta is None else f'{delta:+.2f} pp'
                left = detail['regular'][scope].get(name, 0.0 if comparison['regular_games'] else None)
                right = detail['playoff'][scope].get(name, 0.0 if comparison['playoff_games'] else None)
                entries.append(f'{name}: {percent(left)} / {percent(right)} / {change}')
            print(f'{target} ({scope}): ' + '; '.join(entries))
    for kind, overlap in report['vocabulary_overlap'].items():
        print(f'\n[{kind} vocabulary] train={overlap["train_unique"]}, test={overlap["test_unique"]}, '
              f'shared={overlap["shared_count"]}, train-only={overlap["train_only_count"]}, test-only={overlap["test_only_count"]}')
        print(f'Jaccard={percent(overlap["jaccard_overlap"])}; test token coverage={percent(overlap["test_vocabulary_coverage"])}; '
              f'test OOV slots={overlap["test_oov_slots"]} ({percent(overlap["test_oov_slot_rate"])})')
        print('Test-only tokens and appearances:', overlap['test_only_appearances'])
    print('\nSplit checks:', report['split_checks'])


def save_report(report, output, paths):
    output = Path(output).resolve()
    if output.suffix.lower() != '.json':
        raise ValueError('Use a .json output path')
    protected = {Path(path).resolve() for path in paths.values()}
    # Keep all existing pipeline data outside the report writer's scope.
    if output in protected or output.is_relative_to((ROOT / 'data').resolve()):
        raise ValueError('Output must be outside data/ and must not overwrite an input file')
    write_json(output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--games', type=Path, default=ROOT / 'data/processed/games.csv')
    parser.add_argument('--train', type=Path, default=ROOT / 'data/splits/train.csv')
    parser.add_argument('--test', type=Path, default=ROOT / 'data/splits/test.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/eda_report.json')
    parser.add_argument('--top', type=int, default=20, help='Console appearance ranking length; 0 prints all. JSON always includes all.')
    args = parser.parse_args()
    if args.top < 0:
        parser.error('--top must be >= 0')
    paths = {'games': args.games, 'train': args.train, 'test': args.test}
    try:
        report = analyze(paths)
        save_report(report, args.output, paths)
    except (OSError, ValueError, csv.Error) as exc:
        parser.exit(1, f'EDA error: {exc}\n')
    print_report(report, args.top)
    print(f'\nJSON saved: {args.output.resolve()}')


if __name__ == '__main__':
    main()
