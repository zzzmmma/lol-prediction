"""Flexible post-processing splits; no random split or model training."""
from pathlib import Path
from .preprocess import column_roles, build_vocab, fingerprint, read_games, validate, write_csv, write_json


def select(rows, years=None, splits=None, stages=None, date_from=None, date_to=None):
    return [r for r in rows
            if (years is None or int(r['year']) in years)
            and (splits is None or r['split'] in splits)
            and (stages is None or r['stage'] in stages)
            and (date_from is None or r['date'][:10] >= date_from)
            and (date_to is None or r['date'][:10] <= date_to)]


def create_splits(dataset, output_dir, train_filter=None, test_filter=None):
    rows = read_games(dataset)
    validate(rows)
    train_filter = train_filter if train_filter is not None else {
        'years': list(range(2015, 2027)), 'stages': ['Regular Season']}
    test_filter = test_filter if test_filter is not None else {
        'years': list(range(2015, 2027)), 'stages': ['Playoff']}
    eligible = [r for r in rows if r['winner_side'] in ('BLUE', 'RED')]
    train, test = select(eligible, **train_filter), select(eligible, **test_filter)
    overlap = {r['game_id'] for r in train} & {r['game_id'] for r in test}
    if overlap:
        raise ValueError(f'Train/test overlap: {len(overlap)} game IDs')
    if not train or not test:
        raise ValueError('Empty train or test split; check filters')
    output_dir = Path(output_dir)
    for name, subset in [('train', train), ('test', test)]:
        write_csv(output_dir / f'{name}.csv', subset, tuple(rows[0]))
    # Fit vocabularies on train only for subsequent nn.Embedding use.
    for kind in ('player', 'champion'):
        write_json(output_dir / f'{kind}_vocab.json', build_vocab(train, kind))
    report = {'dataset': fingerprint(dataset), 'train_filter': train_filter,
              'test_filter': test_filter, 'train_games': len(train), 'test_games': len(test),
              'excluded_games': len(rows) - len(train) - len(test),
              'unresolved_winner_games': len(rows) - len(eligible), 'overlap_games': 0,
              'vocabulary_scope': 'train_only', 'column_roles': column_roles(),
              'note': 'Stage split is not chronological: later regular games can follow earlier test games. Missing targets retained; mask per target before training.'}
    write_json(output_dir / 'split_manifest.json', report)
    return report
