import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from src.preprocess import (COLUMNS, INPUTS, build_vocab, classify_stage, derive_targets,
                            patch_tuple, read_games, validate, write_csv)
from src.splits import create_splits, select


def teams(blue=4, red=2, blue_elders=0, red_elders=0):
    return {s: {'result': str(s == 'BLUE' and 1 or 0), 'firstbaron': str(s == 'BLUE' and 1 or 0),
                'firstdragon': str(int(d > 0 and (s == 'BLUE' or blue == 0))),
                'dragons': str(d + e), 'elementaldrakes': str(d), 'elders': str(e)}
            for s, d, e in [('BLUE', blue, blue_elders), ('RED', red, red_elders)]}


def game(game_id='a', stage='Regular Season'):
    row = dict.fromkeys(COLUMNS, '')
    row.update(game_id=game_id, date='2024-01-01 10:00:00', year=2024,
               split='Spring', stage=stage, patch='14.01', blue_team='Blue Team', red_team='Red Team')
    row.update({c: c for c in INPUTS})
    row.update(derive_targets(teams(), (14, 1))[0])
    return row


class ObjectivesTest(unittest.TestCase):
    def test_patch_numeric_order(self):
        self.assertLess(patch_tuple('6.9'), patch_tuple('6.10'))
        self.assertLess(patch_tuple('9.22'), patch_tuple('9.23'))

    def test_legacy_zero_is_missing_not_none(self):
        t = teams(0, 0)
        for row in t.values():
            row['elementaldrakes'] = ''
        values, _ = derive_targets(t, (5, 24))
        self.assertIsNone(values['first_four_dragon_side'])
        self.assertEqual(values['dragon_soul_side'], 'N/A')
        self.assertEqual(values['elder_dragon_side'], 'N/A')

    def test_system_introductions(self):
        before, _ = derive_targets(teams(3, 2), (6, 8))
        after, _ = derive_targets(teams(3, 2), (6, 9))
        self.assertEqual(before['elder_dragon_side'], 'N/A')
        self.assertEqual(after['elder_dragon_side'], 'NONE')
        self.assertEqual(derive_targets(teams(), (9, 22))[0]['dragon_soul_side'], 'N/A')
        self.assertEqual(derive_targets(teams(), (9, 23))[0]['dragon_soul_side'], 'BLUE')

    def test_subtract_elders(self):
        t = teams(3, 4, 1, 0)
        for row in t.values():
            row['elementaldrakes'] = ''
        v, _ = derive_targets(t, (10, 2))
        self.assertEqual(v['first_four_dragon_side'], 'RED')
        self.assertEqual(v['dragon_soul_side'], 'RED')
        self.assertEqual(v['elder_dragon_side'], 'BLUE')

    def test_both_elders_require_order(self):
        v, _ = derive_targets(teams(4, 2, 1, 1), (14, 1))
        self.assertIsNone(v['elder_dragon_side'])

    def test_unknown_type_not_negative_label(self):
        t = teams(2, 2)
        t['BLUE']['dragons (type unknown)'] = '3'
        self.assertIsNone(derive_targets(t, (12, 1))[0]['dragon_soul_side'])

    def test_missing_count_not_none(self):
        t = teams()
        t['BLUE']['elders'] = ''
        self.assertIsNone(derive_targets(t, (14, 1))[0]['elder_dragon_side'])

    def test_complete_timeline_resolves_legacy(self):
        t = teams(0, 0)
        for r in t.values():
            r['elementaldrakes'] = ''
        t['BLUE']['firstdragon'], t['RED']['firstdragon'] = '0', '1'
        events = [{'timestamp_ms': i, 'monster': 'DRAGON', 'side': 'RED'} for i in range(4)]
        events.append({'timestamp_ms': 4, 'monster': 'BARON', 'side': 'BLUE'})
        timeline = {'source': 'unit-test-fixture', 'complete': True, 'events': events}
        v, _ = derive_targets(t, (5, 1), timeline)
        self.assertEqual(v['first_four_dragon_side'], 'RED')
        self.assertEqual(v['dragon_soul_side'], 'N/A')
        timeline['complete'] = False
        with self.assertRaises(ValueError):
            derive_targets(t, (5, 1), timeline)

    def test_none_no_baron(self):
        t = teams(3, 2)
        for r in t.values():
            r['firstbaron'] = '0'
        v, _ = derive_targets(t, (14, 1))
        self.assertEqual(v['first_baron_side'], 'NONE')
        self.assertEqual(v['first_four_dragon_side'], 'NONE')


class ExpandedDragonTest(unittest.TestCase):
    def test_first_dragon_flags(self):
        for flags, expected in [(('1', '0'), 'BLUE'), (('0', '1'), 'RED'),
                                (('1', '1'), None), (('', '1'), None),
                                (('0', ''), None), (('2', '0'), None),
                                (('garbage', '0'), None)]:
            with self.subTest(flags=flags):
                t = teams(2, 2)
                for side, value in zip(('BLUE', 'RED'), flags):
                    t[side]['firstdragon'] = value
                self.assertEqual(derive_targets(t, (14, 1))[0]['first_dragon_side'], expected)
        self.assertEqual(derive_targets(teams(0, 0), (14, 1))[0]['first_dragon_side'], 'NONE')

    def test_first_dragon_conflicts_not_guessed(self):
        t = teams(2, 2)
        t['BLUE']['firstdragon'] = '0'
        self.assertIsNone(derive_targets(t, (14, 1))[0]['first_dragon_side'])
        t = teams(0, 3)
        t['BLUE']['firstdragon'], t['RED']['firstdragon'] = '1', '0'
        self.assertIsNone(derive_targets(t, (14, 1))[0]['first_dragon_side'])

    def test_more_dragons_including_zero_tie(self):
        for blue, red, expected in [(4, 2, 'BLUE'), (2, 4, 'RED'), (2, 2, 'TIE'), (0, 0, 'TIE')]:
            result, _ = derive_targets(teams(blue, red), (14, 1))
            self.assertEqual(result['more_dragons_side'], expected)
            self.assertEqual(result['blue_dragon_count'], blue)
            self.assertEqual(result['red_dragon_count'], red)

    def test_elemental_preferred_and_elders_excluded(self):
        t = teams(3, 2, 1, 0)
        result, reasons = derive_targets(t, (14, 1))
        self.assertEqual(result['blue_dragon_count'], 3)
        self.assertEqual(reasons['blue_dragon_count'], 'elementaldrakes')
        t['BLUE']['elementaldrakes'] = ''
        result, reasons = derive_targets(t, (14, 1))
        self.assertEqual(result['blue_dragon_count'], 3)
        self.assertEqual(reasons['blue_dragon_count'], 'dragons_minus_elders')

    def test_unrecoverable_count_propagates_missing(self):
        for field, value in [('elementaldrakes', '-1'), ('elementaldrakes', 'bad'),
                             ('elementaldrakes', '2.5'), ('dragons (type unknown)', '1')]:
            t = teams()
            t['BLUE'][field] = value
            r, _ = derive_targets(t, (14, 1))
            self.assertIsNone(r['blue_dragon_count'])
            self.assertIsNone(r['more_dragons_side'])
        t = teams()
        t['BLUE'].update(elementaldrakes='', elders='bad')
        self.assertIsNone(derive_targets(t, (14, 1))[0]['blue_dragon_count'])
        t = teams()
        t['BLUE'].update(elementaldrakes='', dragons='0', elders='1')
        self.assertIsNone(derive_targets(t, (14, 1))[0]['blue_dragon_count'])

    def test_old_season_first_flag_still_usable(self):
        t = teams(0, 0)
        for row in t.values():
            row['elementaldrakes'] = ''
        t['RED']['firstdragon'] = '1'
        r, _ = derive_targets(t, (5, 1))
        self.assertEqual(r['first_dragon_side'], 'RED')
        for column in ('blue_dragon_count', 'red_dragon_count', 'more_dragons_side', 'first_four_dragon_side'):
            self.assertIsNone(r[column])
        self.assertEqual(r['dragon_soul_side'], 'N/A')
        self.assertEqual(r['elder_dragon_side'], 'N/A')

    def test_both_four_needs_order(self):
        t = teams(4, 4)
        r, _ = derive_targets(t, (8, 1))
        self.assertEqual(r['more_dragons_side'], 'TIE')
        self.assertIsNone(r['first_four_dragon_side'])
        events = [{'timestamp_ms': i, 'monster': 'DRAGON', 'side': side}
                  for i, side in enumerate(['BLUE'] * 4 + ['RED'] * 4)]
        events.append({'timestamp_ms': 9, 'monster': 'BARON', 'side': 'BLUE'})
        r, _ = derive_targets(t, (8, 1), dict(source='fixture', complete=True, events=events))
        self.assertEqual(r['first_four_dragon_side'], 'BLUE')
        self.assertEqual(r['more_dragons_side'], 'TIE')

    def test_wrong_tie_class_and_count_relation_rejected(self):
        r = game()
        r['more_dragons_side'] = 'NONE'
        with self.assertRaises(ValueError):
            validate([r])
        r['more_dragons_side'] = 'TIE'
        with self.assertRaises(ValueError):
            validate([r])
        r = game()
        r['winner_side'] = 'TIE'
        with self.assertRaises(ValueError):
            validate([r])

    def test_inputs_and_audit_coverage(self):
        from src.preprocess import DRAGON_COUNTS, TARGETS, column_roles
        self.assertEqual(len(INPUTS), 20)
        self.assertFalse(set(INPUTS) & set((*TARGETS, *DRAGON_COUNTS)))
        self.assertEqual(len(COLUMNS), 37)
        r = game()
        report = validate([r])
        self.assertEqual(set(report['target_distributions']), set(TARGETS))
        self.assertEqual(report['dragon_count_statistics']['blue_dragon_count']['missing'], 0)
        self.assertEqual(column_roles()['analysis_only_columns'], list(DRAGON_COUNTS))


class DatasetTest(unittest.TestCase):
    def test_stage_contamination(self):
        def stage(year, split, date, playoffs='0'):
            return classify_stage(dict(year=str(year), split=split, date=date, playoffs=playoffs))
        self.assertEqual(stage(2025, 'Cup', '2025-02-01'), 'Cup')
        self.assertEqual(stage(2016, 'Spring', '2015-09-11'), 'Promotion')
        self.assertEqual(stage(2016, 'Summer', '2016-04-28'), 'Promotion')
        self.assertEqual(stage(2021, '', '2021-09-01'), 'Regional Qualifier')
        self.assertEqual(stage(2025, 'Rounds 1-2', '2025-06-07', '1'), 'Road to MSI')
        self.assertEqual(stage(2025, 'Rounds 3-5', '2025-09-07', '1'), 'Play-In')
        self.assertEqual(stage(2025, 'Rounds 3-5', '2025-09-10', '1'), 'Playoff')
        self.assertEqual(stage(2026, 'Rounds 3-4', '2026-08-28', '1'), 'Play-In')
        self.assertEqual(stage(2026, 'Rounds 3-4', '2026-08-29', '1'), 'Playoff')

    def test_csv_preserves_na_and_missing(self):
        r = game()
        r['patch'] = '6.9'
        r['dragon_soul_side'], r['elder_dragon_side'] = 'N/A', None
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'games.csv'
            write_csv(path, [r], COLUMNS)
            actual = read_games(path)[0]
            self.assertEqual(actual['dragon_soul_side'], 'N/A')
            self.assertEqual(actual['elder_dragon_side'], '')
            self.assertEqual(validate([actual])['total_missing_cells'], 1)

    def test_duplicate_game_rejected(self):
        with self.assertRaises(ValueError):
            validate([game(), game()])

    def test_splits_and_train_only_vocab(self):
        a, b, c = game(), game('b', 'Playoff'), game('c', 'Cup')
        b['blue_player_top'] = 'UnseenPlayer'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'games.csv'
            write_csv(path, [a, b, c], COLUMNS)
            report = create_splits(path, Path(directory) / 'splits')
            self.assertEqual((report['train_games'], report['test_games'], report['excluded_games']), (1, 1, 1))
            self.assertNotIn('UnseenPlayer', build_vocab([a], 'player'))
            self.assertEqual(build_vocab([a], 'player')['<UNK>'], 0)
            with self.assertRaises(ValueError):
                create_splits(path, Path(directory) / 'bad', {}, {})
        self.assertEqual(select([a], years=[2023]), [])
        self.assertEqual(select([a], date_to='2023-12-31'), [])

class IngestionTest(unittest.TestCase):
    def fixture(self, directory):
        from src.preprocess import ROLES, SIDES
        rows = []
        for side in SIDES:
            for role in (*ROLES, 'team'):
                rows.append(dict(gameid='fixture', date='2024-01-01 10:00:00', year='2024',
                                 split='Spring', playoffs='0', patch='14.01', league='LCK',
                                 side=side.title(), position=role, playername=f'{side}-{role}',
                                 teamname=f'{side} Team',
                                 champion=f'champion-{side}-{role}', result='1' if side == 'BLUE' else '0',
                                 firstbaron='1' if side == 'BLUE' else '0',
                                 firstdragon='1' if side == 'BLUE' else '0',
                                 dragons='4' if side == 'BLUE' else '2', elders='0'))
        path = Path(directory) / '2024_LoL_esports_match_data_from_OraclesElixir.csv'
        write_csv(path, rows, tuple(rows[0]))
        return path, rows

    def test_full_ingestion_and_original_preserved(self):
        import contextlib
        import io
        from src.preprocess import preprocess, fingerprint
        with tempfile.TemporaryDirectory() as directory:
            path, _ = self.fixture(directory)
            before = fingerprint(path)
            with contextlib.redirect_stdout(io.StringIO()):
                rows, report = preprocess(directory, Path(directory) / 'out', 2024, 2024)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['blue_player_jungle'], 'BLUE-jng')
            self.assertEqual(rows[0]['blue_team'], 'BLUE Team')
            self.assertEqual(rows[0]['red_team'], 'RED Team')
            self.assertEqual(report['total_missing_cells'], 0)
            from src.preprocess import TARGETS, DRAGON_COUNTS
            audit = read_games(Path(directory) / 'out/target_audit.csv')
            self.assertEqual({r['target'] for r in audit}, set((*TARGETS, *DRAGON_COUNTS)))
            self.assertEqual(len(audit), 9)
            self.assertEqual(before, fingerprint(path))
            self.assertTrue((Path(directory) / 'out/validation_summary.md').exists())

    def test_duplicate_slot_rejected_before_dataset_written(self):
        from src.preprocess import preprocess
        with tempfile.TemporaryDirectory() as directory:
            path, rows = self.fixture(directory)
            rows[1] = dict(rows[0])
            write_csv(path, rows, tuple(rows[0]))
            with self.assertRaises(ValueError):
                preprocess(directory, Path(directory) / 'out', 2024, 2024)
            self.assertFalse((Path(directory) / 'out/games.csv').exists())

    def test_collect_preserves_existing_without_network(self):
        from src.collect import collect
        with tempfile.TemporaryDirectory() as directory:
            path, _ = self.fixture(directory)
            before = path.read_bytes()
            report = collect(directory, 'https://invalid.example/{year}.csv', [2024])
            self.assertEqual(report[0]['status'], 'preserved_existing')
            self.assertEqual(before, path.read_bytes())

    def test_unrecorded_winner_not_in_train(self):
        a, b, c = game(), game('b', 'Playoff'), game('c')
        c['winner_side'] = ''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'games.csv'
            write_csv(path, [a, b, c], COLUMNS)
            report = create_splits(path, Path(directory) / 'splits')
            self.assertEqual(report['train_games'], 1)
            self.assertEqual(report['unresolved_winner_games'], 1)


class RegressionTest(unittest.TestCase):
    def test_conflicting_elemental_and_total_counts_are_missing(self):
        t = teams()
        t['BLUE']['dragons'] = '3'
        values, reasons = derive_targets(t, (14, 1))
        self.assertIsNone(values['blue_dragon_count'])
        self.assertEqual(values['red_dragon_count'], 2)
        self.assertIsNone(values['more_dragons_side'])
        self.assertEqual(reasons['blue_dragon_count'], 'conflicting_dragon_totals')

    def test_impossible_two_souls_not_a_tie(self):
        values, _ = derive_targets(teams(4, 4), (14, 1))
        for c in ('blue_dragon_count', 'red_dragon_count', 'more_dragons_side',
                  'first_four_dragon_side', 'dragon_soul_side'):
            self.assertIsNone(values[c], c)
        row = game()
        row.update(blue_dragon_count=4, red_dragon_count=4, more_dragons_side='TIE',
                   first_four_dragon_side=None, dragon_soul_side=None)
        with self.assertRaises(ValueError):
            validate([row])

    def test_baron_conflicts_are_missing(self):
        cases = [('1', '0', '0', '1'), ('0', '0', '1', '0'),
                 ('1', '1', '1', '1'), ('bad', '0', '1', '0'),
                 ('0', '', '1', '0')]
        for blue_flag, red_flag, blue_count, red_count in cases:
            with self.subTest(flags=(blue_flag, red_flag)):
                t = teams()
                t['BLUE'].update(firstbaron=blue_flag, barons=blue_count)
                t['RED'].update(firstbaron=red_flag, barons=red_count)
                self.assertIsNone(derive_targets(t, (14, 1))[0]['first_baron_side'])

    def test_baron_missing_flags_only_use_conclusive_totals(self):
        for blue, red, expected in [(2, 0, 'BLUE'), (0, 1, 'RED'), (0, 0, 'NONE'), (1, 1, None), ('bad', 0, None)]:
            t = teams()
            for side, n in zip(('BLUE', 'RED'), (blue, red)):
                t[side].update(firstbaron='', barons=str(n))
            self.assertEqual(derive_targets(t, (14, 1))[0]['first_baron_side'], expected)

    def test_invalid_timeline_order_and_timestamps(self):
        for timestamp in (-1, True):
            with self.assertRaises(ValueError):
                derive_targets(teams(), (14, 1), dict(source='fixture', complete=True,
                    events=[dict(timestamp_ms=timestamp, monster='DRAGON', side='BLUE')]))
        events = [dict(timestamp_ms=i, monster='DRAGON', side='BLUE') for i in range(4)]
        events.append(dict(timestamp_ms=4, monster='DRAGON', side='RED'))
        with self.assertRaisesRegex(ValueError, 'after soul'):
            derive_targets(teams(), (14, 1), dict(source='fixture', complete=True, events=events))
        with self.assertRaisesRegex(ValueError, 'before soul'):
            derive_targets(teams(), (14, 1), dict(source='fixture', complete=True,
                events=[dict(timestamp_ms=0, monster='ELDER', side='BLUE')]))

    def test_validation_requires_schema_metadata_and_nonempty_dataset(self):
        with self.assertRaises(ValueError):
            validate([])
        for column, value in [('blue_team', ''), ('red_team', 'Blue Team'), ('stage', 'Unknown'), ('year', 'bad')]:
            row = game()
            row[column] = value
            with self.assertRaises(ValueError):
                validate([row])
        row = game()
        del row['blue_team']
        with self.assertRaisesRegex(ValueError, 'rerun preprocess'):
            validate([row])

    def test_unknown_count_cannot_prove_first_four_none(self):
        row = game()
        row.update(blue_dragon_count='', more_dragons_side='',
                   first_four_dragon_side='NONE', dragon_soul_side='NONE')
        with self.assertRaises(ValueError):
            validate([row])

    def test_future_year_in_default_split_and_train_only_vocab_files(self):
        rows = [game(), game('test', 'Playoff')]
        for row in rows:
            row.update(year=2027, date='2027-01-01 10:00:00', patch='17.1')
        rows[1]['blue_player_top'] = 'Test Only Player'
        rows[1]['blue_champion_top'] = 'Test Only Champion'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'games.csv'
            write_csv(path, rows, COLUMNS)
            out = Path(directory) / 'split'
            report = create_splits(path, out)
            self.assertEqual((report['train_games'], report['test_games']), (1, 1))
            for kind in ('player', 'champion'):
                vocab = json.loads((out / f'{kind}_vocab.json').read_text(encoding='utf-8'))
                self.assertNotIn(f'Test Only {kind.title()}', vocab)
                self.assertEqual(vocab['<UNK>'], 0)
                self.assertEqual(sorted(vocab.values()), list(range(len(vocab))))

    def test_future_calendar_is_configurable_and_never_guessed(self):
        from src.preprocess import load_stage_calendar
        row = dict(year='2027', split='Rounds 3-6', date='2027-09-10', playoffs='1')
        with self.assertRaisesRegex(ValueError, 'No postseason calendar'):
            classify_stage(row)
        calendar = load_stage_calendar()
        calendar['playoff_start']['2027'] = '2027-09-10'  # Synthetic test boundary.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'calendar.json'
            path.write_text(json.dumps(calendar), encoding='utf-8-sig')
            calendar = load_stage_calendar(path)
        self.assertEqual(classify_stage(row, calendar), 'Playoff')
        self.assertEqual(classify_stage(row | {'date': '2027-09-09'}, calendar), 'Play-In')
        self.assertEqual(classify_stage(row | {'playoffs': '0'}, calendar), 'Regular Season')

    def test_download_suffix_auto_end_year_and_raw_preservation(self):
        from src.preprocess import preprocess
        from src.collect import collect
        with tempfile.TemporaryDirectory() as directory:
            path, _ = IngestionTest().fixture(directory)
            suffixed = path.with_name(path.stem + ' (1).csv')
            path.rename(suffixed)
            original = suffixed.read_bytes()
            with contextlib.redirect_stdout(io.StringIO()):
                rows, report = preprocess(directory, Path(directory) / 'out', start_year=2024)
            self.assertEqual(report['coverage']['requested_years'], [2024])
            self.assertEqual(len(rows), 1)
            self.assertEqual(collect(directory, 'https://invalid.example/{year}', [2024])[0]['status'], 'preserved_existing')
            self.assertEqual(suffixed.read_bytes(), original)
            self.assertFalse(path.exists())
            path.write_bytes(original)
            with self.assertRaisesRegex(ValueError, 'Ambiguous'):
                preprocess(directory, Path(directory) / 'bad', 2024, 2024)

    def test_missing_raw_year_fails_before_processing(self):
        from src.preprocess import preprocess
        with tempfile.TemporaryDirectory() as directory:
            IngestionTest().fixture(directory)
            with self.assertRaisesRegex(FileNotFoundError, '2025'):
                preprocess(directory, Path(directory) / 'out', 2024, 2025)
            self.assertFalse((Path(directory) / 'out/games.csv').exists())

    def test_ogn_only_in_2015_and_other_leagues_excluded(self):
        from src.preprocess import preprocess
        for year in (2015, 2024):
            with self.subTest(year=year), tempfile.TemporaryDirectory() as directory:
                path, source = IngestionTest().fixture(directory)
                rows = [row | dict(gameid=league, league=league, year=str(year),
                        date=f'{year}-01-01 10:00:00', patch='5.1' if year == 2015 else '14.1')
                        for league in ('OGN', 'LCK', 'LCK CL', 'LPL') for row in source]
                path.unlink()
                path = Path(directory) / f'{year}_LoL_esports_match_data_from_OraclesElixir.csv'
                write_csv(path, rows, tuple(rows[0]))
                with contextlib.redirect_stdout(io.StringIO()):
                    games, _ = preprocess(directory, Path(directory) / 'out', year, year)
                self.assertEqual({r['game_id'] for r in games}, {'OGN', 'LCK'} if year == 2015 else {'LCK'})

    def test_team_metadata_conflicts_are_structural_errors(self):
        from src.preprocess import preprocess
        with tempfile.TemporaryDirectory() as directory:
            path, rows = IngestionTest().fixture(directory)
            rows[0]['teamname'] = 'Wrong Team'
            write_csv(path, rows, tuple(rows[0]))
            out = Path(directory) / 'out'
            with self.assertRaisesRegex(ValueError, 'structural errors'):
                preprocess(directory, out, 2024, 2024)
            self.assertIn('teamname', json.loads((out / 'structural_errors.json').read_text())[0]['error'])
            self.assertFalse((out / 'games.csv').exists())


if __name__ == '__main__':
    unittest.main()
