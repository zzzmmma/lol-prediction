"""Check EDA denominators, unseen vocabulary and read-only behavior."""
import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

# Support VS Code's direct-file runner as well as unittest discovery.
if __name__ == '__main__' and __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eda import (ROOT, analyze, compare_stages, print_report, save_report,
                     summarize, target_distribution, vocabulary_overlap)
from src.preprocess import COLUMNS, INPUTS, TARGETS, write_csv


def fixture(game_id, stage='Regular Season'):
    row = dict.fromkeys(COLUMNS, '')
    row.update(game_id=game_id, date='2024-01-01 10:00:00', year='2024',
               split='Spring', stage=stage, patch='14.1', blue_team='A', red_team='B',
               winner_side='BLUE', first_dragon_side='BLUE', more_dragons_side='BLUE',
               first_four_dragon_side='NONE', dragon_soul_side='NONE',
               elder_dragon_side='NONE', first_baron_side='NONE',
               blue_dragon_count='2', red_dragon_count='1')
    row.update({c: c for c in INPUTS})
    return row


class EdaTest(unittest.TestCase):
    def test_direct_execution_from_another_working_directory(self):
        self.assertEqual(ROOT, Path(__file__).resolve().parents[1])
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            arguments = []
            for name in ('games', 'train', 'test'):
                path = base / f'{name}.csv'
                write_csv(path, [fixture(name)], COLUMNS)
                arguments.extend([f'--{name}', str(path)])
            output = base / 'report.json'
            result = subprocess.run(
                [sys.executable, str(ROOT / 'src/eda.py'), *arguments,
                 '--output', str(output), '--top', '1'],
                cwd=base, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(output.read_text(encoding='utf-8'))['datasets']['games']['total_games'], 1)

    def test_missing_na_none_invalid_and_denominators(self):
        rows = [{'dragon_soul_side': value} for value in ('BLUE', 'RED', 'NONE', 'N/A', '', 'bad')]
        result = target_distribution(rows, 'dragon_soul_side')
        self.assertEqual(result['counts']['MISSING'], 1)
        self.assertEqual(result['counts']['N/A'], 1)
        self.assertEqual(result['counts']['NONE'], 1)
        self.assertEqual(result['missing_rate'], 1 / 6)
        self.assertEqual(result['na_rate'], 1 / 6)
        self.assertEqual(result['invalid_class_counts'], {'bad': 1})
        self.assertEqual(result['known_applicable_games'], 3)
        self.assertEqual(result['known_applicable_proportions']['BLUE'], 1 / 3)
        self.assertAlmostEqual(sum(result['proportions'].values()), 1)

    def test_empty_populations_have_undefined_rates(self):
        result = target_distribution([], 'winner_side')
        self.assertIsNone(result['missing_rate'])
        self.assertIsNone(result['proportions']['BLUE'])
        self.assertIsNone(vocabulary_overlap({}, {})['jaccard_overlap'])
        stages = compare_stages([fixture('a')])
        self.assertIsNone(stages['targets']['winner_side']['playoff_minus_regular_percentage_points']['proportions']['BLUE'])
        self.assertIsNotNone(summarize([])['checks']['existing_validator_first_error'])

    def test_stage_comparison_uses_games_and_correct_delta(self):
        a, b, c = fixture('a'), fixture('b'), fixture('c', 'Playoff')
        b['winner_side'], c['winner_side'] = '', 'RED'
        detail = compare_stages([a, b, c, fixture('cup', 'Cup')])
        self.assertEqual((detail['regular_games'], detail['playoff_games']), (2, 1))
        changes = detail['targets']['winner_side']['playoff_minus_regular_percentage_points']
        self.assertEqual(changes['proportions']['BLUE'], -50)
        self.assertEqual(changes['known_applicable_proportions']['BLUE'], -100)
        self.assertEqual(changes['proportions']['RED'], 100)

    def test_overlap_distinguishes_unique_tokens_from_slots(self):
        result = vocabulary_overlap({'a': 2, 'b': 1}, {'b': 3, 'c': 2})
        self.assertEqual(result['shared_count'], 1)
        self.assertEqual(result['jaccard_overlap'], 1 / 3)
        self.assertEqual(result['test_vocabulary_coverage'], 0.5)
        self.assertEqual(result['test_only_count'], 1)
        self.assertEqual(result['test_only_tokens'], ['c'])
        self.assertEqual(result['test_oov_slots'], 2)
        self.assertEqual(result['test_oov_slot_rate'], 0.4)

    def test_full_report_does_not_modify_csvs(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            a, b = fixture('a'), fixture('b', 'Playoff')
            b['blue_player_top'], b['blue_champion_top'] = 'Unseen Player', 'Unseen Champion'
            paths = {name: base / f'{name}.csv' for name in ('games', 'train', 'test')}
            for name, rows in [('games', [a, b]), ('train', [a]), ('test', [b])]:
                write_csv(paths[name], rows, COLUMNS)
            originals = {name: path.read_bytes() for name, path in paths.items()}
            report = analyze(paths)
            save_report(report, base / 'eda.json', paths)
            saved = json.loads((base / 'eda.json').read_text(encoding='utf-8'))
            self.assertEqual(saved, report)
            for name, path in paths.items():
                self.assertEqual(path.read_bytes(), originals[name])
                self.assertEqual(set(report['datasets'][name]['targets']), set(TARGETS))
                self.assertEqual(sum(report['datasets'][name]['player_appearances'].values()),
                                 report['datasets'][name]['total_games'] * 10)
            self.assertEqual(report['vocabulary_overlap']['player']['test_only_tokens'], ['Unseen Player'])
            self.assertEqual(report['vocabulary_overlap']['champion']['test_only_tokens'], ['Unseen Champion'])
            self.assertEqual(report['split_checks']['train_test_overlap_game_ids'], [])
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                print_report(report, top=0)
            self.assertIn('Unseen Champion', output.getvalue())
            self.assertIn('known_applicable_proportions', output.getvalue())

    def test_anomalies_are_reported_without_dropping_rows(self):
        a = fixture('a')
        a['blue_player_jungle'] = a['blue_player_top']
        a['red_champion_mid'] = ''
        result = summarize([a, a])
        self.assertEqual(result['total_games'], 2)
        self.assertEqual(result['checks']['duplicate_game_ids'], {'a': 2})
        self.assertEqual(result['checks']['missing_by_column']['red_champion_mid'], 2)
        self.assertEqual(result['checks']['duplicate_tokens_within_game']['player'], ['a', 'a'])
        self.assertIsNotNone(result['checks']['existing_validator_first_error'])

    def test_bad_header_and_protected_output_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'empty.csv'
            path.write_text('game_id\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'missing columns'):
                analyze(dict.fromkeys(('games', 'train', 'test'), path))
            with self.assertRaises(ValueError):
                save_report({}, ROOT / 'data/processed/player_vocab.json', {})
            with self.assertRaises(ValueError):
                save_report({}, path, {'games': path})


if __name__ == '__main__':
    unittest.main()
