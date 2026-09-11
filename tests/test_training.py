"""Numerical and integration checks for the masked multitask baseline."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

if __name__ == '__main__' and __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.dataset import (GameDataset, IGNORE_INDEX, TASK_CLASSES, compute_class_weights,
                         encode_label, validate_training_inputs, validate_vocab)
from src.metrics import MetricAccumulator, classification_metrics
from src.model import MatchTransformer, masked_multitask_loss
from src.preprocess import INPUTS, TARGETS, build_vocab, write_csv, write_json
from src.utils import ROOT, load_checkpoint, save_checkpoint


def row(game_id, stage='Regular Season'):
    result = {'game_id': game_id, 'stage': stage}
    result.update({c: c for c in INPUTS})
    result.update({t: 'BLUE' for t in TARGETS})
    return result


class TrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_balanced_weights_exclude_masked_and_absent_classes(self):
        labels = torch.full((6, len(TARGETS)), IGNORE_INDEX)
        labels[:4, 0] = torch.tensor([0, 0, 0, 1])
        labels[:3, 2] = torch.tensor([0, 0, 2])
        weights = compute_class_weights(labels)
        torch.testing.assert_close(weights['winner_side'], torch.tensor([2 / 3, 2.0]))
        torch.testing.assert_close(weights['more_dragons_side'], torch.tensor([0.75, 0., 1.5]))
        self.assertEqual(weights['dragon_soul_side'].sum().item(), 0)
        self.assertTrue(all(bool(torch.isfinite(w).all()) for w in weights.values()))

    def test_weighted_cross_entropy_and_off_switch(self):
        labels = torch.full((4, len(TARGETS)), IGNORE_INDEX)
        labels[:3, 0] = torch.tensor([0, 0, 1])
        logits = {t: torch.zeros(4, len(c), requires_grad=True) for t, c in TASK_CLASSES.items()}
        logits['winner_side'] = torch.tensor([[2., 0.], [2., 0.], [2., 0.], [0., 100.]], requires_grad=True)
        weights = compute_class_weights(labels)
        weighted = masked_multitask_loss(logits, labels, weights)
        expected = torch.nn.CrossEntropyLoss(weight=weights['winner_side'])(logits['winner_side'][:3], labels[:3, 0])
        torch.testing.assert_close(weighted, expected)
        unweighted = masked_multitask_loss(logits, labels, None)
        torch.testing.assert_close(unweighted, torch.nn.CrossEntropyLoss()(logits['winner_side'][:3], labels[:3, 0]))
        self.assertGreater(weighted.item(), unweighted.item())
        weighted.backward()
        self.assertEqual(logits['winner_side'].grad[3].abs().sum().item(), 0)
        zero_weights = {t: torch.zeros_like(w) for t, w in weights.items()}
        self.assertIsNone(masked_multitask_loss(logits, labels, zero_weights))

    def test_first_dragon_is_binary_and_none_is_excluded(self):
        self.assertEqual(TASK_CLASSES['first_dragon_side'], ('BLUE', 'RED'))
        self.assertEqual(encode_label('NONE', 'first_dragon_side'), IGNORE_INDEX)
        self.assertEqual(encode_label('BLUE', 'first_dragon_side'), 0)
        self.assertEqual(encode_label('RED', 'first_dragon_side'), 1)
        model = MatchTransformer(3, 4)
        self.assertEqual(model.heads['first_dragon_side'].out_features, 2)
        self.assertEqual(model.player_embedding.embedding_dim, 256)
        logits = {t: torch.zeros(2, len(c)) for t, c in TASK_CLASSES.items()}
        labels = torch.full((2, len(TARGETS)), IGNORE_INDEX)
        labels[0, TARGETS.index('first_dragon_side')] = 0
        metrics = MetricAccumulator()
        metrics.update(logits, labels)
        result = metrics.compute()['tasks']['first_dragon_side']
        self.assertEqual(result['class_order'], ['BLUE', 'RED'])
        self.assertEqual(result['ignored_samples'], 1)

    def test_exact_input_order_unk_and_masking(self):
        r = row('a')
        player, champion = build_vocab([r], 'player'), build_vocab([r], 'champion')
        r['red_player_support'] = 'Unseen'
        r.update(dragon_soul_side='N/A', elder_dragon_side='MISSING', first_baron_side='')
        # Analysis columns cannot become features, regardless of their values.
        r.update(blue_dragon_count='999', red_dragon_count='888', blue_team='Ignored')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'games.csv'
            write_csv(path, [r], tuple(r))
            data = GameDataset(path, player, champion, 'Regular Season')
        self.assertEqual(data.player_ids.shape, (1, 10))
        self.assertEqual(data.player_ids[0].tolist(), [player.get(r[c], 0) for c in INPUTS[:10]])
        self.assertEqual(data.champion_ids[0].tolist(), [champion[r[c]] for c in INPUTS[10:]])
        self.assertEqual(data.player_ids[0, 9], 0)
        for task in ('dragon_soul_side', 'elder_dragon_side', 'first_baron_side'):
            self.assertEqual(data.labels[0, TARGETS.index(task)], IGNORE_INDEX)
        self.assertEqual(encode_label('NONE', 'elder_dragon_side'), 2)
        self.assertEqual(encode_label('TIE', 'more_dragons_side'), 2)
        with self.assertRaises(ValueError):
            encode_label('TIE', 'winner_side')

    def test_masked_rows_have_zero_gradient_and_all_masked_batch_is_skipped(self):
        logits = {t: torch.randn(3, len(c), requires_grad=True) for t, c in TASK_CLASSES.items()}
        labels = torch.full((3, 7), IGNORE_INDEX)
        self.assertIsNone(masked_multitask_loss(logits, labels))
        labels[0, 0], labels[1, 1] = 0, 1
        loss = masked_multitask_loss(logits, labels)
        expected = (torch.nn.functional.cross_entropy(logits['winner_side'][:1], torch.tensor([0]))
                    + torch.nn.functional.cross_entropy(logits['first_dragon_side'][1:2], torch.tensor([1]))) / 2
        self.assertTrue(torch.allclose(loss, expected))
        loss.backward()
        self.assertEqual(logits['winner_side'].grad[1:].abs().sum().item(), 0)
        self.assertEqual(logits['first_dragon_side'].grad[[0, 2]].abs().sum().item(), 0)
        self.assertIsNone(logits['dragon_soul_side'].grad)

    def test_metrics_hand_calculated_and_absent_classes(self):
        result = classification_metrics([[2, 1, 0], [0, 1, 0], [0, 0, 0]])
        self.assertEqual(result['accuracy'], 0.75)
        self.assertAlmostEqual(result['macro_f1'], (0.8 + 2 / 3) / 3)
        self.assertEqual(result['per_class'][2]['f1'], 0)
        self.assertIsNone(classification_metrics([[0, 0], [0, 0]])['macro_f1'])
        logits = {t: torch.tensor([[10.] + [0.] * (len(c) - 1)] * 2) for t, c in TASK_CLASSES.items()}
        labels = torch.full((2, 7), IGNORE_INDEX)
        labels[0, 0] = 0
        metrics = MetricAccumulator()
        metrics.update(logits, labels)
        report = metrics.compute()
        self.assertEqual(report['tasks']['winner_side']['accuracy'], 1)
        self.assertEqual(report['tasks']['winner_side']['valid_samples'], 1)
        self.assertEqual(report['tasks']['winner_side']['ignored_samples'], 1)
        self.assertIsNone(report['tasks']['dragon_soul_side']['accuracy'])

    def test_model_structure_gradients_and_checkpoint_roundtrip(self):
        model = MatchTransformer(3, 4)
        ids = torch.tensor([[0, 1, 2, 1, 0] * 2])
        self.assertEqual(model.embed_tokens(ids, ids).shape, (1, 20, 256))
        self.assertEqual(model.sides.tolist(), ([0] * 5 + [1] * 5) * 2)
        self.assertEqual(model.roles.tolist(), list(range(5)) * 4)
        self.assertNotEqual(model.player_embedding.weight.data_ptr(), model.champion_embedding.weight.data_ptr())
        self.assertFalse(torch.equal(model.encoder.layers[0].self_attn.in_proj_weight,
                                     model.encoder.layers[1].self_attn.in_proj_weight))
        optimizer = torch.optim.AdamW(model.parameters())
        logits = model(ids, ids)
        self.assertEqual({t: tuple(v.shape) for t, v in logits.items()},
                         {t: (1, len(c)) for t, c in TASK_CLASSES.items()})
        masked_multitask_loss(logits, torch.zeros((1, 7), dtype=torch.long)).backward()
        self.assertGreater(model.player_embedding.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.champion_embedding.weight.grad.abs().sum().item(), 0)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            expected = model(ids, ids)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'
            save_checkpoint(path, model, optimizer, 1, {'<UNK>': 0, 'a': 1, 'b': 2},
                            {'<UNK>': 0, 'x': 1, 'y': 2, 'z': 3}, {})
            actual_model, saved = load_checkpoint(path, torch.device('cpu'))
            actual_model.eval()
            with torch.no_grad():
                actual = actual_model(ids, ids)
        self.assertEqual(saved['epoch'], 1)
        for task in TARGETS:
            self.assertTrue(torch.equal(actual[task], expected[task]))

    def test_split_and_vocabulary_leakage_rejected(self):
        with self.assertRaises(ValueError):
            validate_vocab({'<UNK>': 0, 'a': 0})
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            r, test_row = row('a'), row('b', 'Playoff')
            player, champion = build_vocab([r], 'player'), build_vocab([r], 'champion')
            for name, item in [('train', r), ('test', test_row)]:
                write_csv(base / f'{name}.csv', [item], tuple(item))
            train = GameDataset(base / 'train.csv', player, champion, 'Regular Season')
            test = GameDataset(base / 'test.csv', player, champion, 'Playoff')
            validate_training_inputs(train, test)
            test.game_ids = train.game_ids[:]
            with self.assertRaisesRegex(ValueError, 'overlap'):
                validate_training_inputs(train, test)
            test.game_ids = ['b']
            train.player_vocab = {**player, 'TestOnly': len(player)}
            with self.assertRaisesRegex(ValueError, 'Vocabulary'):
                validate_training_inputs(train, test)
            with self.assertRaises(ValueError):
                GameDataset(base / 'test.csv', player, champion, 'Regular Season')

    def test_direct_train_and_evaluate_cli_from_external_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            train_rows, test_rows = [row('a'), row('b')], [row('test', 'Playoff')]
            train_rows[1]['winner_side'] = 'RED'
            for r in train_rows + test_rows:
                r['elder_dragon_side'], r['dragon_soul_side'] = '', 'N/A'
            for name, rows in [('train', train_rows), ('test', test_rows)]:
                write_csv(base / f'{name}.csv', rows, tuple(rows[0]))
            for kind in ('player', 'champion'):
                write_json(base / f'{kind}.json', build_vocab(train_rows, kind))
            originals = {p: p.read_bytes() for p in base.iterdir()}
            result = subprocess.run([sys.executable, str(ROOT / 'src/train.py'), '--train', str(base / 'train.csv'),
                        '--test', str(base / 'test.csv'), '--player-vocab', str(base / 'player.json'),
                        '--champion-vocab', str(base / 'champion.json'), '--epochs', '1', '--device', 'cpu',
                        '--output-dir', str(base / 'run')], cwd=base, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            history = [json.loads(line) for line in (base / 'run/history.jsonl').read_text().splitlines()]
            self.assertEqual([r['split'] for r in history], ['train', 'test'])
            config = json.loads((base / 'run/config.json').read_text())
            self.assertTrue(config['class_weighting']['enabled'])
            self.assertEqual(config['class_weighting']['frequencies']['winner_side'], [1, 1])
            self.assertEqual(config['class_weighting']['weights']['winner_side'], [1., 1.])
            self.assertEqual(config['class_weighting']['weights']['dragon_soul_side'], [0., 0., 0.])
            self.assertIn('optimization_loss', history[0])
            result = subprocess.run([sys.executable, str(ROOT / 'src/evaluate.py'),
                        '--checkpoint', str(base / 'run/last.pt'), '--test', str(base / 'test.csv'),
                        '--device', 'cpu', '--output-dir', str(base / 'evaluation')],
                        cwd=base, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            a = json.loads((base / 'run/test_metrics.json').read_text())
            b = json.loads((base / 'evaluation/test_metrics.json').read_text())
            self.assertEqual(a['tasks'], b['tasks'])
            self.assertIsNone(b['tasks']['dragon_soul_side']['accuracy'])
            off = subprocess.run([sys.executable, str(ROOT / 'src/train.py'), '--train', str(base / 'train.csv'),
                        '--test', str(base / 'test.csv'), '--player-vocab', str(base / 'player.json'),
                        '--champion-vocab', str(base / 'champion.json'), '--epochs', '1', '--device', 'cpu',
                        '--no-class-weight', '--output-dir', str(base / 'unweighted')],
                        cwd=base, capture_output=True, text=True)
            self.assertEqual(off.returncode, 0, off.stderr)
            config = json.loads((base / 'unweighted/config.json').read_text())
            self.assertFalse(config['class_weighting']['enabled'])
            self.assertIsNone(config['class_weighting']['weights'])
            for path, data in originals.items():
                self.assertEqual(path.read_bytes(), data)


if __name__ == '__main__':
    unittest.main()
