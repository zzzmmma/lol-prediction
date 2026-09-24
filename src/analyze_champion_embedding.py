"""Analyze champion embedding neighbors across five 6L / FFN512 seeds."""

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.nn import functional as F

from src.dataset import load_vocab


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = PROJECT_ROOT / 'runs/layers_6_20260921_111124_289760'


@torch.inference_mode()
def load_champion_similarities(checkpoint_path):
    vocab = load_vocab(checkpoint_path.parent / 'champion_vocab.json')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    config = checkpoint.get('model_config', {})
    expected = {'embedding_dim': 256, 'num_layers': 6, 'feedforward_dim': 512}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError(f'Expected a 6L / FFN512 / embedding_dim=256 checkpoint: {checkpoint_path}')
    if checkpoint.get('champion_vocab') != vocab:
        raise ValueError(f'Champion vocabulary does not match checkpoint: {checkpoint_path}')
    weights = checkpoint['model_state_dict']['champion_embedding.weight'].detach().float()
    if weights.shape != (len(vocab), 256) or not bool(torch.isfinite(weights).all()):
        raise ValueError(f'Invalid champion embedding weights: {checkpoint_path}')

    champions = sorted((name for name in vocab if name != '<UNK>'), key=vocab.__getitem__)
    embeddings = weights[[vocab[name] for name in champions]]
    if bool((embeddings.norm(dim=1) == 0).any()):
        raise ValueError(f'Zero-length champion embedding: {checkpoint_path}')
    normalized = F.normalize(embeddings, p=2, dim=1)
    similarities = (normalized @ normalized.T).clamp(-1, 1)
    return champions, similarities, checkpoint.get('epoch')


def find_champion(champions, query):
    query = query.strip()
    if query in champions:
        return query
    matches = [name for name in champions if name.casefold() == query.casefold()]
    if len(matches) != 1:
        raise ValueError(f'Champion {query!r} was not found uniquely in champion_vocab.json')
    return matches[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--champion', required=True, help='Champion name, for example Azir')
    parser.add_argument('--run-dir', type=Path, default=DEFAULT_RUN_DIR,
                        help='Directory containing five seed_*/last.pt checkpoints')
    args = parser.parse_args()

    try:
        checkpoints = sorted(args.run_dir.glob('seed_*/last.pt'),
                             key=lambda path: int(path.parent.name.removeprefix('seed_')))
        if len(checkpoints) != 5:
            raise ValueError(f'Expected five seed checkpoints, found {len(checkpoints)} in {args.run_dir}')

        appearances = Counter()
        candidate_scores = defaultdict(list)
        reference_champions = None
        for checkpoint_path in checkpoints:
            champions, similarities, epoch = load_champion_similarities(checkpoint_path)
            if reference_champions is None:
                reference_champions = champions
                query = find_champion(champions, args.champion)
                print(f'{query}\n')
            elif champions != reference_champions:
                raise ValueError('All five checkpoints must use the same champion vocabulary')

            query_index = champions.index(query)
            candidates = []
            for index, name in enumerate(champions):
                if index != query_index:
                    score = float(similarities[query_index, index])
                    candidate_scores[name].append(score)
                    candidates.append((name, score))
            candidates.sort(key=lambda item: (-item[1], item[0]))
            top_ten = candidates[:10]
            appearances.update(name for name, _ in top_ten)
            print(f'Seed {checkpoint_path.parent.name.removeprefix("seed_")} (epoch={epoch}):')
            for rank, (name, score) in enumerate(top_ten, 1):
                print(f'{rank}. {name} {score:.6f}')
            print()

        print('Across 5 seeds:')
        print('Champion | TOP10 appearances | Mean cosine ± sample std (all seeds)')
        for name, count in sorted(appearances.items(),
                                  key=lambda item: (-item[1], -mean(candidate_scores[item[0]]), item[0])):
            scores = candidate_scores[name]
            print(f'{name} | {count}/5 | {mean(scores):.6f} ± {stdev(scores):.6f}')
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        parser.exit(1, f'Champion embedding analysis error: {exc}\n')


if __name__ == '__main__':
    main()
