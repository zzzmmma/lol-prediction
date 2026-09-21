"""Find similar players in a trained 6-layer / FFN-512 / 256-dimensional checkpoint."""
import argparse
import sys
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.nn import functional as F

from src.dataset import load_vocab


@torch.inference_mode()
def load_player_similarities(checkpoint_path, vocab_path=None):
    checkpoint_path = Path(checkpoint_path)
    vocab_path = Path(vocab_path) if vocab_path is not None else checkpoint_path.parent / 'player_vocab.json'
    vocab = load_vocab(vocab_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    config = checkpoint.get('model_config', {})
    expected = {'embedding_dim': 256, 'num_layers': 6, 'feedforward_dim': 512}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError('Expected a 6-layer / FFN-512 / embedding_dim=256 checkpoint')
    if checkpoint.get('player_vocab') != vocab:
        raise ValueError('player_vocab.json does not match the checkpoint player IDs')
    weights = checkpoint['model_state_dict']['player_embedding.weight'].detach().float()
    if weights.shape != (len(vocab), 256):
        raise ValueError('Player embedding shape does not match vocabulary size and dimension 256')
    if not bool(torch.isfinite(weights).all()):
        raise ValueError('Player embedding contains non-finite values')

    # <UNK> is a fallback token, not an actual player.
    players = sorted((name for name in vocab if name != '<UNK>'), key=vocab.__getitem__)
    embeddings = weights[[vocab[name] for name in players]]
    if bool((embeddings.norm(dim=1) == 0).any()):
        raise ValueError('Cosine similarity is undefined for a zero-length player embedding')
    normalized = F.normalize(embeddings, p=2, dim=1)
    similarities = (normalized @ normalized.T).clamp(-1, 1)
    return players, similarities, checkpoint.get('epoch')


def nearest_players(players, similarities, query, top_k=10):
    query = query.strip()
    if query in players:
        index = players.index(query)
    else:
        matches = [i for i, name in enumerate(players) if name.casefold() == query.casefold()]
        if len(matches) != 1:
            raise ValueError(f'Player {query!r} was not found uniquely; use the exact name in player_vocab.json')
        index = matches[0]
    candidates = [i for i in range(len(players)) if i != index]
    candidates.sort(key=lambda i: (-float(similarities[index, i]), players[i]))
    return players[index], [(players[i], float(similarities[index, i])) for i in candidates[:top_k]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True, help='Final trained last.pt for one seed')
    parser.add_argument('--player-vocab', type=Path,
                        help='Defaults to player_vocab.json next to the checkpoint')
    parser.add_argument('--player', required=True, help='Player name, for example Faker')
    args = parser.parse_args()
    try:
        players, similarities, epoch = load_player_similarities(args.checkpoint, args.player_vocab)
        name, neighbors = nearest_players(players, similarities, args.player)
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        parser.exit(1, f'Embedding analysis error: {exc}\n')
    print(f'Checkpoint: {args.checkpoint.resolve()} (epoch={epoch})')
    print(f'All-player cosine similarity matrix: {tuple(similarities.shape)}; <UNK> excluded')
    print(f'\n{name}: TOP {len(neighbors)} similar players (self excluded)')
    print('Rank | Player | Cosine similarity')
    for rank, (player, score) in enumerate(neighbors, 1):
        print(f'{rank:>4} | {player} | {score:.6f}')


if __name__ == '__main__':
    main()
