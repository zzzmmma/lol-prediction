"""Analyze player cosine similarities in 6-layer / FFN-512 / 256D checkpoints."""
import argparse
import csv
from collections import Counter, defaultdict
from statistics import mean, stdev
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


def nearest_players(players, similarities, query, top_k=10, roles=None):
    query = query.strip()
    if query in players:
        index = players.index(query)
    else:
        matches = [i for i, name in enumerate(players) if name.casefold() == query.casefold()]
        if len(matches) != 1:
            raise ValueError(f'Player {query!r} was not found uniquely; use the exact name in player_vocab.json')
        index = matches[0]
    candidates = [i for i in range(len(players))
                  if i != index and (roles is None or roles[players[i]] == roles[players[index]])]
    candidates.sort(key=lambda i: (-float(similarities[index, i]), players[i]))
    return players[index], [(players[i], float(similarities[index, i])) for i in candidates[:top_k]]


ROLES = ('top', 'jungle', 'mid', 'adc', 'support')
ROLE_LABELS = ('TOP', 'JG', 'MID', 'ADC', 'SUP')


def load_player_roles(games_path, players):
    """Assign the most frequent observed role; ties use ROLES order."""
    counts = defaultdict(Counter)
    columns = [(f'{side}_player_{role}', role)
               for side in ('blue', 'red') for role in ROLES]
    with Path(games_path).open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        missing = {column for column, _ in columns} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f'Missing player role columns: {sorted(missing)}')
        for row in reader:
            for column, role in columns:
                name = (row[column] or '').strip()
                if name and name != '<UNK>':
                    counts[name][role] += 1
    missing = [name for name in players if not counts[name]]
    if missing:
        raise ValueError(f'No observed role for players: {missing}')
    roles = {name: max(ROLES, key=lambda role: counts[name][role]) for name in players}
    print(f'Role data: {Path(games_path).resolve()}')
    print('Role assignment: most appearances; ties use TOP/JG/MID/ADC/SUP order')
    print('Players by role: ' + ', '.join(
        f'{label}={sum(role == value for role in roles.values())}'
        for value, label in zip(ROLES, ROLE_LABELS)))
    for name in players:
        if len(counts[name]) > 1:
            observed = ', '.join(f'{role.upper()}={counts[name][role]}'
                                 for role in ROLES if counts[name][role])
            print(f'Multi-role: {name}: {observed} -> {roles[name].upper()}')
    return roles


def role_similarities(players, similarities, roles):
    """Each unordered pair contributes once; same-role mean is pair-weighted."""
    left, right = torch.triu_indices(len(players), len(players), offset=1)
    values = similarities[left, right].double()
    role_ids = torch.tensor([ROLES.index(roles[name]) for name in players])
    same = role_ids[left] == role_ids[right]
    masks = {f'{label}-{label}': same & (role_ids[left] == index)
             for index, label in enumerate(ROLE_LABELS)}
    masks.update({'same-role': same, 'different-role': ~same})
    return {label: (float(values[mask].mean()) if bool(mask.any()) else float('nan'),
                    int(mask.sum())) for label, mask in masks.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--checkpoint', type=Path, help='Final trained last.pt for one seed')
    source.add_argument('--run-dir', type=Path,
                        help='Experiment directory containing exactly five seed_*/last.pt files')
    parser.add_argument('--player-vocab', type=Path,
                        help='Defaults to player_vocab.json next to each checkpoint')
    parser.add_argument('--player', help='Analyze same-role TOP 10 across seeds, for example Faker')
    parser.add_argument('--role-data', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'data/splits/train.csv',
                        help='Game CSV with actual blue/red player role slots (default: train.csv)')
    args = parser.parse_args()
    try:
        if args.run_dir:
            checkpoints = sorted(args.run_dir.glob('seed_*/last.pt'),
                                 key=lambda path: int(path.parent.name.removeprefix('seed_')))
            if len(checkpoints) != 5:
                raise ValueError(f'Expected exactly five seed checkpoints, found {len(checkpoints)}')
        else:
            checkpoints = [args.checkpoint]
        results = []
        neighbor_lists = []
        candidate_scores = defaultdict(list)
        reference_players = None
        for path in checkpoints:
            players, similarities, epoch = load_player_similarities(path, args.player_vocab)
            if reference_players is None:
                reference_players = players
                roles = load_player_roles(args.role_data, players)
            elif players != reference_players:
                raise ValueError('All seed checkpoints must use the same player vocabulary')
            if args.player:
                name, neighbors = nearest_players(players, similarities, args.player, roles=roles)
                if not neighbor_lists:
                    role_label = ROLE_LABELS[ROLES.index(roles[name])]
                    print(f'\n{name} ({role_label})')
                neighbor_lists.append(neighbors)
                index = players.index(name)
                for i, candidate in enumerate(players):
                    if candidate != name and roles[candidate] == roles[name]:
                        candidate_scores[candidate].append(float(similarities[index, i]))
                print(f'\nSeed {path.parent.name.removeprefix("seed_")} (epoch={epoch}):')
                for rank, (player, score) in enumerate(neighbors, 1):
                    print(f'{rank}. {player} {score:.6f}')
            else:
                print(f'\nCheckpoint: {path.resolve()} (epoch={epoch})')
                print(f'Players: {len(players)}; <UNK> excluded; unique pairs i < j only')
                result = role_similarities(players, similarities, roles)
                results.append(result)
                print('Pair group | Mean cosine similarity | Pairs')
                for label, (score, count) in result.items():
                    print(f'{label} | {score:.6f} | {count}')
        if args.player:
            appearances = Counter(player for neighbors in neighbor_lists for player, _ in neighbors)
            print(f'\nAcross {len(checkpoints)} seeds:')
            print('Player | TOP10 appearances | Mean cosine ± sample std (all seeds)')
            for candidate, count in sorted(appearances.items(),
                                           key=lambda item: (-item[1], -mean(candidate_scores[item[0]]), item[0])):
                scores = candidate_scores[candidate]
                spread = f'{stdev(scores):.6f}' if len(scores) > 1 else 'N/A'
                print(f'{candidate} | {count}/{len(checkpoints)} | {mean(scores):.6f} ± {spread}')
        elif len(results) > 1:
            print(f'\nAcross {len(results)} seeds: mean ± sample standard deviation (ddof=1)')
            for label in results[0]:
                scores = [result[label][0] for result in results]
                if all(result[label][1] > 0 for result in results):
                    print(f'{label} | {mean(scores):.6f} ± {stdev(scores):.6f}')
                else:
                    print(f'{label} | N/A (no pairs)')
    except (ValueError, OSError, KeyError, RuntimeError, csv.Error) as exc:
        parser.exit(1, f'Embedding analysis error: {exc}\n')


if __name__ == '__main__':
    main()
