"""Compare raw embedding geometry using matched holdout checkpoints."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
from statistics import mean, stdev

import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from threadpoolctl import threadpool_limits
from torch.nn.functional import normalize

from visualize_embeddings import ROLES, save_plot

from .analyze_champion_role_similarity import role_pair_means
from .analyze_player_embedding import nearest_players
from .preprocess import write_csv, write_json
from .train_masked import (BASELINE, MASKED_MODELS, OUTPUT, SEEDS, prepare_extended_split,
                           seed_dir, validation_result)
from .utils import ROOT


SPECIAL = {'<unk>', '<pad>', '<mask>', '[unk]', '[pad]', '[mask]'}


def training_roles(rows, kind, names):
    counts = defaultdict(Counter)
    columns = [(f'{side}_{kind}_{role.lower()}', role) for side in ('blue', 'red') for role in ROLES]
    for row in rows:
        for column, role in columns:
            counts[row[column]][role] += 1
    roles = {name: max(ROLES, key=counts[name].__getitem__) for name in names}
    queries = [min((name for name in names if roles[name] == role),
                   key=lambda name: (-counts[name][role], name)) for role in ROLES]
    return {name: role.lower() for name, role in roles.items()}, queries


def load_embeddings(path, train, variant, best_epoch):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    width = 16 if variant == '10l_16d' else 256
    config = checkpoint['model_config']
    if (config['num_layers'] != MASKED_MODELS[variant][1] or config['embedding_dim'] != 256
            or config.get('token_embedding_dim', 256) != width or checkpoint['epoch'] != best_epoch):
        raise ValueError(f'Checkpoint does not match the selected architecture or epoch: {path}')
    state = checkpoint['model_state_dict']
    result = {}
    for kind in ('player', 'champion'):
        vocab = getattr(train, f'{kind}_vocab')
        if checkpoint[f'{kind}_vocab'] != vocab:
            raise ValueError(f'{kind} vocabulary differs from the holdout training data: {path}')
        names = [name for name in sorted(vocab) if name.casefold() not in SPECIAL]
        weights = state[f'{kind}_embedding.weight']
        # The extra input-only mask row lies outside the real champion vocabulary.
        expected_rows = len(vocab) + int(kind == 'champion' and 'champion_head.weight' in state)
        if weights.shape != (expected_rows, width):
            raise ValueError(f'Unexpected {kind} embedding shape: {path}')
        vectors = weights[[vocab[name] for name in names]].float()
        if not torch.isfinite(vectors).all() or (vectors.norm(dim=1) == 0).any():
            raise ValueError(f'Invalid {kind} embedding values: {path}')
        result[kind] = names, normalize(vectors, dim=1)
    return result


def plots(output, vectors, names, roles, seed, title):
    labels = [roles[name].upper() for name in names]
    points = vectors.numpy()
    pca = PCA(n_components=2).fit_transform(points)
    tsne = TSNE(n_components=2, random_state=seed, init='pca', learning_rate='auto',
                perplexity=min(30, len(names) - 1)).fit_transform(points)
    save_plot(pca, names, labels, f'{title} | PCA', ('PC1', 'PC2'), output / 'pca.png', False)
    save_plot(tsne, names, labels, f'{title} | t-SNE', ('t-SNE 1', 't-SNE 2'), output / 'tsne.png', False)


def analyze_kind(output, names, by_seed, roles, queries, plot_seed, title):
    output.mkdir(parents=True)
    matrices = {seed: (vectors @ vectors.T).clamp(-1, 1) for seed, vectors in by_seed.items()}
    left, right = torch.triu_indices(len(names), len(names), offset=1)
    pairs = torch.stack([matrix[left, right] for matrix in matrices.values()])
    averages = pairs.mean(dim=0).tolist()
    spreads = pairs.std(dim=0, correction=1).tolist()
    pair_rows = (
        dict(name_a=names[i], name_b=names[j], role_a=roles[names[i]].upper(),
             role_b=roles[names[j]].upper(), mean_cosine=average, std_ddof1=spread)
        for i, j, average, spread in zip(left.tolist(), right.tolist(), averages, spreads)
    )
    write_csv(output / 'pairwise_cosine.csv', pair_rows,
              ('name_a', 'name_b', 'role_a', 'role_b', 'mean_cosine', 'std_ddof1'))

    neighbors, frequency = [], []
    name_to_index = {name: i for i, name in enumerate(names)}
    for query in queries:
        counts = Counter()
        for seed, matrix in matrices.items():
            name, top10 = nearest_players(names, matrix, query)
            counts.update(candidate for candidate, _ in top10)
            for rank, (candidate, score) in enumerate(top10, 1):
                neighbors.append(dict(seed=seed, query=name, rank=rank, neighbor=candidate, cosine=score))
        for candidate, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            i, j = name_to_index[name], name_to_index[candidate]
            scores = [matrix[i, j].item() for matrix in matrices.values()]
            frequency.append(dict(query=name, neighbor=candidate, top10_seed_count=count,
                                  seeds=len(SEEDS), mean_cosine=mean(scores), std_ddof1=stdev(scores)))
    write_csv(output / 'top10_by_seed.csv', neighbors, tuple(neighbors[0]))
    write_csv(output / 'top10_frequency.csv', frequency, tuple(frequency[0]))

    plots(output, by_seed[plot_seed], names, roles, plot_seed, f'{title} | Seed {plot_seed}')
    return {(seed, group.replace('JG', 'JUNGLE').replace('SUP', 'SUPPORT')): (score, count)
            for seed, matrix in matrices.items()
            for group, (score, count) in role_pair_means(names, matrix, roles).items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lambda', dest='coefficient', type=float, choices=(0.1, 0.3, 1.0), default=1.0)
    parser.add_argument('--plot-seed', type=int, choices=SEEDS, default=137)
    parser.add_argument('--player', action='append', help='Representative player name; repeat to add more')
    parser.add_argument('--champion', action='append', help='Representative champion name; repeat to add more')
    args = parser.parse_args()
    torch.set_num_threads(1)
    train, _, split = prepare_extended_split()
    condition = f'lambda_{args.coefficient:g}'
    output = ROOT / 'runs/masked_embedding_analysis' / condition / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True)
    sources, role_rows, summary = [], [], []
    with threadpool_limits(limits=1):
        for variant in MASKED_MODELS:
            scores = {}
            for label, coefficient in (('baseline', 0.0), (condition, args.coefficient)):
                root = OUTPUT / f'lambda_{coefficient:g}' / 'validation'
                embeddings = {}
                for seed in SEEDS:
                    report = validation_result(root, variant, seed, split, coefficient)
                    checkpoint = report.get('checkpoint', {})
                    # The validated run directory is authoritative even after moving a workspace.
                    run_root = root
                    if not coefficient and (seed_dir(BASELINE, variant, seed) / 'metrics.json').is_file():
                        run_root = BASELINE
                    path = seed_dir(run_root, variant, seed) / 'best.pt'
                    embeddings[seed] = load_embeddings(path, train, variant, report['best_epoch'])
                    sources.append(dict(model=variant, condition=label, seed=seed, path=str(path),
                                        epoch=report['best_epoch'], sha256=checkpoint.get('sha256')))
                for kind in ('player', 'champion'):
                    names = embeddings[SEEDS[0]][kind][0]
                    by_seed = {seed: embeddings[seed][kind][1] for seed in SEEDS}
                    roles, defaults = training_roles(train.rows, kind, names)
                    queries = getattr(args, kind) or defaults
                    result = analyze_kind(output / variant / label / kind, names, by_seed, roles, queries,
                                          args.plot_seed, f'{variant} | {label} | {kind.capitalize()}')
                    scores[label, kind] = result
                    for (seed, group), (score, count) in result.items():
                        role_rows.append(dict(model=variant, condition=label, kind=kind, seed=seed,
                                              group=group, pairs=count, mean_cosine=score))
            for kind in ('player', 'champion'):
                baseline, masked = (scores[label, kind] for label in ('baseline', condition))
                for group in dict.fromkeys(group for _, group in baseline):
                    base = [baseline[seed, group][0] for seed in SEEDS]
                    joint = [masked[seed, group][0] for seed in SEEDS]
                    differences = [right - left for left, right in zip(base, joint)]
                    summary.append(dict(model=variant, kind=kind, group=group, seeds=len(SEEDS),
                                        baseline_mean=mean(base), baseline_std_ddof1=stdev(base),
                                        masked_mean=mean(joint), masked_std_ddof1=stdev(joint),
                                        paired_delta_mean=mean(differences), paired_delta_std_ddof1=stdev(differences)))
    write_csv(output / 'role_cosine_by_seed.csv', role_rows, tuple(role_rows[0]))
    write_csv(output / 'role_cosine_comparison.csv', summary, tuple(summary[0]))
    write_json(output / 'analysis_config.json', {
        'lambda': args.coefficient, 'seeds': list(SEEDS), 'plot_seed': args.plot_seed, 'split': split,
        'sources': sources, 'role_rule': 'Most appearances in holdout train; ties follow TOP/JUNGLE/MID/ADC/SUPPORT',
        'queries': {'player': args.player, 'champion': args.champion},
        'default_query_rule': 'Most appearances in each primary role',
        'method': 'Raw, L2-normalized embeddings; cosine statistics aggregated across seeds, no vector averaging',
        'plots': 'Separate fits for each architecture and condition; axes do not align across fits',
    })
    print(f'Output: {output}')


if __name__ == '__main__':
    main()
