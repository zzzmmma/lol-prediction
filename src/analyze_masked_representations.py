"""Compare saved embedding spaces and training-only champion frequency baselines."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
from statistics import mean, stdev

import torch
from threadpoolctl import threadpool_limits
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .analyze_champion_role_similarity import role_pair_means
from .analyze_masked_embeddings import ROLES, SPECIAL, plots, training_roles
from .dataset import IGNORE_INDEX
from .experiment_extended import read_json
from .masking import EVAL_MASK_SEED, MASK_COUNT, mask_champions
from .preprocess import write_csv, write_json
from .train_extended import BATCH_SIZE, prepare_extended_split
from .train_masked import BASELINE, MASKED_MODELS, OUTPUT as JOINT, SEEDS, seed_dir, validation_result
from .train_masked_pretrain import OUTPUT as PRETRAIN
from .utils import ROOT


def checkpoint_groups(variant, split):
    joint = JOINT / 'lambda_1/validation'
    groups = defaultdict(list)
    for seed in SEEDS:
        zero = JOINT / 'lambda_0/validation'
        baseline = BASELINE if (seed_dir(BASELINE, variant, seed) / 'metrics.json').is_file() else zero
        for method, root, coefficient in (('baseline', baseline, 0), ('joint_lambda_1', joint, 1)):
            report = validation_result(zero if not coefficient else joint, variant, seed, split, coefficient)
            groups[method].append((seed, seed_dir(root, variant, seed) / 'best.pt', report))
        for stage, method in (('finetune', 'pretrain_finetune'), ('pretrain', 'pretrain_only')):
            run = seed_dir(PRETRAIN, variant, seed) / stage
            checkpoint = run / 'best.pt'
            if stage == 'pretrain' and not checkpoint.is_file():
                continue
            report_path = run / 'metrics.json'
            report = read_json(report_path) if report_path.is_file() or stage == 'finetune' else None
            groups[method].append((seed, checkpoint, report))
    return groups


def load_spaces(path, variant, seed, report, train, split):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    config = checkpoint['model_config']
    run = checkpoint['run_config']['config']
    width = 16 if variant == '10l_16d' else 256
    if (run['split'] != split or run['seed'] != seed or run['variant'] != variant
            or config['num_layers'] != MASKED_MODELS[variant][1] or config['embedding_dim'] != 256
            or config.get('token_embedding_dim', 256) != width
            or (report and checkpoint['epoch'] != report['best_epoch'])):
        raise ValueError(f'Incompatible architecture, split, seed, or epoch: {path}')
    if report and 'masked_champion' in report['validation']:
        if (run['eval_mask_seed'], run['mask_count'], run['batch_size']) != (EVAL_MASK_SEED, MASK_COUNT, BATCH_SIZE):
            raise ValueError(f'Incompatible reconstruction evaluation settings: {path}')
    state = checkpoint['model_state_dict']
    spaces = {}
    for kind in ('player', 'champion'):
        vocab = getattr(train, f'{kind}_vocab')
        if vocab != checkpoint[f'{kind}_vocab']:
            raise ValueError(f'Incompatible {kind} vocabulary: {path}')
        names = [name for name in sorted(vocab) if name.casefold() not in SPECIAL]
        embeddings = state[f'{kind}_embedding.weight'][[vocab[name] for name in names]].float()
        spaces[kind, f'raw_{width}d'] = names, embeddings
        if width == 16:
            spaces[kind, 'projected_256d'] = names, F.linear(
                embeddings, state[f'{kind}_projection.weight'].float(), state[f'{kind}_projection.bias'].float())
    return spaces, checkpoint['epoch']


def analyze_space(output, vectors, names, roles, metadata, plot_seeds):
    if not torch.isfinite(vectors).all() or (vectors.norm(dim=1) == 0).any():
        raise ValueError(f'Invalid embedding vectors: {metadata}')
    vectors = F.normalize(vectors, dim=1)
    similarity = (vectors @ vectors.T).clamp(-1, 1)
    groups = role_pair_means(names, similarity, roles)
    left, right = torch.triu_indices(len(names), len(names), offset=1)
    groups['all'] = (similarity[left, right].double().mean().item(), left.numel())
    metrics = [
        dict(**metadata, group=group.replace('JG', 'JUNGLE').replace('SUP', 'SUPPORT'),
             metric='mean_cosine', value=value, observations=count)
        for group, (value, count) in groups.items()
    ]

    # Names are sorted, so stable sorting breaks cosine ties by name.
    similarity.fill_diagonal_(-torch.inf)
    neighbors = similarity.argsort(dim=1, descending=True, stable=True)[:, :10].tolist()
    role_counts = Counter(roles.values())
    role_fraction = defaultdict(list)
    query_rows, neighbor_rows = [], []
    for i, indices in enumerate(neighbors):
        query = names[i]
        role = roles[query]
        hits = sum(role == roles[names[j]] for j in indices)
        observed = hits / len(indices)
        expected = (role_counts[role] - 1) / (len(names) - 1)
        role_fraction[role.upper()].append((observed, expected))
        role_fraction['all'].append((observed, expected))
        query_rows.append(dict(**metadata, query=query, role=role.upper(), same_role_ratio=observed,
                               random_same_role_ratio=expected, delta=observed - expected))
        for rank, j in enumerate(indices, 1):
            neighbor_rows.append(dict(**metadata, query=query, query_role=role.upper(), rank=rank,
                                      neighbor=names[j], neighbor_role=roles[names[j]].upper(),
                                      cosine=similarity[i, j].item(), same_role=role == roles[names[j]]))
    for group, values in role_fraction.items():
        observed, expected = map(mean, zip(*values))
        for metric, value in (('same_role_top10_ratio', observed), ('random_same_role_ratio', expected),
                              ('same_role_top10_delta', observed - expected)):
            metrics.append(dict(**metadata, group=group, metric=metric, value=value, observations=len(values)))
    if metadata['seed'] in plot_seeds:
        title = ' | '.join(str(value) for value in metadata.values())
        path = output.joinpath(*(str(value) for value in metadata.values()))
        path.mkdir(parents=True)
        plots(path, vectors, names, roles, metadata['seed'], title)
    return metrics, query_rows, neighbor_rows


def seed_summary(rows):
    keys = ('model', 'method', 'kind', 'space', 'group', 'metric')
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row['value'])
    return [dict(zip(keys, key), seeds=len(values), mean=mean(values),
                 std_ddof1=stdev(values) if len(values) > 1 else None)
            for key, values in groups.items()]


def rank_counts(counts):
    return sorted(counts, key=lambda champion: (-counts[champion], champion))


def frequency_baselines(train, validation):
    role_counts = [Counter() for _ in ROLES]
    player_counts = defaultdict(Counter)
    global_counts = Counter()
    for players, champions in zip(train.player_ids.tolist(), train.champion_ids.tolist()):
        for position, (player, champion) in enumerate(zip(players, champions)):
            if champion:
                role_counts[position % len(ROLES)][champion] += 1
                player_counts[player][champion] += 1
                global_counts[champion] += 1
    global_rank = rank_counts(global_counts)
    role_ranks = [list(dict.fromkeys(rank_counts(counts) + global_rank))[:5] for counts in role_counts]
    player_ranks = {player: rank_counts(counts)[:5] for player, counts in player_counts.items()}

    hits = {'role_frequency': [0, 0], 'player_frequency': [0, 0]}
    valid_samples = 0
    generator = torch.Generator().manual_seed(EVAL_MASK_SEED)
    for batch in DataLoader(validation, batch_size=BATCH_SIZE, shuffle=False):
        _, positions, targets = mask_champions(batch['champion_ids'], len(train.champion_vocab), generator)
        players = batch['player_ids'].gather(1, positions)
        for position, target, player in zip(positions.flatten().tolist(), targets.flatten().tolist(),
                                            players.flatten().tolist()):
            if target == IGNORE_INDEX:
                continue
            valid_samples += 1
            role_rank = role_ranks[position % len(ROLES)]
            player_rank = list(dict.fromkeys(player_ranks.get(player, []) + role_rank))[:5]
            for method, rank in (('role_frequency', role_rank), ('player_frequency', player_rank)):
                hits[method][0] += target == rank[0]
                hits[method][1] += target in rank
    return {method: dict(top1_accuracy=top1 / valid_samples, top5_accuracy=top5 / valid_samples,
                         valid_samples=valid_samples) for method, (top1, top5) in hits.items()}


def save_masked_comparison(output, rows):
    write_csv(output / 'masked_champion_by_seed.csv', rows, tuple(rows[0]))
    groups = defaultdict(list)
    for row in rows:
        groups[row['model'], row['method']].append(row)
    summary = []
    for (model, method), reports in groups.items():
        for key in ('top1_accuracy', 'top5_accuracy'):
            values = [row[key] for row in reports]
            summary.append(dict(model=model, method=method, metric=key, seeds=len(values) if model != 'shared' else 0,
                                mean=mean(values), std_ddof1=stdev(values) if len(values) > 1 else None,
                                valid_samples=reports[0]['valid_samples']))
    write_csv(output / 'masked_champion_baseline_comparison.csv', summary, tuple(summary[0]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plot-seeds', nargs='+', type=int, choices=SEEDS, default=[137])
    args = parser.parse_args()
    torch.set_num_threads(1)
    train, validation, split = prepare_extended_split()
    output = ROOT / 'runs/masked_analysis' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True)
    frequency = frequency_baselines(train, validation)
    masked_rows = [dict(model='shared', method=method, seed=None, **metrics) for method, metrics in frequency.items()]
    expected_samples = frequency['role_frequency']['valid_samples']
    metrics, queries, neighbors, sources = [], [], [], []
    repeats = Counter()
    roles_by_kind = {}
    with threadpool_limits(limits=1):
        for variant in MASKED_MODELS:
            for method, runs in checkpoint_groups(variant, split).items():
                for seed, path, report in runs:
                    spaces, epoch = load_spaces(path, variant, seed, report, train, split)
                    sources.append(dict(model=variant, method=method, seed=seed, epoch=epoch, path=str(path)))
                    reconstruction = report['validation'].get('masked_champion') if report else None
                    if reconstruction:
                        if reconstruction['valid_samples'] != expected_samples:
                            raise ValueError(f'Validation mask sample count differs: {path}')
                        masked_rows.append(dict(model=variant, method=method, seed=seed,
                                                **{key: reconstruction[key] for key in frequency['role_frequency']}))
                    for (kind, space), (names, vectors) in spaces.items():
                        if kind not in roles_by_kind:
                            roles_by_kind[kind], _ = training_roles(train.rows, kind, names)
                        labels = dict(model=variant, method=method, kind=kind, space=space, seed=seed)
                        scores, fractions, top10 = analyze_space(output / 'plots', vectors, names,
                                                                 roles_by_kind[kind], labels, args.plot_seeds)
                        metrics.extend(scores)
                        queries.extend(fractions)
                        neighbors.extend(top10)
                        repeats.update((variant, method, kind, space, row['query'], row['neighbor']) for row in top10)
    seed_counts = Counter((row['model'], row['method']) for row in sources)
    repeated = [dict(model=model, method=method, kind=kind, space=space, query=query, neighbor=neighbor,
                     top10_seed_count=count, seeds=seed_counts[model, method])
                for (model, method, kind, space, query, neighbor), count in sorted(
                    repeats.items(), key=lambda item: (*item[0][:-1], -item[1], item[0][-1]))]
    summary = seed_summary(metrics)
    tables = {
        'embedding_metrics_by_seed.csv': metrics,
        'cosine_similarity_summary.csv': [row for row in summary if row['metric'] == 'mean_cosine'],
        'role_analysis_summary.csv': summary,
        'top10_same_role_by_query.csv': queries,
        'top10_neighbors_by_seed.csv': neighbors,
        'top10_neighbor_frequency.csv': repeated,
    }
    for filename, rows in tables.items():
        write_csv(output / filename, rows, tuple(rows[0]))
    save_masked_comparison(output, masked_rows)
    write_json(output / 'analysis_config.json', {
        'split': split, 'sources': sources, 'requested_seeds': list(SEEDS), 'plot_seeds': args.plot_seeds,
        'embedding_method': 'Raw weights or learned linear projection including bias, then L2 normalization',
        'role_rule': 'Most appearances in train; ties in TOP/JUNGLE/MID/ADC/SUPPORT order',
        'random_top10_rule': 'Uniform neighbors without replacement excluding the query: (role size - 1)/(vocab size - 1)',
        'frequency_rule': 'Train counts; player ties by champion ID, then role and global fallback',
        'eval_mask_seed': EVAL_MASK_SEED, 'eval_batch_size': BATCH_SIZE,
    })
    print(f'Output: {output}')


if __name__ == '__main__':
    main()
