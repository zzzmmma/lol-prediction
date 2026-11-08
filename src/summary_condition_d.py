"""Compare Condition D with archived A/B/C validation results without rewriting them."""
from statistics import mean, stdev

from .experiment_extended import read_json
from .preprocess import write_csv
from .train_masked import OUTPUT as CHAMPION_OUTPUT, seed_dir
from .train_masked_player import OUTPUT as PLAYER_OUTPUT
from .train_masked_pretrain import OUTPUT as PRETRAIN_OUTPUT


OUTPUT = PLAYER_OUTPUT / 'condition_d'
STAGES = ('joint', 'pretrain', 'finetune')
COMBINATIONS = tuple(f'player_{p}_champion_{3 - p}_count' for p in range(4))
TYPES = ('masked_player', 'masked_champion', 'masked_token')
METRICS = ('best_epoch', 'winner_accuracy', 'winner_macro_f1',
           *(f'{kind}_{metric}' for kind in TYPES for metric in ('top1_accuracy', 'top5_accuracy', 'valid_samples')),
           'player_0_champion_2_count', *COMBINATIONS)
MASK_PROTOCOL = {
    'A': 'Fixed 0 Player + 2 Champion. Unknown targets excluded.',
    'B': 'Fixed 0 Player + 3 Champion. Unknown and visible duplicate targets excluded.',
    'C': 'Fixed 1 Player + 2 Champion. Player sampling prefers unique known IDs. Visible duplicate Champions excluded.',
    'D': 'Three uniformly sampled positions from 20. The type mix varies. Unknown targets excluded from scoring only.',
}
COMPARISON_NOTE = ('Masked accuracy depends on mask count, type mix and target eligibility. '
                   'Pooled top-k is weighted by scored token counts within each seed.')


def result_dir(condition, stage, variant, seed):
    if condition == 'D':
        return seed_dir(OUTPUT, variant, seed) / stage
    if condition == 'A':
        if stage == 'joint':
            return seed_dir(CHAMPION_OUTPUT / 'lambda_1/validation', variant, seed)
        return seed_dir(PRETRAIN_OUTPUT, variant, seed) / stage
    return seed_dir(PLAYER_OUTPUT / stage / condition, variant, seed)


def validation_row(report, condition, split):
    val = report['validation']
    winner = val.get('tasks', {}).get('winner_side', {})
    row = dict(best_epoch=report['best_epoch'], winner_accuracy=winner.get('accuracy'),
               winner_macro_f1=winner.get('macro_f1'))
    masked = {kind: val[kind] for kind in TYPES[:2] if kind in val}
    if masked:
        count = sum(stats['valid_samples'] for stats in masked.values())
        pooled = {'valid_samples': count}
        for key in ('top1_accuracy', 'top5_accuracy'):
            pooled[key] = (sum(stats[key] * stats['valid_samples'] for stats in masked.values()
                               if stats['valid_samples']) / count if count else None)
        for kind, stats in {**masked, 'masked_token': val.get('masked_token', pooled)}.items():
            row.update({f'{kind}_{key}': stats[key] for key in ('top1_accuracy', 'top5_accuracy', 'valid_samples')})
        if condition == 'D':
            row.update(val['mask_combinations'])
        else:
            fixed_count = {'A': 'player_0_champion_2_count', 'B': COMBINATIONS[0], 'C': COMBINATIONS[1]}
            row.update({key: split['validation_games'] if key == fixed_count[condition] else 0
                        for key in ('player_0_champion_2_count', *COMBINATIONS)})
    return row


def write_summary(path, models, seeds, split):
    rows = []
    for variant in models:
        for stage in STAGES:
            for condition in MASK_PROTOCOL:
                results, selection = [], set()
                for seed in seeds:
                    run = result_dir(condition, stage, variant, seed)
                    if not (run / 'metrics.json').is_file():
                        continue
                    config = read_json(run / 'config.json')
                    if any(config['split'][key] != split[key] for key in ('train_sha256', 'validation_sha256')):
                        raise ValueError(f'Incompatible Regular holdout: {run}')
                    selection.add(config['selection_metric'])
                    results.append(validation_row(read_json(run / 'metrics.json'), condition, split))

                for metric in METRICS:
                    values = [row[metric] for row in results if row.get(metric) is not None]
                    rows.append(dict(condition=condition, model=variant, stage=stage, metric=metric,
                                     seeds=len(values), completed_runs=len(results), requested_seeds=len(seeds),
                                     mean=mean(values) if values else None,
                                     std_ddof1=stdev(values) if len(values) > 1 else None,
                                     winner_head_trained=stage != 'pretrain',
                                     selection_metric=' / '.join(sorted(selection)),
                                     masking_protocol=MASK_PROTOCOL[condition], comparison_note=COMPARISON_NOTE))
    write_csv(path, rows, tuple(rows[0]))
