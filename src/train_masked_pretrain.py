"""Pretrain on champion reconstruction, then fine-tune the existing tasks."""
import argparse

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .dataset import IGNORE_INDEX, compute_class_weights
from .evaluate import evaluate
from .experiment import SEEDS
from .experiment_extended import read_json
from .masking import EVAL_MASK_SEED, MASK_COUNT, mask_champions
from .metrics import print_metrics
from .model_masked import MASKED_MODELS, load_masked_checkpoint
from .preprocess import fingerprint, write_json
from .train import train_epoch
from .train_extended import (BATCH_SIZE, GRAD_CLIP, LR, MAX_EPOCHS, MODEL_OPTIONS, MODELS, PATIENCE,
                             WEIGHT_DECAY, experiment_config, prepare_extended_split, publish_epoch, save_epoch)
from .train_masked import OUTPUT as JOINT_OUTPUT, seed_dir, summarize, task_scores, validation_result
from .utils import ROOT, load_checkpoint, seed_everything, select_device


OUTPUT = ROOT / 'runs/masked_pretraining'


def masked_pass(model, loader, device, generator, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_sum = top1 = top5 = count = 0
    with torch.set_grad_enabled(training):
        for batch in loader:
            masked, positions, targets = mask_champions(batch['champion_ids'], model.mask_id, generator)
            valid = targets != IGNORE_INDEX
            if not valid.any():
                continue
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model.forward_masked(batch['player_ids'].to(device), masked.to(device), positions.to(device))
            scores, truth = logits[valid.to(device)], targets[valid].to(device)
            loss = F.cross_entropy(scores, truth)
            if not torch.isfinite(loss):
                raise ValueError('Non-finite masked champion loss')
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP, error_if_nonfinite=True)
                optimizer.step()
            matches = scores.detach().topk(min(5, model.mask_id - 1)).indices == truth[:, None]
            loss_sum += loss.item() * truth.numel()
            top1 += matches[:, 0].sum().item()
            top5 += matches.any(dim=1).sum().item()
            count += truth.numel()
    if not count:
        raise ValueError('No known champion targets in this split')
    return {'loss': loss_sum / count, 'top1_accuracy': top1 / count,
            'top5_accuracy': top5 / count, 'valid_samples': count}


def transfer_backbone(model, pretrained, train, variant, seed, split):
    checkpoint = torch.load(pretrained, map_location='cpu', weights_only=True)
    config = checkpoint['run_config']['config']
    if (config['stage'] != 'pretrain' or config['variant'] != variant or config['seed'] != seed
            or config['split'] != split or checkpoint['player_vocab'] != train.player_vocab
            or checkpoint['champion_vocab'] != train.champion_vocab):
        raise ValueError(f'Pretraining checkpoint does not match this run: {pretrained}')
    state = checkpoint['model_state_dict']
    backbone = {name: state[name] for name in model.state_dict() if not name.startswith('heads.')}
    backbone['champion_embedding.weight'] = backbone['champion_embedding.weight'][:len(train.champion_vocab)]
    model.load_state_dict(backbone, strict=False)


def run_stage(output, stage, variant, seed, train, validation, split, device, pretrained=None):
    pretrain = stage == 'pretrain'
    config = {**experiment_config(variant, seed, split, device), 'stage': stage}
    if pretrain:
        config.update(masked_lambda=None, mask_count=MASK_COUNT, eval_mask_seed=EVAL_MASK_SEED,
                      selection_metric='validation_masked_champion_loss')
    else:
        config['pretrained_checkpoint'] = fingerprint(pretrained)
    load = load_masked_checkpoint if pretrain else load_checkpoint
    if (output / 'metrics.json').is_file():
        saved = read_json(output / 'config.json')
        if any(saved[key] != value for key, value in config.items() if key != 'device'):
            raise ValueError(f'Completed stage settings differ: {output}')
        print(f'Already complete: {output}', flush=True)
        return

    seed_everything(seed)
    generator = torch.Generator().manual_seed(seed)
    history = []
    if output.exists():
        model, checkpoint = load(output / 'last.pt', device)
        state = checkpoint['run_config']
        if state['config'] != config:
            raise ValueError(f'Resume settings or data differ: {output}')
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        history = state['history']
        if history:
            publish_epoch(output, history)
        generator.set_state(state['rng']['shuffle'])
        torch.set_rng_state(state['rng']['cpu'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state(state['rng']['device'], device)
        elif device.type == 'mps':
            torch.mps.set_rng_state(state['rng']['device'])
        del checkpoint
    else:
        model_class, layers = (MASKED_MODELS if pretrain else MODELS)[variant]
        model = model_class(player_vocab_size=len(train.player_vocab), champion_vocab_size=len(train.champion_vocab),
                            num_layers=layers, **MODEL_OPTIONS)
        if not pretrain:
            transfer_backbone(model, pretrained, train, variant, seed, split)
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        output.mkdir(parents=True)
        write_json(output / 'config.json', {**config, 'model': model.config})
        save_epoch(output, model, optimizer, config, train, history, generator, device)

    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, generator=generator)
    eval_train = DataLoader(train, batch_size=BATCH_SIZE, shuffle=False)
    eval_val = DataLoader(validation, batch_size=BATCH_SIZE, shuffle=False)
    if not pretrain:
        weights = {task: weight.to(device) for task, weight in compute_class_weights(train.labels).items()}

    best_epoch = history[-1]['best_epoch'] if history else 0
    best_score = history[best_epoch - 1]['selection_score'] if best_epoch else -float('inf')
    for epoch in range(len(history) + 1, MAX_EPOCHS + 1):
        if epoch - 1 - best_epoch >= PATIENCE:
            break
        if pretrain:
            online = masked_pass(model, train_loader, device,
                                 torch.Generator().manual_seed(seed * MAX_EPOCHS + epoch), optimizer)
            val = masked_pass(model, eval_val, device, torch.Generator().manual_seed(EVAL_MASK_SEED))
            score = -val['loss']
        else:
            train_report = train_epoch(model, train_loader, optimizer, device, GRAD_CLIP, weights)
            online = {'loss': train_report['loss'], 'optimization_loss': train_report['optimization_loss'],
                      **task_scores(train_report)}
            eval_report = evaluate(model, eval_train, device)
            online.update({f'eval_{key}': value for key, value in task_scores(eval_report).items()})
            val = task_scores(evaluate(model, eval_val, device))
            score = val['winner_side_macro_f1']
        if score > best_score:
            best_score, best_epoch = score, epoch
        history.append({
            'epoch': epoch, 'best_epoch': best_epoch, 'selection_score': score,
            'lr': optimizer.param_groups[0]['lr'],
            **{f'train_{key}': value for key, value in online.items()},
            **{f'validation_{key}': value for key, value in val.items()},
        })
        save_epoch(output, model, optimizer, config, train, history, generator, device)
        display = f'masked loss {val["loss"]:.4f}' if pretrain else f'Winner Macro F1 {score:.4f}'
        print(f'{variant} seed {seed} | {stage} | epoch {epoch} | validation {display} | best {best_epoch}',
              flush=True)

    best_path = output / 'best.pt'
    best, _ = load(best_path, device)
    report = {'best_epoch': best_epoch, 'last_epoch': history[-1]['epoch'], 'checkpoint': fingerprint(best_path)}
    for name, loader in (('training', eval_train), ('validation', eval_val)):
        if pretrain:
            report[name] = {'masked_champion': masked_pass(
                best, loader, device, torch.Generator().manual_seed(EVAL_MASK_SEED))}
        else:
            report[name] = evaluate(best, loader, device)
            print_metrics(report[name], per_class=False)
    write_json(output / 'metrics.json', report)
    print(f'Output: {output}', flush=True)


def compare(variants, seeds, split, coefficient):
    groups = {}
    joint = JOINT_OUTPUT / f'lambda_{coefficient:g}' / 'validation'
    for variant in variants:
        groups['baseline', variant] = [
            validation_result(JOINT_OUTPUT / 'lambda_0/validation', variant, seed, split, 0) for seed in seeds]
        groups[f'joint_lambda_{coefficient:g}', variant] = [
            validation_result(joint, variant, seed, split, coefficient) for seed in seeds]
        reports = []
        for seed in seeds:
            run = seed_dir(OUTPUT, variant, seed) / 'finetune'
            config = read_json(run / 'config.json')
            expected = experiment_config(variant, seed, split, config['device'])
            if any(config[key] != value for key, value in expected.items()) or config['stage'] != 'finetune':
                raise ValueError(f'Incompatible fine-tuning run: {run}')
            reports.append(read_json(run / 'metrics.json'))
        groups['pretrain_finetune', variant] = reports
    suffix = '_'.join([f'lambda_{coefficient:g}', *variants, *map(str, seeds)])
    summarize(OUTPUT / f'validation_comparison_{suffix}.csv', groups, ('training', 'validation'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('train', 'pretrain', 'finetune', 'compare'))
    parser.add_argument('--model', nargs='+', choices=tuple(MODELS), default=list(MODELS))
    parser.add_argument('--seeds', nargs='+', type=int, choices=SEEDS)
    parser.add_argument('--joint-lambda', type=float, choices=(0.1, 0.3, 1.0), default=1.0,
                        help='Existing joint run for the compare command')
    parser.add_argument('--device', choices=('auto', 'cuda', 'mps', 'cpu'), default='auto')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.threads <= 0:
        parser.error('Use positive --threads')
    seeds = tuple(dict.fromkeys(args.seeds or (SEEDS if args.stage == 'compare' else (137,))))
    variants = tuple(dict.fromkeys(args.model))
    try:
        torch.set_num_threads(args.threads)
        train, validation, split = prepare_extended_split()
        if args.stage == 'compare':
            compare(variants, seeds, split, args.joint_lambda)
            return
        device = select_device(args.device)
        for variant in variants:
            for seed in seeds:
                run = seed_dir(OUTPUT, variant, seed)
                if args.stage in ('train', 'pretrain'):
                    run_stage(run / 'pretrain', 'pretrain', variant, seed, train, validation, split, device)
                if args.stage in ('train', 'finetune'):
                    pretrain = run / 'pretrain'
                    read_json(pretrain / 'metrics.json')
                    run_stage(run / 'finetune', 'finetune', variant, seed, train, validation, split, device,
                              pretrained=pretrain / 'best.pt')
    except (KeyError, ValueError, OSError) as exc:
        parser.exit(1, f'Pretraining experiment error: {exc}\n')


if __name__ == '__main__':
    main()
