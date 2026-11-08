"""Evaluate ordinary predictions and a fixed set of masked champion targets."""
import torch
from torch.nn import functional as F

from .dataset import IGNORE_INDEX
from .masking import EVAL_MASK_SEED, mask_champions
from .metrics import MetricAccumulator


@torch.inference_mode()
def evaluate_masked(model, loader, device, evaluate_champions=True):
    model.eval()
    metrics = MetricAccumulator()
    generator = torch.Generator().manual_seed(EVAL_MASK_SEED)
    champion_loss = top1 = top5 = valid_samples = 0
    for batch in loader:
        players = batch['player_ids'].to(device)
        champions = batch['champion_ids'].to(device)
        metrics.update(model(players, champions), batch['labels'].to(device))
        if evaluate_champions:
            masked, positions, targets = mask_champions(batch['champion_ids'], model.mask_id, generator)
            valid = targets != IGNORE_INDEX
            if not valid.any():
                continue
            logits = model.forward_masked(players, masked.to(device), positions.to(device))
            scores, truth = logits[valid.to(device)], targets[valid].to(device)
            champion_loss += F.cross_entropy(scores, truth, reduction='sum').item()
            matches = scores.topk(min(5, model.mask_id - 1)).indices == truth[:, None]
            top1 += matches[:, 0].sum().item()
            top5 += matches.any(dim=1).sum().item()
            valid_samples += truth.numel()

    report = metrics.compute()
    if evaluate_champions:
        report['masked_champion'] = {
            'loss': champion_loss / valid_samples if valid_samples else None,
            'top1_accuracy': top1 / valid_samples if valid_samples else None,
            'top5_accuracy': top5 / valid_samples if valid_samples else None,
            'valid_samples': valid_samples,
        }
    return report
