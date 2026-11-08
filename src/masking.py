"""Choose two distinct champion positions and keep their targets separate."""
import torch

from .dataset import IGNORE_INDEX


UNK_ID = 0
MASK_COUNT = 2
EVAL_MASK_SEED = 42


def mask_champions(champion_ids, mask_id, generator):
    # Give known champions priority. If fewer than two are known, ignore the
    # unknown targets selected to fill the remaining positions.
    priorities = torch.rand(champion_ids.shape, generator=generator, device='cpu').to(champion_ids.device)
    priorities.masked_fill_(champion_ids == UNK_ID, float('inf'))
    positions = priorities.topk(MASK_COUNT, dim=1, largest=False).indices
    targets = champion_ids.gather(1, positions)
    targets.masked_fill_(targets == UNK_ID, IGNORE_INDEX)
    return champion_ids.scatter(1, positions, mask_id), positions, targets
