"""Add player reconstruction to the existing masked-champion models."""
import torch
from torch import nn

from .dataset import IGNORE_INDEX
from .masking import MASK_COUNT, UNK_ID, mask_champions
from .model_masked import MaskedMatchTransformer, MaskedMatchTransformerEmb16


class PlayerMaskMixin:
    def __init__(self, **config):
        super().__init__(**config)
        self.player_mask_id = self.config['player_vocab_size']
        with torch.random.fork_rng(devices=[]):
            self.player_head = nn.Linear(self.config['embedding_dim'], self.player_mask_id)
            nn.init.xavier_uniform_(self.player_head.weight)
            nn.init.zeros_(self.player_head.bias)
            weights = self.player_embedding.weight.detach()
            mask = weights.new_empty((1, weights.shape[1])).normal_(std=0.02)
            self.player_embedding = nn.Embedding.from_pretrained(torch.cat((weights, mask)), freeze=False)

    def forward_masked_both(self, players, champions, player_positions, champion_positions):
        encoded = self.encoder(self.embed_tokens(players, champions))
        outputs = {}
        for kind, positions, head in (
            ('player', player_positions, self.player_head),
            ('champion', champion_positions + players.shape[1], self.champion_head),
        ):
            indices = positions.unsqueeze(-1).expand(-1, -1, encoded.shape[-1])
            logits = head(encoded.gather(1, indices))
            logits[..., UNK_ID] = -torch.inf
            outputs[kind] = logits
        return outputs


class MaskedPlayerTransformer(PlayerMaskMixin, MaskedMatchTransformer):
    pass


class MaskedPlayerTransformerEmb16(PlayerMaskMixin, MaskedMatchTransformerEmb16):
    pass


PLAYER_MODELS = {'6l_256d': MaskedPlayerTransformer, '10l_16d': MaskedPlayerTransformerEmb16}


def mask_tokens(ids, mask_id, count, generator, prefer_unique=False):
    priorities = torch.rand(ids.shape, generator=generator, device='cpu').to(ids.device)
    valid = (ids != UNK_ID) & (ids < mask_id)
    if prefer_unique:
        valid &= (ids[:, :, None] == ids[:, None, :]).sum(dim=-1) == 1
    priorities.masked_fill_(~valid, torch.inf)
    positions = priorities.topk(count, dim=1, largest=False).indices
    targets = ids.gather(1, positions).masked_fill(~valid.gather(1, positions), IGNORE_INDEX)
    return ids.scatter(1, positions, mask_id), positions, targets


def masked_batch(model, batch, device, champion_count, champion_rng, player_rng=None):
    players, champions = batch['player_ids'], batch['champion_ids']
    if champion_count == MASK_COUNT:
        champions, positions, targets = mask_champions(champions, model.mask_id, champion_rng)
    else:
        champions, positions, targets = mask_tokens(champions, model.mask_id, champion_count, champion_rng)
    # A duplicated ID left in another slot is ineligible as a reconstruction target.
    visible = (champions[:, None, :] == targets[:, :, None]).any(dim=-1)
    labels = {'champion': targets.masked_fill(visible, IGNORE_INDEX).to(device)}
    if player_rng is None:
        logits = {'champion': model.forward_masked(players.to(device), champions.to(device), positions.to(device))}
    else:
        players, player_positions, player_targets = mask_tokens(
            players, model.player_mask_id, 1, player_rng, prefer_unique=True)
        labels['player'] = player_targets.to(device)
        logits = model.forward_masked_both(players.to(device), champions.to(device),
                                           player_positions.to(device), positions.to(device))
    return logits, labels
