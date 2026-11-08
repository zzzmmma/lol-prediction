"""Baseline transformers with a shared encoder for masked champion prediction."""
import torch
from torch import nn

from .masking import UNK_ID
from .model import MatchTransformer
from .model_emb16 import MatchTransformerEmb16


class MaskedChampionMixin:
    def __init__(self, **config):
        super().__init__(**config)
        self.mask_id = self.config['champion_vocab_size']
        # Retain baseline initialization and RNG state for the unmasked path.
        with torch.random.fork_rng(devices=[]):
            existing = self.champion_embedding.weight.detach()
            mask_embedding = existing.new_empty((1, existing.shape[1])).normal_(std=0.02)
            self.champion_embedding = nn.Embedding.from_pretrained(
                torch.cat((existing, mask_embedding)), freeze=False)
            self.champion_head = nn.Linear(self.config['embedding_dim'], self.mask_id)
            nn.init.xavier_uniform_(self.champion_head.weight)
            nn.init.zeros_(self.champion_head.bias)

    def forward_masked(self, player_ids, masked_champion_ids, positions):
        encoded = self.encoder(self.embed_tokens(player_ids, masked_champion_ids))
        # Champion slots follow the ten player slots.
        indices = (positions + player_ids.shape[1]).unsqueeze(-1).expand(-1, -1, encoded.shape[-1])
        logits = self.champion_head(encoded.gather(1, indices))
        # The input-only mask ID has no output class. Exclude UNK before CE,
        # top-k and softmax. The existing vocabulary has no PAD class.
        logits[..., UNK_ID] = -torch.inf
        return logits


class MaskedMatchTransformer(MaskedChampionMixin, MatchTransformer):
    pass


class MaskedMatchTransformerEmb16(MaskedChampionMixin, MatchTransformerEmb16):
    pass


MASKED_MODELS = {
    '10l_16d': (MaskedMatchTransformerEmb16, 10),
    '6l_256d': (MaskedMatchTransformer, 6),
}


def load_masked_checkpoint(path, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    config = checkpoint.get('run_config', {}).get('config', {})
    if checkpoint.get('format_version') != 1 or 'masked_lambda' not in config:
        raise ValueError(f'Expected a Masked Champion checkpoint: {path}')
    model_class, _ = MASKED_MODELS[config['variant']]
    model = model_class(**checkpoint['model_config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    return model.to(device), checkpoint
