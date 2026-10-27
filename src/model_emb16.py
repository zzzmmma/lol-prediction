"""Low-dimensional player/champion embeddings projected into the shared Transformer."""
import torch
from torch import nn

from .model import MatchTransformer


class MatchTransformerEmb16(MatchTransformer):
    def __init__(self, player_vocab_size, champion_vocab_size, embedding_dim=256,
                 num_layers=6, num_heads=8, feedforward_dim=512, dropout=0.1,
                 token_embedding_dim=16):
        if token_embedding_dim != 16 or embedding_dim != 256:
            raise ValueError('Emb16 requires 16D token embeddings and d_model=256')
        super().__init__(player_vocab_size, champion_vocab_size, embedding_dim,
                         num_layers, num_heads, feedforward_dim, dropout)
        self.config['token_embedding_dim'] = token_embedding_dim
        self.player_embedding = nn.Embedding(player_vocab_size, token_embedding_dim)
        self.champion_embedding = nn.Embedding(champion_vocab_size, token_embedding_dim)
        self.player_projection = nn.Linear(token_embedding_dim, embedding_dim)
        self.champion_projection = nn.Linear(token_embedding_dim, embedding_dim)
        for embedding in (self.player_embedding, self.champion_embedding):
            nn.init.normal_(embedding.weight, std=0.02)
        for projection in (self.player_projection, self.champion_projection):
            nn.init.xavier_uniform_(projection.weight)
            nn.init.zeros_(projection.bias)

    def embed_tokens(self, player_ids, champion_ids):
        if player_ids.ndim != 2 or player_ids.shape[1] != 10 or champion_ids.shape != player_ids.shape:
            raise ValueError('Expected player_ids and champion_ids of shape [batch, 10]')
        tokens = torch.cat((self.player_projection(self.player_embedding(player_ids)),
                            self.champion_projection(self.champion_embedding(champion_ids))), dim=1)
        tokens = tokens + self.position_embedding(self.positions) + self.side_embedding(self.sides)
        tokens = tokens + self.role_embedding(self.roles) + self.type_embedding(self.types)
        return self.input_dropout(self.input_norm(tokens))
