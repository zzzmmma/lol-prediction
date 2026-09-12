"""A small 20-token Transformer with seven independent classification heads."""
import torch
from torch import nn
from torch.nn import functional as F

from .dataset import IGNORE_INDEX, TASK_CLASSES
from .preprocess import TARGETS


class MatchTransformer(nn.Module):
    def __init__(self, player_vocab_size, champion_vocab_size, embedding_dim=256,
                 num_layers=2, num_heads=8, feedforward_dim=512, dropout=0.1):
        super().__init__()
        if embedding_dim != 256 or num_layers not in (2, 3, 4):
            raise ValueError('Baseline requires embedding_dim=256 and 2, 3 or 4 layers')
        if num_heads <= 0 or embedding_dim % num_heads or feedforward_dim <= 0 or not 0 <= dropout < 1:
            raise ValueError('Invalid attention/feedforward/dropout configuration')
        self.config = dict(player_vocab_size=player_vocab_size, champion_vocab_size=champion_vocab_size,
                           embedding_dim=embedding_dim, num_layers=num_layers, num_heads=num_heads,
                           feedforward_dim=feedforward_dim, dropout=dropout)
        # UNK is a real token, not padding: all games have exactly 20 slots.
        self.player_embedding = nn.Embedding(player_vocab_size, embedding_dim)
        self.champion_embedding = nn.Embedding(champion_vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(20, embedding_dim)
        self.side_embedding = nn.Embedding(2, embedding_dim)
        self.role_embedding = nn.Embedding(5, embedding_dim)
        self.type_embedding = nn.Embedding(2, embedding_dim)
        self.register_buffer('positions', torch.arange(20))
        self.register_buffer('sides', torch.tensor(([0] * 5 + [1] * 5) * 2))
        self.register_buffer('roles', torch.tensor(list(range(5)) * 4))
        self.register_buffer('types', torch.tensor([0] * 10 + [1] * 10))
        self.input_norm = nn.LayerNorm(embedding_dim)
        self.input_dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads,
                    dim_feedforward=feedforward_dim, dropout=dropout, activation='gelu',
                    batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers,
                    norm=nn.LayerNorm(embedding_dim), enable_nested_tensor=False)
        self.heads = nn.ModuleDict({task: nn.Linear(embedding_dim, len(classes))
                                    for task, classes in TASK_CLASSES.items()})
        # Encoder layers are cloned by PyTorch; initialize each independently.
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
        for encoder_layer in self.encoder.layers:
            nn.init.xavier_uniform_(encoder_layer.self_attn.in_proj_weight)
            nn.init.zeros_(encoder_layer.self_attn.in_proj_bias)

    def embed_tokens(self, player_ids, champion_ids):
        if player_ids.ndim != 2 or player_ids.shape[1] != 10 or champion_ids.shape != player_ids.shape:
            raise ValueError('Expected player_ids and champion_ids of shape [batch, 10]')
        tokens = torch.cat((self.player_embedding(player_ids), self.champion_embedding(champion_ids)), dim=1)
        tokens = tokens + self.position_embedding(self.positions) + self.side_embedding(self.sides)
        tokens = tokens + self.role_embedding(self.roles) + self.type_embedding(self.types)
        return self.input_dropout(self.input_norm(tokens))

    def forward(self, player_ids, champion_ids):
        encoded = self.encoder(self.embed_tokens(player_ids, champion_ids))
        pooled = encoded.mean(dim=1)  # No additional CLS token; encoder input stays 20.
        return {task: head(pooled) for task, head in self.heads.items()}


def masked_multitask_loss(logits, labels, class_weights=None):
    """Mean of per-task (optionally weighted) means; skip unsupervised tasks."""
    losses = []
    for i, task in enumerate(TARGETS):
        valid = labels[:, i] != IGNORE_INDEX
        weight = None if class_weights is None or task == 'winner_side' else class_weights.get(task)
        if weight is not None:
            weight = weight.to(logits[task])
        if bool(valid.any()):
            if weight is not None and not bool(weight[labels[valid, i]].sum() > 0):
                continue  # CrossEntropy's weighted-mean denominator would be zero.
            losses.append(F.cross_entropy(logits[task][valid], labels[valid, i], weight=weight))
    return torch.stack(losses).mean() if losses else None
