#graphormer.py


import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseGNN

class GraphormerFull(BaseGNN):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=3, num_heads=4, dropout=0.2,
                 max_degree=40, max_dist=10, activation='relu', **kwargs):
        super().__init__()

        self.num_heads = num_heads
        self.dropout = dropout
        self.hidden_channels = hidden_channels

        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)

        # Degree and distance encodings
        self.degree_embedding = nn.Embedding(max_degree + 1, hidden_channels)
        self.dist_embedding = nn.Embedding(max_dist + 1, num_heads)

        # Transformer layers with custom attention
        self.layers = nn.ModuleList([
            GraphormerEncoderLayer(hidden_channels, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.output_proj = nn.Linear(hidden_channels, out_channels)
        self.activation = self._get_activation(activation)

    def forward(self, x, edge_index=None):
        N = x.size(0)
        device = x.device

        # === Degree ===
        degree = torch.zeros(N, dtype=torch.long, device=device)
        if edge_index is not None:
            src, dst = edge_index
            degree.scatter_add_(0, dst, torch.ones_like(dst))
        
        
        # Print before clamping to analyze true stats
        # print(f"Max degree: {degree.max().item()}")
        # print(f"Mean degree: {degree.float().mean().item():.2f}")


        degree = degree.clamp(max=self.degree_embedding.num_embeddings - 1)

        # Inspect how many nodes hit the max degree cap
        max_bucket = self.degree_embedding.num_embeddings - 1
        high_deg_count = (degree == max_bucket).sum().item()
        total_nodes = degree.size(0)
        percent = 100 * high_deg_count / total_nodes
        # print(f"Nodes with degree = {max_bucket} (clipped): {high_deg_count}/{total_nodes} ({percent:.2f}%)")

        

        deg_embed = self.degree_embedding(degree)

        x = self.input_proj(x) + deg_embed  # [N, H]

        # === Pairwise distances (shortest path or euclidean) ===
        with torch.no_grad():
            dist = torch.cdist(x, x, p=2.0)  # [N, N]
            dist = torch.clamp(dist, max=10).long()
        dist_bias = self.dist_embedding(dist.to(device))  # [N, N, num_heads]

        # === Apply Transformer layers ===
        x = x.unsqueeze(0)  # [1, N, H]
        for layer in self.layers:
            x = layer(x, dist_bias)

        x = x.squeeze(0)  # [N, H]
        return self.output_proj(x)

    def _get_activation(self, name):
        if name == 'relu':
            return nn.ReLU()
        elif name == 'leakyrelu':
            return nn.LeakyReLU()
        elif name == 'prelu':
            return nn.PReLU()
        else:
            raise ValueError(f"Unsupported activation function: {name}")


class GraphormerEncoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.2):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim)
        )

    def forward(self, x, attn_bias=None):
        B, N, H = x.shape

        if attn_bias is not None:
            # attention bias shape: [N, N, num_heads] → [B*num_heads, N, N]
            bias = attn_bias.permute(2, 0, 1)  # [heads, N, N]
            bias = bias.unsqueeze(0).repeat(B, 1, 1, 1).reshape(B * self.attn.num_heads, N, N)
        else:
            bias = None

        # Multihead attention with optional additive bias
        x_attn, _ = self.attn(x, x, x, attn_mask=bias, need_weights=False)
        x = self.norm1(x + self.dropout(x_attn))
        x_ff = self.ff(x)
        x = self.norm2(x + self.dropout(x_ff))
        return x
