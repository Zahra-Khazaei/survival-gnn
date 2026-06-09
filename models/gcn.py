#models/gcn.py

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from .base import BaseGNN


class GCN(BaseGNN):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.2, activation='relu'):
        super(GCN, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.dropout = dropout
        self.activation = self._get_activation(activation)

        self.layers.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.layers.append(GCNConv(hidden_channels, hidden_channels))
        self.layers.append(GCNConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        for conv in self.layers[:-1]:
            x = conv(x, edge_index)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x, edge_index)
        return x

    def _get_activation(self, name):
        if name == 'relu':
            return torch.nn.ReLU()
        elif name == 'leakyrelu':
            return torch.nn.LeakyReLU()
        elif name == 'prelu':
            return torch.nn.PReLU()
        else:
            raise ValueError(f"Unsupported activation: {name}")
