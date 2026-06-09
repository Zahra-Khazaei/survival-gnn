#gin.py:

import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d
from .base import BaseGNN

class GIN(BaseGNN):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.2, activation='relu', train_eps=True):
        super(GIN, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.dropout = dropout

        act = self._get_activation(activation)

        self.layers.append(GINConv(Sequential(
            Linear(in_channels, hidden_channels),
            BatchNorm1d(hidden_channels),
            act,
            Linear(hidden_channels, hidden_channels),
            act
        ), train_eps=train_eps)) 

        for _ in range(num_layers - 2):
            self.layers.append(GINConv(Sequential(
                Linear(hidden_channels, hidden_channels),
                BatchNorm1d(hidden_channels),
                act,
                Linear(hidden_channels, hidden_channels),
                act
            ), train_eps=train_eps))

        self.layers.append(GINConv(Sequential(
            Linear(hidden_channels, out_channels),
            act
        ), train_eps=train_eps))

    def forward(self, x, edge_index):
        for conv in self.layers[:-1]:
            x = conv(x, edge_index)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x, edge_index)
        return x

    def _get_activation(self, name):
        if name == 'relu':
            return ReLU()
        elif name == 'leakyrelu':
            return torch.nn.LeakyReLU()
        elif name == 'prelu':
            return torch.nn.PReLU()
        else:
            raise ValueError(f"Unsupported activation: {name}")
