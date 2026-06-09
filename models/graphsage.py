#graphsage.py

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from .base import BaseGNN

class GraphSAGE(BaseGNN):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.2, activation='relu', **kwargs):
        super(GraphSAGE, self).__init__()

        self.dropout = dropout
        self.activation = self._get_activation(activation)

        # Optional: support custom aggregator
        aggr = kwargs.get('aggr', 'max')  # defaults to 'mean'

        self.layers = torch.nn.ModuleList()
        self.layers.append(SAGEConv(in_channels, hidden_channels, aggr=aggr))
        for _ in range(num_layers - 2):
            self.layers.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))
        self.layers.append(SAGEConv(hidden_channels, out_channels, aggr=aggr))

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
