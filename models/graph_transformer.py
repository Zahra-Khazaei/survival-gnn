# graph_transformer.py

import torch
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv
from .base import BaseGNN

class GraphTransformer(BaseGNN):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.2, activation='relu', num_heads=4, **kwargs):
        super(GraphTransformer, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(TransformerConv(in_channels, hidden_channels, heads=num_heads, concat=False))
        for _ in range(num_layers - 2):
            self.layers.append(TransformerConv(hidden_channels, hidden_channels, heads=num_heads, concat=False))
        self.layers.append(TransformerConv(hidden_channels, out_channels, heads=num_heads, concat=False))
        self.dropout = dropout

        # Set activation
        if activation.lower() == 'prelu':
            self.activation = torch.nn.PReLU()
        elif activation.lower() == 'relu':
            self.activation = torch.nn.ReLU()
        elif activation.lower() == 'leakyrelu':
            self.activation = torch.nn.LeakyReLU()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

    def forward(self, x, edge_index):
        for layer in self.layers[:-1]:
            x = layer(x, edge_index)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x, edge_index)
        return x
