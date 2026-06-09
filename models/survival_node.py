# models/survival_node.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, SAGEConv, GINConv, GATConv, TransformerConv
from models.graphormer import GraphormerFull


class SurvivalNodeGNN(nn.Module):
    """
    Node-level survival model: outputs discrete-time hazards per node.

    Output:
      hazards: [N, num_bins] in (0,1)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        num_bins: int,
        model_name: str = "GCN",
        dropout: float = 0.2,
        num_heads: int = 4,
        max_degree: int = 40,
    ):
        super().__init__()
        self.model_name = str(model_name).upper()
        self.dropout = float(dropout)
        self.num_bins = int(num_bins)

        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.convs = nn.ModuleList()

        if self.model_name == "GCN":
            self.convs.append(GCNConv(in_channels, hidden_channels, add_self_loops=False))
            for _ in range(num_layers - 1):
                self.convs.append(GCNConv(hidden_channels, hidden_channels, add_self_loops=False))

        elif self.model_name in ["SAGE", "GRAPHSAGE"]:
            self.convs.append(SAGEConv(in_channels, hidden_channels, aggr="max"))
            for _ in range(num_layers - 1):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr="max"))

        elif self.model_name == "GIN":
            mlp0 = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
            )
            self.convs.append(GINConv(mlp0, train_eps=True))
            for _ in range(num_layers - 1):
                mlp = nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.ReLU(),
                    nn.Linear(hidden_channels, hidden_channels),
                )
                self.convs.append(GINConv(mlp, train_eps=True))

        elif self.model_name == "GAT":
            # concat=False keeps output dim = hidden_channels regardless of num_heads
            self.convs.append(GATConv(in_channels, hidden_channels, heads=num_heads, concat=False, add_self_loops=False))
            for _ in range(num_layers - 1):
                self.convs.append(GATConv(hidden_channels, hidden_channels, heads=num_heads, concat=False, add_self_loops=False))

        elif self.model_name in ["GRAPHTRANSFORMER", "TRANSFORMERCONV"]:
            # TransformerConv with concat=False: output dim = hidden_channels
            self.convs.append(TransformerConv(in_channels, hidden_channels, heads=num_heads, concat=False))
            for _ in range(num_layers - 1):
                self.convs.append(TransformerConv(hidden_channels, hidden_channels, heads=num_heads, concat=False))

        elif self.model_name == "GRAPHORMER":
            self.graphormer = GraphormerFull(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                out_channels=hidden_channels,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=float(dropout),
                max_degree=max_degree,
            )

        else:
            raise ValueError(
                f"Unsupported model_name: {model_name}. "
                "Use GCN, GraphSAGE, GIN, GAT, or GraphTransformer."
            )

        self.norms = nn.ModuleList([nn.LayerNorm(hidden_channels) for _ in self.convs])
        self.head = nn.Linear(hidden_channels, num_bins)

    def forward(self, data, edge_index=None):
        x = data.x
        if edge_index is None:
            edge_index = data.edge_index

        if self.model_name == "GRAPHORMER":
            x = self.graphormer(x, edge_index)
        else:
            for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
                h = conv(x, edge_index)
                h = norm(h)
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
                x = h + x if i > 0 else h  # residual from 2nd layer onwards (same dim)

        logits = self.head(x)              # [N, num_bins]
        hazards = torch.sigmoid(logits)    # probabilities in (0,1)
        return hazards