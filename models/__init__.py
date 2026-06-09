from .graph_transformer import GraphTransformer
from .graphormer import GraphormerFull

from .gin import GIN
from .graphsage import GraphSAGE
from .gcn import GCN

# Optional: mapping string names to classes
MODEL_REGISTRY = {
    "GraphTransformer": GraphTransformer,
    "Graphormer": GraphormerFull,
    "graphormer_full": GraphormerFull,
    "GIN": GIN,
    "GraphSAGE": GraphSAGE,
    "GCN": GCN
}
