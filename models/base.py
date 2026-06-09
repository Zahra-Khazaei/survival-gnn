import torch
import abc

class BaseGNN(torch.nn.Module, metaclass=abc.ABCMeta):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def forward(self, x, edge_index):
        pass
