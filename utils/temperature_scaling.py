# utils/temperature_scaling.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.T = nn.Parameter(torch.ones(1))

    def forward(self, logits):
        return logits / self.T

def fit_temperature_on_val(logits, labels, max_iter=50):
    model = TemperatureScaler().to(logits.device)
    optimizer = torch.optim.LBFGS(model.parameters(), lr=0.01, max_iter=max_iter)
    nll = nn.CrossEntropyLoss()
    def closure():
        optimizer.zero_grad()
        loss = nll(model(logits), labels)
        loss.backward()
        return loss
    optimizer.step(closure)
    return model

# ---------- Vector Scaling (recommended for better ECE) ----------
class VectorScaler(nn.Module):
    def __init__(self, C=2):
        super().__init__()
        self.s = nn.Parameter(torch.ones(C))   # per-class scale
        self.b = nn.Parameter(torch.zeros(C))  # per-class bias

    def forward(self, logits):
        return logits * self.s + self.b

def fit_vector_scaler_on_val(logits, labels, max_iter=200, reg=1e-4):
    """
    Fit vector scaling on validation logits.
    Args:
        logits: [N, C] torch.float (VAL)
        labels: [N] torch.long (VAL)
        reg: small L2 on scales to avoid extreme values
    """
    model = VectorScaler(C=logits.size(1)).to(logits.device)
    opt = torch.optim.LBFGS(model.parameters(), lr=0.1, max_iter=max_iter)
    nll = nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = nll(model(logits), labels)
        if reg and reg > 0:
            loss = loss + reg * (model.s ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return model
