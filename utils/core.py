# utils/core.py

import os
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import contextlib

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, 
    auc, roc_curve, confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import json

import yaml


@torch.no_grad()
def mc_predict_probs(model, x, edge_index, T=50, pos_col=1):
    """
    MC-dropout prediction: run T stochastic passes with dropout enabled.
    Returns:
      mean_prob: (N,) mean P(y=pos_col)
      var_prob:  (N,) variance across passes
      entropy:   (N,) predictive entropy of the mean Bernoulli
    """
    device = next(model.parameters()).device
    x = x.to(device); edge_index = edge_index.to(device)

    probs_T = []
    model.eval()
    with enable_dropout(model):
        for _ in range(T):
            logits = model(x, edge_index)               # [N, C]
            p = F.softmax(logits, dim=1)[:, pos_col]    # [N]
            probs_T.append(p.unsqueeze(0))
    P = torch.cat(probs_T, dim=0)   # [T, N]
    mean_prob = P.mean(dim=0)       # [N]
    var_prob  = P.var(dim=0, unbiased=False)
    p = mean_prob.clamp(1e-8, 1 - 1e-8)
    entropy = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
    return mean_prob, var_prob, entropy


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if "extends" in cfg:
        base_path = os.path.join(os.path.dirname(os.path.abspath(path)), cfg.pop("extends"))
        base = load_config(base_path)
        base.update(cfg)
        return base
    return cfg
    

def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# utils.py
def compute_class_weights(y, device, beta=0.999):
    counts = torch.bincount(y).float()
    effective_num = 1.0 - torch.pow(torch.tensor([beta, beta], device=y.device), counts)
    weights = (1.0 - beta) / (effective_num + 1e-12)
    weights = weights / weights.sum()  # normalize
    return weights.to(device)



def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(model, path, device='cpu'):
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


# def save_metrics(metrics, path):
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     with open(path, 'w') as f:
#         json.dump(metrics, f, indent=4)

def save_metrics_plot(results_dict, save_path="results", prefix=""):
    os.makedirs(save_path, exist_ok=True)
    
    folds = [k for k in results_dict.keys() if k.startswith("fold_")]
    val_accs = [results_dict[f]["val_acc"] for f in folds]
    test_accs = [results_dict[f]["test_acc"] for f in folds]
    val_f1s = [results_dict[f]["val_f1"] for f in folds]
    test_f1s = [results_dict[f]["test_f1"] for f in folds]

    # Accuracy
    plt.figure()
    plt.bar(folds, val_accs, label='Val Accuracy')
    plt.bar(folds, test_accs, label='Test Accuracy', alpha=0.7)
    plt.ylabel("Accuracy")
    plt.title("Accuracy per Fold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{prefix}accuracy_per_fold.png"))
    plt.close()

    # F1 Score
    plt.figure()
    plt.bar(folds, val_f1s, label='Val F1 Score')
    plt.bar(folds, test_f1s, label='Test F1 Score', alpha=0.7)
    plt.ylabel("F1 Score")
    plt.title("F1 Score per Fold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{prefix}f1_per_fold.png"))
    plt.close()


def plot_roc_curve(y_true, y_probs, save_path='results/roc_curve.png'):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, save_path='results/confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig(save_path)
    plt.close()


def plot_learning_curve(train_losses, val_losses, save_path='results/learning_curve.png'):
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()



def get_max_degree(edge_index, num_nodes):
    device = edge_index.device if edge_index is not None else torch.device("cpu")
    degree = torch.zeros(num_nodes, dtype=torch.long, device=device)
    if edge_index is not None:
        _, dst = edge_index
        degree.scatter_add_(0, dst, torch.ones_like(dst))
    return int(degree.max().item())



def find_best_summary_path(optuna_root, model_name, label_col, graph_method, graph_feature):
    model_key = model_name.lower()  
    task = label_col.lower()        
    method = graph_method           
    feature = graph_feature         

    model_dir = os.path.join(optuna_root, f"{model_key}_results")
    prefix = f"{task}_{method}_{feature}_"

    candidates = [d for d in os.listdir(model_dir) if d.startswith(prefix)]
    if not candidates:
        raise ValueError(f"No folder starting with {prefix} found in {model_dir}")

    latest = sorted(candidates)[-1]
    return os.path.join(model_dir, latest, "logs", "best_summary.json")



class FocalLoss(nn.Module):
    """
    Focal Loss for binary or multi-class classification.
    Args:
        alpha (Tensor or list): class weighting factor (e.g., tensor([w_neg, w_pos]))
        gamma (float): focusing parameter (default = 2.0)
        reduction (str): 'mean' or 'sum'
    """
    def __init__(self, alpha=None, gamma=0.5, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits: [N, num_classes]
        # targets: [N]
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)  # p_t = probability of the true class
        loss = ((1 - pt) ** self.gamma) * ce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss





@contextlib.contextmanager
def enable_dropout(model: torch.nn.Module):
    was_training = model.training
    try:
        model.train(True)   # activate dropout during inference
        yield
    finally:
        model.train(was_training)

