# train.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from data_loader import load_data, load_splits, build_knn_graph, build_graph, create_masks
from model import GraphTransformer
import datetime
from utils import (
    set_seed,
    compute_class_weights,
    load_config,
    get_device,
    save_metrics_plot,
    plot_roc_curve,
    plot_confusion_matrix,
    plot_learning_curve
)

config = load_config()

DATA_PATH = config['data_path']
SPLIT_PATH = config['split_path']
device = get_device()

# FEATURES = ['AGE', 'PSA', 'GLEASON_GLOBAL', 'GLEASON_PRIMARY', 'GLEASON_SECONDARY']
# I added this: test
data = pd.read_csv(DATA_PATH)
FEATURES = [c for c in config.get("features", []) if c in data.columns]

###################

LABEL_COL = config['LABEL_COL']
ID_COL = 'ID'
SAVE_DIR = 'checkpoints'

if config['Graph_feature'] == 'all_feature':
    SAVE_DIR = f"checkpoints_train_{LABEL_COL}_{config['graph_method']}"
    RESULTS_DIR = f"results_train_{LABEL_COL}_{config['graph_method']}"
elif config['Graph_feature'] == 'PSA' or config['Graph_feature'] == 'GLEASON_GLOBAL': 
    SAVE_DIR = f"checkpoints_train_{LABEL_COL}_{config['graph_method']}_{config['Graph_feature']}"
    RESULTS_DIR = f"results_train_{LABEL_COL}_{config['graph_method']}_{config['Graph_feature']}"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ====== Load Everything ======
set_seed(42)
df, X, y, id_to_index = load_data(DATA_PATH, FEATURES, LABEL_COL, ID_COL)




edge_index = build_knn_graph(
    X,
    df=df,
    k=10,
    metric=config["graph_method"],
    graph_feature=config["Graph_feature"]
)



splits = load_splits(SPLIT_PATH)
graph = build_graph(X, edge_index, y)
graph.x = graph.x.to(device)
graph.y = graph.y.to(device)
graph.edge_index = graph.edge_index.to(device)

criterion = torch.nn.CrossEntropyLoss(weight=compute_class_weights(y))

def train(model, data, mask, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[mask], data.y[mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(model, data, mask):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out[mask].argmax(dim=1)
        acc = accuracy_score(data.y[mask].cpu(), pred.cpu())
        f1 = f1_score(data.y[mask].cpu(), pred.cpu())
        probs = F.softmax(out[mask], dim=1)[:, 1].cpu().numpy()
        true = data.y[mask].cpu().numpy()
        auc = roc_auc_score(true, probs)
    return acc, f1, auc

# ====== Cross-Validation Loop ======
all_val_acc, all_val_f1, all_val_auc = [], [], []
all_test_acc, all_test_f1, all_test_auc = [], [], []
results = {}

for fold_name, fold_data in splits.items():
    print(f"\n=== Outer Fold: {fold_name} ===")
    test_idx = torch.tensor([id_to_index[i] for i in fold_data["test"]])

    fold_val_acc, fold_val_f1, fold_val_auc = [], [], []
    fold_test_acc, fold_test_f1, fold_test_auc = [], [], []

    for i in range(1, 5):
        print(f"\n-- Inner Fold {i} --")
        inner = fold_data[f"inner_folds_{i}"][0]
        train_idx = torch.tensor([id_to_index[i] for i in inner["train"]])
        val_idx = torch.tensor([id_to_index[i] for i in inner["validation"]])
        set_seed(42 + i)

        create_masks(graph, train_idx, val_idx, test_idx)

        model = GraphTransformer(
            in_channels=X.size(1),
            hidden_channels=32,
            out_channels=2,
            num_layers=3,
            dropout=0.2
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

        best_val_loss = float('inf')
        patience = 20
        counter = 0
        best_model_state = None
        train_losses, val_losses = [], []

        for epoch in tqdm(range(1, 101)):
            loss = train(model, graph, graph.train_mask, optimizer)
            train_losses.append(loss)
            scheduler.step()

            model.eval()
            with torch.no_grad():
                val_out = model(graph.x, graph.edge_index)
                val_loss = criterion(val_out[graph.val_mask], graph.y[graph.val_mask])
                val_losses.append(val_loss.item())

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        if best_model_state:
            model.load_state_dict(best_model_state)
            fold_dir = os.path.join(SAVE_DIR, fold_name)
            os.makedirs(fold_dir, exist_ok=True)
            torch.save(best_model_state, os.path.join(fold_dir, f'best_model_fold{i}.pth'))
            plot_learning_curve(train_losses, val_losses, save_path=os.path.join(fold_dir, f'learning_curve_fold{i}.png'))

        val_acc, val_f1, val_auc = evaluate(model, graph, graph.val_mask)
        test_acc, test_f1, test_auc = evaluate(model, graph, graph.test_mask)

        fold_val_acc.append(val_acc)
        fold_val_f1.append(val_f1)
        fold_val_auc.append(val_auc)

        fold_test_acc.append(test_acc)
        fold_test_f1.append(test_f1)
        fold_test_auc.append(test_auc)

        print(f"Val Acc: {val_acc:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")
        print(f"Test Acc: {test_acc:.4f}, F1: {test_f1:.4f}, AUC: {test_auc:.4f}")

        with torch.no_grad():
            test_probs = model(graph.x, graph.edge_index)[graph.test_mask]
            test_preds = test_probs.argmax(dim=1).cpu().numpy()
            test_true = graph.y[graph.test_mask].cpu().numpy()
            test_probs_np = test_probs.softmax(dim=1)[:, 1].cpu().numpy()
            fold_plot_dir = os.path.join(fold_dir, "plots")
            os.makedirs(fold_plot_dir, exist_ok=True)
            plot_roc_curve(test_true, test_probs_np, save_path=os.path.join(fold_plot_dir, f"roc_curve_fold{i}.png"))
            plot_confusion_matrix(test_true, test_preds, save_path=os.path.join(fold_plot_dir, f"conf_matrix_fold{i}.png"))

    all_val_acc.append(np.mean(fold_val_acc))
    all_val_f1.append(np.mean(fold_val_f1))
    all_val_auc.append(np.mean(fold_val_auc))

    all_test_acc.append(np.mean(fold_test_acc))
    all_test_f1.append(np.mean(fold_test_f1))
    all_test_auc.append(np.mean(fold_test_auc))

    results[fold_name] = {
        "val_acc": float(np.mean(fold_val_acc)),
        "val_f1": float(np.mean(fold_val_f1)),
        "val_auc": float(np.mean(fold_val_auc)),
        "test_acc": float(np.mean(fold_test_acc)),
        "test_f1": float(np.mean(fold_test_f1)),
        "test_auc": float(np.mean(fold_test_auc))
    }

# ====== Save Metrics and Summary ======
# os.makedirs('results', exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
final_results = {
    "final_val_acc": f"{np.mean(all_val_acc):.4f} ± {np.std(all_val_acc):.4f}",
    "final_val_f1": f"{np.mean(all_val_f1):.4f} ± {np.std(all_val_f1):.4f}",
    "final_val_auc": f"{np.mean(all_val_auc):.4f} ± {np.std(all_val_auc):.4f}",
    "final_test_acc": f"{np.mean(all_test_acc):.4f} ± {np.std(all_test_acc):.4f}",
    "final_test_f1": f"{np.mean(all_test_f1):.4f} ± {np.std(all_test_f1):.4f}",
    "final_test_auc": f"{np.mean(all_test_auc):.4f} ± {np.std(all_test_auc):.4f}"
}

results["final_results"] = final_results

with open(os.path.join(RESULTS_DIR, 'metrics.json'), 'w') as f:
    json.dump(results, f, indent=4)

save_metrics_plot(results, save_path=RESULTS_DIR, prefix=f'{timestamp}_')

print("\n=== FINAL RESULTS ===")
print(f"Validation Accuracy: {final_results['final_val_acc']}")
print(f"Validation F1: {final_results['final_val_f1']}")
print(f"Validation AUC: {final_results['final_val_auc']}")
print(f"Test Accuracy: {final_results['final_test_acc']}")
print(f"Test F1: {final_results['final_test_f1']}")
print(f"Test AUC: {final_results['final_test_auc']}")
