# evaluate.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import torch

from metrics.survival_metrics import evaluate_survival


def evaluate_survival_model_on_graph(
    model,
    graph,
    cfg: dict,
    device: str = "cuda",
    save_path: str | None = None,
):
    """
    Evaluate a trained survival GNN on train/val/test splits of a single graph.

    NOTE (logits version):
      model(graph) returns hazard_logits [N, T]
      we apply sigmoid here to get hazards for evaluate_survival(...)
    """

    bin_width = float(cfg.get("bin_width", 6.0))
    horizons_months = cfg.get("horizons_months", [60.0, 120.0])
    horizons_months = [float(h) for h in horizons_months]

    model.eval()
    graph = graph.to(device)

    with torch.no_grad():
        hazards = model(graph)  # [N, T] in (0,1) — model already applies sigmoid

    results = {}

    splits = {
        "train": getattr(graph, "train_mask", None),
        "val": getattr(graph, "val_mask", None),
        "test": getattr(graph, "test_mask", None),
    }

    for split_name, mask in splits.items():
        if mask is None:
            continue

        mask = mask.to(device)
        if mask.sum().item() == 0:
            continue

        metrics = evaluate_survival(
            hazards=hazards[mask],
            times=graph.time[mask],
            events=graph.event[mask],
            bin_width=bin_width,
            horizons_months=horizons_months,
        )
        results[split_name] = metrics

    if save_path is not None:

        def _to_python(obj):
            if isinstance(obj, dict):
                return {k: _to_python(v) for k, v in obj.items()}
            if hasattr(obj, "item"):
                try:
                    return obj.item()
                except Exception:
                    return obj
            if isinstance(obj, (list, tuple)):
                return [_to_python(v) for v in obj]
            return obj

        with open(save_path, "w") as f:
            json.dump(_to_python(results), f, indent=2)

    return results