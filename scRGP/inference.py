import torch
import numpy as np

from sklearn.metrics import mean_squared_error as mse
from scipy.stats import pearsonr
from typing import Dict, Tuple


def evaluate(loader, model, device, no_perturb=False) -> Dict:

    model.eval()
    model.to(device)

    pert_cat, pred, truth, ctrl = [], [], [], []
    pred_de, truth_de, ctrl_de = {20: [], 50: [], 100: [], 200: []}, {20: [], 50: [], 100: [], 200: []}, {20: [], 50: [], 100: [], 200: []}

    # Iterate over batches
    for batch in loader:
        batch.to(device)
        pert_cat.extend(batch.pert)

        with torch.no_grad():
            if no_perturb:
                p = batch.x.view(-1, 1)
                p = torch.stack(torch.split(torch.flatten(p), int(batch.x.shape[0] / batch.num_graphs)))
            else:
                p = model(batch)

            x = batch.x.view(p.shape)
            t = batch.y.view(p.shape)

            pred.extend(p.cpu())
            truth.extend(t.cpu())
            ctrl.extend(x.cpu())

            # Save DE gene subsets
            for k in [20, 50, 100, 200]:
                idx_list = getattr(batch, f"de_idx_{k}")
                for i, de_idx in enumerate(idx_list):
                    pred_de[k].append(p[i, de_idx])
                    truth_de[k].append(t[i, de_idx])
                    ctrl_de[k].append(x[i, de_idx])

    # Stack tensors
    results = {
        "pert_cat": np.array(pert_cat),
        "pred": torch.stack(pred).cpu().numpy(),
        "truth": torch.stack(truth).cpu().numpy(),
        "ctrl": torch.stack(ctrl).cpu().numpy()
    }

    for k in [20, 50, 100, 200]:
        results[f"pred_de_{k}"] = torch.stack(pred_de[k]).cpu().numpy()
        results[f"truth_de_{k}"] = torch.stack(truth_de[k]).cpu().numpy()
        results[f"ctrl_de_{k}"] = torch.stack(ctrl_de[k]).cpu().numpy()

    return results


def acc(truth, pred, ctrl) -> float:
    """
    Accuracy of direction (up/down regulation) compared to control.
    """
    truth_ctrl = ((truth - ctrl) >= 0).astype(int)
    pred_ctrl = ((pred - ctrl) >= 0).astype(int)
    return (truth_ctrl == pred_ctrl).sum() / len(truth_ctrl)


def compute_acc(results: Dict) -> Tuple[float, float, float, float]:
    """
    Compute accuracy for DE gene subsets (20, 50, 100, 200).
    """
    accs = []
    for k in [20, 50, 100, 200]:
        truth_k = np.concatenate(results[f"truth_de_{k}"])
        pred_k = np.concatenate(results[f"pred_de_{k}"])
        ctrl_k = np.concatenate(results[f"ctrl_de_{k}"])
        accs.append(acc(truth_k, pred_k, ctrl_k))
    return tuple(accs)


def compute_metrics(results: Dict) -> Tuple[Dict, Dict]:
    """
    Compute performance metrics (MSE, Pearson) for predictions.
    """
    metrics, metrics_pert = {}, {}
    metric2fct = {"mse": mse, "pearson": pearsonr}

    # Init storage
    for m in metric2fct:
        for suffix in ["", "_de_20", "_de_50", "_de_100", "_de_200"]:
            metrics[m + suffix] = []

    for pert in np.unique(results["pert_cat"]):
        metrics_pert[pert] = {}
        p_idx = np.where(results["pert_cat"] == pert)[0]

        # Compute all-genes metrics
        for m, fct in metric2fct.items():
            vals = []
            for idx in p_idx:
                if m == "pearson":
                    vals.append(fct(results["pred"][idx], results["truth"][idx])[0])
                else:
                    vals.append(fct(results["pred"][idx], results["truth"][idx]))
            metrics_pert[pert][m] = np.mean(vals)
            metrics[m].append(metrics_pert[pert][m])

        # Compute DE-gene metrics
        if pert != "ctrl":
            for m, fct in metric2fct.items():
                for k in [20, 50, 100, 200]:
                    vals = []
                    for idx in p_idx:
                        if m == "pearson":
                            vals.append(fct(results[f"pred_de_{k}"][idx], results[f"truth_de_{k}"][idx])[0])
                        else:
                            vals.append(fct(results[f"pred_de_{k}"][idx], results[f"truth_de_{k}"][idx]))
                    metrics_pert[pert][f"{m}_de_{k}"] = np.mean(vals)
                    metrics[f"{m}_de_{k}"].append(metrics_pert[pert][f"{m}_de_{k}"])
        else:
            for m in metric2fct:
                for k in [20, 50, 100, 200]:
                    metrics_pert[pert][f"{m}_de_{k}"] = 0

    # Average over perturbations
    for m in metric2fct:
        metrics[m] = np.mean(metrics[m])
        for k in [20, 50, 100, 200]:
            metrics[f"{m}_de_{k}"] = np.mean(metrics[f"{m}_de_{k}"])

    return metrics, metrics_pert


def print_result(metrics: Dict):
    """
    Print metrics in human-readable format.
    """
    print(f"Overall MSE: {metrics['mse']:.4f}")
    print(f"Overall PCC: {metrics['pearson']:.4f}")
    print(f"Overall R2score: {metrics['r2_score']:.4f}")

    print(
        "Top 20, 50, 100, 200 DEG MSE: "
        f"{metrics['mse_de_20']:.4f}, {metrics['mse_de_50']:.4f}, "
        f"{metrics['mse_de_100']:.4f}, {metrics['mse_de_200']:.4f}"
    )
    print(
        "Top 20, 50, 100, 200 DEG PCC: "
        f"{metrics['pearson_de_20']:.4f}, {metrics['pearson_de_50']:.4f}, "
        f"{metrics['pearson_de_100']:.4f}, {metrics['pearson_de_200']:.4f}"
    )
    print(
        "Top 20, 50, 100, 200 DEG R2score: "
        f"{metrics['r2_score_de_20']:.4f}, {metrics['r2_score_de_50']:.4f}, "
        f"{metrics['r2_score_de_100']:.4f}, {metrics['r2_score_de_200']:.4f}"
    )
