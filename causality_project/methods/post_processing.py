from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


METHOD_CATEGORIES = {
    "PC-Stable": "Constraint-based",
    "PC-Parallel": "Constraint-based",
    "ANM-NCPOLR": "Function-based",
    "ANM-GPR": "Function-based",
    "ANM-GPR-Kernel": "Function-based",
    "Direct-LiNGAM": "Function-based",
    "ICA-LiNGAM": "Function-based",
    "PNL": "Function-based",
    "GES": "Score-based",
    "ExMAG": "Score-based",
    "ExDBN": "Score-based",
    "DyNotear": "Score-based",
    "Notear-Linear": "Gradient-based",
    "Notear-NonLinear": "Gradient-based",
    "Notear-Lowrank": "Gradient-based",
    "DAG-GNN": "Gradient-based",
    "GOLEM": "Gradient-based",
    "GraNDAG": "Gradient-based",
    "MCSL": "Gradient-based",
    "GAE": "Gradient-based",
    "RL": "Gradient-based",
    "CORL": "Gradient-based",
}


def ensure_output_layout(output_root: Path, dataset_name: str) -> dict[str, Path]:
    result_root = output_root / f"Results_{dataset_name}"
    paths = {
        "root": result_root,
        "adj": result_root / "adj_matrices",
        "score": result_root / "score",
        "heatmap": result_root / "heatmap",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def save_adjacency_matrix(path: Path, adj_matrix: np.ndarray) -> None:
    pd.DataFrame(np.asarray(adj_matrix)).to_csv(path, header=False, index=False)


def save_metrics(
    path: Path,
    method: str,
    adj_matrix: np.ndarray,
    true_matrix: np.ndarray,
    runtime_seconds: float,
) -> dict[str, float | str]:
    metrics = compute_metrics(np.asarray(adj_matrix), np.asarray(true_matrix))
    metrics["Method"] = method
    metrics["RuntimeSeconds"] = runtime_seconds
    pd.DataFrame([metrics]).to_csv(path, index=False)
    return metrics


def save_heatmap(path: Path, adj_matrix: np.ndarray, true_matrix: np.ndarray, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(true_matrix, ax=axes[0], cmap="Blues", cbar=False, square=True)
    axes[0].set_title("Ground Truth")
    sns.heatmap(adj_matrix, ax=axes[1], cmap="Reds", cbar=False, square=True)
    axes[1].set_title("Estimated")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def compute_metrics(adj_matrix: np.ndarray, true_matrix: np.ndarray) -> dict[str, float]:
    estimated = (adj_matrix != 0).astype(int)
    truth = (true_matrix != 0).astype(int)

    mask = ~np.eye(truth.shape[0], dtype=bool)
    estimated_edges = estimated[mask]
    true_edges = truth[mask]

    tp = int(np.sum((estimated_edges == 1) & (true_edges == 1)))
    fp = int(np.sum((estimated_edges == 1) & (true_edges == 0)))
    fn = int(np.sum((estimated_edges == 0) & (true_edges == 1)))
    tn = int(np.sum((estimated_edges == 0) & (true_edges == 0)))
    nnz = int(np.sum(estimated_edges))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f_score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fdr = fp / (tp + fp) if tp + fp else 0.0
    tpr = recall
    fpr = fp / (fp + tn) if fp + tn else 0.0
    shd = int(np.sum(np.abs(estimated - truth)))

    return {
        "FDR": fdr,
        "TPR": tpr,
        "FPR": fpr,
        "SHD": shd,
        "nnz": nnz,
        "Precision": precision,
        "Recall": recall,
        "F-score": f_score,
    }


def merge_scores(
    output_root: Path,
    dataset_name: str,
    methods: list[str],
    sid_file: Path | None = None,
) -> None:
    result_root = output_root / f"Results_{dataset_name}"
    score_dir = result_root / "score"
    score_files = sorted(score_dir.glob("Scores_*.csv"))
    if not score_files:
        return

    frames = [pd.read_csv(file) for file in score_files]
    merged = pd.concat(frames, ignore_index=True)
    merged_scores_path = result_root / f"merged_scores_{dataset_name}.csv"
    merged.to_csv(merged_scores_path, index=False)

    if sid_file and sid_file.exists():
        sid_df = pd.read_csv(sid_file)
    else:
        sid_df = pd.DataFrame(columns=["Method", "SID"])
    sid_output_path = result_root / f"sid_{dataset_name}.csv"
    sid_df.to_csv(sid_output_path, index=False)

    category_df = pd.DataFrame(
        {"Method": methods, "Category": [METHOD_CATEGORIES.get(method, "Unknown") for method in methods]}
    )
    final_df = category_df.merge(merged, on="Method", how="left").merge(sid_df, on="Method", how="left")
    final_df.to_csv(result_root / f"output_{dataset_name}.csv", index=False)
