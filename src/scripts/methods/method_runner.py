from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from ._external_sources import ensure_external_method_paths


@dataclass
class MethodContext:
    method: str
    X: np.ndarray
    anm_data: np.ndarray
    true_dag: np.ndarray


LOCAL_GCASTLE_METHODS = {
    "PC-Stable": "pc_stable",
    "PC-Parallel": "pc_parallel",
    "ANM-NCPOLR": "anm_ncpolr",
    "ANM-GPR": "anm_gpr",
    "ANM-GPR-Kernel": "anm_gpr_kernel",
    "Direct-LiNGAM": "direct_lingam",
    "ICA-LiNGAM": "ica_lingam",
    "PNL": "pnl",
    "GES": "ges",
    "Notear-Linear": "notear_linear",
    "Notear-NonLinear": "notear_nonlinear",
    "Notear-Lowrank": "notear_lowrank",
    "DAG-GNN": "dag_gnn",
    "GOLEM": "golem",
    "GraNDAG": "grandag",
    "MCSL": "mcsl",
    "GAE": "gae",
    "RL": "rl",
    "CORL": "corl",
}


def adj_binary(weight_matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(weight_matrix)
    binary = np.zeros_like(matrix, dtype=int)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] > 0:
                binary[i, j] = 1
            elif matrix[i, j] < 0:
                binary[j, i] = 1
    return binary


def dag_adj_binary(weight_matrix: np.ndarray) -> np.ndarray:
    graph = nx.DiGraph()
    matrix = np.asarray(weight_matrix)
    graph.add_nodes_from(range(matrix.shape[0]))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] != 0:
                graph.add_edge(i, j, weight=matrix[i, j])

    while True:
        try:
            cycle = nx.find_cycle(graph, orientation="original")
        except nx.NetworkXNoCycle:
            break
        weakest = min(cycle, key=lambda edge: abs(graph[edge[0]][edge[1]]["weight"]))
        graph.remove_edge(weakest[0], weakest[1])

    return nx.to_numpy_array(graph, weight=None).astype(int)


def run_method(context: MethodContext) -> np.ndarray:
    method = context.method
    if method in LOCAL_GCASTLE_METHODS:
        return _run_local_gcastle_method(context)
    if method == "ExMAG":
        return _run_exmag(context)
    if method == "ExDBN":
        return _run_exdbn(context)
    if method == "ExDAG":
        return _run_exdag(context)
    if method == "DyNotear":
        return _run_dynotear(context)
    if method.startswith("ANM-"):
        return _run_anm_placeholder(context)
    raise ValueError(f"Unknown method: {method}")


def _run_local_gcastle_method(context: MethodContext) -> np.ndarray:
    module_name = LOCAL_GCASTLE_METHODS[context.method]
    module = import_module(f"{__package__}.{module_name}")
    return np.asarray(module.run(context))


def _run_exmag(context: MethodContext) -> np.ndarray:
    ensure_external_method_paths()
    solve = _import_from_candidates(
        "solve",
        ["dagsolvers.solve_exmag", "solve_exmag"],
    )
    W_est, *_ = solve(
        context.X,
        lambda1=0,
        loss_type="l2",
        reg_type="l1",
        w_threshold=0.1,
        B_ref=context.true_dag,
        mode="all_cycles",
    )
    return dag_adj_binary(W_est)


def _run_exdbn(context: MethodContext) -> np.ndarray:
    ensure_external_method_paths()
    solve_milp = _import_from_candidates("solve", ["dagsolvers.solve_milp", "solve_milp"])
    cfg = _build_notebook_milp_cfg()
    print(
        "[STAGE] ExDBN: starting MILP solve "
        f"(time_limit={cfg.time_limit}, mip_gap={cfg.target_mip_gap})."
    )
    W_est, *_ = solve_milp(context.X, cfg, 0, Y=[], B_ref=context.true_dag)
    return dag_adj_binary(W_est)


def _run_exdag(context: MethodContext) -> np.ndarray:
    ensure_external_method_paths()
    solve_dagma = _import_from_candidates("solve_dagma", ["dagsolvers.solve_dagma", "solve_dagma"])
    cfg = _build_exdag_dagma_cfg()
    W_est = solve_dagma(context.X, cfg)
    return dag_adj_binary(W_est)


def _run_dynotear(context: MethodContext) -> np.ndarray:
    ensure_external_method_paths()
    from_pandas_dynamic = _import_from_candidates(
        "from_pandas_dynamic",
        ["structure.dynotears", "dynotears"],
    )
    import pandas as pd

    X = pd.DataFrame(context.X)
    _, W_est, _ = from_pandas_dynamic(time_series=X, p=2)
    return dag_adj_binary(W_est)


def _build_notebook_milp_cfg() -> Any:
    omegaconf = import_module("omegaconf")
    cfg = omegaconf.DictConfig({})
    cfg.time_limit = 1800
    cfg.constraints_mode = "weights"
    cfg.callback_mode = "all_cycles"
    cfg.lambda1 = 1
    cfg.lambda2 = 1
    cfg.loss_type = "l2"
    cfg.reg_type = "l1"
    cfg.a_reg_type = "l1"
    cfg.robust = False
    cfg.weights_bound = 100
    cfg.target_mip_gap = 0.001
    cfg.tabu_edges = False
    cfg.plot_dpi = 100
    return cfg


def _build_exdag_dagma_cfg() -> Any:
    omegaconf = import_module("omegaconf")
    cfg_path = (
        Path(__file__).resolve().parent
        / "project_bestdagsolverintheworld"
        / "dagsolvers"
        / "experiments_conf"
        / "solver"
        / "dagma.yaml"
    )
    return omegaconf.OmegaConf.load(cfg_path)


def _run_anm_placeholder(context: MethodContext) -> np.ndarray:
    raise NotImplementedError(
        f"{context.method} still depends on the notebook-local ANM implementation. "
        "Move that class into a dedicated module before enabling it."
    )


def _import_from_candidates(attr: str, module_names: list[str]) -> Any:
    errors: list[str] = []
    for module_name in module_names:
        try:
            module = import_module(module_name)
        except Exception as exc:
            errors.append(f"{module_name}: {exc}")
            continue
        if hasattr(module, attr):
            return getattr(module, attr)
        errors.append(f"{module_name}: missing {attr}")
    joined = "; ".join(errors)
    raise ImportError(f"Unable to import {attr}. Tried {module_names}. Details: {joined}")
