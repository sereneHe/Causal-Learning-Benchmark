from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any

import networkx as nx
import numpy as np


@dataclass
class MethodContext:
    method: str
    X: np.ndarray
    anm_data: np.ndarray
    true_dag: np.ndarray


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
    if method in {"PC-Stable", "PC-Parallel", "Direct-LiNGAM", "ICA-LiNGAM", "PNL", "GES",
                  "Notear-Linear", "Notear-NonLinear", "Notear-Lowrank", "DAG-GNN",
                  "GOLEM", "GraNDAG", "MCSL", "GAE", "RL", "CORL"}:
        return _run_gcastle_method(context)
    if method == "ExMAG":
        return _run_exmag(context)
    if method == "ExDBN":
        return _run_exdbn(context)
    if method == "DyNotear":
        return _run_dynotear(context)
    if method.startswith("ANM-"):
        return _run_anm_placeholder(context)
    raise ValueError(f"Unknown method: {method}")


def _run_gcastle_method(context: MethodContext) -> np.ndarray:
    castle_algorithms = import_module("castle.algorithms")
    if context.method == "PC-Stable":
        model = castle_algorithms.PC(variant="stable")
        model.learn(context.X)
        return np.asarray(model.causal_matrix)
    if context.method == "PC-Parallel":
        model = castle_algorithms.PC(variant="parallel")
        model.learn(context.X, p_cores=2)
        return np.asarray(model.causal_matrix)
    if context.method == "Direct-LiNGAM":
        model = castle_algorithms.DirectLiNGAM()
    elif context.method == "ICA-LiNGAM":
        model = castle_algorithms.ICALiNGAM()
    elif context.method == "PNL":
        model = castle_algorithms.PNL()
    elif context.method == "GES":
        model = castle_algorithms.GES()
    elif context.method == "Notear-Linear":
        model = castle_algorithms.Notears()
    elif context.method == "Notear-NonLinear":
        model = castle_algorithms.NotearsNonlinear()
    elif context.method == "Notear-Lowrank":
        model = castle_algorithms.NotearsLowRank()
        model.learn(context.X, rank=np.linalg.matrix_rank(context.true_dag))
        return np.asarray(model.causal_matrix)
    elif context.method == "GOLEM":
        model = castle_algorithms.GOLEM()
    elif context.method == "GraNDAG":
        model = castle_algorithms.GraNDAG(input_dim=context.X.shape[1])
    elif context.method == "MCSL":
        model = castle_algorithms.MCSL(iter_step=1000, init_rho=1e-5, rho_multiply=10, l1_graph_penalty=2e-3)
    elif context.method == "GAE":
        model = castle_algorithms.GAE(input_dim=context.X.shape[1])
    elif context.method == "RL":
        model = castle_algorithms.RL()
    elif context.method == "DAG-GNN":
        model = import_module("castle.algorithms.gradient.dag_gnn.torch").DAG_GNN()
    elif context.method == "CORL":
        model = import_module("castle.algorithms.gradient.corl.torch").CORL()
    else:
        raise ValueError(context.method)
    model.learn(context.X)
    return np.asarray(model.causal_matrix)


def _run_exmag(context: MethodContext) -> np.ndarray:
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
    return adj_binary(W_est)


def _run_exdbn(context: MethodContext) -> np.ndarray:
    solve_milp = _import_from_candidates(
        "solve_milp",
        ["dagsolvers.solve_exdbn", "dagsolvers.solve_exmag", "solve_exdbn"],
    )
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
    W_est, *_ = solve_milp(context.X, cfg, 0, Y=[], B_ref=context.true_dag)
    return adj_binary(W_est)


def _run_dynotear(context: MethodContext) -> np.ndarray:
    from_pandas_dynamic = _import_from_candidates(
        "from_pandas_dynamic",
        ["notears", "dynotears", "castle.algorithms"],
    )
    import pandas as pd

    X = pd.DataFrame(context.X)
    variables = list(pd.DataFrame(X[0]).columns)
    tabu_edges = [(0, u, v) for u in variables for v in variables]
    _, W_est, _ = from_pandas_dynamic(time_series=X, p=1, tabu_edges=tabu_edges, lambda_a=1e-5)
    return adj_binary(W_est)


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
