from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.preprocessing import scale

from ._vendored_castle import ensure_vendored_castle

ensure_vendored_castle()

import gurobipy as gp
from gurobipy import GRB

from castle.common import BaseLearner
from castle.common.independence_tests import hsic_test


class NCPOLR:
    """Notebook-local nonlinear regressor backed by a small Gurobi model."""

    def __init__(self, time_limit_seconds: int = 6 * 60) -> None:
        self.time_limit_seconds = time_limit_seconds

    def estimate(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        X = np.asarray(x, dtype=float)
        Y = np.asarray(y, dtype=float)

        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()

        model = gp.Model(env=env)
        model.setParam("TimeLimit", self.time_limit_seconds)

        upper_bound = 3
        regularization = 0.125

        if X.ndim < 2:
            variable_count = 1
            sample_count = X.shape[0]
            X = X.reshape((-1, 1))
        else:
            variable_count = X.shape[1]
            sample_count = X.shape[0]

        f = model.addVars(variable_count, sample_count, name="f", vtype=GRB.CONTINUOUS)
        phi = model.addVars(upper_bound, sample_count + 1, name="phi", vtype=GRB.CONTINUOUS)
        G = model.addVars(upper_bound, upper_bound, name="G", vtype=GRB.CONTINUOUS)
        F = model.addVars(variable_count, upper_bound, name="F", vtype=GRB.CONTINUOUS)
        q = model.addVars(upper_bound, sample_count, name="q", vtype=GRB.CONTINUOUS)
        p = model.addVars(variable_count, sample_count, name="p", vtype=GRB.CONTINUOUS)
        quatr = model.addVars(variable_count, sample_count + 1, name="quatr", vtype=GRB.CONTINUOUS)
        quatr_hidden = model.addVars(upper_bound, sample_count, name="quatr_hidden", vtype=GRB.CONTINUOUS)
        z_squared = model.addVars(sample_count, variable_count, name="z_squared", vtype=GRB.CONTINUOUS)

        objective = gp.quicksum(
            z_squared[t, m] for m in range(variable_count) for t in range(sample_count)
        )
        objective += gp.quicksum(
            regularization * q[n, t] ** 2 for n in range(upper_bound) for t in range(sample_count)
        )
        objective += gp.quicksum(
            regularization * p[m, t] ** 2 for m in range(variable_count) for t in range(sample_count)
        )
        model.setObjective(objective, GRB.MINIMIZE)

        model.addConstrs(
            z_squared[t, m] == (float(X[t, m]) - f[m, t]) ** 2
            for t in range(sample_count)
            for m in range(variable_count)
        )
        for t in range(sample_count):
            for m in range(variable_count):
                model.addConstr(
                    gp.quicksum(F[m, n] * phi[n, t + 1] for n in range(upper_bound)) == quatr[m, t + 1]
                )
        model.addConstr(
            gp.quicksum(
                f[m, t] - quatr[m, t + 1] - p[m, t]
                for t in range(sample_count)
                for m in range(variable_count)
            )
            == 0
        )
        for t in range(sample_count):
            for n in range(upper_bound):
                model.addConstr(
                    gp.quicksum(G[n, nn] * phi[nn, t] for nn in range(upper_bound))
                    == quatr_hidden[n, t]
                )
        model.addConstr(
            gp.quicksum(
                phi[n, t + 1] - quatr_hidden[n, t] - q[n, t]
                for t in range(sample_count)
                for n in range(upper_bound)
            )
            == 0
        )

        model.Params.NonConvex = 2
        model.optimize()

        solution = model.getAttr("x", f)
        y_pred = np.array([solution[m, t] for m in range(variable_count) for t in range(sample_count)])
        return y_pred.reshape(variable_count, sample_count).T


class NotebookANMNonlinear(BaseLearner):
    """Notebook ANM implementation that operates on [node, sample, time] tensors."""

    def __init__(self, alpha: float = 0.05) -> None:
        super().__init__()
        self.alpha = alpha

    def learn(
        self,
        data: np.ndarray,
        columns: Any = None,
        regressor: NCPOLR | None = None,
        test_method=hsic_test,
        **kwargs: Any,
    ) -> None:
        del columns, kwargs
        regressor = regressor or NCPOLR()

        data_array = np.asarray(data, dtype=float)
        if data_array.ndim != 3:
            raise ValueError(
                "Notebook ANM-NCPOLR expects data with shape [node, sample, time]. "
                f"Got {data_array.shape}."
            )

        node_num = data_array.shape[0]
        self.causal_matrix = np.zeros((node_num, node_num), dtype=int)

        for i in range(node_num):
            for j in range(i + 1, node_num):
                x = data_array[i, :, :]
                y = data_array[j, :, :]

                if test_method(x, y, alpha=self.alpha):
                    continue

                if self.anm_estimate(x, y, regressor=regressor, test_method=test_method):
                    self.causal_matrix[i, j] = 1
                if self.anm_estimate(y, x, regressor=regressor, test_method=test_method):
                    self.causal_matrix[j, i] = 1

    def anm_estimate(
        self,
        x: np.ndarray,
        y: np.ndarray,
        regressor: NCPOLR | None = None,
        test_method=hsic_test,
    ) -> bool:
        regressor = regressor or NCPOLR()
        x_scaled = scale(x).reshape((-1, 1))
        y_scaled = scale(y).reshape((-1, 1))
        y_predict = regressor.estimate(x_scaled, y_scaled)
        return bool(test_method(y_scaled - y_predict, x_scaled, alpha=self.alpha))


def run(context: Any) -> np.ndarray:
    model = NotebookANMNonlinear(alpha=0.05)
    model.learn(data=context.anm_data, regressor=NCPOLR())
    return np.asarray(model.causal_matrix)
