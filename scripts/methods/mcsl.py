from __future__ import annotations

from typing import Any

import numpy as np

from ._vendored_castle import ensure_vendored_castle

ensure_vendored_castle()

from castle.algorithms import MCSL


def run(context: Any) -> np.ndarray:
    model = MCSL(iter_step=1000, init_rho=1e-5, rho_multiply=10, l1_graph_penalty=2e-3)
    model.learn(context.X)
    return np.asarray(model.causal_matrix)
