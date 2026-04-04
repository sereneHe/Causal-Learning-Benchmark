from __future__ import annotations

from typing import Any

import numpy as np

from ._vendored_castle import ensure_vendored_castle

ensure_vendored_castle()

from castle.algorithms.anm._anm import ANMNonlinear, GPR


def run(context: Any) -> np.ndarray:
    model = ANMNonlinear(alpha=0.05)
    model.learn(data=context.X, regressor=GPR())
    return np.asarray(model.causal_matrix)
