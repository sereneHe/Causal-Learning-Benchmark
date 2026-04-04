from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.gaussian_process.kernels import RBF

from ._vendored_castle import ensure_vendored_castle

ensure_vendored_castle()

from castle.algorithms.anm._anm import ANMNonlinear, GPR


def run(context: Any) -> np.ndarray:
    kernel = 1.0 * RBF(1.0)
    model = ANMNonlinear(alpha=0.05)
    model.learn(data=context.X, regressor=GPR(kernel=kernel))
    return np.asarray(model.causal_matrix)
