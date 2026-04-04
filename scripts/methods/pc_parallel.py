from __future__ import annotations

from typing import Any

import numpy as np

from ._vendored_castle import ensure_vendored_castle

ensure_vendored_castle()

from castle.algorithms import PC


def run(context: Any) -> np.ndarray:
    model = PC(variant="parallel")
    model.learn(context.X, p_cores=2)
    return np.asarray(model.causal_matrix)
