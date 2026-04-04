from __future__ import annotations

from typing import Any

import numpy as np

from ._vendored_castle import ensure_vendored_castle

ensure_vendored_castle()

from castle.algorithms import PNL


def run(context: Any) -> np.ndarray:
    model = PNL()
    model.learn(context.X)
    return np.asarray(model.causal_matrix)
