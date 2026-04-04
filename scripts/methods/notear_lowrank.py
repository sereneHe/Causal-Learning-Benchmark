from __future__ import annotations

from typing import Any

import numpy as np

from ._vendored_castle import ensure_vendored_castle

ensure_vendored_castle()

from castle.algorithms import NotearsLowRank


def run(context: Any) -> np.ndarray:
    model = NotearsLowRank()
    model.learn(context.X, rank=np.linalg.matrix_rank(context.true_dag))
    return np.asarray(model.causal_matrix)
