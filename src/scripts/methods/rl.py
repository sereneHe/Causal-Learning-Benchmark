from __future__ import annotations

from typing import Any

import numpy as np

from ._vendored_castle import ensure_vendored_castle

ensure_vendored_castle()

from castle.algorithms import RL


def run(context: Any) -> np.ndarray:
    print("[STAGE] RL: initializing model with reg_type=GPR.")
    model = RL(reg_type="GPR")
    print("[STAGE] RL: starting learn().")
    model.learn(context.X)
    print("[STAGE] RL: learn() finished.")
    return np.asarray(model.causal_matrix)
