from __future__ import annotations

from typing import Any

import numpy as np

from ._vendored_castle import ensure_vendored_castle

ensure_vendored_castle()

from castle.algorithms.gradient.corl.torch import CORL


def run(context: Any) -> np.ndarray:
    print("[STAGE] CORL: initializing model with reward_regression_type=GPR.")
    model = CORL(reward_regression_type="GPR")
    print("[STAGE] CORL: starting learn().")
    model.learn(context.X)
    print("[STAGE] CORL: learn() finished.")
    return np.asarray(model.causal_matrix)
