from __future__ import annotations

import sys
from pathlib import Path


def ensure_external_method_paths() -> None:
    """Expose local solver sources used by the notebook-style methods."""

    vendored_root = Path(__file__).resolve().parent / "project_bestdagsolverintheworld"
    candidate_roots = [
        vendored_root,
        Path("/Users/xiaoyuhe/project-bestdagsolverintheworld"),
        Path("/Users/xiaoyuhe/Recommender_Pavel"),
        Path("/Users/xiaoyuhe/Causal-Methods/bestdagsolverintheworld-main/src/exdbn"),
        Path("/Users/xiaoyuhe/EXDBN-LLM/OLD-workspace/bestdagsolverintheworld"),
    ]

    for root in reversed(candidate_roots):
        root_str = str(root)
        if root.exists() and root_str not in sys.path:
            sys.path.insert(0, root_str)
