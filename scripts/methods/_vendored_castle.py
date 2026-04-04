from __future__ import annotations

import sys
from importlib import import_module


def ensure_vendored_castle():
    """Expose methods.gcastle under the canonical 'castle' package name."""

    castle_module = sys.modules.get("castle")
    if castle_module is not None:
        return castle_module

    vendored = import_module("methods.gcastle")
    sys.modules["castle"] = vendored
    return vendored
