from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class Timer:
    started_at: float = field(default=0.0, init=False)
    elapsed: float = field(default=0.0, init=False)

    def __enter__(self) -> "Timer":
        self.started_at = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        self.elapsed = time.perf_counter() - self.started_at
