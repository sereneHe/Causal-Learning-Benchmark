from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterator


def _flatten_dict(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if is_dataclass(value):
            flat.update(_flatten_dict(asdict(value), prefix=full_key))
        elif isinstance(value, dict):
            flat.update(_flatten_dict(value, prefix=full_key))
        elif isinstance(value, (list, tuple)):
            flat[full_key] = ",".join(str(item) for item in value)
        elif value is None:
            flat[full_key] = "null"
        else:
            flat[full_key] = value
    return flat


class MlflowLogger:
    def __init__(self, config: Any):
        self.config = config
        self.enabled = bool(config.enabled)
        self._mlflow = None
        self._active_run = None

        if self.enabled:
            import mlflow

            self._mlflow = mlflow
            mlflow.set_tracking_uri(config.tracking_uri)
            mlflow.set_experiment(config.experiment_name)

    def start_run(self) -> None:
        if not self.enabled:
            return

        run_name = self.config.run_name or None
        tags = dict(self.config.tags or {})
        self._active_run = self._mlflow.start_run(run_name=run_name, tags=tags)

    def end_run(self) -> None:
        if not self.enabled or self._active_run is None:
            return

        self._mlflow.end_run()
        self._active_run = None

    def log_params(self, params: dict[str, Any]) -> None:
        if not self.enabled:
            return

        for key, value in _flatten_dict(params).items():
            self._mlflow.log_param(key, value)

    def log_metrics(self, metrics: dict[str, Any], prefix: str | None = None) -> None:
        if not self.enabled:
            return

        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                metric_key = f"{prefix}.{key}" if prefix else key
                self._mlflow.log_metric(metric_key, float(value))

    def log_artifact(self, path: Path, artifact_path: str | None = None) -> None:
        if not self.enabled or not path.exists():
            return

        self._mlflow.log_artifact(str(path), artifact_path=artifact_path)

    def log_artifacts(self, path: Path, artifact_path: str | None = None) -> None:
        if not self.enabled or not path.exists():
            return

        self._mlflow.log_artifacts(str(path), artifact_path=artifact_path)

    @contextmanager
    def method_run(self, method: str) -> Iterator[None]:
        if not self.enabled or not self.config.nested_method_runs:
            yield
            return

        self._mlflow.start_run(run_name=method, nested=True)
        try:
            yield
        finally:
            self._mlflow.end_run()
