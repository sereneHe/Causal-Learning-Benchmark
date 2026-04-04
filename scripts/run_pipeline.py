from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from methods.method_runner import MethodContext, run_method
from methods.post_processing import (
    ensure_output_layout,
    merge_scores,
    save_adjacency_matrix,
    save_heatmap,
    save_metrics,
)
from utils.data_loader import Real_Data_Standardization
from utils.mlflow_logger import MlflowLogger
from utils.timer import Timer


@dataclass
class MlflowConfig:
    enabled: bool = True
    tracking_uri: str = "file:./mlruns"
    experiment_name: str = "causality_project"
    run_name: str | None = None
    nested_method_runs: bool = True
    log_artifacts: bool = True
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class CausalPipelineConfig:
    dataset_name: str
    data_root: Path
    output_root: Path
    methods: Sequence[str]
    sample_index: int = 0
    time_start: int = 100
    time_end: int = 200
    single_time_step: int | None = None
    anm_features: int = 3
    anm_samples: int = 8
    time_limit: int = 3600
    skip_existing: bool = False
    sid_file: Path | None = None
    mlflow: MlflowConfig = field(default_factory=MlflowConfig)


class CausalPipeline:
    def __init__(self, config: CausalPipelineConfig):
        self.config = config
        self.result_root = config.output_root / f"Results_{config.dataset_name}"
        self.output_dirs = ensure_output_layout(config.output_root, config.dataset_name)
        self.mlflow = MlflowLogger(config.mlflow)

    def run(self) -> None:
        self.mlflow.start_run()
        self.mlflow.log_params(
            {
                "dataset_name": self.config.dataset_name,
                "data_root": str(self.config.data_root),
                "output_root": str(self.config.output_root),
                "methods": list(self.config.methods),
                "sample_index": self.config.sample_index,
                "time_start": self.config.time_start,
                "time_end": self.config.time_end,
                "single_time_step": self.config.single_time_step,
                "anm_features": self.config.anm_features,
                "anm_samples": self.config.anm_samples,
                "time_limit": self.config.time_limit,
                "skip_existing": self.config.skip_existing,
                "sid_file": str(self.config.sid_file) if self.config.sid_file else None,
                "mlflow": self.config.mlflow,
            }
        )

        loader = Real_Data_Standardization(self.config.data_root, self.config.dataset_name)
        raw_data, true_dag = loader.standardize_data()
        X, anm_data = self._build_inputs(raw_data)
        true_matrix = true_dag.values if isinstance(true_dag, pd.DataFrame) else np.asarray(true_dag)
        self.mlflow.log_params(
            {
                "data_shape.raw": list(np.asarray(raw_data).shape),
                "data_shape.X": list(np.asarray(X).shape),
                "data_shape.anm_data": list(np.asarray(anm_data).shape),
                "data_shape.true_dag": list(np.asarray(true_matrix).shape),
            }
        )

        try:
            for method in self.config.methods:
                adj_path = self.output_dirs["adj"] / f"{method}_adj.csv"
                score_path = self.output_dirs["score"] / f"Scores_{method}.csv"
                heatmap_path = self.output_dirs["heatmap"] / f"Heatmap_{method}.png"
                if self.config.skip_existing and adj_path.exists():
                    self.mlflow.log_metrics({"skipped": 1.0}, prefix=f"method.{method}")
                    continue

                context = MethodContext(
                    method=method,
                    X=X,
                    anm_data=anm_data,
                    true_dag=true_matrix,
                )

                with self.mlflow.method_run(method):
                    with Timer() as timer:
                        adj_matrix = self._run_with_timeout(context)

                    if adj_matrix is None:
                        self.mlflow.log_metrics({"success": 0.0}, prefix=f"method.{method}")
                        continue

                    save_adjacency_matrix(adj_path, adj_matrix)
                    metrics = save_metrics(
                        score_path,
                        method,
                        adj_matrix,
                        true_matrix,
                        timer.elapsed,
                    )
                    save_heatmap(
                        heatmap_path,
                        adj_matrix,
                        true_matrix,
                        title=f"{self.config.dataset_name} - {method}",
                    )
                    self.mlflow.log_metrics({"success": 1.0, **metrics}, prefix=f"method.{method}")
                    if self.config.mlflow.log_artifacts:
                        self.mlflow.log_artifact(adj_path, artifact_path=f"{method}/adj")
                        self.mlflow.log_artifact(score_path, artifact_path=f"{method}/score")
                        self.mlflow.log_artifact(heatmap_path, artifact_path=f"{method}/heatmap")
                    print(f"[OK] {method}: {metrics}")

            merge_scores(
                output_root=self.config.output_root,
                dataset_name=self.config.dataset_name,
                methods=list(self.config.methods),
                sid_file=self.config.sid_file,
            )

            if self.config.mlflow.log_artifacts:
                self.mlflow.log_artifacts(self.result_root, artifact_path="results")
        finally:
            self.mlflow.end_run()

    def _run_with_timeout(self, context: MethodContext) -> np.ndarray | None:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_method, context)
            try:
                return future.result(timeout=self.config.time_limit)
            except concurrent.futures.TimeoutError:
                print(f"[TIMEOUT] {context.method} exceeded {self.config.time_limit}s.")
            except Exception as exc:
                print(f"[FAIL] {context.method}: {exc}")
        return None

    def _build_inputs(self, raw_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        data = np.asarray(raw_data)
        if data.ndim == 2:
            return data, data

        if data.ndim != 3:
            raise ValueError(f"Unsupported raw data shape: {data.shape}")

        if self.config.single_time_step is not None:
            X = data[:, :, self.config.single_time_step].T
        else:
            time_stop = min(self.config.time_end, data.shape[2])
            time_start = min(self.config.time_start, max(0, time_stop - 1))
            X = data[:, self.config.sample_index, time_start:time_stop].T

        # Notebook-local ANM methods use all nodes, then trim sample/time axes
        # like `data[:, :3, :8]` in the original notebook.
        anm_data = data[
            :,
            : min(self.config.anm_features, data.shape[1]),
            : min(self.config.anm_samples, data.shape[2]),
        ]
        return X, anm_data
