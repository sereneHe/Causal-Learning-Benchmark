from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass
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
from utils.timer import Timer


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


class CausalPipeline:
    def __init__(self, config: CausalPipelineConfig):
        self.config = config
        self.result_root = config.output_root / f"Results_{config.dataset_name}"
        self.output_dirs = ensure_output_layout(config.output_root, config.dataset_name)

    def run(self) -> None:
        loader = Real_Data_Standardization(self.config.data_root, self.config.dataset_name)
        raw_data, true_dag = loader.standardize_data()
        X, anm_data = self._build_inputs(raw_data)
        true_matrix = true_dag.values if isinstance(true_dag, pd.DataFrame) else np.asarray(true_dag)

        for method in self.config.methods:
            adj_path = self.output_dirs["adj"] / f"{method}_adj.csv"
            if self.config.skip_existing and adj_path.exists():
                continue

            context = MethodContext(
                method=method,
                X=X,
                anm_data=anm_data,
                true_dag=true_matrix,
            )

            with Timer() as timer:
                adj_matrix = self._run_with_timeout(context)

            if adj_matrix is None:
                continue

            save_adjacency_matrix(adj_path, adj_matrix)
            metrics = save_metrics(
                self.output_dirs["score"] / f"Scores_{method}.csv",
                method,
                adj_matrix,
                true_matrix,
                timer.elapsed,
            )
            save_heatmap(
                self.output_dirs["heatmap"] / f"Heatmap_{method}.png",
                adj_matrix,
                true_matrix,
                title=f"{self.config.dataset_name} - {method}",
            )
            print(f"[OK] {method}: {metrics}")

        merge_scores(
            output_root=self.config.output_root,
            dataset_name=self.config.dataset_name,
            methods=list(self.config.methods),
            sid_file=self.config.sid_file,
        )

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

        anm_data = data[
            : min(self.config.anm_features, data.shape[0]),
            : min(self.config.anm_samples, data.shape[1]),
            :,
        ]
        return X, anm_data
