from __future__ import annotations

import re
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd


class Real_Data_Standardization:
    def __init__(self, file_path: str | Path, filename: str):
        self.file_path = Path(file_path)
        self.filename = filename

    def standardize_data(self) -> tuple[np.ndarray, pd.DataFrame | np.ndarray]:
        raw_data, true_dag = self._produce_raw_data()
        result_dir = self.file_path / f"Result_{self.filename}" / f"Datasets_{self.filename}"
        result_dir.mkdir(parents=True, exist_ok=True)

        dag_array = true_dag.values if isinstance(true_dag, pd.DataFrame) else np.asarray(true_dag)
        data_name = f"{self.filename}_{dag_array.shape[0]}Nodes_{int(np.count_nonzero(dag_array))}Edges_TS.npz"
        np.savez(result_dir / data_name, x=np.asarray(raw_data), y=dag_array)
        return np.asarray(raw_data), true_dag

    def _produce_raw_data(self) -> tuple[np.ndarray, pd.DataFrame | np.ndarray]:
        direct_npz = self.file_path / f"{self.filename}.npz"
        if direct_npz.exists():
            data = np.load(direct_npz, allow_pickle=True)
            return np.asarray(data["x"]), np.asarray(data["y"])

        direct_csv = self.file_path / f"{self.filename}.csv"
        if direct_csv.exists():
            true_graph_csv = self.file_path / "true_graph.csv"
            if not true_graph_csv.exists():
                raise FileNotFoundError(f"Missing ground truth file: {true_graph_csv}")
            return pd.read_csv(direct_csv), pd.read_csv(true_graph_csv, index_col=0)

        tar_path = self.file_path / f"{self.filename}.tar.gz"
        if tar_path.exists():
            return self._load_tar_gz(tar_path)

        ts_dir = self.file_path / f"{self.filename}_TS"
        if ts_dir.exists():
            return self._load_time_series(ts_dir)

        legacy_ts_dir = self.file_path / self.filename
        if legacy_ts_dir.is_dir() and legacy_ts_dir.name.endswith("_TS"):
            return self._load_time_series(legacy_ts_dir)

        raise FileNotFoundError(
            f"Unable to find dataset '{self.filename}' under {self.file_path}. "
            "Expected .npz, .csv, .tar.gz, or a *_TS directory."
        )

    def _load_tar_gz(self, tar_path: Path) -> tuple[np.ndarray, pd.DataFrame]:
        with tarfile.open(tar_path) as archive:
            archive.extractall(self.file_path)
            names = archive.getnames()
        npy_candidates = [self.file_path / name for name in names if name.endswith(".npy")]
        csv_candidates = [self.file_path / name for name in names if name.endswith(".csv")]
        if not npy_candidates or not csv_candidates:
            raise FileNotFoundError(f"{tar_path} did not contain the expected .npy/.csv files.")
        raw_data = np.load(npy_candidates[0], allow_pickle=True)
        true_dag = pd.read_csv(csv_candidates[0])
        return np.asarray(raw_data), true_dag

    def _load_time_series(self, ts_dir: Path) -> tuple[np.ndarray, pd.DataFrame]:
        series_files = sorted([path for path in ts_dir.iterdir() if path.is_file()])
        if not series_files:
            raise FileNotFoundError(f"No readable files found under {ts_dir}")

        true_graph = self._find_true_graph(ts_dir)
        true_dag = pd.DataFrame(np.load(true_graph, allow_pickle=True)["arr_0"])

        first = pd.read_csv(series_files[0], delimiter="\t", index_col=0, header=None)
        feature_names = np.array(first.index)
        feature_num = len(feature_names)
        sample_num = len(series_files)
        time_num = first.shape[1]
        raw_data = np.zeros((feature_num, sample_num, time_num))

        for sample_index, file_path in enumerate(series_files):
            sample_frame = pd.read_csv(file_path, delimiter="\t", index_col=0, header=None).T
            for feature_index, feature_name in enumerate(feature_names):
                raw_data[feature_index, sample_index, :] = sample_frame[feature_name].to_numpy()
        return raw_data, true_dag

    def _find_true_graph(self, ts_dir: Path) -> Path:
        candidates = [
            ts_dir.parent / "true_graph.npz",
            ts_dir / "true_graph.npz",
            self.file_path / "true_graph.npz",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Missing true_graph.npz near {ts_dir}")


def split_filename(path: Path) -> tuple[str, str]:
    matched = re.match(r"(.+)\.(.+)", path.name)
    if not matched:
        return path.stem, ""
    return matched.group(1), matched.group(2)
