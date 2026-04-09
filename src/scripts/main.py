from __future__ import annotations

from pathlib import Path

import hydra
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig

from run_pipeline import CausalPipelineConfig, CausalPipeline
from run_pipeline import MlflowConfig


def _default_data_root(project_root: Path) -> Path:
    return project_root / "data"


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    project_root = Path(get_original_cwd())
    paths_cfg = cfg.paths
    problem_cfg = cfg.problem
    solver_cfg = cfg.solver
    mlflow_cfg = cfg.mlflow

    data_root = (
        Path(to_absolute_path(paths_cfg.data_root))
        if paths_cfg.data_root
        else _default_data_root(project_root)
    )
    output_root = (
        Path(to_absolute_path(paths_cfg.output_root))
        if paths_cfg.output_root
        else project_root / "output"
    )
    sid_file = Path(to_absolute_path(paths_cfg.sid_file)) if paths_cfg.sid_file else None

    config = CausalPipelineConfig(
        dataset_name=problem_cfg.dataset_name,
        data_root=data_root,
        output_root=output_root,
        methods=list(solver_cfg.methods),
        sample_index=problem_cfg.sample_index,
        time_start=problem_cfg.time_start,
        time_end=problem_cfg.time_end,
        single_time_step=problem_cfg.single_time_step,
        anm_features=problem_cfg.anm_features,
        anm_samples=problem_cfg.anm_samples,
        time_limit=solver_cfg.time_limit,
        skip_existing=solver_cfg.skip_existing,
        sid_file=sid_file,
        mlflow=MlflowConfig(
            enabled=mlflow_cfg.enabled,
            tracking_uri=mlflow_cfg.tracking_uri,
            experiment_name=mlflow_cfg.experiment_name,
            run_name=mlflow_cfg.run_name,
            nested_method_runs=mlflow_cfg.nested_method_runs,
            log_artifacts=mlflow_cfg.log_artifacts,
            tags=dict(mlflow_cfg.tags or {}),
        ),
    )
    CausalPipeline(config).run()


if __name__ == "__main__":
    main()
