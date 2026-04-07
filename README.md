# causality_project

This project is a cleaned-up, runnable Python pipeline extracted from `scripts/SereneHE_gCastle_project.ipynb`.

## Overview

- The notebook method set contains 22 methods:
  `PC-Stable`, `PC-Parallel`, `ANM-NCPOLR`, `ANM-GPR`, `ANM-GPR-Kernel`,
  `Direct-LiNGAM`, `ICA-LiNGAM`, `PNL`, `GES`, `ExMAG`, `ExDBN`, `DyNotear`,
  `Notear-Linear`, `Notear-NonLinear`, `Notear-Lowrank`, `DAG-GNN`, `GOLEM`,
  `GraNDAG`, `MCSL`, `GAE`, `RL`, `CORL`.
- gCastle source is vendored under `scripts/methods/gcastle/` from commit
  `f96c9b52d5e38c74c14969dbf6c48b380f3c3dab`.
- Notebook-local external methods are vendored under
  `scripts/methods/project_bestdagsolverintheworld/`.
- Hydra is the main entrypoint format and MLflow is enabled by default.

## Evaluation

- Metrics include `F-score`, `SHD`, `SID` when provided, `FDR`, `TPR`, `FPR`, `nnz`, `Precision`, and `Recall`.
- Heatmaps compare `est_graph` and `truth_graph`.

## Current Structure

```text
.
├── README.md
├── background_logs/
├── conf/
│   ├── config.yaml
│   ├── config-cluster.yaml
│   ├── problem/
│   │   ├── krebs_cycle_1.yaml
│   │   ├── krebs_cycle_3.yaml
│   │   ├── krebs_cycle_normalised_1.yaml
│   │   └── krebs_cycle_normalised_3.yaml
│   ├── solver/
│   │   ├── default.yaml
│   │   ├── gcastle_all_17.yaml
│   │   ├── lightweight.yaml
│   │   ├── local_plus_corl.yaml
│   │   └── notebook_all.yaml
│   └── hydra/
│       └── launcher/
│           └── configured_submitit_slurm.yaml
├── data/
│   ├── Krebs_Cycle_1_TS/
│   ├── Krebs_Cycle_3_TS/
│   ├── Krebs_Cycle_Normalised_1_TS/
│   ├── Krebs_Cycle_Normalised_3_TS/
│   └── true_graph.npz
├── hydra_runs/
├── mlruns/
├── output/
│   ├── Results_Krebs_Cycle_1/
│   └── __plot_smoke__/
└── scripts/
    ├── SereneHE_gCastle_project.ipynb
    ├── main.py
    ├── run_pipeline.py
    ├── plot.py
    ├── data_loader.py
    ├── methods/
    │   ├── method_runner.py
    │   ├── post_processing.py
    │   ├── gcastle/
    │   └── project_bestdagsolverintheworld/
    └── utils/
        ├── mlflow_logger.py
        └── timer.py
```

## Pipeline Layout

The notebook logic is split into these runtime pieces:

- `scripts/main.py`: Hydra entrypoint.
- `scripts/run_pipeline.py`: dataset loading, method execution, timeout handling, metrics, and artifact writing.
- `scripts/data_loader.py`: `Real_Data_Standardization`.
- `scripts/methods/method_runner.py`: routes method names to wrappers.
- `scripts/methods/post_processing.py`: score CSVs, merged outputs, and heatmaps.
- `scripts/plot.py`: notebook-derived `circle_barplot`, `barplot`, and `heatmap` helpers.
- `scripts/utils/mlflow_logger.py`: MLflow run and artifact logging.

## Method Sources

### gCastle wrappers

Local wrappers in `scripts/methods/` expose vendored gCastle implementations for:

- `PC-Stable`
- `PC-Parallel`
- `ANM-GPR`
- `ANM-GPR-Kernel`
- `Direct-LiNGAM`
- `ICA-LiNGAM`
- `PNL`
- `GES`
- `Notear-Linear`
- `Notear-NonLinear`
- `Notear-Lowrank`
- `DAG-GNN`
- `GOLEM`
- `GraNDAG`
- `MCSL`
- `GAE`
- `RL`
- `CORL`

### Notebook-local methods

Project-local implementations or vendored external sources are used for:

- `ANM-NCPOLR`
- `ExMAG`
- `ExDBN`
- `DyNotear`

`ExDAG` is also wired in `local_plus_corl`, but it is not part of the original 22-method `notebook_all` set.

## Data Layout

The loader accepts these forms under `paths.data_root`:

- `<dataset>.npz` with `x` and `y`
- `<dataset>.csv` with `true_graph.csv`
- `<dataset>.tar.gz`
- `<dataset>_TS/*.tsv` with `true_graph.npz`

Typical local setup:

- `conf/config.yaml` points `paths.data_root` to `/Users/xiaoyuhe/Downloads/KrebsCycle`
- repo-local sample data also exists under `data/`

## Hydra Usage

Default run:

```bash
cd /Users/xiaoyuhe/Causal-Methods/krebcycle
python3 scripts/main.py
```

Run `Krebs_Cycle_3`:

```bash
python3 scripts/main.py problem=krebs_cycle_3
```

Run `Krebs_Cycle_Normalised_3`:

```bash
python3 scripts/main.py problem=krebs_cycle_normalised_3
```

Run the 17-method gCastle set:

```bash
python3 scripts/main.py solver=gcastle_all_17
```

Run all 22 notebook methods:

```bash
python3 scripts/main.py solver=notebook_all
```

Run notebook-local methods plus `CORL` and `ExDAG`:

```bash
python3 scripts/main.py solver=local_plus_corl
```

Use a custom subset:

```bash
python3 scripts/main.py \
  'solver.methods=[PC-Stable,Direct-LiNGAM,ExDBN]' \
  solver.time_limit=600
```

Use repo-local data instead of Downloads:

```bash
python3 scripts/main.py \
  paths.data_root='${hydra:runtime.cwd}/data' \
  mlflow.enabled=false
```

Multirun sweep:

```bash
python3 scripts/main.py -m \
  problem=krebs_cycle_1,krebs_cycle_3 \
  solver=lightweight \
  mlflow.enabled=false
```

Cluster config:

```bash
python3 scripts/main.py --config-name config-cluster
```

## Outputs

- Hydra run metadata: `hydra_runs/<dataset>/<timestamp>/`
- MLflow tracking: `mlruns/`
- Method outputs: `output/Results_<dataset>/`
- Background process logs: `background_logs/`

Inside each `Results_<dataset>/` folder:

- `adj_matrices/`
- `score/`
- `heatmap/`
- `merged_scores_<dataset>.csv`
- `sid_<dataset>.csv` if available
- `output_<dataset>.csv`

## Notes

- `config.yaml` uses Hydra `joblib` locally.
- `config-cluster.yaml` switches Hydra to `configured_submitit_slurm`.
- MLflow is enabled by default. Disable it with `mlflow.enabled=false`.
- `solver=notebook_all` is the notebook 22-method set.
- `solver=local_plus_corl` is a project convenience set and includes `ExDAG`.
- `ExMAG`, `ExDBN`, `ExDAG`, and `DyNotear` prefer the vendored copy under `scripts/methods/project_bestdagsolverintheworld/`.
- `ExDAG` currently still needs the external `dagma` Python package at runtime.
- There is no formal `pytest` suite in this project yet; validation is currently based on config expansion, import checks, `py_compile`, and smoke runs.
