# causality_project

This project is a cleaned-up Python version of `SereneHE_gCastle_project.ipynb`.

## Structure

```text
causality_project/
├── main.py
├── run_pipeline.py
├── methods/
│   ├── __init__.py
│   ├── method_runner.py
│   └── post_processing.py
├── data/
│   ├── Krebs_Cycle_1_TS/
│   ├── Krebs_Cycle_Normalized_1_TS/
│   ├── Krebs_Cycle_3_TS/
│   └── Krebs_Cycle_Normalized_3_TS/
├── utils/
│   ├── data_loader.py
│   └── timer.py
└── output/
    └── Results_Krebs_Cycle_1/
        ├── adj_matrices/
        ├── score/
        ├── heatmap/
        ├── merged_scores_Krebs_Cycle_1.csv
        ├── sid_Krebs_Cycle_1.csv
        └── output_Krebs_Cycle_1.csv
```

## What was moved out of the notebook

- Data loading and standardization: `utils/data_loader.py`
- Method dispatching: `methods/method_runner.py`
- Metrics, merging, and plotting: `methods/post_processing.py`
- Pipeline orchestration: `run_pipeline.py`
- CLI entrypoint: `main.py`

## Expected data layout

Put Krebs data under `data/` in one of these forms:

- `data/<dataset>.npz` with `x` and `y`
- `data/<dataset>.csv` with `data/true_graph.csv`
- `data/<dataset>.tar.gz`
- `data/<dataset>_TS/*.tsv` with `data/true_graph.npz` nearby

Example:

```text
data/
├── Krebs_Cycle_1_TS/
│   ├── sample_1.tsv
│   ├── sample_2.tsv
│   └── ...
└── true_graph.npz
```

Current local setup:

- Krebs data already exists under `/Users/xiaoyuhe/Downloads/KrebsCycle`
- `main.py` now uses that path as the default `--data-root` when it exists

## Run

```bash
cd /Users/xiaoyuhe/Causal-Methods/krebcycle/causality_project
python3 main.py --dataset-name Krebs_Cycle_1 --methods ExDBN
```

Run a wider method set:

```bash
python3 main.py \
  --dataset-name Krebs_Cycle_3 \
  --methods PC-Stable PC-Parallel Direct-LiNGAM ICA-LiNGAM GES ExMAG ExDBN
```

## Notes

- `ExMAG`, `ExDBN`, `DyNotear`, and ANM variants still depend on external modules that were only referenced in the notebook. The new code now fails fast with a clear import or implementation error instead of silently relying on Colab state.
- `sid_<dataset>.csv` is optional. If you have a precomputed SID file, pass it with `--sid-file`.
