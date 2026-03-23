from __future__ import annotations

import argparse
from pathlib import Path

from run_pipeline import CausalPipelineConfig, CausalPipeline


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parent
    download_root = Path("/Users/xiaoyuhe/Downloads/KrebsCycle")
    default_data_root = download_root if download_root.exists() else project_root / "data"
    parser = argparse.ArgumentParser(description="Run the causality project pipeline.")
    parser.add_argument("--dataset-name", default="Krebs_Cycle_1")
    parser.add_argument("--data-root", default=str(default_data_root))
    parser.add_argument("--output-root", default=str(project_root / "output"))
    parser.add_argument("--methods", nargs="*", default=["ExDBN"])
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--time-start", type=int, default=100)
    parser.add_argument("--time-end", type=int, default=200)
    parser.add_argument("--single-time-step", type=int, default=None)
    parser.add_argument("--anm-features", type=int, default=3)
    parser.add_argument("--anm-samples", type=int, default=8)
    parser.add_argument("--time-limit", type=int, default=3600)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--sid-file", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = CausalPipelineConfig(
        dataset_name=args.dataset_name,
        data_root=Path(args.data_root),
        output_root=Path(args.output_root),
        methods=args.methods,
        sample_index=args.sample_index,
        time_start=args.time_start,
        time_end=args.time_end,
        single_time_step=args.single_time_step,
        anm_features=args.anm_features,
        anm_samples=args.anm_samples,
        time_limit=args.time_limit,
        skip_existing=args.skip_existing,
        sid_file=Path(args.sid_file) if args.sid_file else None,
    )
    CausalPipeline(config).run()


if __name__ == "__main__":
    main()
