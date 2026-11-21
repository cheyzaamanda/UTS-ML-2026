"""Unified CLI entrypoint to reproduce Task 1 (Regression) and Task 2 (Classification)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Any

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kode.config import (
    DEFAULT_KLASIFIKASI_CSV,
    DEFAULT_REGRESI_CSV,
    REVIEW_COST_IDR,
    SEED,
    configure_logging,
    ensure_dir,
    format_library_versions,
    get_data_dir,
    get_library_versions,
    reconfigure_stdio_utf8,
    resolve_output_path,
    save_json,
    set_global_seed,
    to_json_compatible,
)
from kode.klasifikasi import run_klasifikasi_pipeline
from kode.regresi import run_regresi_pipeline


def run_all_pipelines(
    data_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    artifacts_dir: str | Path | None = None,
    only: str = "all",
    smoke_test: bool = False,
    sample_n: int | None = None,
    seed: int = SEED,
) -> dict[str, Any]:
    """Run Task 1 and/or Task 2 pipelines deterministically and return combined summary."""
    t0 = time.perf_counter()
    reconfigure_stdio_utf8()
    set_global_seed(seed)
    logger = configure_logging(name="uts_ml.run_all")

    resolved_data_dir = get_data_dir(data_dir, must_exist=True)
    lib_versions = get_library_versions()
    lib_summary = format_library_versions(lib_versions)

    logger.info(
        "UTS ML 2026 Reproduction Runner | SEED=%d | REVIEW_COST_IDR=%d | smoke_test=%s | only=%s",
        seed,
        REVIEW_COST_IDR,
        smoke_test,
        only,
    )
    logger.info("Data directory: %s", resolved_data_dir)
    logger.info("Library versions: %s", lib_summary)
    print(f"UTS ML 2026 | SEED={seed} | smoke_test={smoke_test} | only={only}")
    print(f"Libraries: {lib_summary}")

    resolved_artifacts_dir: Path | None = None
    if artifacts_dir is not None:
        resolved_artifacts_dir = ensure_dir(artifacts_dir)

    results: dict[str, Any] = {
        "seed": int(seed),
        "smoke_test": bool(smoke_test),
        "only": only,
        "data_dir": str(resolved_data_dir),
        "library_versions": lib_versions,
        "tasks": {},
    }

    if only in ("all", "regresi"):
        reg_csv = resolve_output_path(None, DEFAULT_REGRESI_CSV.name, output_dir=output_dir)
        reg_art = (resolved_artifacts_dir / "regresi") if resolved_artifacts_dir is not None else None
        reg_res = run_regresi_pipeline(
            data_dir=resolved_data_dir,
            output_csv=reg_csv,
            artifacts_dir=reg_art,
            smoke_test=smoke_test,
            sample_n=sample_n,
            seed=seed,
        )
        results["tasks"]["regresi"] = reg_res
        print(f"[Task 1 Regresi] Saved: {reg_csv}")

    if only in ("all", "klasifikasi"):
        cls_csv = resolve_output_path(None, DEFAULT_KLASIFIKASI_CSV.name, output_dir=output_dir)
        cls_art = (resolved_artifacts_dir / "klasifikasi") if resolved_artifacts_dir is not None else None
        cls_res = run_klasifikasi_pipeline(
            data_dir=resolved_data_dir,
            output_csv=cls_csv,
            artifacts_dir=cls_art,
            smoke_test=smoke_test,
            sample_n=sample_n,
            seed=seed,
        )
        results["tasks"]["klasifikasi"] = cls_res
        print(f"[Task 2 Klasifikasi] Saved: {cls_csv}")

    elapsed = float(time.perf_counter() - t0)
    results["elapsed_seconds"] = round(elapsed, 3)

    if resolved_artifacts_dir is not None:
        summary_path = save_json(results, resolved_artifacts_dir / "run_all_summary.json")
        results["summary_json"] = str(summary_path)

    logger.info("All requested pipelines completed in %.2fs", elapsed)
    return to_json_compatible(results)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="One-command reproduction runner for UTS ML 2026 (Task 1 Regression & Task 2 Classification)."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to data_uts_ml directory (defaults to UTS_DATA_DIR or ~/Downloads/data_uts_ml).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for prediksi_regresi.csv and prediksi_klasifikasi.csv (defaults to repository root).",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default=None,
        help="Optional directory to store JSON metric summaries and diagnostic plots.",
    )
    parser.add_argument(
        "--only",
        choices=("all", "regresi", "klasifikasi"),
        default="all",
        help="Run both tasks ('all') or a single task ('regresi' or 'klasifikasi').",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run fast deterministic subsample verification mode.",
    )
    parser.add_argument(
        "--sample-n",
        type=int,
        default=None,
        help="Optional training subsample size when running with --smoke-test.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Global random seed (default: {SEED}).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    reconfigure_stdio_utf8()
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        run_all_pipelines(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            artifacts_dir=args.artifacts_dir,
            only=args.only,
            smoke_test=args.smoke_test,
            sample_n=args.sample_n,
            seed=args.seed,
        )
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
