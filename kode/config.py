"""Shared configuration, path resolution, seeding, logging, and I/O helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import importlib
import importlib.metadata
import importlib.util
import json
import logging
import math
import os
from pathlib import Path
import platform
import random
import sys
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SEED: int = 42
REVIEW_COST_IDR: int = 25_000
DEFAULT_DATA_DIR: Path = Path.home() / "Downloads" / "data_uts_ml"
OUTPUT_DIR: Path = PROJECT_ROOT / "output"
DEFAULT_REGRESI_CSV: Path = PROJECT_ROOT / "prediksi_regresi.csv"
DEFAULT_KLASIFIKASI_CSV: Path = PROJECT_ROOT / "prediksi_klasifikasi.csv"

CORE_PACKAGES: tuple[tuple[str, str], ...] = (
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("scikit-learn", "sklearn"),
)

OPTIONAL_ML_PACKAGES: tuple[tuple[str, str], ...] = (
    ("lightgbm", "lightgbm"),
    ("xgboost", "xgboost"),
    ("catboost", "catboost"),
    ("statsmodels", "statsmodels"),
    ("shap", "shap"),
    ("imbalanced-learn", "imblearn"),
    ("optuna", "optuna"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("joblib", "joblib"),
    ("pytest", "pytest"),
)


def _clean_path_input(raw: str | Path | None) -> Path | None:
    """Normalize a path-like input, treating empty/whitespace strings as None."""
    if raw is None:
        return None
    if isinstance(raw, Path):
        if not str(raw).strip():
            return None
        return raw.expanduser().resolve()
    if isinstance(raw, str):
        cleaned = raw.strip()
        while len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in ("'", '"'):
            cleaned = cleaned[1:-1].strip()
        if not cleaned:
            return None
        return Path(cleaned).expanduser().resolve()
    raise TypeError(f"Expected str, Path, or None for path input, got {type(raw).__name__}")


def get_data_dir(data_dir: str | Path | None = None, *, must_exist: bool = False) -> Path:
    """Resolve dataset directory from explicit arg, UTS_DATA_DIR env var, or DEFAULT_DATA_DIR."""
    resolved = _clean_path_input(data_dir)
    if resolved is None:
        resolved = _clean_path_input(os.environ.get("UTS_DATA_DIR"))
    if resolved is None:
        resolved = DEFAULT_DATA_DIR.expanduser().resolve()

    if must_exist:
        if not resolved.exists():
            raise FileNotFoundError(
                f"Data directory not found: {resolved}. "
                f"Set UTS_DATA_DIR or place dataset files in {DEFAULT_DATA_DIR}."
            )
        if not resolved.is_dir():
            raise NotADirectoryError(f"Data path is not a directory: {resolved}")
    return resolved


def get_library_versions() -> dict[str, str]:
    """Return installed versions for Python, core scientific libs, and optional ML packages."""
    versions: dict[str, str] = {"python": platform.python_version()}
    for dist_name, module_name in CORE_PACKAGES + OPTIONAL_ML_PACKAGES:
        try:
            spec = importlib.util.find_spec(module_name)
        except (ImportError, ValueError, AttributeError):
            spec = None
        if spec is None:
            continue

        try:
            versions[dist_name] = str(importlib.metadata.version(dist_name))
            continue
        except importlib.metadata.PackageNotFoundError:
            pass
        except Exception:
            continue

        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", None)
            if version is not None:
                versions[dist_name] = str(version)
        except Exception:
            continue

    return versions


def format_library_versions(versions: dict[str, str] | None = None) -> str:
    """Format library versions dict into a concise single-line ASCII summary."""
    resolved = get_library_versions() if versions is None else versions
    return ", ".join(f"{name}={ver}" for name, ver in resolved.items())


def _validate_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError(f"seed must be an integer, got {type(seed).__name__}")
    seed_int = int(seed)
    if not (0 <= seed_int < 2**32):
        raise ValueError(f"seed must be in [0, {2**32 - 1}], got {seed_int}")
    return seed_int


def set_global_seed(seed: int = SEED) -> None:
    """Set PYTHONHASHSEED, stdlib random, and NumPy RNG seeds deterministically."""
    seed_int = _validate_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed_int)
    random.seed(seed_int)
    np.random.seed(seed_int)


def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if missing and return its resolved Path."""
    resolved = Path(path).expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def resolve_output_path(
    output_path: str | Path | None,
    default_filename: str,
    output_dir: str | Path | None = None,
) -> Path:
    """Resolve output file path with priority: explicit output_path > output_dir > PROJECT_ROOT."""
    explicit = _clean_path_input(output_path)
    if explicit is not None:
        return explicit
    target_dir = _clean_path_input(output_dir)
    if target_dir is not None:
        return target_dir / default_filename
    return PROJECT_ROOT / default_filename


def reconfigure_stdio_utf8() -> None:
    """Safely switch stdout/stderr to UTF-8 with replacement on Windows consoles."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def configure_logging(
    level: int = logging.INFO,
    name: str = "uts_ml",
    log_file: str | Path | None = None,
) -> logging.Logger:
    """Configure and return an idempotent logger without duplicate handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    has_stream = False
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            handler.setLevel(level)
            has_stream = True
    if not has_stream:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    log_path = _clean_path_input(log_file)
    if log_path is not None:
        if log_path.exists() and log_path.is_dir():
            raise IsADirectoryError(f"log_file path is a directory, not a file: {log_path}")
        ensure_dir(log_path.parent)
        has_file = False
        for handler in logger.handlers:
            if (
                isinstance(handler, logging.FileHandler)
                and Path(handler.baseFilename).resolve() == log_path
            ):
                handler.setLevel(level)
                has_file = True
        if not has_file:
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "uts_ml") -> logging.Logger:
    """Return existing logger or configure a default one if no handlers exist."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        return configure_logging(name=name)
    return logger


def _to_json_key(key: Any) -> str:
    """Normalize a dictionary key to its exact RFC 8259 JSON object key string."""
    if isinstance(key, str):
        return key
    if key is None or key is pd.NA or key is pd.NaT:
        return "null"
    if isinstance(key, (np.datetime64, np.timedelta64)):
        return "null" if bool(np.isnat(key)) else str(key)
    if isinstance(key, (bool, np.bool_)):
        return "true" if bool(key) else "false"
    if isinstance(key, (int, np.integer)):
        return str(int(key))
    if isinstance(key, (float, np.floating)):
        val = float(key)
        return repr(val) if math.isfinite(val) else "null"
    if isinstance(key, (pd.Timestamp, datetime, date)):
        try:
            if pd.isna(key):
                return "null"
        except Exception:
            pass
        return key.isoformat()
    if isinstance(key, Path):
        return str(key)
    return str(key)


def _to_json_compatible(obj: Any, seen: set[int]) -> Any:
    if obj is None or obj is pd.NA or obj is pd.NaT:
        return None
    if isinstance(obj, (np.datetime64, np.timedelta64)):
        return None if bool(np.isnat(obj)) else str(obj)
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        val = float(obj)
        return val if math.isfinite(val) else None
    if isinstance(obj, str):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (pd.Timestamp, datetime, date)):
        try:
            if pd.isna(obj):
                return None
        except Exception:
            pass
        return obj.isoformat()
    if isinstance(obj, np.ndarray):
        obj_id = id(obj)
        if obj_id in seen:
            raise ValueError("Circular reference detected during JSON serialization")
        seen.add(obj_id)
        try:
            return [_to_json_compatible(item, seen) for item in obj.tolist()]
        finally:
            seen.remove(obj_id)
    if isinstance(obj, pd.DataFrame):
        obj_id = id(obj)
        if obj_id in seen:
            raise ValueError("Circular reference detected during JSON serialization")
        seen.add(obj_id)
        try:
            return [_to_json_compatible(row, seen) for row in obj.to_dict(orient="records")]
        finally:
            seen.remove(obj_id)
    if isinstance(obj, (pd.Series, pd.Index)):
        obj_id = id(obj)
        if obj_id in seen:
            raise ValueError("Circular reference detected during JSON serialization")
        seen.add(obj_id)
        try:
            return [_to_json_compatible(item, seen) for item in obj.tolist()]
        finally:
            seen.remove(obj_id)
    if isinstance(obj, dict):
        obj_id = id(obj)
        if obj_id in seen:
            raise ValueError("Circular reference detected during JSON serialization")
        seen.add(obj_id)
        try:
            out_dict: dict[str, Any] = {}
            for k, v in obj.items():
                key_str = _to_json_key(k)
                if key_str in out_dict:
                    raise ValueError(
                        f"Duplicate JSON object key after normalization: {key_str!r}"
                    )
                out_dict[key_str] = _to_json_compatible(v, seen)
            return out_dict
        finally:
            seen.remove(obj_id)
    if isinstance(obj, (list, tuple)):
        obj_id = id(obj)
        if obj_id in seen:
            raise ValueError("Circular reference detected during JSON serialization")
        seen.add(obj_id)
        try:
            return [_to_json_compatible(item, seen) for item in obj]
        finally:
            seen.remove(obj_id)
    if isinstance(obj, set):
        obj_id = id(obj)
        if obj_id in seen:
            raise ValueError("Circular reference detected during JSON serialization")
        seen.add(obj_id)
        try:
            return [_to_json_compatible(item, seen) for item in sorted(obj, key=str)]
        finally:
            seen.remove(obj_id)
    try:
        is_na = pd.isna(obj)
        if isinstance(is_na, (bool, np.bool_)) and bool(is_na):
            return None
    except Exception:
        pass
    return str(obj)


def to_json_compatible(obj: Any) -> Any:
    """Recursively convert NumPy/pandas/Path/non-finite objects into RFC 8259 JSON types."""
    return _to_json_compatible(obj, set())


def save_json(data: Any, path: str | Path, indent: int = 2) -> Path:
    """Serialize data cleanly to UTF-8 JSON with NaN/Inf sanitized to null."""
    target = Path(path).expanduser().resolve()
    ensure_dir(target.parent)
    clean_data = to_json_compatible(data)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(clean_data, handle, indent=indent, ensure_ascii=False, allow_nan=False)
    return target


def load_csv_gz(
    filename: str,
    data_dir: str | Path | None = None,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Load a CSV or gzipped CSV file from data_dir while preserving float64 precision."""
    base_dir = get_data_dir(data_dir, must_exist=True)
    file_path = base_dir / filename
    if not file_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    return pd.read_csv(file_path, compression="infer", nrows=nrows, low_memory=False)


@dataclass(frozen=True)
class SmokeTestConfig:
    """Configuration for lightweight smoke-test / dry-run pipeline execution."""

    enabled: bool = False
    sample_n: int = 5_000
    test_sample_n: int = 1_000
    max_iter: int = 30
    n_iter_search: int = 2
    cv_folds: int = 2
    seed: int = SEED


def get_smoke_config(
    smoke_test: bool = False,
    sample_n: int | None = None,
    seed: int = SEED,
) -> SmokeTestConfig:
    """Build a validated SmokeTestConfig for full or smoke-test pipeline runs."""
    seed_int = _validate_seed(seed)
    if sample_n is None:
        return SmokeTestConfig(enabled=bool(smoke_test), seed=seed_int)

    if isinstance(sample_n, bool) or not isinstance(sample_n, (int, np.integer)):
        raise TypeError(f"sample_n must be a positive integer, got {type(sample_n).__name__}")
    resolved_sample_n = int(sample_n)
    if resolved_sample_n <= 0:
        raise ValueError(f"sample_n must be > 0, got {resolved_sample_n}")

    resolved_test_n = (
        1_000
        if resolved_sample_n >= 1_000
        else min(1_000, resolved_sample_n, max(200, resolved_sample_n // 5))
    )
    return SmokeTestConfig(
        enabled=bool(smoke_test),
        sample_n=resolved_sample_n,
        test_sample_n=resolved_test_n,
        seed=seed_int,
    )


if __name__ == "__main__":
    reconfigure_stdio_utf8()
    print(f"PROJECT_ROOT={PROJECT_ROOT}")
    print(f"DEFAULT_DATA_DIR={DEFAULT_DATA_DIR}")
    print(f"SEED={SEED}")
    print(f"REVIEW_COST_IDR={REVIEW_COST_IDR}")
    print(f"LIBRARIES={format_library_versions()}")
