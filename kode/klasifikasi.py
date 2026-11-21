"""Task 2 Classification Pipeline: Fraud Detection (R2a-R2g)."""

from __future__ import annotations

import argparse
import logging
import math
import time
import warnings
from pathlib import Path
from typing import Any

from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, TargetEncoder
import xgboost as xgb

from kode.config import (
    DEFAULT_KLASIFIKASI_CSV,
    REVIEW_COST_IDR,
    SEED,
    SmokeTestConfig,
    configure_logging,
    get_data_dir,
    get_library_versions,
    get_smoke_config,
    load_csv_gz,
    reconfigure_stdio_utf8,
    resolve_output_path,
    save_json,
    set_global_seed,
    to_json_compatible,
)

TARGET_COL = "is_fraud"
ID_COL = "id_transaksi"
TIME_COL = "waktu"

LEAKAGE_COLS: tuple[str, ...] = ("flag_review_manual",)
DRIFT_DROP_COLS: tuple[str, ...] = ("v15",)
RAW_HIGH_CARD_DROP_COLS: tuple[str, ...] = ("waktu", "id_pengguna", "id_merchant")
CAT_COLS: tuple[str, ...] = (
    "kategori_merchant",
    "kanal",
    "tipe_perangkat",
    "email_domain",
)
INTERACTION_COL = "kanal__kategori_merchant"
ALL_CAT_COLS: tuple[str, ...] = (*CAT_COLS, INTERACTION_COL)

MATURE_END_EXCLUSIVE = pd.Timestamp("2025-08-11")
MATURE_CUTOFF = pd.Timestamp("2025-08-10 23:59:59")
AUG_11_START = pd.Timestamp("2025-08-11")
AUG_18_START = pd.Timestamp("2025-08-18")
AUG_25_START = pd.Timestamp("2025-08-25")
AUG_29_START = pd.Timestamp("2025-08-29")

USER_CAUSAL_COLS: tuple[str, ...] = (
    "user_prior_tx_count_cap10",
    "user_prior_nom_mean",
    "user_prior_nom_max",
    "user_lognom_zscore",
    "nominal_to_user_prior_max",
    "user_seen_kanal_before",
    "user_seen_kat_before",
    "user_seen_merchant_before",
    "user_seen_device_before",
    "user_distinct_merchants_prior",
    "user_trx_1j_vs_7h_ratio",
    "user_nom_24j_ratio_to_avg",
)

MERCHANT_CAUSAL_COLS: tuple[str, ...] = (
    "merch_prior_nom_mean",
    "nominal_to_merch_prior_mean",
    "merch_qris_share_prior",
    "merch_night_share_prior",
    "merch_rolling_1h_count",
    "merch_rolling_1h_nom_sum",
    "merch_rolling_24h_count",
    "merch_rolling_24h_nom_sum",
    "merch_rolling_24h_unique_users",
    "merch_rolling_7d_count",
    "merch_rolling_7d_nom_sum",
    "merch_burst_1h_vs_7d",
    "merch_burst_24h_vs_7d",
)

LAGGED_TARGET_COLS: tuple[str, ...] = (
    "lag21d_merch_fraud_rate",
    "lag21d_kat_fraud_rate",
    "lag21d_kanal_kat_fraud_rate",
)

COLD_START_HYGIENE_COLS: tuple[str, ...] = (
    "is_first_user_tx",
    "user_has_prior_tx",
    "umur_merchant_neg_flag",
    "umur_merchant_hari_clipped",
    "persen_saldo_capped_flag",
    "nominal_gt_saldo_flag",
    "nominal_ge_review_cost",
    "log1p_nominal",
    "log1p_saldo",
    "log1p_nominal_24j",
)

TIME_CYCLICAL_COLS: tuple[str, ...] = (
    "hour_sin",
    "hour_cos",
    "day_of_week",
    "is_weekend",
    "is_night",
)


def _to_ts(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def _ts_to_epoch_sec(ts_series: pd.Series) -> np.ndarray:
    """Unit-agnostic conversion from datetime Series to int64 epoch seconds."""
    return ts_series.dt.as_unit("s").astype(np.int64).to_numpy()


def _safe_auc(metric_fn: Any, y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    y_arr = np.asarray(y_true, dtype=np.int64)
    n_pos = int(np.sum(y_arr == 1))
    if n_pos == 0 or n_pos == len(y_arr):
        return None
    val = float(metric_fn(y_arr, y_prob))
    return val if math.isfinite(val) else None


def _safe_brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.0
    p = np.clip(np.asarray(y_prob, dtype=np.float64), 0.0, 1.0)
    return float(brier_score_loss(np.asarray(y_true, dtype=np.int64), p))


def _safe_log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_arr = np.asarray(y_true, dtype=np.int64)
    if len(y_arr) == 0:
        return 0.0
    p = np.clip(np.asarray(y_prob, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return float(log_loss(y_arr, p, labels=[0, 1]))


def _logit(p: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


# ---------------------------------------------------------------------------
# 1. Data Loading & Deterministic Smoke Subsampling
# ---------------------------------------------------------------------------


def _subsample_train_temporal(
    train_df: pd.DataFrame,
    target_n: int,
    seed: int,
) -> pd.DataFrame:
    """Deterministically subsample train preserving months, censoring tail, and shift signals."""
    if len(train_df) <= target_n:
        return train_df.copy()

    rng = np.random.default_rng(seed)
    ts = _to_ts(train_df[TIME_COL])
    y = train_df[TARGET_COL].to_numpy(dtype=np.int64)
    scale = target_n / 2400.0

    # Define temporal buckets matching real dataset structure
    buckets: list[tuple[str, np.ndarray, int, int]] = []
    for m in (1, 2, 3, 4):
        buckets.append((f"m{m:02d}", (ts.dt.month == m).to_numpy(), max(10, round(180 * scale)), max(2, round(12 * scale))))
    buckets.append(("m05", (ts.dt.month == 5).to_numpy(), max(12, round(230 * scale)), max(2, round(15 * scale))))
    buckets.append(("m06", (ts.dt.month == 6).to_numpy(), max(15, round(350 * scale)), max(3, round(20 * scale))))
    buckets.append(("m07", (ts.dt.month == 7).to_numpy(), max(15, round(350 * scale)), max(3, round(22 * scale))))
    buckets.append(
        (
            "aug_1_10",
            ((ts >= pd.Timestamp("2025-08-01")) & (ts < MATURE_END_EXCLUSIVE)).to_numpy(),
            max(20, round(400 * scale)),
            max(4, round(28 * scale)),
        )
    )
    buckets.append(
        (
            "aug_11_17",
            ((ts >= AUG_11_START) & (ts < AUG_18_START)).to_numpy(),
            max(8, round(150 * scale)),
            max(1, round(3 * scale)),
        )
    )
    buckets.append(
        (
            "aug_18_28",
            ((ts >= AUG_18_START) & (ts < AUG_29_START)).to_numpy(),
            max(6, round(120 * scale)),
            max(0, round(1 * scale)),
        )
    )
    buckets.append(
        (
            "aug_29_31",
            ((ts >= AUG_29_START) & (ts < pd.Timestamp("2025-09-01"))).to_numpy(),
            max(4, round(80 * scale)),
            0,
        )
    )

    selected: set[int] = set()
    jan_may_negs: list[int] = []

    for bname, mask, quota_total, quota_pos in buckets:
        idx_all = np.flatnonzero(mask)
        if len(idx_all) == 0:
            continue
        pos_idx = idx_all[y[idx_all] == 1]
        neg_idx = idx_all[y[idx_all] == 0]

        n_pos_take = min(len(pos_idx), quota_pos)
        if n_pos_take > 0:
            if bname == "aug_1_10":
                # Prioritize QRIS + jasa_lainnya frauds so shift stays well-represented
                is_qris = (train_df.iloc[pos_idx]["kanal"] == "qris").to_numpy()
                is_jasa = (train_df.iloc[pos_idx]["kategori_merchant"] == "jasa_lainnya").to_numpy()
                prio = pos_idx[is_qris | is_jasa]
                rest = pos_idx[~(is_qris | is_jasa)]
                chosen_p: list[int] = []
                if len(prio) > 0:
                    take_prio = min(len(prio), max(1, n_pos_take // 2))
                    chosen_p.extend(rng.choice(prio, size=take_prio, replace=False).tolist())
                rem_p = n_pos_take - len(chosen_p)
                if rem_p > 0:
                    avail = np.array([i for i in pos_idx if i not in set(chosen_p)], dtype=np.int64)
                    if len(avail) > 0:
                        chosen_p.extend(rng.choice(avail, size=min(len(avail), rem_p), replace=False).tolist())
                selected.update(chosen_p)
            else:
                selected.update(rng.choice(pos_idx, size=n_pos_take, replace=False).tolist())

        n_neg_take = min(len(neg_idx), max(0, quota_total - n_pos_take))
        if n_neg_take > 0:
            chosen_n = rng.choice(neg_idx, size=n_neg_take, replace=False).tolist()
            selected.update(chosen_n)
            if bname in ("m01", "m02", "m03", "m04", "m05"):
                jan_may_negs.extend(chosen_n)

    # Coverage backfill for categories, channels, negative merchant age, flag_review_manual
    extra_needed: list[int] = []
    for col in ("kategori_merchant", "kanal"):
        present = set(train_df.iloc[list(selected)][col].unique())
        for val in train_df[col].unique():
            if val not in present:
                cand = np.flatnonzero((train_df[col] == val).to_numpy())
                if len(cand) > 0:
                    pick = int(cand[0])
                    if pick not in selected:
                        extra_needed.append(pick)
                        selected.add(pick)

    if int((train_df.iloc[list(selected)]["umur_merchant_hari"] < 0).sum()) == 0:
        neg_age_idx = np.flatnonzero((train_df["umur_merchant_hari"] < 0).to_numpy())
        for pick in neg_age_idx[: min(5, len(neg_age_idx))]:
            if int(pick) not in selected:
                extra_needed.append(int(pick))
                selected.add(int(pick))

    if "flag_review_manual" in train_df.columns:
        if int((train_df.iloc[list(selected)]["flag_review_manual"] == 1).sum()) == 0:
            flag_idx = np.flatnonzero((train_df["flag_review_manual"] == 1).to_numpy())
            for pick in flag_idx[: min(5, len(flag_idx))]:
                if int(pick) not in selected:
                    extra_needed.append(int(pick))
                    selected.add(int(pick))

    # Trim excess from early Jan-May negatives if above target_n
    if len(selected) > target_n and jan_may_negs:
        rng.shuffle(jan_may_negs)
        for drop_i in jan_may_negs:
            if len(selected) <= target_n:
                break
            if drop_i in selected and drop_i not in extra_needed:
                selected.remove(drop_i)

    # Top up if slightly below target_n
    if len(selected) < target_n:
        remaining = np.array([i for i in range(len(train_df)) if i not in selected], dtype=np.int64)
        if len(remaining) > 0:
            need = min(len(remaining), target_n - len(selected))
            selected.update(rng.choice(remaining, size=need, replace=False).tolist())

    sel_idx = np.sort(np.fromiter(selected, dtype=np.int64))
    if len(sel_idx) > target_n:
        sel_idx = sel_idx[:target_n]
    return train_df.iloc[sel_idx].reset_index(drop=True)


def _subsample_test_temporal(
    test_df: pd.DataFrame,
    target_n: int,
    seed: int,
) -> pd.DataFrame:
    """Deterministically subsample test preserving Sep and Oct + QRIS/jasa_lainnya."""
    if len(test_df) <= target_n:
        return test_df.copy()

    rng = np.random.default_rng(seed + 1)
    ts = _to_ts(test_df[TIME_COL])
    sep_idx = np.flatnonzero((ts.dt.month == 9).to_numpy())
    oct_idx = np.flatnonzero((ts.dt.month == 10).to_numpy())

    n_sep = min(len(sep_idx), target_n // 2)
    n_oct = min(len(oct_idx), target_n - n_sep)
    chosen: list[int] = []
    if n_sep > 0:
        chosen.extend(rng.choice(sep_idx, size=n_sep, replace=False).tolist())
    if n_oct > 0:
        chosen.extend(rng.choice(oct_idx, size=n_oct, replace=False).tolist())

    sel_set = set(chosen)
    if len(sel_set) < target_n:
        rem = np.array([i for i in range(len(test_df)) if i not in sel_set], dtype=np.int64)
        if len(rem) > 0:
            sel_set.update(rng.choice(rem, size=min(len(rem), target_n - len(sel_set)), replace=False).tolist())

    sel_idx = np.sort(np.fromiter(sel_set, dtype=np.int64))[:target_n]
    return test_df.iloc[sel_idx].reset_index(drop=True)


def load_klasifikasi_data(
    data_dir: str | Path | None = None,
    smoke_cfg: SmokeTestConfig | None = None,
    smoke_test: bool = False,
    sample_n: int | None = None,
    seed: int = SEED,
    logger: logging.Logger | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load klasifikasi train & test datasets with optional deterministic smoke subsampling."""
    if smoke_cfg is None:
        smoke_cfg = get_smoke_config(smoke_test=smoke_test, sample_n=sample_n, seed=seed)

    resolved_dir = get_data_dir(data_dir, must_exist=True)
    if logger:
        logger.info("Loading klasifikasi datasets from %s", resolved_dir)

    train_full = load_csv_gz("klasifikasi_train.csv.gz", data_dir=resolved_dir)
    test_full = load_csv_gz("klasifikasi_test.csv.gz", data_dir=resolved_dir)

    n_train_raw = int(len(train_full))
    n_test_raw = int(len(test_full))

    if smoke_cfg.enabled:
        # Enforce <= 3,000 total rows in smoke_test mode on large real datasets
        if sample_n is None:
            train_cap = 2_400
            test_cap = 600
        else:
            train_cap = min(int(smoke_cfg.sample_n), 2_400)
            test_cap = min(int(smoke_cfg.test_sample_n), 3_000 - train_cap, 600)
            test_cap = max(test_cap, min(len(test_full), 200))
        train_df = _subsample_train_temporal(train_full, train_cap, smoke_cfg.seed)
        test_df = _subsample_test_temporal(test_full, test_cap, smoke_cfg.seed)
    else:
        train_df = train_full
        test_df = test_full

    meta = {
        "data_dir": str(resolved_dir),
        "raw_train_rows": n_train_raw,
        "raw_test_rows": n_test_raw,
        "loaded_train_rows": int(len(train_df)),
        "loaded_test_rows": int(len(test_df)),
        "n_train_cols": int(train_df.shape[1]),
        "n_test_cols": int(test_df.shape[1]),
        "smoke_test": bool(smoke_cfg.enabled),
    }
    return train_df, test_df, meta


# ---------------------------------------------------------------------------
# 2. R2a + R2b + R2d Audit
# ---------------------------------------------------------------------------


def audit_klasifikasi_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, Any]:
    """Audit leakage, drift, anomalies, label-delay censoring, and temporal shift."""
    ts_tr = _to_ts(train_df[TIME_COL])
    ts_te = _to_ts(test_df[TIME_COL])
    y = train_df[TARGET_COL].to_numpy(dtype=np.int64)
    n_tr = len(train_df)
    n_pos = int(np.sum(y == 1))
    fraud_rate = float(n_pos / max(n_tr, 1))

    # 1. flag_review_manual leakage audit
    flag_tr = train_df["flag_review_manual"].to_numpy(dtype=np.float64)
    flag_te = test_df["flag_review_manual"].to_numpy(dtype=np.float64)
    m0 = flag_tr == 0
    m1 = flag_tr == 1
    rate_0 = float(np.mean(y[m0])) if np.any(m0) else 0.0
    rate_1 = float(np.mean(y[m1])) if np.any(m1) else 0.0
    pct_frauds_flagged = float(np.sum(y[m1] == 1) / max(n_pos, 1))
    corr_flag = float(np.corrcoef(flag_tr, y)[0, 1]) if np.std(flag_tr) > 0 and np.std(y) > 0 else 0.0
    flag_audit = {
        "train_flag_rate": float(np.mean(flag_tr)),
        "test_flag_rate": float(np.mean(flag_te)),
        "fraud_rate_when_flag_0": rate_0,
        "fraud_rate_when_flag_1": rate_1,
        "pct_frauds_flagged": pct_frauds_flagged,
        "corr_with_is_fraud": corr_flag,
        "univariate_pr_auc": _safe_auc(average_precision_score, y, flag_tr),
        "univariate_roc_auc": _safe_auc(roc_auc_score, y, flag_tr),
        "action": "DROP post-transaction leakage column from all production/causal features",
    }

    # 2. v15 monotonic time drift vs v26 legitimate feature
    ts_sec = _ts_to_epoch_sec(ts_tr).astype(np.float64)
    v15_tr = train_df["v15"].to_numpy(dtype=np.float64)
    v15_te = test_df["v15"].to_numpy(dtype=np.float64)
    v15_std = float(np.std(v15_tr)) if float(np.std(v15_tr)) > 1e-12 else 1.0
    v15_monthly = {
        f"2025-{m:02d}": float(v15_tr[(ts_tr.dt.month == m).to_numpy()].mean())
        for m in sorted(ts_tr.dt.month.unique())
    }
    v15_audit = {
        "train_mean": float(np.mean(v15_tr)),
        "test_mean": float(np.mean(v15_te)),
        "shift_z": float((np.mean(v15_te) - np.mean(v15_tr)) / v15_std),
        "corr_with_time": float(np.corrcoef(v15_tr, ts_sec)[0, 1]) if np.std(v15_tr) > 0 else 0.0,
        "corr_with_is_fraud": float(np.corrcoef(v15_tr, y)[0, 1]) if np.std(v15_tr) > 0 and np.std(y) > 0 else 0.0,
        "univariate_pr_auc": _safe_auc(average_precision_score, y, v15_tr),
        "monthly_means": v15_monthly,
        "action": "DROP pure monotonic calendar-time drift feature v15",
    }

    v26_tr = train_df["v26"].to_numpy(dtype=np.float64)
    v26_te = test_df["v26"].to_numpy(dtype=np.float64)
    v26_std = float(np.std(v26_tr)) if float(np.std(v26_tr)) > 1e-12 else 1.0
    v26_audit = {
        "train_mean": float(np.mean(v26_tr)),
        "test_mean": float(np.mean(v26_te)),
        "shift_z": float((np.mean(v26_te) - np.mean(v26_tr)) / v26_std),
        "corr_with_log1p_nominal": float(np.corrcoef(v26_tr, np.log1p(np.clip(train_df["nominal"].to_numpy(), 0, None)))[0, 1])
        if np.std(v26_tr) > 0
        else 0.0,
        "action": "KEEP v26 (genuine risk signal, stabilizes Q2+)",
    }

    # 3. Negative merchant age & cold-start nulls & persen_saldo anomalies
    age_tr = train_df["umur_merchant_hari"].to_numpy(dtype=np.float64)
    age_te = test_df["umur_merchant_hari"].to_numpy(dtype=np.float64)
    neg_mask = age_tr < 0
    umur_audit = {
        "train_neg_count": int(np.sum(neg_mask)),
        "train_neg_fraud_count": int(np.sum(y[neg_mask] == 1)) if np.any(neg_mask) else 0,
        "train_neg_fraud_rate": float(np.mean(y[neg_mask])) if np.any(neg_mask) else 0.0,
        "train_nonneg_fraud_rate": float(np.mean(y[~neg_mask])) if np.any(~neg_mask) else 0.0,
        "test_neg_count": int(np.sum(age_te < 0)),
        "train_min": int(np.min(age_tr)),
        "test_min": int(np.min(age_te)),
        "handling": "Create umur_merchant_neg_flag + clipped umur_merchant_hari_clipped",
    }

    na_tr = train_df["menit_sejak_trx_terakhir"].isna().to_numpy()
    na_te = test_df["menit_sejak_trx_terakhir"].isna().to_numpy()
    na2_tr = train_df["rasio_nominal_rata2"].isna().to_numpy()
    cold_audit = {
        "train_null_count": int(np.sum(na_tr)),
        "train_null_pct": float(np.mean(na_tr)),
        "test_null_count": int(np.sum(na_te)),
        "test_null_pct": float(np.mean(na_te)),
        "identical_null_masks": bool(np.array_equal(na_tr, na2_tr)),
        "first_tx_fraud_rate": float(np.mean(y[na_tr])) if np.any(na_tr) else 0.0,
        "returning_tx_fraud_rate": float(np.mean(y[~na_tr])) if np.any(~na_tr) else 0.0,
        "handling": "Create is_first_user_tx flag before median imputation for linear/SMOTE models",
    }

    ps_tr = train_df["persen_saldo"].to_numpy(dtype=np.float64)
    persen_audit = {
        "gt_1_count": int(np.sum(ps_tr > 1.0)),
        "capped_50_count": int(np.sum(ps_tr >= 49.99)),
        "capped_50_fraud_rate": float(np.mean(y[ps_tr >= 49.99])) if np.any(ps_tr >= 49.99) else 0.0,
    }

    # 4. R2b Label-delay censoring audit (monthly + August windows)
    mature_mask = (ts_tr < MATURE_END_EXCLUSIVE).to_numpy()
    mature_ref_rate = float(np.mean(y[mature_mask])) if np.any(mature_mask) else max(fraud_rate, 1e-6)
    mature_ref_rate = max(mature_ref_rate, 1e-6)

    monthly_rates: dict[str, dict[str, Any]] = {}
    for m in sorted(ts_tr.dt.month.unique()):
        m_mask = (ts_tr.dt.month == m).to_numpy()
        monthly_rates[f"2025-{m:02d}"] = {
            "n_tx": int(np.sum(m_mask)),
            "n_fraud": int(np.sum(y[m_mask] == 1)),
            "fraud_rate": float(np.mean(y[m_mask])),
        }

    aug_windows_spec = [
        ("aug_1_10", (ts_tr >= pd.Timestamp("2025-08-01")) & (ts_tr < MATURE_END_EXCLUSIVE)),
        ("aug_11_17", (ts_tr >= AUG_11_START) & (ts_tr < AUG_18_START)),
        ("aug_18_24", (ts_tr >= AUG_18_START) & (ts_tr < AUG_25_START)),
        ("aug_25_28", (ts_tr >= AUG_25_START) & (ts_tr < AUG_29_START)),
        ("aug_29_31", (ts_tr >= AUG_29_START) & (ts_tr < pd.Timestamp("2025-09-01"))),
        ("aug_11_31_combined", (ts_tr >= AUG_11_START) & (ts_tr < pd.Timestamp("2025-09-01"))),
    ]
    aug_windows: dict[str, dict[str, Any]] = {}
    for name, mask_s in aug_windows_spec:
        mask_arr = mask_s.to_numpy()
        cnt = int(np.sum(mask_arr))
        nf = int(np.sum(y[mask_arr] == 1)) if cnt > 0 else 0
        rate = float(nf / cnt) if cnt > 0 else 0.0
        aug_windows[name] = {
            "n_tx": cnt,
            "n_fraud": nf,
            "fraud_rate": rate,
            "empirical_maturity": float(np.clip(rate / mature_ref_rate, 0.0, 1.0)) if cnt > 0 else 0.0,
        }

    # 5. R2d Temporal shift audit (Jan-Jul vs Aug 1-10 + Test covariate share)
    jan_jul_mask = (ts_tr < pd.Timestamp("2025-08-01")).to_numpy()
    aug1_10_mask = ((ts_tr >= pd.Timestamp("2025-08-01")) & (ts_tr < MATURE_END_EXCLUSIVE)).to_numpy()

    def _slice_rates(col: str) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        vals = sorted(set(train_df[col].unique()) | set(test_df[col].unique()))
        for v in vals:
            b_m = jan_jul_mask & (train_df[col].to_numpy() == v)
            r_m = aug1_10_mask & (train_df[col].to_numpy() == v)
            b_cnt, r_cnt = int(np.sum(b_m)), int(np.sum(r_m))
            b_pos = int(np.sum(y[b_m] == 1)) if b_cnt > 0 else 0
            r_pos = int(np.sum(y[r_m] == 1)) if r_cnt > 0 else 0
            b_rate = float(b_pos / b_cnt) if b_cnt > 0 else 0.0
            r_rate = float(r_pos / r_cnt) if r_cnt > 0 else 0.0
            lift = float(r_rate / max(b_rate, 1e-4)) if r_cnt > 0 else 0.0
            out[str(v)] = {
                "jan_jul_n": b_cnt,
                "jan_jul_fraud": b_pos,
                "jan_jul_rate": b_rate,
                "aug_1_10_n": r_cnt,
                "aug_1_10_fraud": r_pos,
                "aug_1_10_rate": r_rate,
                "lift": lift,
                "test_share": float(np.mean(test_df[col].to_numpy() == v)),
                "train_jan_jul_share": float(np.mean(train_df.loc[jan_jul_mask, col].to_numpy() == v))
                if np.any(jan_jul_mask)
                else 0.0,
            }
        return out

    kanal_shift = _slice_rates("kanal")
    kat_shift = _slice_rates("kategori_merchant")
    emerging_segments = [
        {"dimension": "kanal", "segment": k, **v}
        for k, v in kanal_shift.items()
        if v["lift"] >= 2.5 and v["aug_1_10_fraud"] >= 2
    ] + [
        {"dimension": "kategori_merchant", "segment": k, **v}
        for k, v in kat_shift.items()
        if v["lift"] >= 2.5 and v["aug_1_10_fraud"] >= 2
    ]

    return {
        "train_shape": [int(train_df.shape[0]), int(train_df.shape[1])],
        "test_shape": [int(test_df.shape[0]), int(test_df.shape[1])],
        "train_time_min": str(ts_tr.min()),
        "train_time_max": str(ts_tr.max()),
        "test_time_min": str(ts_te.min()),
        "test_time_max": str(ts_te.max()),
        "n_fraud_train": n_pos,
        "fraud_rate_train": fraud_rate,
        "imbalance_ratio": float((n_tr - n_pos) / max(n_pos, 1)),
        "flag_review_manual": flag_audit,
        "v15": v15_audit,
        "v26": v26_audit,
        "umur_merchant_hari": umur_audit,
        "cold_start_nulls": cold_audit,
        "persen_saldo_anomalies": persen_audit,
        "label_delay_censoring": {
            "mature_cutoff": str(MATURE_CUTOFF),
            "mature_ref_fraud_rate": mature_ref_rate,
            "monthly_rates": monthly_rates,
            "august_windows": aug_windows,
            "aug_29_31_fraud_count": aug_windows["aug_29_31"]["n_fraud"],
        },
        "temporal_shift_audit": {
            "by_kanal": kanal_shift,
            "by_kategori_merchant": kat_shift,
            "emerging_shift_segments": emerging_segments,
        },
    }


# ---------------------------------------------------------------------------
# 3. R2c Strictly Causal Behavioral & Lagged Entity Features
# ---------------------------------------------------------------------------


def compute_lagged_entity_risk(
    train_df: pd.DataFrame,
    query_df: pd.DataFrame,
    train_cutoff_ts: pd.Timestamp | None = None,
    mature_cutoff: pd.Timestamp = MATURE_CUTOFF,
    lag_days: int = 21,
) -> pd.DataFrame:
    """Compute strictly causal >=21-day lagged smoothed target rates using mature train labels only."""
    effective_cutoff = mature_cutoff if train_cutoff_ts is None else min(pd.Timestamp(train_cutoff_ts), mature_cutoff)
    lag_sec = int(lag_days * 86400)

    tr_ts = _to_ts(train_df[TIME_COL])
    src_mask = (tr_ts <= effective_cutoff).to_numpy()
    q_ts = _to_ts(query_df[TIME_COL])
    q_sec = _ts_to_epoch_sec(q_ts)
    n_q = len(query_df)

    default_prior = 0.0065
    if not np.any(src_mask):
        return pd.DataFrame(
            {
                "lag21d_merch_fraud_rate": np.full(n_q, default_prior, dtype=np.float64),
                "lag21d_kat_fraud_rate": np.full(n_q, default_prior, dtype=np.float64),
                "lag21d_kanal_kat_fraud_rate": np.full(n_q, default_prior, dtype=np.float64),
            },
            index=query_df.index,
        )

    src = train_df.loc[src_mask, [TIME_COL, "id_merchant", "kategori_merchant", "kanal", TARGET_COL]].copy()
    src_ts = tr_ts.loc[src_mask]
    src_sec = _ts_to_epoch_sec(src_ts)
    src_y = src[TARGET_COL].to_numpy(dtype=np.float64)

    # Sort source chronologically for causal global prior lookup at t_i - lag_days
    g_order = np.argsort(src_sec, kind="mergesort")
    g_sec_sorted = src_sec[g_order]
    g_y_sorted = src_y[g_order]
    g_pref = np.concatenate(([0.0], np.cumsum(g_y_sorted)))

    q_elig_sec = q_sec - lag_sec
    g_idx = np.searchsorted(g_sec_sorted, q_elig_sec, side="right")
    q_prior = np.where(g_idx > 0, g_pref[g_idx] / np.maximum(g_idx, 1), default_prior)

    def _entity_lag_rate(src_keys: np.ndarray, q_keys: np.ndarray, alpha: float) -> np.ndarray:
        uniq, inv_all = np.unique(np.concatenate([src_keys, q_keys]), return_inverse=True)
        src_codes = inv_all[: len(src_keys)].astype(np.int64)
        q_codes = inv_all[len(src_keys) :].astype(np.int64)

        ts_min = int(min(np.min(src_sec), np.min(q_elig_sec)))
        stride = int(max(int(np.max(src_sec)) - ts_min + 10, 10_000_000))
        s_comp = src_codes * stride + (src_sec - ts_min)
        order = np.argsort(s_comp, kind="mergesort")
        s_comp_sorted = s_comp[order]
        s_y_sorted = src_y[order]
        pref = np.concatenate(([0.0], np.cumsum(s_y_sorted)))

        q_start = np.searchsorted(s_comp_sorted, q_codes * stride, side="left")
        # Clamp query relative time >= -1 so it stays inside the entity's bucket
        q_rel_clamped = np.maximum(-1, q_elig_sec - ts_min)
        q_end = np.searchsorted(s_comp_sorted, q_codes * stride + q_rel_clamped, side="right")
        cnt = np.maximum(0, q_end - q_start).astype(np.float64)
        pos = np.where(q_end > q_start, pref[q_end] - pref[q_start], 0.0)
        return (pos + alpha * q_prior) / (cnt + alpha)

    src_m = src["id_merchant"].astype(str).to_numpy()
    q_m = query_df["id_merchant"].astype(str).to_numpy()
    src_k = src["kategori_merchant"].astype(str).to_numpy()
    q_k = query_df["kategori_merchant"].astype(str).to_numpy()
    src_kk = (src["kanal"].astype(str) + "__" + src["kategori_merchant"].astype(str)).to_numpy()
    q_kk = (query_df["kanal"].astype(str) + "__" + query_df["kategori_merchant"].astype(str)).to_numpy()

    return pd.DataFrame(
        {
            "lag21d_merch_fraud_rate": _entity_lag_rate(src_m, q_m, alpha=20.0),
            "lag21d_kat_fraud_rate": _entity_lag_rate(src_k, q_k, alpha=50.0),
            "lag21d_kanal_kat_fraud_rate": _entity_lag_rate(src_kk, q_kk, alpha=50.0),
        },
        index=query_df.index,
    )


def _compute_merchant_24h_unique_users(
    m_codes_sorted: np.ndarray,
    ts_sorted: np.ndarray,
    u_codes_sorted: np.ndarray,
) -> np.ndarray:
    """Strictly causal (< ts_i, >= ts_i - 86400) unique user count per merchant in O(N)."""
    n = len(m_codes_sorted)
    out = np.zeros(n, dtype=np.float64)
    if n == 0:
        return out

    # Find contiguous merchant group boundaries
    boundaries = np.flatnonzero(np.diff(m_codes_sorted)) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [n]))

    for s, e in zip(starts, ends):
        if e - s <= 1:
            continue
        counts: dict[int, int] = {}
        left = int(s)
        right = int(s)
        for i in range(int(s), int(e)):
            t_i = int(ts_sorted[i])
            t_low = t_i - 86400
            while right < e and int(ts_sorted[right]) < t_i:
                u_r = int(u_codes_sorted[right])
                counts[u_r] = counts.get(u_r, 0) + 1
                right += 1
            while left < right and int(ts_sorted[left]) < t_low:
                u_l = int(u_codes_sorted[left])
                c = counts[u_l] - 1
                if c <= 0:
                    del counts[u_l]
                else:
                    counts[u_l] = c
                left += 1
            out[i] = float(len(counts))
    return out


def build_causal_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    mature_cutoff: pd.Timestamp = MATURE_CUTOFF,
    train_cutoff_ts: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build strictly causal user, merchant, hygiene, and lagged target features without leakage."""
    tr_clean = train_df.drop(columns=[c for c in (*LEAKAGE_COLS, TARGET_COL) if c in train_df.columns]).copy()
    te_clean = test_df.drop(columns=[c for c in (*LEAKAGE_COLS, TARGET_COL) if c in test_df.columns]).copy()

    n_tr = len(tr_clean)
    n_te = len(te_clean)
    tr_clean["_is_train"] = 1
    tr_clean["_orig_idx"] = np.arange(n_tr, dtype=np.int64)
    te_clean["_is_train"] = 0
    te_clean["_orig_idx"] = np.arange(n_te, dtype=np.int64)

    combined = pd.concat([tr_clean, te_clean], axis=0, ignore_index=True)
    ts_all = _to_ts(combined[TIME_COL])
    ts_sec = _ts_to_epoch_sec(ts_all)
    combined["_ts_sec"] = ts_sec

    # Stable chronological sort (ts_sec, id_transaksi)
    combined = combined.sort_values(["_ts_sec", ID_COL], kind="mergesort").reset_index(drop=True)
    ts_sorted_dt = _to_ts(combined[TIME_COL])
    ts_sorted_sec = combined["_ts_sec"].to_numpy(dtype=np.int64)
    n_all = len(combined)

    # Hygiene & domain features
    nom = combined["nominal"].to_numpy(dtype=np.float64)
    saldo = combined["saldo_sebelum"].to_numpy(dtype=np.float64)
    nom24 = combined["nominal_24j"].to_numpy(dtype=np.float64)
    age_m = combined["umur_merchant_hari"].to_numpy(dtype=np.float64)
    ps = combined["persen_saldo"].to_numpy(dtype=np.float64)

    is_first_user_tx = combined["menit_sejak_trx_terakhir"].isna().astype(np.int8).to_numpy()
    user_has_prior_tx = (1 - is_first_user_tx).astype(np.int8)
    umur_merchant_neg_flag = (age_m < 0).astype(np.int8)
    umur_merchant_hari_clipped = np.clip(age_m, 0.0, None)
    persen_saldo_capped_flag = (ps >= 49.99).astype(np.int8)
    nominal_gt_saldo_flag = (nom > saldo).astype(np.int8)
    nominal_ge_review_cost = (nom >= float(REVIEW_COST_IDR)).astype(np.int8)
    log1p_nominal = np.log1p(np.clip(nom, 0.0, None))
    log1p_saldo = np.log1p(np.clip(saldo, 0.0, None))
    log1p_nominal_24j = np.log1p(np.clip(nom24, 0.0, None))

    # Cyclical & calendar features (pandas 3.1 safe: day_of_week)
    hours = combined["jam"].to_numpy(dtype=np.float64)
    hour_sin = np.sin(2.0 * np.pi * hours / 24.0)
    hour_cos = np.cos(2.0 * np.pi * hours / 24.0)
    day_of_week = ts_sorted_dt.dt.day_of_week.to_numpy(dtype=np.int8)
    is_weekend = (day_of_week >= 5).astype(np.int8)
    is_night = (hours <= 5.0).astype(np.int8)

    # Integer codes for fast grouping
    u_codes, _ = pd.factorize(combined["id_pengguna"], sort=False)
    m_codes, _ = pd.factorize(combined["id_merchant"], sort=False)
    k_codes, _ = pd.factorize(combined["kanal"], sort=False)
    kat_codes, _ = pd.factorize(combined["kategori_merchant"], sort=False)
    dev_codes, _ = pd.factorize(combined["tipe_perangkat"], sort=False)

    # Strictly causal user history via vectorized groupby on sorted order
    # Group boundary tie handling: shift within user; also ensure strictly earlier timestamp if ties occur
    u_df = pd.DataFrame(
        {
            "u": u_codes,
            "m": m_codes,
            "k": k_codes,
            "kat": kat_codes,
            "dev": dev_codes,
            "nom": nom,
            "lognom": log1p_nominal,
            "lognom_sq": log1p_nominal * log1p_nominal,
        }
    )
    g_u = u_df.groupby("u", sort=False)
    u_prior_cnt = g_u.cumcount().to_numpy(dtype=np.float64)
    u_prior_cnt_cap10 = np.minimum(u_prior_cnt, 10.0)

    u_prior_nom_sum = g_u["nom"].cumsum().to_numpy(dtype=np.float64) - nom
    u_prior_nom_mean = np.where(u_prior_cnt > 0, u_prior_nom_sum / np.maximum(u_prior_cnt, 1.0), nom)
    u_cummax_prev = g_u["nom"].cummax().groupby(u_df["u"], sort=False).shift(1).to_numpy(dtype=np.float64)
    u_prior_nom_max = np.where(u_prior_cnt > 0, u_cummax_prev, nom)

    u_prior_log_sum = g_u["lognom"].cumsum().to_numpy(dtype=np.float64) - log1p_nominal
    u_prior_log_sq_sum = g_u["lognom_sq"].cumsum().to_numpy(dtype=np.float64) - (log1p_nominal * log1p_nominal)
    u_prior_log_mean = np.where(u_prior_cnt > 0, u_prior_log_sum / np.maximum(u_prior_cnt, 1.0), log1p_nominal)
    u_prior_log_var = np.where(
        u_prior_cnt > 1,
        np.maximum(0.0, (u_prior_log_sq_sum / np.maximum(u_prior_cnt, 1.0)) - (u_prior_log_mean**2)),
        1.0,
    )
    u_prior_log_std = np.sqrt(u_prior_log_var)
    user_lognom_zscore = np.where(u_prior_cnt > 0, (log1p_nominal - u_prior_log_mean) / (u_prior_log_std + 0.1), 0.0)
    nominal_to_user_prior_max = np.where(u_prior_cnt > 0, nom / np.maximum(u_prior_nom_max, 1.0), 1.0)

    user_seen_kanal_before = (u_df.groupby(["u", "k"], sort=False).cumcount().to_numpy() > 0).astype(np.int8)
    user_seen_kat_before = (u_df.groupby(["u", "kat"], sort=False).cumcount().to_numpy() > 0).astype(np.int8)
    user_seen_merchant_before = (u_df.groupby(["u", "m"], sort=False).cumcount().to_numpy() > 0).astype(np.int8)
    user_seen_device_before = (u_df.groupby(["u", "dev"], sort=False).cumcount().to_numpy() > 0).astype(np.int8)

    u_new_merch = (1 - user_seen_merchant_before).astype(np.float64)
    u_df["new_m"] = u_new_merch
    user_distinct_merchants_prior = u_df.groupby("u", sort=False)["new_m"].cumsum().to_numpy(dtype=np.float64) - u_new_merch

    trx_1j = combined["trx_1j"].to_numpy(dtype=np.float64)
    trx_7h = combined["trx_7h"].to_numpy(dtype=np.float64)
    user_trx_1j_vs_7h_ratio = trx_1j / (trx_7h + 0.5)
    user_nom_24j_ratio_to_avg = np.where(u_prior_cnt > 0, nom24 / np.maximum(u_prior_nom_mean, 1000.0), 1.0)

    # Strictly causal merchant history via composite key & np.searchsorted (ts_j < ts_i)
    ts_min = int(np.min(ts_sorted_sec)) if n_all > 0 else 0
    ts_rel = ts_sorted_sec - ts_min
    stride = int(max(int(np.max(ts_rel)) + 100_000, 100_000_000)) if n_all > 0 else 100_000_000
    m_comp = m_codes.astype(np.int64) * stride + ts_rel.astype(np.int64)
    m_order = np.argsort(m_comp, kind="mergesort")
    m_comp_sorted = m_comp[m_order]
    m_codes_sorted = m_codes[m_order]
    m_ts_sorted = ts_sorted_sec[m_order]
    m_u_sorted = u_codes[m_order]

    m_nom_sorted = nom[m_order]
    m_qris_sorted = (combined["kanal"].to_numpy() == "qris")[m_order].astype(np.float64)
    m_night_sorted = is_night[m_order].astype(np.float64)

    pref_nom = np.concatenate(([0.0], np.cumsum(m_nom_sorted)))
    pref_qris = np.concatenate(([0.0], np.cumsum(m_qris_sorted)))
    pref_night = np.concatenate(([0.0], np.cumsum(m_night_sorted)))

    m_start = np.searchsorted(m_comp_sorted, m_codes.astype(np.int64) * stride, side="left")
    m_right = np.searchsorted(m_comp_sorted, m_comp, side="left")
    m_left_1h = np.searchsorted(m_comp_sorted, m_codes.astype(np.int64) * stride + np.maximum(0, ts_rel - 3600), side="left")
    m_left_24h = np.searchsorted(m_comp_sorted, m_codes.astype(np.int64) * stride + np.maximum(0, ts_rel - 86400), side="left")
    m_left_7d = np.searchsorted(m_comp_sorted, m_codes.astype(np.int64) * stride + np.maximum(0, ts_rel - 7 * 86400), side="left")

    m_prior_cnt = np.maximum(0, m_right - m_start).astype(np.float64)
    m_prior_nom_sum = pref_nom[m_right] - pref_nom[m_start]
    merch_prior_nom_mean = np.where(m_prior_cnt > 0, m_prior_nom_sum / np.maximum(m_prior_cnt, 1.0), np.nan)
    nominal_to_merch_prior_mean = np.where(m_prior_cnt > 0, nom / np.maximum(merch_prior_nom_mean, 1.0), 1.0)
    merch_qris_share_prior = np.where(
        m_prior_cnt > 0,
        (pref_qris[m_right] - pref_qris[m_start]) / np.maximum(m_prior_cnt, 1.0),
        0.0,
    )
    merch_night_share_prior = np.where(
        m_prior_cnt > 0,
        (pref_night[m_right] - pref_night[m_start]) / np.maximum(m_prior_cnt, 1.0),
        0.0,
    )

    merch_rolling_1h_count = np.maximum(0, m_right - m_left_1h).astype(np.float64)
    merch_rolling_1h_nom_sum = pref_nom[m_right] - pref_nom[m_left_1h]
    merch_rolling_24h_count = np.maximum(0, m_right - m_left_24h).astype(np.float64)
    merch_rolling_24h_nom_sum = pref_nom[m_right] - pref_nom[m_left_24h]
    merch_rolling_7d_count = np.maximum(0, m_right - m_left_7d).astype(np.float64)
    merch_rolling_7d_nom_sum = pref_nom[m_right] - pref_nom[m_left_7d]

    merch_burst_1h_vs_7d = merch_rolling_1h_count / (merch_rolling_7d_count / 168.0 + 0.05)
    merch_burst_24h_vs_7d = merch_rolling_24h_count / (merch_rolling_7d_count / 7.0 + 0.1)

    uniq_24h_sorted = _compute_merchant_24h_unique_users(m_codes_sorted, m_ts_sorted, m_u_sorted)
    merch_rolling_24h_unique_users = np.empty(n_all, dtype=np.float64)
    merch_rolling_24h_unique_users[m_order] = uniq_24h_sorted

    # Assemble engineered block in single DataFrame concat (zero fragmentation warnings)
    eng_df = pd.DataFrame(
        {
            INTERACTION_COL: (combined["kanal"].astype(str) + "__" + combined["kategori_merchant"].astype(str)),
            "is_first_user_tx": is_first_user_tx,
            "user_has_prior_tx": user_has_prior_tx,
            "umur_merchant_neg_flag": umur_merchant_neg_flag,
            "umur_merchant_hari_clipped": umur_merchant_hari_clipped,
            "persen_saldo_capped_flag": persen_saldo_capped_flag,
            "nominal_gt_saldo_flag": nominal_gt_saldo_flag,
            "nominal_ge_review_cost": nominal_ge_review_cost,
            "log1p_nominal": log1p_nominal,
            "log1p_saldo": log1p_saldo,
            "log1p_nominal_24j": log1p_nominal_24j,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "is_night": is_night,
            "user_prior_tx_count_cap10": u_prior_cnt_cap10,
            "user_prior_nom_mean": u_prior_nom_mean,
            "user_prior_nom_max": u_prior_nom_max,
            "user_lognom_zscore": user_lognom_zscore,
            "nominal_to_user_prior_max": nominal_to_user_prior_max,
            "user_seen_kanal_before": user_seen_kanal_before,
            "user_seen_kat_before": user_seen_kat_before,
            "user_seen_merchant_before": user_seen_merchant_before,
            "user_seen_device_before": user_seen_device_before,
            "user_distinct_merchants_prior": user_distinct_merchants_prior,
            "user_trx_1j_vs_7h_ratio": user_trx_1j_vs_7h_ratio,
            "user_nom_24j_ratio_to_avg": user_nom_24j_ratio_to_avg,
            "merch_prior_nom_mean": merch_prior_nom_mean,
            "nominal_to_merch_prior_mean": nominal_to_merch_prior_mean,
            "merch_qris_share_prior": merch_qris_share_prior,
            "merch_night_share_prior": merch_night_share_prior,
            "merch_rolling_1h_count": merch_rolling_1h_count,
            "merch_rolling_1h_nom_sum": merch_rolling_1h_nom_sum,
            "merch_rolling_24h_count": merch_rolling_24h_count,
            "merch_rolling_24h_nom_sum": merch_rolling_24h_nom_sum,
            "merch_rolling_24h_unique_users": merch_rolling_24h_unique_users,
            "merch_rolling_7d_count": merch_rolling_7d_count,
            "merch_rolling_7d_nom_sum": merch_rolling_7d_nom_sum,
            "merch_burst_1h_vs_7d": merch_burst_1h_vs_7d,
            "merch_burst_24h_vs_7d": merch_burst_24h_vs_7d,
        },
        index=combined.index,
    )

    combined_full = pd.concat([combined, eng_df], axis=1)

    # Restore exact original train/test row order
    tr_part = (
        combined_full.loc[combined_full["_is_train"] == 1]
        .sort_values("_orig_idx", kind="mergesort")
        .reset_index(drop=True)
    )
    te_part = (
        combined_full.loc[combined_full["_is_train"] == 0]
        .sort_values("_orig_idx", kind="mergesort")
        .reset_index(drop=True)
    )

    # Lagged (>=21d) mature-only causal entity risk (strictly from mature train labels)
    lag_tr = compute_lagged_entity_risk(train_df, train_df, train_cutoff_ts=train_cutoff_ts, mature_cutoff=mature_cutoff)
    lag_te = compute_lagged_entity_risk(train_df, test_df, train_cutoff_ts=train_cutoff_ts, mature_cutoff=mature_cutoff)
    tr_part = pd.concat([tr_part, lag_tr.reset_index(drop=True)], axis=1)
    te_part = pd.concat([te_part, lag_te.reset_index(drop=True)], axis=1)

    drop_cols = {
        ID_COL,
        TARGET_COL,
        *LEAKAGE_COLS,
        *DRIFT_DROP_COLS,
        *RAW_HIGH_CARD_DROP_COLS,
        "umur_merchant_hari",
        "_is_train",
        "_orig_idx",
        "_ts_sec",
    }
    keep_cols = [c for c in tr_part.columns if c not in drop_cols]
    X_train = tr_part[keep_cols].copy()
    X_test = te_part[keep_cols].copy()

    cat_cols = [c for c in ALL_CAT_COLS if c in keep_cols]
    num_cols = [c for c in keep_cols if c not in cat_cols]

    feat_meta = {
        "n_features": len(keep_cols),
        "feature_names": keep_cols,
        "numeric_features": num_cols,
        "categorical_features": cat_cols,
        "dropped_columns": sorted(c for c in drop_cols if not c.startswith("_")),
        "user_causal_features": [c for c in USER_CAUSAL_COLS if c in keep_cols],
        "merchant_causal_features": [c for c in MERCHANT_CAUSAL_COLS if c in keep_cols],
        "lagged_target_features": [c for c in LAGGED_TARGET_COLS if c in keep_cols],
        "cold_start_flags": [c for c in COLD_START_HYGIENE_COLS if c in keep_cols],
        "time_cyclical_features": [c for c in TIME_CYCLICAL_COLS if c in keep_cols],
    }
    return X_train, X_test, feat_meta


# ---------------------------------------------------------------------------
# 4. Internal Fold Preprocessing & Model Wrappers (Zero Test Leakage)
# ---------------------------------------------------------------------------


def _prepare_tree_matrices(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_eval_list: list[pd.DataFrame],
    cat_cols: list[str],
    num_cols: list[str],
    seed: int,
) -> tuple[np.ndarray, list[np.ndarray], list[str]]:
    """Fit OrdinalEncoder & fold-internal TargetEncoder strictly on X_tr."""
    ord_enc = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=-1,
        dtype=np.float64,
    )
    n_splits = (2 if len(y_tr) < 3_000 else min(5, max(2, len(y_tr) // 10))) if len(y_tr) >= 4 else 2
    use_te = len(y_tr) >= 10 and int(np.sum(y_tr == 1)) >= 2

    tr_num = X_tr[num_cols].to_numpy(dtype=np.float64)
    tr_cat_ord = ord_enc.fit_transform(X_tr[cat_cols].astype(str))

    te_cols = [f"te_{c}" for c in cat_cols]
    if use_te:
        te_enc = TargetEncoder(
            cv=KFold(n_splits=n_splits, shuffle=True, random_state=seed),
            smooth="auto",
            target_type="binary",
        )
        tr_cat_te = te_enc.fit_transform(X_tr[cat_cols].astype(str), y_tr)
        X_tr_arr = np.hstack([tr_num, tr_cat_ord, tr_cat_te])
        feat_names = [*num_cols, *[f"ord_{c}" for c in cat_cols], *te_cols]
    else:
        te_enc = None
        X_tr_arr = np.hstack([tr_num, tr_cat_ord])
        feat_names = [*num_cols, *[f"ord_{c}" for c in cat_cols]]

    eval_arrs: list[np.ndarray] = []
    for X_ev in X_eval_list:
        ev_num = X_ev[num_cols].to_numpy(dtype=np.float64)
        ev_ord = ord_enc.transform(X_ev[cat_cols].astype(str))
        if te_enc is not None:
            ev_te = te_enc.transform(X_ev[cat_cols].astype(str))
            eval_arrs.append(np.hstack([ev_num, ev_ord, ev_te]))
        else:
            eval_arrs.append(np.hstack([ev_num, ev_ord]))

    return X_tr_arr, eval_arrs, feat_names


def _prepare_linear_matrices(
    X_tr: pd.DataFrame,
    X_eval_list: list[pd.DataFrame],
    cat_cols: list[str],
    num_cols: list[str],
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Fit SimpleImputer + StandardScaler + OneHotEncoder strictly on X_tr."""
    imp = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64)

    tr_num = scaler.fit_transform(imp.fit_transform(X_tr[num_cols].to_numpy(dtype=np.float64)))
    tr_cat = ohe.fit_transform(X_tr[cat_cols].astype(str))
    X_tr_arr = np.hstack([tr_num, tr_cat])

    eval_arrs: list[np.ndarray] = []
    for X_ev in X_eval_list:
        ev_num = scaler.transform(imp.transform(X_ev[num_cols].to_numpy(dtype=np.float64)))
        ev_cat = ohe.transform(X_ev[cat_cols].astype(str))
        eval_arrs.append(np.hstack([ev_num, ev_cat]))
    return X_tr_arr, eval_arrs


def _make_lgbm(
    n_rows: int,
    smoke_test: bool,
    seed: int,
    class_weight: str | None = None,
) -> lgb.LGBMClassifier:
    n_est = 12 if smoke_test else (120 if n_rows <= 60_000 else 220)
    min_child = 5 if n_rows < 5_000 else 50
    num_leaves = 15 if n_rows < 2_000 else (31 if (smoke_test or n_rows <= 60_000) else 63)
    return lgb.LGBMClassifier(
        n_estimators=n_est,
        learning_rate=0.10 if smoke_test else 0.06,
        num_leaves=num_leaves,
        min_child_samples=min_child,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        class_weight=class_weight,
        random_state=seed,
        deterministic=True,
        n_jobs=1 if smoke_test else 4,
        verbosity=-1,
    )


def _make_xgb(
    n_rows: int,
    smoke_test: bool,
    seed: int,
    scale_pos_weight: float = 1.0,
) -> xgb.XGBClassifier:
    n_est = 12 if smoke_test else (120 if n_rows <= 60_000 else 220)
    max_depth = 4 if n_rows < 2_000 else 6
    return xgb.XGBClassifier(
        n_estimators=n_est,
        learning_rate=0.10 if smoke_test else 0.06,
        max_depth=max_depth,
        min_child_weight=2 if n_rows < 2_000 else 10,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=float(scale_pos_weight),
        tree_method="hist",
        eval_metric="logloss",
        random_state=seed,
        n_jobs=1 if smoke_test else 4,
    )


def _make_hgb(
    n_rows: int,
    smoke_test: bool,
    seed: int,
    class_weight: str | None = None,
) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=12 if smoke_test else (120 if n_rows <= 60_000 else 200),
        learning_rate=0.10 if smoke_test else 0.06,
        max_leaf_nodes=15 if n_rows < 2_000 else 31,
        min_samples_leaf=5 if n_rows < 2_000 else 30,
        class_weight=class_weight,
        random_state=seed,
    )


def _calibrate_on_tail(
    base_clf: Any,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    method: str = "sigmoid",
) -> Any:
    """Calibrate fitted classifier on unweighted calibration slice with zero warnings."""
    n_pos = int(np.sum(y_cal == 1))
    n_neg = int(np.sum(y_cal == 0))
    if n_pos < 2 or n_neg < 2:
        return base_clf
    cv_k = min(5, n_pos, n_neg)
    cal_method = method if (method == "sigmoid" or n_pos >= 15) else "sigmoid"
    cal = CalibratedClassifierCV(
        estimator=FrozenEstimator(base_clf),
        method=cal_method,
        cv=cv_k,
    )
    cal.fit(X_cal, y_cal)
    return cal


def _predict_pos_prob(clf: Any, X: np.ndarray) -> np.ndarray:
    probs = clf.predict_proba(X)
    if probs.ndim == 2 and probs.shape[1] >= 2:
        return np.clip(probs[:, 1].astype(np.float64), 0.0, 1.0)
    return np.clip(probs.ravel().astype(np.float64), 0.0, 1.0)


# ---------------------------------------------------------------------------
# 5. Cost Metrics & R2e Rule
# ---------------------------------------------------------------------------


def compute_cost_metrics(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    nominal: np.ndarray | pd.Series,
    tandai: np.ndarray | pd.Series | None = None,
    tau: float = 1.0,
    review_cost_idr: int = REVIEW_COST_IDR,
) -> dict[str, Any]:
    """Compute exact Rp 25.000 fraud cost rule, PR-AUC, ROC-AUC, Brier, and baselines."""
    y_arr = np.asarray(y_true, dtype=np.int64)
    p_arr = np.clip(np.asarray(y_prob, dtype=np.float64), 0.0, 1.0)
    nom_arr = np.asarray(nominal, dtype=np.float64)

    if tandai is None:
        flag_arr = ((p_arr * nom_arr >= float(tau) * float(review_cost_idr)) & (nom_arr >= float(review_cost_idr))).astype(
            np.int64
        )
    else:
        flag_arr = np.asarray(tandai, dtype=np.int64)

    missed_fraud_loss = float(np.sum(nom_arr[(y_arr == 1) & (flag_arr == 0)]))
    n_flagged = int(np.sum(flag_arr == 1))
    review_cost = float(review_cost_idr) * n_flagged
    total_cost = missed_fraud_loss + review_cost

    mark_none_cost = float(np.sum(nom_arr[y_arr == 1]))
    mark_all_cost = float(review_cost_idr) * len(y_arr)
    naive_p05_flags = (p_arr >= 0.5).astype(np.int64)
    naive_p05_cost = float(np.sum(nom_arr[(y_arr == 1) & (naive_p05_flags == 0)])) + float(review_cost_idr) * int(
        np.sum(naive_p05_flags == 1)
    )

    n_pos = int(np.sum(y_arr == 1))
    tp = int(np.sum((y_arr == 1) & (flag_arr == 1)))
    fp = int(np.sum((y_arr == 0) & (flag_arr == 1)))
    fn = int(np.sum((y_arr == 1) & (flag_arr == 0)))

    return {
        "n_tx": int(len(y_arr)),
        "n_fraud": n_pos,
        "fraud_rate": float(n_pos / len(y_arr)) if len(y_arr) > 0 else 0.0,
        "pr_auc": _safe_auc(average_precision_score, y_arr, p_arr),
        "roc_auc": _safe_auc(roc_auc_score, y_arr, p_arr),
        "brier": _safe_brier(y_arr, p_arr),
        "log_loss": _safe_log_loss(y_arr, p_arr),
        "tau": float(tau),
        "n_flagged": n_flagged,
        "pct_flagged": float(n_flagged / len(y_arr)) if len(y_arr) > 0 else 0.0,
        "tp_count": tp,
        "fp_count": fp,
        "fn_count": fn,
        "precision": float(precision_score(y_arr, flag_arr, zero_division=0)) if len(y_arr) > 0 else 0.0,
        "recall": float(recall_score(y_arr, flag_arr, zero_division=0)) if len(y_arr) > 0 else 0.0,
        "f1": float(f1_score(y_arr, flag_arr, zero_division=0)) if len(y_arr) > 0 else 0.0,
        "missed_fraud_loss_idr": missed_fraud_loss,
        "review_cost_idr": review_cost,
        "total_cost_idr": total_cost,
        "mark_none_cost_idr": mark_none_cost,
        "mark_all_cost_idr": mark_all_cost,
        "naive_p05_cost_idr": naive_p05_cost,
        "cost_saved_vs_mark_none_idr": mark_none_cost - total_cost,
    }


# ---------------------------------------------------------------------------
# 6. R2a Leakage Impact & R2b Validation Comparison
# ---------------------------------------------------------------------------


def _build_oot_folds(ts_tr: pd.Series) -> list[dict[str, Any]]:
    """Return 3 expanding-window mature OOT folds excluding Aug 11-31 from validation."""
    specs = [
        ("Jun", pd.Timestamp("2025-06-01"), pd.Timestamp("2025-07-01")),
        ("Jul", pd.Timestamp("2025-07-01"), pd.Timestamp("2025-08-01")),
        ("Aug_1_10", pd.Timestamp("2025-08-01"), MATURE_END_EXCLUSIVE),
    ]
    folds: list[dict[str, Any]] = []
    for name, val_start, val_end in specs:
        tr_idx = np.flatnonzero((ts_tr < val_start).to_numpy())
        va_idx = np.flatnonzero(((ts_tr >= val_start) & (ts_tr < val_end)).to_numpy())
        if len(tr_idx) == 0 or len(va_idx) == 0:
            continue
        folds.append(
            {
                "name": name,
                "train_idx": tr_idx,
                "val_idx": va_idx,
                "train_max": str(ts_tr.iloc[tr_idx].max()),
                "val_min": str(ts_tr.iloc[va_idx].min()),
                "val_max": str(ts_tr.iloc[va_idx].max()),
            }
        )
    return folds


def _make_fold_features(
    raw_train: pd.DataFrame,
    X_base_tr: pd.DataFrame,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    mature_cutoff: pd.Timestamp = MATURE_CUTOFF,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Slice precomputed causal features and recompute lag21d strictly from fold train_idx."""
    fold_tr_raw = raw_train.iloc[tr_idx]
    fold_va_raw = raw_train.iloc[va_idx]
    fold_tr_max = _to_ts(fold_tr_raw[TIME_COL]).max()

    X_tr_f = X_base_tr.iloc[tr_idx].copy()
    X_va_f = X_base_tr.iloc[va_idx].copy()

    lag_tr = compute_lagged_entity_risk(fold_tr_raw, fold_tr_raw, train_cutoff_ts=fold_tr_max, mature_cutoff=mature_cutoff)
    lag_va = compute_lagged_entity_risk(fold_tr_raw, fold_va_raw, train_cutoff_ts=fold_tr_max, mature_cutoff=mature_cutoff)
    for col in LAGGED_TARGET_COLS:
        X_tr_f[col] = lag_tr[col].to_numpy()
        X_va_f[col] = lag_va[col].to_numpy()
    return X_tr_f, X_va_f


def evaluate_leakage_and_drift_impact(
    raw_train: pd.DataFrame,
    X_base_tr: pd.DataFrame,
    cat_cols: list[str],
    num_cols: list[str],
    smoke_test: bool = False,
    seed: int = SEED,
) -> dict[str, Any]:
    """R2a: Quantify PR-AUC, ROC-AUC, and Cost distortion caused by flag_review_manual & v15."""
    ts_tr = _to_ts(raw_train[TIME_COL])
    y_all = raw_train[TARGET_COL].to_numpy(dtype=np.int64)
    nom_all = raw_train["nominal"].to_numpy(dtype=np.float64)
    oot_folds = _build_oot_folds(ts_tr)
    # Use Aug_1_10 in smoke_test or Jul + Aug_1_10 in full mode for fast leakage ablation
    eval_folds = (oot_folds[-1:] if smoke_test else [f for f in oot_folds if f["name"] in ("Jul", "Aug_1_10")]) or oot_folds[-1:]

    def _run_oot_ablation(with_leakage: bool) -> dict[str, Any]:
        y_list, p_list, nom_list = [], [], []
        for f in eval_folds:
            tr_idx, va_idx = f["train_idx"], f["val_idx"]
            X_tr_f, X_va_f = _make_fold_features(raw_train, X_base_tr, tr_idx, va_idx)
            y_tr, y_va = y_all[tr_idx], y_all[va_idx]
            use_num = list(num_cols)
            if with_leakage:
                X_tr_f["flag_review_manual"] = raw_train.iloc[tr_idx]["flag_review_manual"].to_numpy(dtype=np.float64)
                X_va_f["flag_review_manual"] = raw_train.iloc[va_idx]["flag_review_manual"].to_numpy(dtype=np.float64)
                X_tr_f["v15"] = raw_train.iloc[tr_idx]["v15"].to_numpy(dtype=np.float64)
                X_va_f["v15"] = raw_train.iloc[va_idx]["v15"].to_numpy(dtype=np.float64)
                use_num = [*use_num, "flag_review_manual", "v15"]

            X_tr_arr, [X_va_arr], _ = _prepare_tree_matrices(X_tr_f, y_tr, [X_va_f], cat_cols, use_num, seed)
            clf = _make_lgbm(len(X_tr_arr), smoke_test=smoke_test, seed=seed)
            clf.fit(X_tr_arr, y_tr)
            p_va = _predict_pos_prob(clf, X_va_arr)
            y_list.append(y_va)
            p_list.append(p_va)
            nom_list.append(nom_all[va_idx])

        return compute_cost_metrics(np.concatenate(y_list), np.concatenate(p_list), np.concatenate(nom_list))

    oot_leaky = _run_oot_ablation(with_leakage=True)
    oot_clean = _run_oot_ablation(with_leakage=False)

    # Random 5-fold CV WITH leakage (demonstrating double-optimism of random split + flag_review_manual)
    skf = StratifiedKFold(n_splits=2 if smoke_test else 5, shuffle=True, random_state=seed)
    rcv_y, rcv_p, rcv_nom = [], [], []
    use_num_leaky = [*num_cols, "flag_review_manual", "v15"]
    for tr_idx, va_idx in skf.split(X_base_tr, y_all):
        X_tr_f = X_base_tr.iloc[tr_idx].copy()
        X_va_f = X_base_tr.iloc[va_idx].copy()
        X_tr_f["flag_review_manual"] = raw_train.iloc[tr_idx]["flag_review_manual"].to_numpy(dtype=np.float64)
        X_va_f["flag_review_manual"] = raw_train.iloc[va_idx]["flag_review_manual"].to_numpy(dtype=np.float64)
        X_tr_f["v15"] = raw_train.iloc[tr_idx]["v15"].to_numpy(dtype=np.float64)
        X_va_f["v15"] = raw_train.iloc[va_idx]["v15"].to_numpy(dtype=np.float64)
        X_tr_arr, [X_va_arr], _ = _prepare_tree_matrices(X_tr_f, y_all[tr_idx], [X_va_f], cat_cols, use_num_leaky, seed)
        clf = _make_lgbm(len(X_tr_arr), smoke_test=smoke_test, seed=seed)
        clf.fit(X_tr_arr, y_all[tr_idx])
        rcv_p.append(_predict_pos_prob(clf, X_va_arr))
        rcv_y.append(y_all[va_idx])
        rcv_nom.append(nom_all[va_idx])
    rcv_leaky = compute_cost_metrics(np.concatenate(rcv_y), np.concatenate(rcv_p), np.concatenate(rcv_nom))

    pr_with = oot_leaky["pr_auc"] or 0.0
    pr_without = oot_clean["pr_auc"] or 0.0
    roc_with = oot_leaky["roc_auc"] or 0.5
    roc_without = oot_clean["roc_auc"] or 0.5

    return {
        "oot_with_leakage": oot_leaky,
        "oot_without_leakage": oot_clean,
        "random_cv_with_leakage": rcv_leaky,
        "pr_auc_with_leakage": pr_with,
        "pr_auc_without_leakage": pr_without,
        "pr_auc_leakage_delta": float(pr_with - pr_without),
        "roc_auc_with_leakage": roc_with,
        "roc_auc_without_leakage": roc_without,
        "roc_auc_leakage_delta": float(roc_with - roc_without),
        "cost_with_leakage_idr": oot_leaky["total_cost_idr"],
        "cost_without_leakage_idr": oot_clean["total_cost_idr"],
        "conclusion": "Removing flag_review_manual & v15 eliminates artificial PR-AUC inflation and time-drift failure in Sep-Oct.",
    }


def compare_validation_schemes(
    raw_train: pd.DataFrame,
    X_base_tr: pd.DataFrame,
    cat_cols: list[str],
    num_cols: list[str],
    audit_dict: dict[str, Any],
    smoke_test: bool = False,
    seed: int = SEED,
) -> dict[str, Any]:
    """R2b: Compare mature expanding-window OOT vs Random CV vs unguarded Aug 11-31 & censoring policy."""
    ts_tr = _to_ts(raw_train[TIME_COL])
    y_all = raw_train[TARGET_COL].to_numpy(dtype=np.int64)
    nom_all = raw_train["nominal"].to_numpy(dtype=np.float64)

    # 1. Mature expanding-window OOT folds (Jun, Jul, Aug 1-10)
    oot_folds = _build_oot_folds(ts_tr)
    oot_fold_metrics: list[dict[str, Any]] = []
    oot_y, oot_p, oot_nom = [], [], []
    for f in oot_folds:
        tr_idx, va_idx = f["train_idx"], f["val_idx"]
        X_tr_f, X_va_f = _make_fold_features(raw_train, X_base_tr, tr_idx, va_idx)
        X_tr_arr, [X_va_arr], _ = _prepare_tree_matrices(X_tr_f, y_all[tr_idx], [X_va_f], cat_cols, num_cols, seed)
        clf = _make_lgbm(len(X_tr_arr), smoke_test=smoke_test, seed=seed)
        clf.fit(X_tr_arr, y_all[tr_idx])
        p_va = _predict_pos_prob(clf, X_va_arr)
        m = compute_cost_metrics(y_all[va_idx], p_va, nom_all[va_idx])
        m["fold"] = f["name"]
        m["train_max"] = f["train_max"]
        m["val_min"] = f["val_min"]
        m["val_max"] = f["val_max"]
        oot_fold_metrics.append(m)
        oot_y.append(y_all[va_idx])
        oot_p.append(p_va)
        oot_nom.append(nom_all[va_idx])

    oot_pooled = compute_cost_metrics(np.concatenate(oot_y), np.concatenate(oot_p), np.concatenate(oot_nom))

    # 2. Random 5-fold CV (clean features, across Jan-Aug)
    skf = StratifiedKFold(n_splits=2 if smoke_test else 5, shuffle=True, random_state=seed)
    rcv_y, rcv_p, rcv_nom = [], [], []
    for tr_idx, va_idx in skf.split(X_base_tr, y_all):
        X_tr_arr, [X_va_arr], _ = _prepare_tree_matrices(
            X_base_tr.iloc[tr_idx], y_all[tr_idx], [X_base_tr.iloc[va_idx]], cat_cols, num_cols, seed
        )
        clf = _make_lgbm(len(X_tr_arr), smoke_test=smoke_test, seed=seed)
        clf.fit(X_tr_arr, y_all[tr_idx])
        rcv_p.append(_predict_pos_prob(clf, X_va_arr))
        rcv_y.append(y_all[va_idx])
        rcv_nom.append(nom_all[va_idx])
    random_cv = compute_cost_metrics(np.concatenate(rcv_y), np.concatenate(rcv_p), np.concatenate(rcv_nom))

    # 3. Unguarded censored tail OOT (Jan-Jul -> Aug 11-31)
    tr_jul_idx = np.flatnonzero((ts_tr < pd.Timestamp("2025-08-01")).to_numpy())
    va_cens_idx = np.flatnonzero((ts_tr >= AUG_11_START).to_numpy())
    if len(tr_jul_idx) > 0 and len(va_cens_idx) > 0:
        X_tr_c, X_va_c = _make_fold_features(raw_train, X_base_tr, tr_jul_idx, va_cens_idx)
        X_tr_arr, [X_va_arr], _ = _prepare_tree_matrices(X_tr_c, y_all[tr_jul_idx], [X_va_c], cat_cols, num_cols, seed)
        clf = _make_lgbm(len(X_tr_arr), smoke_test=smoke_test, seed=seed)
        clf.fit(X_tr_arr, y_all[tr_jul_idx])
        unguarded_censored = compute_cost_metrics(y_all[va_cens_idx], _predict_pos_prob(clf, X_va_arr), nom_all[va_cens_idx])
    else:
        unguarded_censored = {}

    # 4. Censoring training policy selection (drop_aug11_31 vs maturity_weighted_aug11_17)
    aug_w = audit_dict["label_delay_censoring"]["august_windows"]["aug_11_17"]
    aug11_17_pos = int(aug_w["n_fraud"])
    aug11_17_mat = float(aug_w["empirical_maturity"])
    adopted_policy = (
        "maturity_weighted_aug11_17"
        if (aug11_17_pos >= 5 and aug11_17_mat >= 0.45)
        else "drop_aug11_31"
    )

    return {
        "oot_folds": oot_fold_metrics,
        "oot_pooled": oot_pooled,
        "random_cv_vs_oot": {
            "random_cv_pr_auc": random_cv["pr_auc"],
            "oot_pooled_pr_auc": oot_pooled["pr_auc"],
            "random_cv_roc_auc": random_cv["roc_auc"],
            "oot_pooled_roc_auc": oot_pooled["roc_auc"],
            "unguarded_aug_11_31": unguarded_censored,
        },
        "censoring_policies": {
            "policy_a": "drop_aug11_31 (train strictly on waktu < 2025-08-11)",
            "policy_b": "maturity_weighted_aug11_17 (w=1.0 <= Aug 10; Aug 11-17 pos w=1.0, neg w=empirical_maturity; drop Aug 18-31)",
            "aug_11_17_frauds": aug11_17_pos,
            "aug_11_17_maturity": aug11_17_mat,
            "adopted_policy": adopted_policy,
        },
    }


def _get_mature_train_mask_and_weights(
    raw_train: pd.DataFrame,
    censoring_policy: str,
    aug11_17_maturity: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return boolean mask of training rows and censoring-adjusted weights (no recency yet)."""
    ts_tr = _to_ts(raw_train[TIME_COL])
    y = raw_train[TARGET_COL].to_numpy(dtype=np.int64)

    if censoring_policy == "maturity_weighted_aug11_17":
        mask = (ts_tr < AUG_18_START).to_numpy()
        in_11_17 = ((ts_tr >= AUG_11_START) & (ts_tr < AUG_18_START)).to_numpy()
        w = np.ones(len(raw_train), dtype=np.float64)
        w[in_11_17 & (y == 0)] = float(np.clip(aug11_17_maturity, 0.3, 1.0))
        return mask, w[mask]

    mask = (ts_tr < MATURE_END_EXCLUSIVE).to_numpy()
    return mask, np.ones(int(np.sum(mask)), dtype=np.float64)


# ---------------------------------------------------------------------------
# 7. R2d Imbalance Strategies & Temporal Shift Adaptation
# ---------------------------------------------------------------------------


def _fit_strategy_predict(
    strategy: str,
    X_tr_tree: np.ndarray,
    X_ev_tree: np.ndarray,
    X_tr_lin: np.ndarray,
    X_ev_lin: np.ndarray,
    y_tr: np.ndarray,
    nom_tr: np.ndarray,
    ts_tr_sec: np.ndarray,
    censor_w_tr: np.ndarray,
    smoke_test: bool,
    seed: int,
    recency_lambda: float = 0.015,
    blend_alpha: float = 0.65,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit an imbalance/adaptation strategy and return (raw_ev_prob, oof_or_tail_cal_prob)."""
    n_tr = len(y_tr)
    n_pos = int(np.sum(y_tr == 1))
    n_neg = int(n_tr - n_pos)
    pos_ratio = n_pos / max(n_neg, 1)

    if strategy == "unweighted_lgbm":
        clf = _make_lgbm(n_tr, smoke_test=smoke_test, seed=seed)
        clf.fit(X_tr_tree, y_tr, sample_weight=censor_w_tr)
        return _predict_pos_prob(clf, X_ev_tree), _predict_pos_prob(clf, X_tr_tree)

    if strategy == "class_weight_balanced":
        clf = _make_lgbm(n_tr, smoke_test=smoke_test, seed=seed, class_weight="balanced")
        clf.fit(X_tr_tree, y_tr, sample_weight=censor_w_tr)
        return _predict_pos_prob(clf, X_ev_tree), _predict_pos_prob(clf, X_tr_tree)

    if strategy == "cost_weighted_lgbm":
        cost_w = np.where(y_tr == 1, np.clip(nom_tr / float(REVIEW_COST_IDR), 1.0, 200.0), 1.0)
        clf = _make_lgbm(n_tr, smoke_test=smoke_test, seed=seed)
        clf.fit(X_tr_tree, y_tr, sample_weight=cost_w * censor_w_tr)
        return _predict_pos_prob(clf, X_ev_tree), _predict_pos_prob(clf, X_tr_tree)

    if strategy == "xgb_scale_pos_weight":
        spw = max(1.0, n_neg / max(n_pos, 1))
        clf = _make_xgb(n_tr, smoke_test=smoke_test, seed=seed, scale_pos_weight=spw)
        clf.fit(X_tr_tree, y_tr, sample_weight=censor_w_tr)
        return _predict_pos_prob(clf, X_ev_tree), _predict_pos_prob(clf, X_tr_tree)

    if strategy == "hgb_balanced":
        clf = _make_hgb(n_tr, smoke_test=smoke_test, seed=seed, class_weight="balanced")
        clf.fit(X_tr_tree, y_tr, sample_weight=censor_w_tr)
        return _predict_pos_prob(clf, X_ev_tree), _predict_pos_prob(clf, X_tr_tree)

    if strategy == "logreg_balanced":
        clf = LogisticRegression(
            class_weight="balanced",
            max_iter=150 if smoke_test else 400,
            C=0.5,
            random_state=seed,
        )
        clf.fit(X_tr_lin, y_tr, sample_weight=censor_w_tr)
        return _predict_pos_prob(clf, X_ev_lin), _predict_pos_prob(clf, X_tr_lin)

    if strategy == "rus_bagging_lgbm":
        n_bags = 2 if smoke_test else 7
        target_ratio = min(0.5, max(0.15, pos_ratio * 2.0))
        ev_preds = np.zeros(len(X_ev_tree), dtype=np.float64)
        tr_preds = np.zeros(n_tr, dtype=np.float64)
        for b in range(n_bags):
            if n_pos >= 2 and pos_ratio < target_ratio:
                rus = RandomUnderSampler(sampling_strategy=target_ratio, random_state=seed + b)
                X_res, y_res = rus.fit_resample(X_tr_tree, y_tr)
            else:
                X_res, y_res = X_tr_tree, y_tr
            clf = _make_lgbm(len(X_res), smoke_test=smoke_test, seed=seed + b)
            clf.fit(X_res, y_res)
            ev_preds += _predict_pos_prob(clf, X_ev_tree)
            tr_preds += _predict_pos_prob(clf, X_tr_tree)
        return ev_preds / n_bags, tr_preds / n_bags

    if strategy == "smote_lgbm":
        # Impute NaNs strictly inside train fold before SMOTE
        imp = SimpleImputer(strategy="median")
        X_tr_imp = imp.fit_transform(X_tr_tree)
        X_ev_imp = imp.transform(X_ev_tree)
        target_ratio = min(0.5, max(0.15, pos_ratio * 2.0))
        if n_pos >= 2 and pos_ratio < target_ratio:
            k_nb = min(5, max(1, n_pos - 1))
            if k_nb >= 1 and n_pos > k_nb:
                sampler = SMOTE(sampling_strategy=target_ratio, k_neighbors=k_nb, random_state=seed)
            else:
                sampler = RandomOverSampler(sampling_strategy=target_ratio, random_state=seed)
            X_res, y_res = sampler.fit_resample(X_tr_imp, y_tr)
        else:
            X_res, y_res = X_tr_imp, y_tr
        clf = _make_lgbm(len(X_res), smoke_test=smoke_test, seed=seed)
        clf.fit(X_res, y_res)
        return _predict_pos_prob(clf, X_ev_imp), _predict_pos_prob(clf, X_tr_imp)

    if strategy == "recency_adapted_ensemble":
        max_ts = float(np.max(ts_tr_sec)) if n_tr > 0 else 0.0
        age_days = np.maximum(0.0, (max_ts - ts_tr_sec) / 86400.0)
        rec_w = np.exp(-float(recency_lambda) * age_days) * censor_w_tr

        clf_full = _make_lgbm(n_tr, smoke_test=smoke_test, seed=seed)
        clf_full.fit(X_tr_tree, y_tr, sample_weight=rec_w)

        # Recent-window XGBClassifier (last 90 days or most recent 40% of train)
        recent_cutoff = max_ts - 90.0 * 86400.0
        rec_mask = ts_tr_sec >= recent_cutoff
        if int(np.sum(rec_mask)) < max(60, int(0.35 * n_tr)) or int(np.sum(y_tr[rec_mask] == 1)) < 2:
            q_cut = np.quantile(ts_tr_sec, 0.55)
            rec_mask = ts_tr_sec >= q_cut

        if int(np.sum(y_tr[rec_mask] == 1)) >= 2 and int(np.sum(y_tr[rec_mask] == 0)) >= 2:
            clf_rec = _make_xgb(int(np.sum(rec_mask)), smoke_test=smoke_test, seed=seed + 1, scale_pos_weight=1.0)
            clf_rec.fit(X_tr_tree[rec_mask], y_tr[rec_mask], sample_weight=censor_w_tr[rec_mask])
            p_ev = blend_alpha * _predict_pos_prob(clf_full, X_ev_tree) + (1.0 - blend_alpha) * _predict_pos_prob(
                clf_rec, X_ev_tree
            )
            p_tr = blend_alpha * _predict_pos_prob(clf_full, X_tr_tree) + (1.0 - blend_alpha) * _predict_pos_prob(
                clf_rec, X_tr_tree
            )
        else:
            p_ev = _predict_pos_prob(clf_full, X_ev_tree)
            p_tr = _predict_pos_prob(clf_full, X_tr_tree)
        return p_ev, p_tr

    raise ValueError(f"Unknown imbalance strategy: {strategy}")


def compare_imbalance_and_adaptation(
    raw_train: pd.DataFrame,
    X_base_tr: pd.DataFrame,
    cat_cols: list[str],
    num_cols: list[str],
    audit_dict: dict[str, Any],
    smoke_test: bool = False,
    seed: int = SEED,
) -> dict[str, Any]:
    """R2d: Compare >=4 imbalance strategies & multiple classical ML models on mature OOT + recency gate."""
    ts_tr = _to_ts(raw_train[TIME_COL])
    ts_tr_sec = _ts_to_epoch_sec(ts_tr).astype(np.float64)
    y_all = raw_train[TARGET_COL].to_numpy(dtype=np.int64)
    nom_all = raw_train["nominal"].to_numpy(dtype=np.float64)

    oot_folds = _build_oot_folds(ts_tr)
    # In smoke mode evaluate Aug_1_10 (and Jul for lambda gate if needed) to stay well within <15s; in full mode evaluate all 3 OOT folds
    eval_folds = oot_folds[-1:] if smoke_test else oot_folds
    if not eval_folds:
        eval_folds = oot_folds

    # Prepare fold matrices once per fold
    prepared_folds: list[dict[str, Any]] = []
    for f in eval_folds:
        tr_idx, va_idx = f["train_idx"], f["val_idx"]
        X_tr_f, X_va_f = _make_fold_features(raw_train, X_base_tr, tr_idx, va_idx)
        y_tr, y_va = y_all[tr_idx], y_all[va_idx]
        X_tr_tree, [X_va_tree], feat_names = _prepare_tree_matrices(X_tr_f, y_tr, [X_va_f], cat_cols, num_cols, seed)
        X_tr_lin, [X_va_lin] = _prepare_linear_matrices(X_tr_f, [X_va_f], cat_cols, num_cols)
        prepared_folds.append(
            {
                "name": f["name"],
                "tr_idx": tr_idx,
                "va_idx": va_idx,
                "X_tr_tree": X_tr_tree,
                "X_va_tree": X_va_tree,
                "X_tr_lin": X_tr_lin,
                "X_va_lin": X_va_lin,
                "y_tr": y_tr,
                "y_va": y_va,
                "nom_tr": nom_all[tr_idx],
                "nom_va": nom_all[va_idx],
                "ts_tr_sec": ts_tr_sec[tr_idx],
                "kanal_va": raw_train.iloc[va_idx]["kanal"].to_numpy(),
                "kat_va": raw_train.iloc[va_idx]["kategori_merchant"].to_numpy(),
            }
        )

    # Recency lambda validation gate on Jul vs Aug_1_10
    lambda_grid = (0.0, 0.015) if smoke_test else (0.0, 0.005, 0.010, 0.020)
    recency_tuning: list[dict[str, Any]] = []
    jul_baseline_pr = None
    best_lambda = 0.015 if smoke_test else 0.010
    best_aug_score = -1.0

    for lam in lambda_grid:
        fold_scores: dict[str, float] = {}
        for pf in prepared_folds:
            p_va, _ = _fit_strategy_predict(
                "recency_adapted_ensemble",
                pf["X_tr_tree"],
                pf["X_va_tree"],
                pf["X_tr_lin"],
                pf["X_va_lin"],
                pf["y_tr"],
                pf["nom_tr"],
                pf["ts_tr_sec"],
                np.ones(len(pf["y_tr"]), dtype=np.float64),
                smoke_test=smoke_test,
                seed=seed,
                recency_lambda=lam,
            )
            fold_scores[pf["name"]] = _safe_auc(average_precision_score, pf["y_va"], p_va) or 0.0
        if lam == 0.0 and "Jul" in fold_scores:
            jul_baseline_pr = fold_scores["Jul"]
        aug_sc = fold_scores.get("Aug_1_10", float(np.mean(list(fold_scores.values()))))
        jul_ok = True
        if jul_baseline_pr is not None and "Jul" in fold_scores:
            jul_ok = fold_scores["Jul"] >= jul_baseline_pr - 0.005
        recency_tuning.append({"lambda": lam, "fold_pr_auc": fold_scores, "jul_gate_passed": jul_ok})
        if jul_ok and aug_sc > best_aug_score:
            best_aug_score = aug_sc
            best_lambda = lam

    strategies = (
        "unweighted_lgbm",
        "class_weight_balanced",
        "cost_weighted_lgbm",
        "rus_bagging_lgbm",
        "smote_lgbm",
        "recency_adapted_ensemble",
    )
    models_extra = ("xgb_scale_pos_weight", "hgb_balanced", "logreg_balanced")

    def _eval_named(name: str) -> dict[str, Any]:
        y_list, p_list, nom_list, kan_list, kat_list = [], [], [], [], []
        for pf in prepared_folds:
            p_va, p_tr_in = _fit_strategy_predict(
                name,
                pf["X_tr_tree"],
                pf["X_va_tree"],
                pf["X_tr_lin"],
                pf["X_va_lin"],
                pf["y_tr"],
                pf["nom_tr"],
                pf["ts_tr_sec"],
                np.ones(len(pf["y_tr"]), dtype=np.float64),
                smoke_test=smoke_test,
                seed=seed,
                recency_lambda=best_lambda,
            )
            # Lightweight sigmoid calibration on OOF-safe rank-preserving logit using train slice for fair cost
            y_list.append(pf["y_va"])
            p_list.append(p_va)
            nom_list.append(pf["nom_va"])
            kan_list.append(pf["kanal_va"])
            kat_list.append(pf["kat_va"])

        y_cat = np.concatenate(y_list)
        p_cat = np.concatenate(p_list)
        nom_cat = np.concatenate(nom_list)
        kan_cat = np.concatenate(kan_list)
        kat_cat = np.concatenate(kat_list)

        overall = compute_cost_metrics(y_cat, p_cat, nom_cat)
        qris_m = kan_cat == "qris"
        jasa_m = kat_cat == "jasa_lainnya"
        overall["qris_pr_auc"] = _safe_auc(average_precision_score, y_cat[qris_m], p_cat[qris_m]) if np.any(qris_m) else None
        overall["jasa_lainnya_pr_auc"] = (
            _safe_auc(average_precision_score, y_cat[jasa_m], p_cat[jasa_m]) if np.any(jasa_m) else None
        )
        overall["strategy"] = name
        return overall

    imbalance_comparison = {s: _eval_named(s) for s in strategies}
    model_comparison = {
        "lgbm_balanced": imbalance_comparison["class_weight_balanced"],
        **{m: _eval_named(m) for m in models_extra},
        "recency_lgbm_xgb_ensemble": imbalance_comparison["recency_adapted_ensemble"],
    }

    return {
        "temporal_shift_audit": audit_dict["temporal_shift_audit"],
        "recency_lambda_tuning": recency_tuning,
        "selected_lambda": best_lambda,
        "imbalance_comparison": imbalance_comparison,
        "model_comparison": model_comparison,
        "selected_strategy": "recency_adapted_ensemble",
        "selected_model": "recency_adapted_ensemble (LGBM recency-weighted + recent-window XGBClassifier)",
    }


# ---------------------------------------------------------------------------
# 8. R2e Calibration & Rp 25.000 Cost Decision Rule
# ---------------------------------------------------------------------------


def _fit_calibrated_final_ensemble(
    X_tr_tree: np.ndarray,
    y_tr: np.ndarray,
    ts_tr_sec: np.ndarray,
    censor_w_tr: np.ndarray,
    ts_tr_dt: pd.Series,
    best_lambda: float,
    cal_method: str,
    smoke_test: bool,
    seed: int,
) -> tuple[Any, float]:
    """Fit final recency-adapted ensemble and calibrate cleanly without sample_weight warnings."""
    n_tr = len(y_tr)
    max_ts = float(np.max(ts_tr_sec)) if n_tr > 0 else 0.0
    age_days = np.maximum(0.0, (max_ts - ts_tr_sec) / 86400.0)
    rec_w = np.exp(-float(best_lambda) * age_days) * censor_w_tr

    # Check if Aug 5-10 holdout has enough positives (>= 15) for dedicated temporal tail calibration
    aug5_10_mask = ((ts_tr_dt >= pd.Timestamp("2025-08-05")) & (ts_tr_dt < MATURE_END_EXCLUSIVE)).to_numpy()
    n_pos_tail = int(np.sum(y_tr[aug5_10_mask] == 1)) if np.any(aug5_10_mask) else 0
    n_neg_tail = int(np.sum(y_tr[aug5_10_mask] == 0)) if np.any(aug5_10_mask) else 0

    def _fit_raw_models(X_m: np.ndarray, y_m: np.ndarray, w_m: np.ndarray, t_m: np.ndarray) -> tuple[Any, Any | None]:
        clf_l = _make_lgbm(len(X_m), smoke_test=smoke_test, seed=seed)
        clf_l.fit(X_m, y_m, sample_weight=w_m)

        m_max = float(np.max(t_m)) if len(t_m) > 0 else 0.0
        rec_m = t_m >= (m_max - 90.0 * 86400.0)
        if int(np.sum(rec_m)) < max(60, int(0.35 * len(X_m))) or int(np.sum(y_m[rec_m] == 1)) < 2:
            rec_m = t_m >= np.quantile(t_m, 0.55)
        clf_x = None
        if int(np.sum(y_m[rec_m] == 1)) >= 2 and int(np.sum(y_m[rec_m] == 0)) >= 2:
            clf_x = _make_xgb(int(np.sum(rec_m)), smoke_test=smoke_test, seed=seed + 1, scale_pos_weight=1.0)
            clf_x.fit(X_m[rec_m], y_m[rec_m])
        return clf_l, clf_x

    def _raw_blend_pred(clf_l: Any, clf_x: Any | None, X_q: np.ndarray) -> np.ndarray:
        p_l = _predict_pos_prob(clf_l, X_q)
        if clf_x is None:
            return p_l
        return 0.65 * p_l + 0.35 * _predict_pos_prob(clf_x, X_q)

    # OOF-calibrated full mature fit (ensures 100% of Aug 1-10 informs splits & calibration)
    n_pos_all = int(np.sum(y_tr == 1))
    n_neg_all = int(np.sum(y_tr == 0))
    n_splits = min(2 if smoke_test else 3, n_pos_all, n_neg_all)
    oof_raw = np.zeros(n_tr, dtype=np.float64)
    if n_splits >= 2:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for tr_i, va_i in skf.split(X_tr_tree, y_tr):
            c_l, c_x = _fit_raw_models(X_tr_tree[tr_i], y_tr[tr_i], rec_w[tr_i], ts_tr_sec[tr_i])
            oof_raw[va_i] = _raw_blend_pred(c_l, c_x, X_tr_tree[va_i])
    else:
        c_l, c_x = _fit_raw_models(X_tr_tree, y_tr, rec_w, ts_tr_sec)
        oof_raw = _raw_blend_pred(c_l, c_x, X_tr_tree)

    # Fit unweighted (or censoring-only weighted) OOF calibrator
    if n_pos_all >= 2 and n_neg_all >= 2:
        if cal_method == "isotonic" and n_pos_all >= 30:
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(oof_raw, y_tr, sample_weight=censor_w_tr)
            cal_fn_oof = lambda p: np.clip(iso.predict(p), 0.0, 1.0)
        else:
            platt = LogisticRegression(C=10.0, max_iter=200, random_state=seed)
            platt.fit(_logit(oof_raw).reshape(-1, 1), y_tr, sample_weight=censor_w_tr)
            cal_fn_oof = lambda p: _predict_pos_prob(platt, _logit(p).reshape(-1, 1))
    else:
        cal_fn_oof = lambda p: np.clip(p, 0.0, 1.0)

    full_l, full_x = _fit_raw_models(X_tr_tree, y_tr, rec_w, ts_tr_sec)
    oof_brier = _safe_brier(y_tr, cal_fn_oof(oof_raw))

    # Optional dedicated Aug 5-10 FrozenEstimator calibration branch when n_pos_tail >= 15 (full data)
    tail_cal_pair: tuple[Any, Any | None, Any] | None = None
    if n_pos_tail >= 15 and n_neg_tail >= 15:
        base_mask = (ts_tr_dt < pd.Timestamp("2025-08-05")).to_numpy()
        if int(np.sum(base_mask)) > 100 and int(np.sum(y_tr[base_mask] == 1)) >= 5:
            t_l, t_x = _fit_raw_models(X_tr_tree[base_mask], y_tr[base_mask], rec_w[base_mask], ts_tr_sec[base_mask])
            cal_l = _calibrate_on_tail(t_l, X_tr_tree[aug5_10_mask], y_tr[aug5_10_mask], method=cal_method)
            cal_x = (
                _calibrate_on_tail(t_x, X_tr_tree[aug5_10_mask], y_tr[aug5_10_mask], method=cal_method)
                if t_x is not None
                else None
            )
            tail_cal_pair = (cal_l, cal_x, aug5_10_mask)

    def _predict_fn(X_q: np.ndarray) -> np.ndarray:
        p_oof_branch = cal_fn_oof(_raw_blend_pred(full_l, full_x, X_q))
        if tail_cal_pair is None:
            return np.clip(p_oof_branch, 0.0, 1.0)
        cal_l, cal_x, _ = tail_cal_pair
        p_tail_l = _predict_pos_prob(cal_l, X_q)
        p_tail_branch = (0.65 * p_tail_l + 0.35 * _predict_pos_prob(cal_x, X_q)) if cal_x is not None else p_tail_l
        return np.clip(0.5 * p_oof_branch + 0.5 * p_tail_branch, 0.0, 1.0)

    return (_predict_fn, full_l), oof_brier


def calibrate_and_tune_cost_rule(
    raw_train: pd.DataFrame,
    X_base_tr: pd.DataFrame,
    cat_cols: list[str],
    num_cols: list[str],
    best_lambda: float = 0.015,
    smoke_test: bool = False,
    seed: int = SEED,
) -> dict[str, Any]:
    """R2e: Evaluate temporal calibration (sigmoid vs isotonic), reliability bins, and Rp 25.000 cost rule."""
    ts_tr = _to_ts(raw_train[TIME_COL])
    ts_tr_sec = _ts_to_epoch_sec(ts_tr).astype(np.float64)
    y_all = raw_train[TARGET_COL].to_numpy(dtype=np.int64)
    nom_all = raw_train["nominal"].to_numpy(dtype=np.float64)

    oot_folds = _build_oot_folds(ts_tr)
    oot_y_list, oot_raw_list, oot_sig_list, oot_iso_list, oot_nom_list, oot_fold_names = [], [], [], [], [], []
    fold_preds_detail: list[dict[str, Any]] = []

    for f in oot_folds:
        tr_idx, va_idx = f["train_idx"], f["val_idx"]
        X_tr_f, X_va_f = _make_fold_features(raw_train, X_base_tr, tr_idx, va_idx)
        y_tr, y_va = y_all[tr_idx], y_all[va_idx]
        X_tr_arr, [X_va_arr], _ = _prepare_tree_matrices(X_tr_f, y_tr, [X_va_f], cat_cols, num_cols, seed)

        # Split fold train into ~80% train_base + ~20% unweighted cal_tail (expand backward if < 3 positives)
        n_f = len(tr_idx)
        split_at = int(round(0.80 * n_f))
        # Ensure tr_idx is chronologically ordered
        f_order = np.argsort(ts_tr_sec[tr_idx], kind="mergesort")
        while split_at > int(0.45 * n_f) and int(np.sum(y_tr[f_order[split_at:]] == 1)) < 3:
            split_at -= max(1, int(0.05 * n_f))

        base_pos = f_order[:split_at]
        cal_pos = f_order[split_at:]

        t_base = ts_tr_sec[tr_idx][base_pos]
        max_tb = float(np.max(t_base))
        rec_w_base = np.exp(-float(best_lambda) * np.maximum(0.0, (max_tb - t_base) / 86400.0))

        clf_base = _make_lgbm(len(base_pos), smoke_test=smoke_test, seed=seed)
        clf_base.fit(X_tr_arr[base_pos], y_tr[base_pos], sample_weight=rec_w_base)

        p_raw = _predict_pos_prob(clf_base, X_va_arr)
        y_cal = y_tr[cal_pos]
        X_cal = X_tr_arr[cal_pos]

        cal_sig = _calibrate_on_tail(clf_base, X_cal, y_cal, method="sigmoid")
        cal_iso = _calibrate_on_tail(clf_base, X_cal, y_cal, method="isotonic")
        p_sig = _predict_pos_prob(cal_sig, X_va_arr)
        p_iso = _predict_pos_prob(cal_iso, X_va_arr)

        oot_y_list.append(y_va)
        oot_raw_list.append(p_raw)
        oot_sig_list.append(p_sig)
        oot_iso_list.append(p_iso)
        oot_nom_list.append(nom_all[va_idx])
        oot_fold_names.extend([f["name"]] * len(va_idx))
        fold_preds_detail.append(
            {
                "fold": f["name"],
                "val_idx": va_idx,
                "y_va": y_va,
                "p_sig": p_sig,
                "p_iso": p_iso,
                "nom_va": nom_all[va_idx],
            }
        )

    y_oot = np.concatenate(oot_y_list)
    p_raw_oot = np.concatenate(oot_raw_list)
    p_sig_oot = np.concatenate(oot_sig_list)
    p_iso_oot = np.concatenate(oot_iso_list)
    nom_oot = np.concatenate(oot_nom_list)

    brier_raw = _safe_brier(y_oot, p_raw_oot)
    brier_sig = _safe_brier(y_oot, p_sig_oot)
    brier_iso = _safe_brier(y_oot, p_iso_oot)
    ll_raw = _safe_log_loss(y_oot, p_raw_oot)
    ll_sig = _safe_log_loss(y_oot, p_sig_oot)
    ll_iso = _safe_log_loss(y_oot, p_iso_oot)

    n_pos_oot = int(np.sum(y_oot == 1))
    if n_pos_oot >= 30 and brier_iso < brier_sig:
        best_cal_method = "isotonic"
        p_cal_oot = p_iso_oot
        best_brier = brier_iso
        best_ll = ll_iso
    else:
        best_cal_method = "sigmoid"
        p_cal_oot = p_sig_oot
        best_brier = brier_sig
        best_ll = ll_sig

    # Reliability bins (10 quantile/uniform bins)
    bins = np.linspace(0.0, 1.0, 11)
    reliability_bins: list[dict[str, Any]] = []
    for b_idx in range(10):
        lo, hi = float(bins[b_idx]), float(bins[b_idx + 1])
        m = (p_cal_oot >= lo) & (p_cal_oot <= hi if b_idx == 9 else p_cal_oot < hi)
        cnt = int(np.sum(m))
        if cnt > 0:
            reliability_bins.append(
                {
                    "bin_low": lo,
                    "bin_high": hi,
                    "count": cnt,
                    "mean_pred": float(np.mean(p_cal_oot[m])),
                    "frac_pos": float(np.mean(y_oot[m])),
                }
            )

    # Tune tau on Jun+Jul OOT and validate out-of-sample on Aug_1_10
    fold_names_arr = np.array(oot_fold_names)
    jj_mask = np.isin(fold_names_arr, ["Jun", "Jul"])
    aug_mask = fold_names_arr == "Aug_1_10"

    tau_grid = (0.8, 1.0, 1.25) if smoke_test else (0.6, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0)
    tune_mask = jj_mask if np.any(jj_mask) else np.ones(len(y_oot), dtype=bool)
    best_tau = 1.0
    best_tune_cost = compute_cost_metrics(y_oot[tune_mask], p_cal_oot[tune_mask], nom_oot[tune_mask], tau=1.0)["total_cost_idr"]
    tau_sweep: list[dict[str, Any]] = []
    for tau in tau_grid:
        m_t = compute_cost_metrics(y_oot[tune_mask], p_cal_oot[tune_mask], nom_oot[tune_mask], tau=tau)
        tau_sweep.append({"tau": float(tau), "tune_cost_idr": m_t["total_cost_idr"], "n_flagged": m_t["n_flagged"]})
        if m_t["total_cost_idr"] < best_tune_cost:
            best_tune_cost = m_t["total_cost_idr"]
            best_tau = float(tau)

    # Out-of-sample check on Aug 1-10: adopt tau* only if it beats tau=1.0 on Aug 1-10
    if np.any(aug_mask) and best_tau != 1.0:
        c_tau1 = compute_cost_metrics(y_oot[aug_mask], p_cal_oot[aug_mask], nom_oot[aug_mask], tau=1.0)["total_cost_idr"]
        c_taub = compute_cost_metrics(y_oot[aug_mask], p_cal_oot[aug_mask], nom_oot[aug_mask], tau=best_tau)["total_cost_idr"]
        if c_taub >= c_tau1:
            best_tau = 1.0

    # Strategy cost comparison on pooled mature OOT
    m_tuned = compute_cost_metrics(y_oot, p_cal_oot, nom_oot, tau=best_tau)
    m_bayes = compute_cost_metrics(y_oot, p_cal_oot, nom_oot, tau=1.0)
    m_naive_05 = compute_cost_metrics(
        y_oot, p_cal_oot, nom_oot, tandai=(p_cal_oot >= 0.5).astype(np.int64)
    )
    m_mark_none = compute_cost_metrics(y_oot, p_cal_oot, nom_oot, tandai=np.zeros(len(y_oot), dtype=np.int64))
    m_mark_all = compute_cost_metrics(y_oot, p_cal_oot, nom_oot, tandai=np.ones(len(y_oot), dtype=np.int64))

    return {
        "calibration_method": best_cal_method,
        "uncalibrated_brier": brier_raw,
        "sigmoid_brier": brier_sig,
        "isotonic_brier": brier_iso,
        "calibrated_brier": best_brier,
        "brier": best_brier,
        "uncalibrated_log_loss": ll_raw,
        "calibrated_log_loss": best_ll,
        "reliability_bins": reliability_bins,
        "optimal_tau": best_tau,
        "tau_sweep": tau_sweep,
        "strategy_cost_comparison": {
            "calibrated_cost_rule": m_tuned,
            "bayes_tau_1_0": m_bayes,
            "naive_prob_0_5": m_naive_05,
            "mark_none": m_mark_none,
            "mark_all": m_mark_all,
        },
        "_oot_indices": np.concatenate([d["val_idx"] for d in fold_preds_detail]),
        "_oot_y": y_oot,
        "_oot_prob": p_cal_oot,
        "_oot_nom": nom_oot,
    }


# ---------------------------------------------------------------------------
# 9. R2f Error Breakdown & Interpretability
# ---------------------------------------------------------------------------


def compute_error_breakdown(
    raw_train: pd.DataFrame,
    oot_indices: np.ndarray,
    y_oot: np.ndarray,
    p_oot: np.ndarray,
    tau: float = 1.0,
    fitted_lgbm: lgb.LGBMClassifier | None = None,
    X_sample_arr: np.ndarray | None = None,
    feature_names: list[str] | None = None,
    smoke_test: bool = False,
) -> dict[str, Any]:
    """R2f: Zero-fraud-safe sliced PR-AUC & cost breakdown + SHAP/feature importance."""
    sub = raw_train.iloc[oot_indices].reset_index(drop=True)
    nom = sub["nominal"].to_numpy(dtype=np.float64)
    tandai = ((p_oot * nom >= float(tau) * float(REVIEW_COST_IDR)) & (nom >= float(REVIEW_COST_IDR))).astype(np.int64)

    overall_oot = compute_cost_metrics(y_oot, p_oot, nom, tandai=tandai, tau=tau)

    def _slice_dict(mask: np.ndarray) -> dict[str, Any]:
        return compute_cost_metrics(y_oot[mask], p_oot[mask], nom[mask], tandai=tandai[mask], tau=tau)

    by_kanal: dict[str, dict[str, Any]] = {}
    for k in sorted(sub["kanal"].astype(str).unique()):
        m = (sub["kanal"].astype(str) == k).to_numpy()
        by_kanal[k] = _slice_dict(m)

    by_kategori: dict[str, dict[str, Any]] = {}
    for k in sorted(sub["kategori_merchant"].astype(str).unique()):
        m = (sub["kategori_merchant"].astype(str) == k).to_numpy()
        by_kategori[k] = _slice_dict(m)

    # Fraud pattern slices
    patterns_masks = {
        "nominal_lt_25k": nom < float(REVIEW_COST_IDR),
        "nominal_25k_100k": (nom >= float(REVIEW_COST_IDR)) & (nom < 100_000.0),
        "nominal_100k_500k": (nom >= 100_000.0) & (nom < 500_000.0),
        "nominal_ge_500k": nom >= 500_000.0,
        "ato_signal": ((sub["perangkat_baru"] == 1) | (sub["ganti_password_7h"] == 1)).to_numpy(),
        "velocity_burst": (sub["trx_1j"] >= 1).to_numpy(),
        "card_not_present": (sub["kartu_tidak_hadir"] == 1).to_numpy(),
        "aug_qris_shift": (
            (sub["kanal"].astype(str) == "qris") & sub["kategori_merchant"].astype(str).isin(["jasa_lainnya", "toko_online"])
        ).to_numpy(),
        "cold_start_user": sub["menit_sejak_trx_terakhir"].isna().to_numpy(),
    }
    by_pattern = {pname: _slice_dict(pmask) for pname, pmask in patterns_masks.items() if np.any(pmask)}

    # Interpretability: exact TreeSHAP via LightGBM native pred_contrib (C++ exact TreeSHAP) + shap.TreeExplainer fallback
    top_importance: list[dict[str, Any]] = []
    if fitted_lgbm is not None and X_sample_arr is not None and feature_names is not None:
        max_shap = 60 if smoke_test else 400
        X_sub = X_sample_arr[: min(len(X_sample_arr), max_shap)]
        mean_abs_shap: np.ndarray | None = None
        try:
            if hasattr(fitted_lgbm, "booster_"):
                contrib = np.asarray(fitted_lgbm.booster_.predict(X_sub, pred_contrib=True), dtype=np.float64)
                if contrib.ndim == 2 and contrib.shape[1] >= len(feature_names):
                    mean_abs_shap = np.mean(np.abs(contrib[:, : len(feature_names)]), axis=0)
            else:
                import shap

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    explainer = shap.TreeExplainer(fitted_lgbm)
                    sv = explainer.shap_values(X_sub)
                    if isinstance(sv, list):
                        sv_arr = np.asarray(sv[-1], dtype=np.float64)
                    else:
                        sv_arr = np.asarray(sv, dtype=np.float64)
                        if sv_arr.ndim == 3:
                            sv_arr = sv_arr[:, :, -1]
                    mean_abs_shap = np.mean(np.abs(sv_arr), axis=0)
        except Exception:
            mean_abs_shap = None

        gain_imp = np.asarray(fitted_lgbm.feature_importances_, dtype=np.float64)
        score_vec = mean_abs_shap if mean_abs_shap is not None and len(mean_abs_shap) == len(feature_names) else gain_imp
        order = np.argsort(-score_vec)[:15]
        for idx in order:
            top_importance.append(
                {
                    "feature": str(feature_names[idx]),
                    "mean_abs_shap": float(mean_abs_shap[idx]) if mean_abs_shap is not None else None,
                    "gain_importance": float(gain_imp[idx]),
                }
            )

    return {
        "overall_oot": overall_oot,
        "breakdown_by_kanal": by_kanal,
        "breakdown_by_kategori_merchant": by_kategori,
        "breakdown_by_fraud_pattern": by_pattern,
        "top_feature_importance": top_importance,
    }


# ---------------------------------------------------------------------------
# 10. Main End-to-End Pipeline Entrypoint (R2g)
# ---------------------------------------------------------------------------


def run_klasifikasi_pipeline(
    data_dir: str | Path | None = None,
    output_csv: str | Path | None = None,
    artifacts_dir: str | Path | None = None,
    smoke_test: bool = False,
    sample_n: int | None = None,
    seed: int = SEED,
) -> dict[str, Any]:
    """Run complete leakage-free Task 2 classification pipeline and write prediksi_klasifikasi.csv."""
    t0 = time.perf_counter()
    set_global_seed(seed)
    logger = configure_logging(name="uts_ml.klasifikasi")
    smoke_cfg = get_smoke_config(smoke_test=smoke_test, sample_n=sample_n, seed=seed)
    lib_versions = get_library_versions()

    out_path = resolve_output_path(output_csv, DEFAULT_KLASIFIKASI_CSV.name)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Task 2 Klasifikasi start | smoke_test=%s | seed=%d | output=%s", smoke_cfg.enabled, seed, out_path)

    # Load data
    train_df, test_df, data_summary = load_klasifikasi_data(
        data_dir=data_dir,
        smoke_cfg=smoke_cfg,
        sample_n=sample_n,
        seed=seed,
        logger=logger,
    )

    # Audit R2a/R2b/R2d
    logger.info(
        "[Step 1/7] Loaded train=%d rows, test=%d rows. Running label & leakage audit (R2a)...",
        len(train_df),
        len(test_df),
    )
    audit_dict = audit_klasifikasi_data(train_df, test_df)

    # Causal Features R2c (strictly zero leakage; encoders fit strictly inside folds)
    logger.info("[Step 2/7] Building strictly causal user & merchant behavioral features (R2c)...")
    X_base_tr, X_base_te, feat_meta = build_causal_features(train_df, test_df, mature_cutoff=MATURE_CUTOFF)
    cat_cols = feat_meta["categorical_features"]
    num_cols = feat_meta["numeric_features"]

    # For multi-fold diagnostic comparisons on large full datasets, use a 40,000-row temporal subsample
    # while preserving 100% of mature training data for final ensemble training & test inference
    if (not smoke_cfg.enabled) and len(train_df) > 40_000:
        tmp_df = train_df.copy()
        tmp_df["_orig_row_id"] = np.arange(len(train_df), dtype=np.int64)
        sub_df = _subsample_train_temporal(tmp_df, 40_000, seed)
        abl_idx = sub_df["_orig_row_id"].to_numpy(dtype=np.int64)
        train_abl = train_df.iloc[abl_idx].reset_index(drop=True)
        X_base_abl = X_base_tr.iloc[abl_idx].reset_index(drop=True)
    else:
        train_abl = train_df
        X_base_abl = X_base_tr

    # R2a Leakage impact ablation
    logger.info("[Step 3/7] Measuring post-transaction leakage impact (flag_review_manual & v15) on %d rows...", len(train_abl))
    leakage_impact = evaluate_leakage_and_drift_impact(
        train_abl, X_base_abl, cat_cols, num_cols, smoke_test=smoke_cfg.enabled, seed=seed
    )

    # R2b Time-based validation vs Random CV & censoring policy
    logger.info("[Step 4/7] Comparing Time-Based OOT vs Random CV & label-delay censoring (R2b)...")
    val_r2b = compare_validation_schemes(
        train_abl, X_base_abl, cat_cols, num_cols, audit_dict, smoke_test=smoke_cfg.enabled, seed=seed
    )

    # R2d Imbalance strategies & QRIS/jasa_lainnya shift adaptation
    logger.info("[Step 5/7] Comparing 6 class-imbalance strategies & QRIS/jasa_lainnya shift adaptation (R2d)...")
    imb_r2d = compare_imbalance_and_adaptation(
        train_abl, X_base_abl, cat_cols, num_cols, audit_dict, smoke_test=smoke_cfg.enabled, seed=seed
    )

    # R2e Calibration & Rp 25.000 cost rule
    logger.info("[Step 6/7] Calibrating probabilities & optimizing Rp 25.000 cost decision rule (R2e)...")
    cal_r2e = calibrate_and_tune_cost_rule(
        train_abl,
        X_base_abl,
        cat_cols,
        num_cols,
        best_lambda=imb_r2d["selected_lambda"],
        smoke_test=smoke_cfg.enabled,
        seed=seed,
    )

    # Final training on mature (+ optionally maturity-weighted Aug 11-17) rows & Test inference
    censor_policy = val_r2b["censoring_policies"]["adopted_policy"]
    aug11_17_mat = val_r2b["censoring_policies"]["aug_11_17_maturity"]
    mature_mask, censor_w_tr = _get_mature_train_mask_and_weights(train_df, censor_policy, aug11_17_mat)

    train_mature = train_df.loc[mature_mask].reset_index(drop=True)
    X_tr_mature = X_base_tr.loc[mature_mask].reset_index(drop=True)
    y_tr_mature = train_mature[TARGET_COL].to_numpy(dtype=np.int64)
    ts_tr_mature_dt = _to_ts(train_mature[TIME_COL])
    ts_tr_mature_sec = _ts_to_epoch_sec(ts_tr_mature_dt).astype(np.float64)

    logger.info(
        "[Step 7/7] Fitting final calibrated ensemble on 100%% mature training data (%d rows) & writing CSV...",
        len(train_mature),
    )

    # Recompute lag21d strictly from mature training rows (consistent for train_mature and test_df)
    X_te_final = X_base_te.copy()
    lag_tr_fin = compute_lagged_entity_risk(train_mature, train_mature, mature_cutoff=MATURE_CUTOFF)
    lag_te_fin = compute_lagged_entity_risk(train_mature, test_df, mature_cutoff=MATURE_CUTOFF)
    for col in LAGGED_TARGET_COLS:
        X_tr_mature[col] = lag_tr_fin[col].to_numpy()
        X_te_final[col] = lag_te_fin[col].to_numpy()

    X_tr_fin_arr, [X_te_fin_arr], fin_feat_names = _prepare_tree_matrices(
        X_tr_mature, y_tr_mature, [X_te_final], cat_cols, num_cols, seed
    )

    (predict_fn, fitted_lgbm), final_oof_brier = _fit_calibrated_final_ensemble(
        X_tr_fin_arr,
        y_tr_mature,
        ts_tr_mature_sec,
        censor_w_tr,
        ts_tr_mature_dt,
        best_lambda=imb_r2d["selected_lambda"],
        cal_method=cal_r2e["calibration_method"],
        smoke_test=smoke_cfg.enabled,
        seed=seed,
    )

    test_prob = np.clip(predict_fn(X_te_fin_arr), 0.0, 1.0)
    test_nom = test_df["nominal"].to_numpy(dtype=np.float64)
    opt_tau = float(cal_r2e["optimal_tau"])
    test_tandai = (
        (test_prob * test_nom >= opt_tau * float(REVIEW_COST_IDR)) & (test_nom >= float(REVIEW_COST_IDR))
    ).astype(np.int64)

    pred_df = pd.DataFrame(
        {
            "id_transaksi": test_df[ID_COL].astype(str),
            "prob": np.round(test_prob, 8),
            "tandai": test_tandai,
        }
    )
    pred_df.to_csv(out_path, index=False)

    # R2f Error breakdown & SHAP
    eval_r2f = compute_error_breakdown(
        train_abl,
        cal_r2e["_oot_indices"],
        cal_r2e["_oot_y"],
        cal_r2e["_oot_prob"],
        tau=opt_tau,
        fitted_lgbm=fitted_lgbm,
        X_sample_arr=X_tr_fin_arr,
        feature_names=fin_feat_names,
        smoke_test=smoke_cfg.enabled,
    )

    # Strip private numpy arrays from calibration dict before returning/saving JSON
    cal_public = {k: v for k, v in cal_r2e.items() if not k.startswith("_")}
    cal_public["final_oof_calibrated_brier"] = final_oof_brier

    elapsed_sec = float(time.perf_counter() - t0)
    result: dict[str, Any] = {
        "task": "klasifikasi",
        "seed": int(seed),
        "smoke_test": bool(smoke_cfg.enabled),
        "review_cost_idr": int(REVIEW_COST_IDR),
        "timing_sec": round(elapsed_sec, 3),
        "library_versions": lib_versions,
        "data_summary": data_summary,
        "audit_r2a": {
            "flag_review_manual": audit_dict["flag_review_manual"],
            "v15": audit_dict["v15"],
            "v26": audit_dict["v26"],
            "umur_merchant_hari": audit_dict["umur_merchant_hari"],
            "cold_start_nulls": audit_dict["cold_start_nulls"],
            "persen_saldo_anomalies": audit_dict["persen_saldo_anomalies"],
            "leakage_impact_comparison": leakage_impact,
        },
        "validation_r2b": {
            "label_delay_censoring": audit_dict["label_delay_censoring"],
            "aug_29_31_fraud_count": audit_dict["label_delay_censoring"]["aug_29_31_fraud_count"],
            **val_r2b,
        },
        "features_r2c": feat_meta,
        "imbalance_and_shift_r2d": imb_r2d,
        "calibration_and_cost_r2e": cal_public,
        "evaluation_r2f": eval_r2f,
        "output": {
            "output_csv": str(out_path),
            "schema": ["id_transaksi", "prob", "tandai"],
            "n_test_rows": int(len(pred_df)),
            "n_flagged": int(np.sum(test_tandai == 1)),
            "pct_flagged": float(np.mean(test_tandai == 1)) if len(pred_df) > 0 else 0.0,
            "mean_prob": float(np.mean(pred_df["prob"])) if len(pred_df) > 0 else 0.0,
            "max_prob": float(np.max(pred_df["prob"])) if len(pred_df) > 0 else 0.0,
        },
    }

    if artifacts_dir is not None:
        art_path = Path(artifacts_dir).expanduser().resolve() / "klasifikasi_summary.json"
        save_json(result, art_path)
        result["output"]["summary_json"] = str(art_path)

    logger.info(
        "Task 2 Klasifikasi done in %.2fs | OOT PR-AUC=%.4f | Test flagged=%d/%d (%.2f%%)",
        elapsed_sec,
        eval_r2f["overall_oot"]["pr_auc"] or 0.0,
        result["output"]["n_flagged"],
        result["output"]["n_test_rows"],
        100.0 * result["output"]["pct_flagged"],
    )
    return to_json_compatible(result)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UTS ML Task 2 Classification Pipeline")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-csv", type=str, default=None)
    parser.add_argument("--artifacts-dir", type=str, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--sample-n", type=int, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


if __name__ == "__main__":
    reconfigure_stdio_utf8()
    args = _parse_args()
    res = run_klasifikasi_pipeline(
        data_dir=args.data_dir,
        output_csv=args.output_csv,
        artifacts_dir=args.artifacts_dir,
        smoke_test=args.smoke_test,
        sample_n=args.sample_n,
        seed=args.seed,
    )
    print(
        f"klasifikasi_done timing_sec={res['timing_sec']} "
        f"oot_pr_auc={res['evaluation_r2f']['overall_oot']['pr_auc']} "
        f"output_csv={res['output']['output_csv']}"
    )
