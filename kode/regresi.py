"""Task 1 Regresi: Estimasi Waktu Pengiriman (p50 & p90) + Kalibrasi Conformal.

Modul ini mengimplementasikan pipeline regresi kuantil tanpa kebocoran data
untuk memprediksi ``durasi_jam`` pada paket uji September-Oktober 2025 dengan
hanya menggunakan fitur yang tersedia saat pengambilan paket (pickup).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import time
from typing import Any
from pathlib import Path

from lightgbm import LGBMRegressor
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import PartialDependenceDisplay, partial_dependence
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_pinball_loss, root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, TargetEncoder
from xgboost import XGBRegressor

from kode.config import (
    DEFAULT_REGRESI_CSV,
    SEED,
    SmokeTestConfig,
    configure_logging,
    ensure_dir,
    format_library_versions,
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

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LOGGER = configure_logging(name="uts_ml.regresi")

ID_COL: str = "id_paket"
DATE_COL: str = "tgl_pickup"
TARGET_COL: str = "durasi_jam"
FORBIDDEN_LEAKAGE_COLS: tuple[str, ...] = ("jumlah_scan",)

DURASI_UNIT_THRESHOLD_HOURS: float = 1000.0
JARAK_MAX_SENTINEL: float = 9999.0

# Low/medium-cardinality categoricals (<= 30 levels <= 255 HGBR bin limit).
CAT_COLS: tuple[str, ...] = (
    "layanan",
    "hub_asal",
    "pulau_tujuan",
    "kota_asal",
    "mitra_kurir",
)
# Target-encoded & frequency-encoded columns (includes 405-level kota_tujuan).
TE_COLS: tuple[str, ...] = ("kota_tujuan", "kota_asal", "mitra_kurir")


@dataclass
class PreprocessedViews:
    """Feature representations produced by ``RegresiFeaturePreprocessor``."""

    linear: np.ndarray
    hgbr_df: pd.DataFrame
    native_df: pd.DataFrame
    hgbr_cat_indices: list[int]
    numeric_cols: list[str]
    all_feature_cols_native: list[str]
    all_feature_cols_hgbr: list[str]


class _LogExpPredictor(RegressorMixin, BaseEstimator):
    """Wrap a log1p-space model so PDP evaluates directly in durasi_jam hours."""

    def __init__(self, estimator: Any, mode: str = "single", col_idx: int = 0) -> None:
        self.estimator = estimator
        self.mode = mode
        self.col_idx = col_idx
        self.is_fitted_ = True

    def fit(self, X: Any, y: Any = None) -> "_LogExpPredictor":
        return self

    def predict(self, X: Any) -> np.ndarray:
        raw = np.asarray(self.estimator.predict(X), dtype=np.float64)
        if raw.ndim == 2:
            raw = raw[:, self.col_idx]
        return np.clip(np.expm1(raw), 1e-3, None)


class RouteJarakImputer:
    """Train-only hierarchical route median imputer for ``jarak_km`` sentinels."""

    def __init__(self) -> None:
        self.map_l1_: dict[tuple[str, str], float] = {}
        self.map_l2_: dict[tuple[str, str], float] = {}
        self.map_l3_: dict[tuple[str, str], float] = {}
        self.map_l4_: dict[str, float] = {}
        self.global_median_: float = 500.0

    def fit(self, df: pd.DataFrame) -> "RouteJarakImputer":
        jarak = pd.to_numeric(df["jarak_km"], errors="coerce")
        valid_mask = jarak.notna() & (jarak > 0.0) & (jarak < JARAK_MAX_SENTINEL)
        valid_df = df.loc[valid_mask, ["kota_asal", "kota_tujuan", "hub_asal", "pulau_tujuan"]].copy()
        valid_df["_jarak"] = jarak.loc[valid_mask].to_numpy(dtype=np.float64)

        if len(valid_df) == 0:
            self.global_median_ = 500.0
            self.map_l1_ = {}
            self.map_l2_ = {}
            self.map_l3_ = {}
            self.map_l4_ = {}
            return self

        for col in ("kota_asal", "kota_tujuan", "hub_asal", "pulau_tujuan"):
            valid_df[col] = valid_df[col].fillna("__MISSING__").astype(str)

        self.global_median_ = float(valid_df["_jarak"].median())
        self.map_l1_ = {
            (str(k[0]), str(k[1])): float(v)
            for k, v in valid_df.groupby(["kota_asal", "kota_tujuan"])["_jarak"].median().items()
        }
        self.map_l2_ = {
            (str(k[0]), str(k[1])): float(v)
            for k, v in valid_df.groupby(["hub_asal", "kota_tujuan"])["_jarak"].median().items()
        }
        self.map_l3_ = {
            (str(k[0]), str(k[1])): float(v)
            for k, v in valid_df.groupby(["kota_asal", "pulau_tujuan"])["_jarak"].median().items()
        }
        self.map_l4_ = {
            str(k): float(v)
            for k, v in valid_df.groupby("pulau_tujuan")["_jarak"].median().items()
        }
        return self

    def transform_with_stats(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
        jarak_raw = pd.to_numeric(df["jarak_km"], errors="coerce").to_numpy(dtype=np.float64)
        bad_mask = np.isnan(jarak_raw) | (jarak_raw <= 0.0) | (jarak_raw >= JARAK_MAX_SENTINEL)
        anomalous_flag = bad_mask.astype(np.int8)
        out = jarak_raw.copy()

        stats = {
            "total_anomalous": int(bad_mask.sum()),
            "level_1_kota_pair": 0,
            "level_2_hub_kota": 0,
            "level_3_kota_pulau": 0,
            "level_4_pulau": 0,
            "level_5_global": 0,
        }
        if not np.any(bad_mask):
            return out, anomalous_flag, stats

        sub = df.loc[bad_mask, ["kota_asal", "kota_tujuan", "hub_asal", "pulau_tujuan"]].fillna(
            "__MISSING__"
        )
        ka = sub["kota_asal"].astype(str).tolist()
        kt = sub["kota_tujuan"].astype(str).tolist()
        hb = sub["hub_asal"].astype(str).tolist()
        pl = sub["pulau_tujuan"].astype(str).tolist()

        filled = np.empty(len(ka), dtype=np.float64)
        for i in range(len(ka)):
            pair_1 = (ka[i], kt[i])
            if pair_1 in self.map_l1_:
                filled[i] = self.map_l1_[pair_1]
                stats["level_1_kota_pair"] += 1
                continue
            pair_2 = (hb[i], kt[i])
            if pair_2 in self.map_l2_:
                filled[i] = self.map_l2_[pair_2]
                stats["level_2_hub_kota"] += 1
                continue
            pair_3 = (ka[i], pl[i])
            if pair_3 in self.map_l3_:
                filled[i] = self.map_l3_[pair_3]
                stats["level_3_kota_pulau"] += 1
                continue
            if pl[i] in self.map_l4_:
                filled[i] = self.map_l4_[pl[i]]
                stats["level_4_pulau"] += 1
                continue
            filled[i] = self.global_median_
            stats["level_5_global"] += 1

        out[bad_mask] = filled
        return out, anomalous_flag, stats


class BeratMNARImputer:
    """Train-only MNAR imputer for ``berat_kg`` conditioned on ``layanan`` and ``volume_liter``."""

    def __init__(self) -> None:
        self.vol_edges_: np.ndarray | None = None
        self.map_layanan_bin_: dict[tuple[str, int], float] = {}
        self.map_layanan_: dict[str, float] = {}
        self.global_median_: float = 5.0

    def _assign_bins(self, vol: np.ndarray) -> np.ndarray:
        if self.vol_edges_ is None or len(self.vol_edges_) < 3:
            return np.zeros(len(vol), dtype=np.int64)
        safe_vol = np.where(np.isfinite(vol), vol, self.vol_edges_[0])
        return np.searchsorted(self.vol_edges_[1:-1], safe_vol, side="right")

    def fit(self, df: pd.DataFrame) -> "BeratMNARImputer":
        berat = pd.to_numeric(df["berat_kg"], errors="coerce")
        vol = pd.to_numeric(df["volume_liter"], errors="coerce").to_numpy(dtype=np.float64)
        valid_vol = vol[np.isfinite(vol)]
        if len(valid_vol) >= 8:
            edges = np.unique(np.quantile(valid_vol, [0.0, 0.25, 0.50, 0.75, 1.0]))
            self.vol_edges_ = edges if len(edges) >= 3 else None
        else:
            self.vol_edges_ = None

        valid_mask = berat.notna() & np.isfinite(berat)
        if not bool(valid_mask.any()):
            self.global_median_ = 5.0
            self.map_layanan_ = {}
            self.map_layanan_bin_ = {}
            return self

        sub = pd.DataFrame(
            {
                "layanan": df.loc[valid_mask, "layanan"].fillna("__MISSING__").astype(str),
                "vol_bin": self._assign_bins(vol[valid_mask.to_numpy()]),
                "berat": berat.loc[valid_mask].to_numpy(dtype=np.float64),
            }
        )
        self.global_median_ = float(sub["berat"].median())
        self.map_layanan_ = {
            str(k): float(v) for k, v in sub.groupby("layanan")["berat"].median().items()
        }
        self.map_layanan_bin_ = {
            (str(k[0]), int(k[1])): float(v)
            for k, v in sub.groupby(["layanan", "vol_bin"])["berat"].median().items()
        }
        return self

    def transform_series(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        berat = pd.to_numeric(df["berat_kg"], errors="coerce").to_numpy(dtype=np.float64)
        missing_mask = ~np.isfinite(berat)
        missing_flag = missing_mask.astype(np.int8)
        out = berat.copy()
        if not np.any(missing_mask):
            return out, missing_flag

        vol = pd.to_numeric(df.loc[missing_mask, "volume_liter"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        bins = self._assign_bins(vol)
        layanan_list = df.loc[missing_mask, "layanan"].fillna("__MISSING__").astype(str).tolist()
        filled = np.empty(len(layanan_list), dtype=np.float64)
        for i, (lay, b) in enumerate(zip(layanan_list, bins)):
            key = (lay, int(b))
            if key in self.map_layanan_bin_:
                filled[i] = self.map_layanan_bin_[key]
            elif lay in self.map_layanan_:
                filled[i] = self.map_layanan_[lay]
            else:
                filled[i] = self.global_median_
        out[missing_mask] = filled
        return out, missing_flag


class RegresiFeaturePreprocessor:
    """Zero-leakage fold preprocessor for pickup-time features & high-card encoding."""

    def __init__(self, seed: int = SEED) -> None:
        self.seed = int(seed)
        self.jarak_imputer = RouteJarakImputer()
        self.berat_imputer = BeratMNARImputer()
        self.freq_maps_: dict[str, dict[str, float]] = {}
        self.te_encoder_: TargetEncoder | None = None
        self.num_imputer_ = SimpleImputer(strategy="median")
        self.scaler_ = StandardScaler()
        self.ohe_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.ord_ = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-2,
        )
        self.train_cats_: dict[str, list[str]] = {}
        self.numeric_cols_: list[str] = []
        self.cat_cols_: list[str] = list(CAT_COLS)

    @staticmethod
    def _build_pickup_base_frame(df: pd.DataFrame) -> pd.DataFrame:
        out = df.drop(
            columns=[ID_COL, TARGET_COL, *FORBIDDEN_LEAKAGE_COLS],
            errors="ignore",
        ).copy()

        dt = pd.to_datetime(out.get(DATE_COL), errors="coerce")
        dow = dt.dt.day_of_week.fillna(0).to_numpy(dtype=np.float64)
        day = dt.dt.day.fillna(15).to_numpy(dtype=np.float64)
        doy = dt.dt.day_of_year.fillna(180).to_numpy(dtype=np.float64)
        hour = pd.to_numeric(out.get("jam_pickup"), errors="coerce").fillna(12.0).to_numpy(
            dtype=np.float64
        )

        out["pickup_dayofweek"] = dow
        out["pickup_is_weekend"] = (dow >= 5.0).astype(np.float64)
        out["pickup_day"] = day
        out["pickup_doy_sin"] = np.sin(2.0 * np.pi * doy / 365.25)
        out["pickup_doy_cos"] = np.cos(2.0 * np.pi * doy / 365.25)
        out["pickup_hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
        out["pickup_hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
        out["pickup_is_peak_hour"] = ((hour >= 9.0) & (hour <= 17.0)).astype(np.float64)

        out = out.drop(columns=[DATE_COL], errors="ignore")
        return out

    def _apply_imputers_and_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        jarak_clean, jarak_flag, _ = self.jarak_imputer.transform_with_stats(df)
        berat_clean, berat_flag = self.berat_imputer.transform_series(df)

        df = df.copy()
        df["jarak_km"] = jarak_clean
        df["jarak_km_is_anomalous"] = jarak_flag.astype(np.float64)
        df["berat_kg"] = berat_clean
        df["berat_kg_is_missing"] = berat_flag.astype(np.float64)

        vol = np.maximum(pd.to_numeric(df["volume_liter"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float64), 0.1)
        items = np.maximum(pd.to_numeric(df["jumlah_item"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float64), 1.0)
        nilai = np.maximum(pd.to_numeric(df["nilai_barang"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64), 0.0)
        h_asal = np.maximum(pd.to_numeric(df["hujan_asal_mm"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64), 0.0)
        h_tuj = np.maximum(pd.to_numeric(df["hujan_tujuan_mm"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64), 0.0)
        antrean = np.maximum(pd.to_numeric(df["antrean_hub_asal"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64), 0.0)
        kemacetan = np.maximum(pd.to_numeric(df["indeks_kemacetan"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64), 0.0)
        pengalaman = np.maximum(pd.to_numeric(df["pengalaman_kurir_bulan"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float64), 1.0)

        df["density_kg_per_liter"] = berat_clean / vol
        df["berat_per_item"] = berat_clean / items
        df["log_jarak_km"] = np.log1p(np.maximum(jarak_clean, 0.0))
        df["log_berat_kg"] = np.log1p(np.maximum(berat_clean, 0.0))
        df["log_volume_liter"] = np.log1p(vol)
        df["log_nilai_barang"] = np.log1p(nilai)
        df["total_hujan_mm"] = h_asal + h_tuj
        df["max_hujan_mm"] = np.maximum(h_asal, h_tuj)
        df["jarak_x_hujan"] = jarak_clean * (h_asal + h_tuj)
        df["antrean_x_kemacetan"] = antrean * kemacetan
        df["log_antrean_hub"] = np.log1p(antrean)
        df["jarak_per_pengalaman"] = jarak_clean / pengalaman
        return df

    def _add_freq_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in TE_COLS:
            s = df[col].fillna("__MISSING__").astype(str)
            df[f"{col}_freq"] = s.map(self.freq_maps_[col]).fillna(0.0).to_numpy(dtype=np.float64)
        route_s = (
            df["kota_asal"].fillna("__MISSING__").astype(str)
            + "__"
            + df["kota_tujuan"].fillna("__MISSING__").astype(str)
        )
        df["route_freq"] = route_s.map(self.freq_maps_["route"]).fillna(0.0).to_numpy(dtype=np.float64)
        return df

    def _build_views(self, df_num_ready: pd.DataFrame, df_cats_raw: pd.DataFrame) -> PreprocessedViews:
        X_num = self.num_imputer_.transform(df_num_ready[self.numeric_cols_])
        X_num_scaled = self.scaler_.transform(X_num)

        cat_strs = df_cats_raw[self.cat_cols_].fillna("__MISSING__").astype(str)
        X_ohe = self.ohe_.transform(cat_strs)
        X_linear = np.hstack([X_num_scaled, X_ohe])

        X_ord = self.ord_.transform(cat_strs)
        hgbr_df = pd.DataFrame(
            np.hstack([X_num, X_ord]),
            columns=[*self.numeric_cols_, *self.cat_cols_],
            index=df_num_ready.index,
        )
        hgbr_cat_indices = list(range(len(self.numeric_cols_), len(self.numeric_cols_) + len(self.cat_cols_)))

        native_df = pd.DataFrame(X_num, columns=self.numeric_cols_, index=df_num_ready.index)
        for col in self.cat_cols_:
            cats = self.train_cats_[col]
            s = cat_strs[col]
            # Mask unseen values to NaN before Categorical construction to avoid Pandas4Warning / XGBoostError
            native_df[col] = pd.Categorical(s.where(s.isin(cats)), categories=cats)

        return PreprocessedViews(
            linear=X_linear,
            hgbr_df=hgbr_df,
            native_df=native_df,
            hgbr_cat_indices=hgbr_cat_indices,
            numeric_cols=list(self.numeric_cols_),
            all_feature_cols_native=[*self.numeric_cols_, *self.cat_cols_],
            all_feature_cols_hgbr=[*self.numeric_cols_, *self.cat_cols_],
        )

    def fit_transform(self, train_df: pd.DataFrame, y_raw: np.ndarray) -> PreprocessedViews:
        base = self._build_pickup_base_frame(train_df)
        self.jarak_imputer.fit(base)
        self.berat_imputer.fit(base)
        fe_df = self._apply_imputers_and_domain_features(base)

        n_rows = max(1, len(fe_df))
        self.freq_maps_ = {}
        for col in TE_COLS:
            counts = fe_df[col].fillna("__MISSING__").astype(str).value_counts(dropna=False)
            self.freq_maps_[col] = {str(k): float(v / n_rows) for k, v in counts.items()}
        route_s = (
            fe_df["kota_asal"].fillna("__MISSING__").astype(str)
            + "__"
            + fe_df["kota_tujuan"].fillna("__MISSING__").astype(str)
        )
        r_counts = route_s.value_counts(dropna=False)
        self.freq_maps_["route"] = {str(k): float(v / n_rows) for k, v in r_counts.items()}
        fe_df = self._add_freq_features(fe_df)

        y_log = np.log1p(np.maximum(np.asarray(y_raw, dtype=np.float64), 0.0))
        te_input = fe_df[list(TE_COLS)].fillna("__MISSING__").astype(str)
        n_splits = min(5, max(2, len(fe_df) // 20))
        n_splits = min(n_splits, max(2, len(fe_df)))
        self.te_encoder_ = TargetEncoder(
            target_type="continuous",
            smooth="auto",
            cv=KFold(n_splits=n_splits, shuffle=True, random_state=self.seed),
        )
        te_vals = self.te_encoder_.fit_transform(te_input, y_log)
        for idx, col in enumerate(TE_COLS):
            fe_df[f"{col}_te"] = te_vals[:, idx]

        # Keep kota_tujuan numeric-only (_te + _freq) so HGBR never exceeds max_bins=255.
        drop_cols = set(self.cat_cols_) | {"kota_tujuan"}
        self.numeric_cols_ = [
            c
            for c in fe_df.columns
            if c not in drop_cols and pd.api.types.is_numeric_dtype(fe_df[c])
        ]

        X_num = self.num_imputer_.fit_transform(fe_df[self.numeric_cols_])
        self.scaler_.fit(X_num)

        cat_strs = fe_df[self.cat_cols_].fillna("__MISSING__").astype(str)
        self.ohe_.fit(cat_strs)
        self.ord_.fit(cat_strs)
        self.train_cats_ = {
            col: sorted(cat_strs[col].unique().tolist()) for col in self.cat_cols_
        }
        return self._build_views(fe_df, fe_df)

    def transform(self, df: pd.DataFrame) -> PreprocessedViews:
        if self.te_encoder_ is None:
            raise RuntimeError("RegresiFeaturePreprocessor must be fitted before transform().")
        base = self._build_pickup_base_frame(df)
        fe_df = self._apply_imputers_and_domain_features(base)
        fe_df = self._add_freq_features(fe_df)

        te_input = fe_df[list(TE_COLS)].fillna("__MISSING__").astype(str)
        te_vals = self.te_encoder_.transform(te_input)
        for idx, col in enumerate(TE_COLS):
            fe_df[f"{col}_te"] = te_vals[:, idx]

        return self._build_views(fe_df, fe_df)


def _stratified_smoke_sample(
    df: pd.DataFrame,
    target_n: int,
    seed: int,
    preserve_masks: list[np.ndarray] | None = None,
    max_preserve_per_mask: int = 10,
) -> pd.DataFrame:
    if len(df) <= target_n:
        return df.reset_index(drop=True)

    rng = np.random.default_rng(seed)
    must_idx: list[int] = []
    seen_pos: set[int] = set()
    if preserve_masks:
        for mask in preserve_masks:
            pos_list = np.flatnonzero(mask)
            if len(pos_list) == 0:
                continue
            take = min(max_preserve_per_mask, len(pos_list))
            chosen = rng.choice(pos_list, size=take, replace=False) if len(pos_list) > take else pos_list
            for p in chosen:
                pi = int(p)
                if pi not in seen_pos:
                    seen_pos.add(pi)
                    must_idx.append(pi)

    if len(must_idx) >= target_n:
        sel = sorted(must_idx[:target_n])
        return df.iloc[sel].reset_index(drop=True)

    months = df[DATE_COL].astype(str).str.slice(0, 7).to_numpy()
    remaining_budget = target_n - len(must_idx)
    avail_mask = np.ones(len(df), dtype=bool)
    avail_mask[must_idx] = False
    avail_positions = np.flatnonzero(avail_mask)
    avail_months = months[avail_positions]

    unique_months, counts = np.unique(avail_months, return_counts=True)
    # Allocate at least 1 sample per month when possible, proportional thereafter
    alloc = np.floor(counts / counts.sum() * remaining_budget).astype(int)
    alloc = np.maximum(alloc, np.minimum(counts, 1))
    while alloc.sum() > remaining_budget:
        idx_max = int(np.argmax(alloc))
        if alloc[idx_max] > 1:
            alloc[idx_max] -= 1
        else:
            break
    while alloc.sum() < remaining_budget:
        deficits = counts - alloc
        valid = np.flatnonzero(deficits > 0)
        if len(valid) == 0:
            break
        best = int(valid[np.argmax(deficits[valid])])
        alloc[best] += 1

    sampled_idx = list(must_idx)
    for m_val, n_take in zip(unique_months, alloc):
        if n_take <= 0:
            continue
        m_pos = avail_positions[avail_months == m_val]
        if len(m_pos) <= n_take:
            sampled_idx.extend(m_pos.tolist())
        else:
            sampled_idx.extend(rng.choice(m_pos, size=int(n_take), replace=False).tolist())

    sampled_idx = sorted(set(sampled_idx))[:target_n]
    return df.iloc[sampled_idx].reset_index(drop=True)


def load_regresi_datasets(
    data_dir: str | Path | None = None,
    smoke_cfg: SmokeTestConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load ``regresi_train.csv.gz`` and ``regresi_test.csv.gz`` with optional smoke subsampling."""
    resolved_dir = get_data_dir(data_dir, must_exist=True)
    tr_nrows = None
    te_nrows = None
    if smoke_cfg is not None and smoke_cfg.enabled:
        # Real CSVs are pre-shuffled; bounded nrows avoids 250k decompression in smoke runs
        # while still reading 100% of synthetic micro-datasets (<= 5,000 rows).
        tr_nrows = max(int(smoke_cfg.sample_n) * 4, 10_000)
        te_nrows = max(int(smoke_cfg.test_sample_n) * 4, 5_000)

    train_df = load_csv_gz("regresi_train.csv.gz", data_dir=resolved_dir, nrows=tr_nrows)
    test_df = load_csv_gz("regresi_test.csv.gz", data_dir=resolved_dir, nrows=te_nrows)

    if smoke_cfg is not None and smoke_cfg.enabled:
        dur = pd.to_numeric(train_df[TARGET_COL], errors="coerce").to_numpy(dtype=np.float64)
        jrk_tr = pd.to_numeric(train_df["jarak_km"], errors="coerce").to_numpy(dtype=np.float64)
        train_df = _stratified_smoke_sample(
            train_df,
            target_n=smoke_cfg.sample_n,
            seed=smoke_cfg.seed,
            preserve_masks=[
                dur > DURASI_UNIT_THRESHOLD_HOURS,
                np.isnan(jrk_tr) | (jrk_tr <= 0.0) | (jrk_tr >= JARAK_MAX_SENTINEL),
            ],
            max_preserve_per_mask=10,
        )

        if len(test_df) > smoke_cfg.test_sample_n:
            jrk_te = pd.to_numeric(test_df["jarak_km"], errors="coerce").to_numpy(dtype=np.float64)
            k13_mask = test_df["mitra_kurir"].astype(str).to_numpy() == "K13"
            test_df = _stratified_smoke_sample(
                test_df,
                target_n=smoke_cfg.test_sample_n,
                seed=smoke_cfg.seed + 1,
                preserve_masks=[
                    k13_mask,
                    np.isnan(jrk_te) | (jrk_te <= 0.0) | (jrk_te >= JARAK_MAX_SENTINEL),
                ],
                max_preserve_per_mask=15,
            )

    return train_df, test_df


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return 0.0
    xv = x[mask]
    yv = y[mask]
    if float(np.std(xv)) < 1e-12 or float(np.std(yv)) < 1e-12:
        return 0.0
    return float(np.corrcoef(xv, yv)[0, 1])


def _measure_audit_impacts(
    train_df: pd.DataFrame,
    seed: int,
    smoke_cfg: SmokeTestConfig | None,
) -> dict[str, Any]:
    """Measure OOT impact of jumlah_scan leakage and durasi_jam > 1000 unit errors."""
    max_probe_rows = 2_000 if (smoke_cfg is not None and smoke_cfg.enabled) else 25_000
    if len(train_df) > max_probe_rows:
        probe_df = _stratified_smoke_sample(train_df, target_n=max_probe_rows, seed=seed + 77)
    else:
        probe_df = train_df.copy()

    probe_df = probe_df.sort_values(DATE_COL).reset_index(drop=True)
    split_idx = max(10, int(len(probe_df) * 0.75))
    tr_df = probe_df.iloc[:split_idx].reset_index(drop=True)
    va_df = probe_df.iloc[split_idx:].reset_index(drop=True)
    if len(va_df) < 5:
        return {"probe_skipped": True}

    y_tr_raw = pd.to_numeric(tr_df[TARGET_COL], errors="coerce").to_numpy(dtype=np.float64)
    y_va_raw = pd.to_numeric(va_df[TARGET_COL], errors="coerce").to_numpy(dtype=np.float64)
    y_tr_clean = np.where(y_tr_raw > DURASI_UNIT_THRESHOLD_HOURS, y_tr_raw / 60.0, y_tr_raw)
    y_va_clean = np.where(y_va_raw > DURASI_UNIT_THRESHOLD_HOURS, y_va_raw / 60.0, y_va_raw)

    # Clean preprocessor (drops jumlah_scan)
    prep = RegresiFeaturePreprocessor(seed=seed)
    v_tr = prep.fit_transform(tr_df, y_tr_clean)
    v_va = prep.transform(va_df)

    # 1. Clean baseline model
    is_smoke = bool(smoke_cfg is not None and smoke_cfg.enabled)
    probe_n_est = 15 if is_smoke else 30
    m_clean = LGBMRegressor(
        objective="quantile",
        alpha=0.5,
        n_estimators=probe_n_est,
        num_leaves=15,
        n_jobs=1,
        random_state=seed,
        verbose=-1,
    )
    m_clean.fit(v_tr.native_df, np.log1p(np.maximum(y_tr_clean, 0.0)))
    pred_clean = np.clip(np.expm1(m_clean.predict(v_va.native_df)), 1e-3, None)

    # 2. Leaky model WITH jumlah_scan
    X_tr_leak = v_tr.native_df.copy()
    X_va_leak = v_va.native_df.copy()
    X_tr_leak["jumlah_scan"] = pd.to_numeric(tr_df["jumlah_scan"], errors="coerce").fillna(0.0).to_numpy()
    X_va_leak["jumlah_scan"] = pd.to_numeric(va_df["jumlah_scan"], errors="coerce").fillna(0.0).to_numpy()
    m_leak = LGBMRegressor(
        objective="quantile",
        alpha=0.5,
        n_estimators=probe_n_est,
        num_leaves=15,
        n_jobs=1,
        random_state=seed,
        verbose=-1,
    )
    m_leak.fit(X_tr_leak, np.log1p(np.maximum(y_tr_clean, 0.0)))
    pred_leak = np.clip(np.expm1(m_leak.predict(X_va_leak)), 1e-3, None)

    # Simulate test-like distribution where jumlah_scan is fixed low (e.g., 1..5)
    X_va_shifted = X_va_leak.copy()
    X_va_shifted["jumlah_scan"] = np.minimum(X_va_shifted["jumlah_scan"], 3.0)
    pred_leak_shifted = np.clip(np.expm1(m_leak.predict(X_va_shifted)), 1e-3, None)

    # 3. Unit-error model trained on RAW durasi_jam (uncleaned >1000 outliers)
    m_raw_y = LGBMRegressor(
        objective="quantile",
        alpha=0.5,
        n_estimators=probe_n_est,
        num_leaves=15,
        n_jobs=1,
        random_state=seed,
        verbose=-1,
    )
    m_raw_y.fit(v_tr.native_df, np.log1p(np.maximum(y_tr_raw, 0.0)))
    pred_raw_y = np.clip(np.expm1(m_raw_y.predict(v_va.native_df)), 1e-3, None)

    return {
        "probe_train_n": int(len(tr_df)),
        "probe_val_n": int(len(va_df)),
        "jumlah_scan_impact": {
            "oot_pinball_p50_with_leak": float(mean_pinball_loss(y_va_clean, pred_leak, alpha=0.5)),
            "oot_pinball_p50_without_leak": float(mean_pinball_loss(y_va_clean, pred_clean, alpha=0.5)),
            "oot_mae_with_leak": float(mean_absolute_error(y_va_clean, pred_leak)),
            "oot_mae_without_leak": float(mean_absolute_error(y_va_clean, pred_clean)),
            "oot_mae_under_test_scan_shift": float(mean_absolute_error(y_va_clean, pred_leak_shifted)),
            "mean_pred_collapse_under_scan_shift": float(np.mean(pred_leak_shifted) - np.mean(pred_clean)),
        },
        "durasi_unit_error_impact": {
            "oot_pinball_p50_raw_target_model": float(mean_pinball_loss(y_va_clean, pred_raw_y, alpha=0.5)),
            "oot_pinball_p50_cleaned_target_model": float(mean_pinball_loss(y_va_clean, pred_clean, alpha=0.5)),
            "oot_mae_raw_target_model": float(mean_absolute_error(y_va_clean, pred_raw_y)),
            "oot_mae_cleaned_target_model": float(mean_absolute_error(y_va_clean, pred_clean)),
            "oot_rmse_raw_target_model": float(root_mean_squared_error(y_va_clean, pred_raw_y)),
            "oot_rmse_cleaned_target_model": float(root_mean_squared_error(y_va_clean, pred_clean)),
        },
    }


def audit_and_clean_regresi(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int = SEED,
    smoke_cfg: SmokeTestConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Audit leakage, unit errors, sentinels, MNAR, drift, & categories; clean train/test."""
    y_raw = pd.to_numeric(train_df[TARGET_COL], errors="coerce").to_numpy(dtype=np.float64)
    gt_1000 = y_raw > DURASI_UNIT_THRESHOLD_HOURS
    y_clean = np.where(gt_1000, y_raw / 60.0, y_raw)

    n_gt_1000 = int(gt_1000.sum())
    n_gap_700_1000 = int(((y_raw > 700.0) & (y_raw <= DURASI_UNIT_THRESHOLD_HOURS)).sum())
    frac_int_gt1000 = (
        float(np.mean(np.isclose(y_raw[gt_1000] % 1.0, 0.0))) if n_gt_1000 > 0 else 0.0
    )
    frac_div60_tenths = (
        float(np.mean((np.round(y_raw[gt_1000] * 10.0) % 60.0) == 0.0)) if n_gt_1000 > 0 else 0.0
    )

    jarak_tr = pd.to_numeric(train_df["jarak_km"], errors="coerce").to_numpy(dtype=np.float64)
    jarak_te = pd.to_numeric(test_df["jarak_km"], errors="coerce").to_numpy(dtype=np.float64)
    valid_j_mask = np.isfinite(jarak_tr) & (jarak_tr > 0.0) & (jarak_tr < JARAK_MAX_SENTINEL)

    scan_tr = pd.to_numeric(train_df["jumlah_scan"], errors="coerce").to_numpy(dtype=np.float64)
    scan_te = pd.to_numeric(test_df["jumlah_scan"], errors="coerce").to_numpy(dtype=np.float64)
    scan_tr_std = max(float(np.nanstd(scan_tr)), 1e-9)

    route_imp = RouteJarakImputer().fit(train_df)
    _, _, te_recovery_stats = route_imp.transform_with_stats(test_df)

    # Berat MNAR audit
    berat_tr = pd.to_numeric(train_df["berat_kg"], errors="coerce")
    berat_te = pd.to_numeric(test_df["berat_kg"], errors="coerce")
    berat_by_layanan = {}
    for lay, grp in train_df.groupby("layanan"):
        b_grp = pd.to_numeric(grp["berat_kg"], errors="coerce")
        berat_by_layanan[str(lay)] = {
            "n": int(len(grp)),
            "missing_rate": float(b_grp.isna().mean()),
        }
    miss_mask = berat_tr.isna().to_numpy()
    med_dur_missing = float(np.median(y_clean[miss_mask])) if bool(miss_mask.any()) else 0.0
    med_dur_present = float(np.median(y_clean[~miss_mask])) if bool((~miss_mask).any()) else 0.0

    # Temporal drift audit (f30, f35, antrean_hub_asal)
    drift_summary: dict[str, Any] = {}
    for col in ("f30", "f35", "antrean_hub_asal"):
        tr_vals = pd.to_numeric(train_df[col], errors="coerce").to_numpy(dtype=np.float64)
        te_vals = pd.to_numeric(test_df[col], errors="coerce").to_numpy(dtype=np.float64)
        tr_mean = float(np.nanmean(tr_vals))
        tr_std = max(float(np.nanstd(tr_vals)), 1e-9)
        te_mean = float(np.nanmean(te_vals))
        drift_summary[col] = {
            "train_mean": tr_mean,
            "train_std": tr_std,
            "test_mean": te_mean,
            "test_std": float(np.nanstd(te_vals)),
            "shift_z": float((te_mean - tr_mean) / tr_std),
            "corr_clean_durasi": _safe_pearson(tr_vals, y_clean),
        }

    # Categorical cardinality & unseen categories
    train_mitra = set(train_df["mitra_kurir"].astype(str).unique())
    test_mitra = set(test_df["mitra_kurir"].astype(str).unique())
    unseen_mitra = sorted(test_mitra - train_mitra)
    unseen_mitra_rows = int(test_df["mitra_kurir"].astype(str).isin(unseen_mitra).sum())

    train_kota = set(train_df["kota_tujuan"].astype(str).unique())
    test_kota = set(test_df["kota_tujuan"].astype(str).unique())
    unseen_kota = sorted(test_kota - train_kota)

    impact_summary = _measure_audit_impacts(train_df, seed=seed, smoke_cfg=smoke_cfg)

    audit_summary: dict[str, Any] = {
        "dropped_leakage_columns": list(FORBIDDEN_LEAKAGE_COLS),
        "jumlah_scan": {
            "train_mean": float(np.nanmean(scan_tr)),
            "train_median": float(np.nanmedian(scan_tr)),
            "train_max": float(np.nanmax(scan_tr)),
            "test_mean": float(np.nanmean(scan_te)),
            "test_median": float(np.nanmedian(scan_te)),
            "test_max": float(np.nanmax(scan_te)),
            "shift_z": float((np.nanmean(scan_te) - np.nanmean(scan_tr)) / scan_tr_std),
            "corr_clean_durasi": _safe_pearson(scan_tr, y_clean),
            "action": "dropped_post_pickup_leakage",
            "impact": impact_summary.get("jumlah_scan_impact", {}),
        },
        "durasi_jam_unit_errors": {
            "threshold_hours": DURASI_UNIT_THRESHOLD_HOURS,
            "count_gt_1000": n_gt_1000,
            "count_gap_700_1000": n_gap_700_1000,
            "frac_integer_gt_1000": frac_int_gt1000,
            "frac_div60_tenths_gt_1000": frac_div60_tenths,
            "raw_max": float(np.nanmax(y_raw)),
            "raw_mean": float(np.nanmean(y_raw)),
            "clean_max": float(np.nanmax(y_clean)),
            "clean_mean": float(np.nanmean(y_clean)),
            "corr_jarak_before": _safe_pearson(jarak_tr[valid_j_mask], y_raw[valid_j_mask]),
            "corr_jarak_after": _safe_pearson(jarak_tr[valid_j_mask], y_clean[valid_j_mask]),
            "action": "divided_gt_1000_by_60_on_train_only",
            "impact": impact_summary.get("durasi_unit_error_impact", {}),
        },
        "jarak_km_sentinels": {
            "train_zero_count": int((jarak_tr <= 0.0).sum()),
            "train_9999_count": int((jarak_tr >= JARAK_MAX_SENTINEL).sum()),
            "test_zero_count": int((jarak_te <= 0.0).sum()),
            "test_9999_count": int((jarak_te >= JARAK_MAX_SENTINEL).sum()),
            "test_recovery_breakdown": te_recovery_stats,
            "action": "train_only_hierarchical_route_median_plus_flag",
        },
        "berat_kg_mnar": {
            "train_missing_rate": float(berat_tr.isna().mean()),
            "test_missing_rate": float(berat_te.isna().mean()),
            "by_layanan": berat_by_layanan,
            "median_durasi_when_missing": med_dur_missing,
            "median_durasi_when_present": med_dur_present,
            "action": "train_only_layanan_volume_bin_median_plus_flag",
        },
        "temporal_drift": drift_summary,
        "categorical_profile": {
            "kota_tujuan_train_nunique": int(len(train_kota)),
            "kota_tujuan_test_nunique": int(len(test_kota)),
            "unseen_kota_tujuan": unseen_kota[:20],
            "unseen_kota_tujuan_count": int(len(unseen_kota)),
            "mitra_kurir_train_levels": sorted(train_mitra),
            "unseen_mitra_kurir": unseen_mitra,
            "unseen_mitra_kurir_test_rows": unseen_mitra_rows,
            "unseen_mitra_kurir_test_pct": float(unseen_mitra_rows / max(1, len(test_df))),
        },
    }

    # Drop post-pickup leakage column immediately and fix 60x unit error in train target.
    train_clean = train_df.drop(columns=list(FORBIDDEN_LEAKAGE_COLS), errors="ignore").copy()
    train_clean[TARGET_COL] = y_clean
    test_clean = test_df.drop(columns=list(FORBIDDEN_LEAKAGE_COLS), errors="ignore").copy()

    return train_clean, test_clean, audit_summary


def build_time_cv_splits(
    tgl_series: pd.Series,
    max_folds: int = 4,
) -> list[dict[str, Any]]:
    """Build chronological expanding-window folds where ``train_max < val_min`` strictly."""
    dates_str = pd.to_datetime(tgl_series, errors="coerce").dt.strftime("%Y-%m-%d").to_numpy()
    months_str = np.array([str(d)[:7] for d in dates_str])
    unique_months = sorted(set(months_str.tolist()))

    folds: list[dict[str, Any]] = []
    if len(unique_months) >= max(3, max_folds + 1):
        val_months = unique_months[-max_folds:]
        for fold_idx, val_m in enumerate(val_months, start=1):
            tr_idx = np.flatnonzero(months_str < val_m)
            va_idx = np.flatnonzero(months_str == val_m)
            if len(tr_idx) == 0 or len(va_idx) == 0:
                continue
            tr_dates = dates_str[tr_idx]
            va_dates = dates_str[va_idx]
            fold_info = {
                "fold": fold_idx,
                "train_idx": tr_idx,
                "val_idx": va_idx,
                "train_n": int(len(tr_idx)),
                "val_n": int(len(va_idx)),
                "train_min": str(tr_dates.min()),
                "train_max": str(tr_dates.max()),
                "val_min": str(va_dates.min()),
                "val_max": str(va_dates.max()),
                "val_label": str(val_m),
            }
            assert fold_info["train_max"] < fold_info["val_min"], (
                f"Temporal ordering violated: {fold_info}"
            )
            folds.append(fold_info)
        if folds:
            return folds

    # Fallback for tiny/few-month datasets: chronological expanding splits by exact date
    unique_dates = sorted(set(dates_str.tolist()))
    if len(unique_dates) < 3:
        raise ValueError("Need at least 3 distinct dates to build chronological Time CV folds.")

    n_folds = min(max_folds, max(2, len(unique_dates) // 3))
    start_idx = max(1, len(unique_dates) // 2)
    step = max(1, (len(unique_dates) - start_idx) // n_folds)
    boundaries = [start_idx + i * step for i in range(n_folds)]
    boundaries.append(len(unique_dates))

    for fold_idx in range(len(boundaries) - 1):
        cut_start = boundaries[fold_idx]
        cut_end = boundaries[fold_idx + 1]
        if cut_start >= len(unique_dates):
            break
        val_start_date = unique_dates[cut_start]
        val_end_date = unique_dates[min(cut_end - 1, len(unique_dates) - 1)]
        tr_idx = np.flatnonzero(dates_str < val_start_date)
        va_idx = np.flatnonzero((dates_str >= val_start_date) & (dates_str <= val_end_date))
        if len(tr_idx) == 0 or len(va_idx) == 0:
            continue
        tr_dates = dates_str[tr_idx]
        va_dates = dates_str[va_idx]
        fold_info = {
            "fold": len(folds) + 1,
            "train_idx": tr_idx,
            "val_idx": va_idx,
            "train_n": int(len(tr_idx)),
            "val_n": int(len(va_idx)),
            "train_min": str(tr_dates.min()),
            "train_max": str(tr_dates.max()),
            "val_min": str(va_dates.min()),
            "val_max": str(va_dates.max()),
            "val_label": f"{val_start_date}..{val_end_date}",
        }
        assert fold_info["train_max"] < fold_info["val_min"]
        folds.append(fold_info)

    return folds


def _compute_quantile_metrics(
    y_true: np.ndarray,
    p50: np.ndarray,
    p90: np.ndarray,
) -> dict[str, float]:
    p50_safe = np.clip(np.asarray(p50, dtype=np.float64), 1e-3, None)
    p90_safe = np.maximum(np.asarray(p90, dtype=np.float64), p50_safe)
    pin50 = float(mean_pinball_loss(y_true, p50_safe, alpha=0.5))
    pin90 = float(mean_pinball_loss(y_true, p90_safe, alpha=0.9))
    return {
        "pinball_p50": pin50,
        "pinball_p90": pin90,
        "pinball_avg": float(0.5 * (pin50 + pin90)),
        "mae": float(mean_absolute_error(y_true, p50_safe)),
        "rmse": float(root_mean_squared_error(y_true, p50_safe)),
        "p90_coverage": float(np.mean(y_true <= p90_safe)),
        "mean_width": float(np.mean(p90_safe - p50_safe)),
    }


def _fit_predict_quantile_pair(
    family: str,
    params: dict[str, Any],
    v_tr: PreprocessedViews,
    y_tr: np.ndarray,
    v_va: PreprocessedViews,
    seed: int,
    n_jobs: int = 1,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Train a quantile pair (p50, p90) on log1p(y) and return original-hour predictions + models."""
    z_tr = np.log1p(np.maximum(y_tr, 0.0))

    if family == "Baseline_Dummy":
        m50 = DummyRegressor(strategy="quantile", quantile=0.5).fit(v_tr.linear, z_tr)
        m90 = DummyRegressor(strategy="quantile", quantile=0.9).fit(v_tr.linear, z_tr)
        p50 = np.expm1(m50.predict(v_va.linear))
        p90 = np.expm1(m90.predict(v_va.linear))
        bundle = {"family": family, "m50": m50, "m90": m90, "view": "linear"}
    elif family == "Baseline_RidgeLogQuantile":
        alpha = float(params.get("alpha", 10.0))
        ridge = Ridge(alpha=alpha, random_state=seed).fit(v_tr.linear, z_tr)
        res_tr = z_tr - ridge.predict(v_tr.linear)
        q50_res = float(np.quantile(res_tr, 0.50))
        q90_res = float(np.quantile(res_tr, 0.90))
        z_va_hat = ridge.predict(v_va.linear)
        p50 = np.expm1(z_va_hat + q50_res)
        p90 = np.expm1(z_va_hat + q90_res)
        bundle = {
            "family": family,
            "m50": ridge,
            "m90": ridge,
            "q50_res": q50_res,
            "q90_res": q90_res,
            "view": "linear",
        }
    elif family == "HistGradientBoosting":
        common = {
            "max_iter": int(params.get("max_iter", 200)),
            "learning_rate": float(params.get("learning_rate", 0.06)),
            "max_leaf_nodes": int(params.get("max_leaf_nodes", 31)),
            "min_samples_leaf": int(params.get("min_samples_leaf", 30)),
            "l2_regularization": float(params.get("l2_regularization", 0.5)),
            "categorical_features": v_tr.hgbr_cat_indices,
            "random_state": seed,
        }
        m50 = HistGradientBoostingRegressor(loss="quantile", quantile=0.5, **common).fit(
            v_tr.hgbr_df, z_tr
        )
        m90 = HistGradientBoostingRegressor(loss="quantile", quantile=0.9, **common).fit(
            v_tr.hgbr_df, z_tr
        )
        p50 = np.expm1(m50.predict(v_va.hgbr_df))
        p90 = np.expm1(m90.predict(v_va.hgbr_df))
        bundle = {"family": family, "m50": m50, "m90": m90, "view": "hgbr"}
    elif family == "LightGBM":
        common = {
            "n_estimators": int(params.get("n_estimators", 300)),
            "learning_rate": float(params.get("learning_rate", 0.05)),
            "num_leaves": int(params.get("num_leaves", 31)),
            "min_child_samples": int(params.get("min_child_samples", 40)),
            "subsample": float(params.get("subsample", 0.8)),
            "subsample_freq": 1,
            "colsample_bytree": float(params.get("colsample_bytree", 0.8)),
            "reg_lambda": float(params.get("reg_lambda", 1.0)),
            "random_state": seed,
            "n_jobs": n_jobs,
            "verbose": -1,
        }
        m50 = LGBMRegressor(objective="quantile", alpha=0.5, **common).fit(v_tr.native_df, z_tr)
        m90 = LGBMRegressor(objective="quantile", alpha=0.9, **common).fit(v_tr.native_df, z_tr)
        p50 = np.expm1(m50.predict(v_va.native_df))
        p90 = np.expm1(m90.predict(v_va.native_df))
        bundle = {"family": family, "m50": m50, "m90": m90, "view": "native"}
    elif family == "XGBoost":
        common = {
            "n_estimators": int(params.get("n_estimators", 300)),
            "learning_rate": float(params.get("learning_rate", 0.06)),
            "max_depth": int(params.get("max_depth", 6)),
            "subsample": float(params.get("subsample", 0.8)),
            "colsample_bytree": float(params.get("colsample_bytree", 0.8)),
            "reg_lambda": float(params.get("reg_lambda", 2.0)),
            "objective": "reg:quantileerror",
            "quantile_alpha": [0.5, 0.9],
            "enable_categorical": True,
            "tree_method": "hist",
            "random_state": seed,
            "n_jobs": n_jobs,
            "verbosity": 0,
        }
        m_multi = XGBRegressor(**common).fit(v_tr.native_df, z_tr)
        preds = np.asarray(m_multi.predict(v_va.native_df), dtype=np.float64)
        p50 = np.expm1(preds[:, 0])
        p90 = np.expm1(preds[:, 1])
        bundle = {"family": family, "m50": m_multi, "m90": m_multi, "view": "native"}
    else:
        raise ValueError(f"Unknown model family: {family}")

    p50 = np.clip(p50, 1e-3, None)
    p90 = np.maximum(p90, p50)
    return p50, p90, bundle


def _predict_with_bundle(
    bundle: dict[str, Any],
    views: PreprocessedViews,
) -> tuple[np.ndarray, np.ndarray]:
    family = bundle["family"]
    if family == "Baseline_Dummy":
        p50 = np.expm1(bundle["m50"].predict(views.linear))
        p90 = np.expm1(bundle["m90"].predict(views.linear))
    elif family == "Baseline_RidgeLogQuantile":
        z_hat = bundle["m50"].predict(views.linear)
        p50 = np.expm1(z_hat + bundle["q50_res"])
        p90 = np.expm1(z_hat + bundle["q90_res"])
    elif family == "HistGradientBoosting":
        p50 = np.expm1(bundle["m50"].predict(views.hgbr_df))
        p90 = np.expm1(bundle["m90"].predict(views.hgbr_df))
    elif family == "LightGBM":
        p50 = np.expm1(bundle["m50"].predict(views.native_df))
        p90 = np.expm1(bundle["m90"].predict(views.native_df))
    elif family == "XGBoost":
        preds = np.asarray(bundle["m50"].predict(views.native_df), dtype=np.float64)
        p50 = np.expm1(preds[:, 0])
        p90 = np.expm1(preds[:, 1])
    else:
        raise ValueError(f"Unknown bundle family: {family}")

    p50 = np.clip(p50, 1e-3, None)
    p90 = np.maximum(p90, p50)
    return p50, p90


def evaluate_time_vs_random_cv(
    train_clean: pd.DataFrame,
    seed: int = SEED,
    smoke_cfg: SmokeTestConfig | None = None,
) -> dict[str, Any]:
    """Compare Expanding-Window Time-Based CV against Random K-Fold CV on identical models."""
    is_smoke = bool(smoke_cfg is not None and smoke_cfg.enabled)
    max_time_folds = smoke_cfg.cv_folds if is_smoke else 3
    n_random_splits = smoke_cfg.cv_folds if is_smoke else 3
    n_jobs = 1 if is_smoke else 4

    cv_df = (
        _stratified_smoke_sample(train_clean, target_n=40_000, seed=seed + 11)
        if (not is_smoke and len(train_clean) > 40_000)
        else train_clean
    )
    LOGGER.info(
        "[Step 3/6] Evaluating Time-Based CV (%d folds) vs Random CV (%d folds) on %d rows...",
        max_time_folds,
        n_random_splits,
        len(cv_df),
    )

    y_all = pd.to_numeric(cv_df[TARGET_COL], errors="coerce").to_numpy(dtype=np.float64)
    time_folds_raw = build_time_cv_splits(cv_df[DATE_COL], max_folds=max_time_folds)

    kf = KFold(n_splits=n_random_splits, shuffle=True, random_state=seed)
    random_folds_raw = [
        {"fold": idx + 1, "train_idx": tr_idx, "val_idx": va_idx}
        for idx, (tr_idx, va_idx) in enumerate(kf.split(cv_df))
    ]

    ref_models = [
        ("Baseline_RidgeLogQuantile", {"alpha": 10.0}),
        (
            "HistGradientBoosting",
            {"max_iter": 15 if is_smoke else 100, "learning_rate": 0.08, "max_leaf_nodes": 31},
        ),
        (
            "LightGBM",
            {"n_estimators": 15 if is_smoke else 100, "learning_rate": 0.08, "num_leaves": 31},
        ),
    ]

    def _run_scheme(folds: list[dict[str, Any]], include_dates: bool) -> dict[str, Any]:
        fold_views: list[tuple[PreprocessedViews, np.ndarray, PreprocessedViews, np.ndarray]] = []
        fold_metadata: list[dict[str, Any]] = []
        for f in folds:
            tr_idx, va_idx = f["train_idx"], f["val_idx"]
            prep = RegresiFeaturePreprocessor(seed=seed)
            v_tr = prep.fit_transform(cv_df.iloc[tr_idx], y_all[tr_idx])
            v_va = prep.transform(cv_df.iloc[va_idx])
            fold_views.append((v_tr, y_all[tr_idx], v_va, y_all[va_idx]))
            meta: dict[str, Any] = {
                "fold": int(f["fold"]),
                "train_n": int(len(tr_idx)),
                "val_n": int(len(va_idx)),
            }
            if include_dates:
                meta["train_min"] = str(f["train_min"])
                meta["train_max"] = str(f["train_max"])
                meta["val_min"] = str(f["val_min"])
                meta["val_max"] = str(f["val_max"])
                meta["val_label"] = str(f["val_label"])
            fold_metadata.append(meta)

        model_rows: list[dict[str, Any]] = []
        for family, params in ref_models:
            per_fold = []
            for v_tr, y_tr, v_va, y_va in fold_views:
                p50, p90, _ = _fit_predict_quantile_pair(
                    family, params, v_tr, y_tr, v_va, seed=seed, n_jobs=n_jobs
                )
                per_fold.append(_compute_quantile_metrics(y_va, p50, p90))
            agg = {
                "model": family,
                "pinball_p50": float(np.mean([m["pinball_p50"] for m in per_fold])),
                "pinball_p90": float(np.mean([m["pinball_p90"] for m in per_fold])),
                "pinball_avg": float(np.mean([m["pinball_avg"] for m in per_fold])),
                "mae": float(np.mean([m["mae"] for m in per_fold])),
                "rmse": float(np.mean([m["rmse"] for m in per_fold])),
                "p90_coverage": float(np.mean([m["p90_coverage"] for m in per_fold])),
            }
            model_rows.append(agg)

        return {"folds": fold_metadata, "models": model_rows}

    time_res = _run_scheme(time_folds_raw, include_dates=True)
    LOGGER.info("  -> Time-Based CV completed.")
    random_res = _run_scheme(random_folds_raw, include_dates=False)
    LOGGER.info("  -> Random CV completed.")

    # Optimism gap (Time CV error minus Random CV error for each reference model)
    optimism_gaps: list[dict[str, Any]] = []
    for t_row, r_row in zip(time_res["models"], random_res["models"]):
        optimism_gaps.append(
            {
                "model": t_row["model"],
                "pinball_avg_gap": float(t_row["pinball_avg"] - r_row["pinball_avg"]),
                "mae_gap": float(t_row["mae"] - r_row["mae"]),
                "rmse_gap": float(t_row["rmse"] - r_row["rmse"]),
                "p90_cov_diff": float(t_row["p90_coverage"] - r_row["p90_coverage"]),
            }
        )

    return {
        "time_cv": {
            "scheme": "expanding_window_next_month",
            "n_folds": int(len(time_res["folds"])),
            "folds": time_res["folds"],
            "models": time_res["models"],
        },
        "random_cv": {
            "scheme": "random_kfold",
            "n_splits": int(n_random_splits),
            "folds": random_res["folds"],
            "models": random_res["models"],
        },
        "optimism_gap": optimism_gaps,
    }


def train_and_tune_quantile_models(
    train_clean: pd.DataFrame,
    seed: int = SEED,
    smoke_cfg: SmokeTestConfig | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Train baselines & tune >=3 advanced quantile models strictly inside Time-Based CV."""
    is_smoke = bool(smoke_cfg is not None and smoke_cfg.enabled)
    max_folds = smoke_cfg.cv_folds if is_smoke else 3
    n_jobs = 1 if is_smoke else 4

    y_all_full = pd.to_numeric(train_clean[TARGET_COL], errors="coerce").to_numpy(dtype=np.float64)
    use_tune_subsample = (not is_smoke) and (len(train_clean) > 50_000)
    tune_df = (
        _stratified_smoke_sample(train_clean, target_n=50_000, seed=seed + 23)
        if use_tune_subsample
        else train_clean
    )
    y_tune = pd.to_numeric(tune_df[TARGET_COL], errors="coerce").to_numpy(dtype=np.float64)
    time_folds = build_time_cv_splits(tune_df[DATE_COL], max_folds=max_folds)

    LOGGER.info(
        "[Step 4/6] Tuning 5 model families across %d Time-CV folds (%d tuning rows, %d full rows)...",
        len(time_folds),
        len(tune_df),
        len(train_clean),
    )

    # Precompute fold preprocessors once so tuning is fast and zero-leakage
    cached_folds: list[dict[str, Any]] = []
    for f in time_folds:
        tr_idx, va_idx = f["train_idx"], f["val_idx"]
        prep = RegresiFeaturePreprocessor(seed=seed)
        v_tr = prep.fit_transform(tune_df.iloc[tr_idx], y_tune[tr_idx])
        v_va = prep.transform(tune_df.iloc[va_idx])
        cached_folds.append(
            {
                "fold": int(f["fold"]),
                "val_idx": va_idx,
                "v_tr": v_tr,
                "y_tr": y_tune[tr_idx],
                "v_va": v_va,
                "y_va": y_tune[va_idx],
            }
        )

    if is_smoke:
        candidate_grid: dict[str, list[dict[str, Any]]] = {
            "Baseline_Dummy": [{}],
            "Baseline_RidgeLogQuantile": [{"alpha": 10.0}],
            "HistGradientBoosting": [
                {"max_iter": 12, "learning_rate": 0.12, "max_leaf_nodes": 15, "min_samples_leaf": 15},
            ],
            "LightGBM": [
                {"n_estimators": 12, "learning_rate": 0.12, "num_leaves": 15, "min_child_samples": 15},
            ],
            "XGBoost": [
                {"n_estimators": 12, "learning_rate": 0.12, "max_depth": 4},
            ],
        }
    else:
        candidate_grid = {
            "Baseline_Dummy": [{}],
            "Baseline_RidgeLogQuantile": [{"alpha": 1.0}, {"alpha": 10.0}],
            "HistGradientBoosting": [
                {"max_iter": 220, "learning_rate": 0.06, "max_leaf_nodes": 31, "min_samples_leaf": 30, "l2_regularization": 0.5},
                {"max_iter": 280, "learning_rate": 0.05, "max_leaf_nodes": 63, "min_samples_leaf": 40, "l2_regularization": 1.0},
            ],
            "LightGBM": [
                {"n_estimators": 250, "learning_rate": 0.06, "num_leaves": 31, "min_child_samples": 40, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0},
                {"n_estimators": 320, "learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 50, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0},
            ],
            "XGBoost": [
                {"n_estimators": 220, "learning_rate": 0.06, "max_depth": 6, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 2.0},
                {"n_estimators": 280, "learning_rate": 0.05, "max_depth": 7, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 2.0},
            ],
        }

    comparison_table: list[dict[str, Any]] = []
    best_per_family: dict[str, dict[str, Any]] = {}

    for family, param_list in candidate_grid.items():
        best_cfg_entry: dict[str, Any] | None = None
        for cfg in param_list:
            fold_metrics: list[dict[str, float]] = []
            last_fold_preds: tuple[np.ndarray, np.ndarray] | None = None
            last_fold_bundle: dict[str, Any] | None = None
            for cf in cached_folds:
                p50, p90, bundle = _fit_predict_quantile_pair(
                    family,
                    cfg,
                    cf["v_tr"],
                    cf["y_tr"],
                    cf["v_va"],
                    seed=seed,
                    n_jobs=n_jobs,
                )
                fold_metrics.append(_compute_quantile_metrics(cf["y_va"], p50, p90))
                last_fold_preds = (p50, p90)
                last_fold_bundle = bundle

            summary = {
                "model": family,
                "is_baseline": family.startswith("Baseline_"),
                "params": dict(cfg),
                "pinball_p50": float(np.mean([m["pinball_p50"] for m in fold_metrics])),
                "pinball_p90": float(np.mean([m["pinball_p90"] for m in fold_metrics])),
                "pinball_avg": float(np.mean([m["pinball_avg"] for m in fold_metrics])),
                "mae": float(np.mean([m["mae"] for m in fold_metrics])),
                "rmse": float(np.mean([m["rmse"] for m in fold_metrics])),
                "p90_coverage": float(np.mean([m["p90_coverage"] for m in fold_metrics])),
                "mean_width": float(np.mean([m["mean_width"] for m in fold_metrics])),
            }
            if best_cfg_entry is None or summary["pinball_avg"] < best_cfg_entry["summary"]["pinball_avg"]:
                best_cfg_entry = {
                    "summary": summary,
                    "last_fold_preds": last_fold_preds,
                    "last_fold_bundle": last_fold_bundle,
                }

        assert best_cfg_entry is not None
        best_per_family[family] = best_cfg_entry
        comparison_table.append(best_cfg_entry["summary"])
        LOGGER.info(
            "  -> [Tuned] %-26s | pinball_avg=%.4f (p50=%.4f, p90=%.4f) | P90_cov=%.3f",
            family,
            best_cfg_entry["summary"]["pinball_avg"],
            best_cfg_entry["summary"]["pinball_p50"],
            best_cfg_entry["summary"]["pinball_p90"],
            best_cfg_entry["summary"]["p90_coverage"],
        )

    # Select winning model minimizing mean Time-CV pinball_avg (preferring advanced models if tied)
    comparison_sorted = sorted(
        comparison_table,
        key=lambda r: (r["pinball_avg"], 1 if r["is_baseline"] else 0),
    )
    selected_family = str(comparison_sorted[0]["model"])
    selected_params = dict(comparison_sorted[0]["params"])
    winner = best_per_family[selected_family]

    if use_tune_subsample:
        full_time_folds = build_time_cv_splits(train_clean[DATE_COL], max_folds=max_folds)
        last_f = full_time_folds[-1]
        tr_idx_oot, va_idx_oot = last_f["train_idx"], last_f["val_idx"]
        LOGGER.info(
            "  -> Fitting winning model (%s) on full OOT fold (%d train -> %d OOT val rows)...",
            selected_family,
            len(tr_idx_oot),
            len(va_idx_oot),
        )
        oot_prep = RegresiFeaturePreprocessor(seed=seed)
        v_oot_tr = oot_prep.fit_transform(train_clean.iloc[tr_idx_oot], y_all_full[tr_idx_oot])
        v_oot_va = oot_prep.transform(train_clean.iloc[va_idx_oot])
        oot_p50_raw, oot_p90_raw, oot_bundle = _fit_predict_quantile_pair(
            selected_family,
            selected_params,
            v_oot_tr,
            y_all_full[tr_idx_oot],
            v_oot_va,
            seed=seed,
            n_jobs=n_jobs,
        )
        oot_val_idx = va_idx_oot
        oot_y = y_all_full[va_idx_oot]
        oot_views = v_oot_va
    else:
        last_cf = cached_folds[-1]
        oot_p50_raw, oot_p90_raw = winner["last_fold_preds"]
        oot_val_idx = last_cf["val_idx"]
        oot_y = last_cf["y_va"]
        oot_views = last_cf["v_va"]
        oot_bundle = winner["last_fold_bundle"]

    # Refit winning configuration on 100% of cleaned training data for test prediction
    LOGGER.info(
        "[Step 5/6] Refitting winning model (%s) on 100%% training data (%d rows) + Conformal Calibration...",
        selected_family,
        len(train_clean),
    )
    full_prep = RegresiFeaturePreprocessor(seed=seed)
    v_full = full_prep.fit_transform(train_clean, y_all_full)
    _, _, full_bundle = _fit_predict_quantile_pair(
        selected_family,
        selected_params,
        v_full,
        y_all_full,
        v_full,
        seed=seed,
        n_jobs=n_jobs,
    )

    model_results = {
        "encoding_strategy": (
            "OOF TargetEncoder(cv=KFold, smooth='auto') + train-only freq for "
            "kota_tujuan(405)/kota_asal/mitra_kurir; OrdinalEncoder for HGBR; "
            "masked pd.Categorical for LGBM/XGB; OHE+StandardScaler for linear baseline; "
            "unseen K13/kota_tujuan fallback to global prior & freq=0"
        ),
        "tuning_cv_scheme": "expanding_window_next_month",
        "comparison_table": comparison_sorted,
        "selected_model": selected_family,
        "selected_params": selected_params,
    }
    best_model_bundle = {
        "selected_family": selected_family,
        "selected_params": selected_params,
        "full_preprocessor": full_prep,
        "full_bundle": full_bundle,
        "oot_val_idx": oot_val_idx,
        "oot_y": oot_y,
        "oot_p50_raw": oot_p50_raw,
        "oot_p90_raw": oot_p90_raw,
        "oot_views": oot_views,
        "oot_bundle": oot_bundle,
    }
    return model_results, best_model_bundle


def conformal_quantile(scores: np.ndarray, alpha: float = 0.90) -> float:
    """Exact finite-sample split-conformal order statistic at level ``alpha``."""
    arr = np.asarray(scores, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return 0.0
    k = min(n, max(1, int(math.ceil((n + 1) * alpha))))
    return float(np.sort(arr)[k - 1])


def apply_conformal_calibration(
    p50_raw: np.ndarray,
    p90_raw: np.ndarray,
    q_hat: float,
    mode: str = "log_space",
) -> tuple[np.ndarray, np.ndarray]:
    """Apply conformal adjustment and strictly enforce ``p50 > 0`` and ``p90 >= p50``."""
    p50 = np.clip(np.asarray(p50_raw, dtype=np.float64), 1e-3, None)
    p90_base = np.clip(np.asarray(p90_raw, dtype=np.float64), 1e-3, None)
    if mode == "log_space":
        p90_adj = np.expm1(np.log1p(p90_base) + float(q_hat))
    elif mode == "additive":
        p90_adj = p90_base + float(q_hat)
    else:
        raise ValueError(f"Unknown conformal calibration mode: {mode}")
    p90 = np.maximum(p90_adj, p50)
    return p50, p90


def calibrate_conformal_intervals(
    y_cal: np.ndarray,
    p50_cal: np.ndarray,
    p90_cal_raw: np.ndarray,
    tgl_cal: pd.Series | np.ndarray | None = None,
    alpha: float = 0.90,
) -> tuple[float, np.ndarray, dict[str, Any]]:
    """Evaluate honest OOT before-vs-after split conformal calibration and fit production q_hat."""
    y_arr = np.asarray(y_cal, dtype=np.float64)
    p50_arr = np.clip(np.asarray(p50_cal, dtype=np.float64), 1e-3, None)
    p90_arr = np.maximum(np.asarray(p90_cal_raw, dtype=np.float64), p50_arr)

    n = len(y_arr)
    if n >= 20:
        if tgl_cal is not None:
            order = np.argsort(np.asarray(tgl_cal, dtype=str))
        else:
            order = np.arange(n)
        half = n // 2
        idx_cal = order[:half]
        idx_eval = order[half:]

        # Compare additive vs log-space CQR honestly on D_eval
        s_add_cal = y_arr[idx_cal] - p90_arr[idx_cal]
        q_add_eval = conformal_quantile(s_add_cal, alpha=alpha)
        _, p90_add_eval = apply_conformal_calibration(
            p50_arr[idx_eval], p90_arr[idx_eval], q_add_eval, mode="additive"
        )
        m_add = _compute_quantile_metrics(y_arr[idx_eval], p50_arr[idx_eval], p90_add_eval)

        s_log_cal = np.log1p(y_arr[idx_cal]) - np.log1p(p90_arr[idx_cal])
        q_log_eval = conformal_quantile(s_log_cal, alpha=alpha)
        _, p90_log_eval = apply_conformal_calibration(
            p50_arr[idx_eval], p90_arr[idx_eval], q_log_eval, mode="log_space"
        )
        m_log = _compute_quantile_metrics(y_arr[idx_eval], p50_arr[idx_eval], p90_log_eval)

        raw_eval_metrics = _compute_quantile_metrics(
            y_arr[idx_eval], p50_arr[idx_eval], p90_arr[idx_eval]
        )

        # Select mode closest to target coverage with tie-break on pinball_p90
        score_add = abs(m_add["p90_coverage"] - alpha) + 0.01 * m_add["pinball_p90"]
        score_log = abs(m_log["p90_coverage"] - alpha) + 0.01 * m_log["pinball_p90"]
        chosen_mode = "log_space" if score_log <= score_add else "additive"
        eval_cal_metrics = m_log if chosen_mode == "log_space" else m_add
        eval_q_hat = q_log_eval if chosen_mode == "log_space" else q_add_eval
    else:
        chosen_mode = "log_space"
        raw_eval_metrics = _compute_quantile_metrics(y_arr, p50_arr, p90_arr)
        eval_cal_metrics = dict(raw_eval_metrics)
        eval_q_hat = 0.0

    # Fit production conformal q_hat on all OOT validation predictions
    if chosen_mode == "log_space":
        full_scores = np.log1p(y_arr) - np.log1p(p90_arr)
    else:
        full_scores = y_arr - p90_arr

    q_hat_full = conformal_quantile(full_scores, alpha=alpha)
    _, p90_cal_full = apply_conformal_calibration(p50_arr, p90_arr, q_hat_full, mode=chosen_mode)

    raw_full_metrics = _compute_quantile_metrics(y_arr, p50_arr, p90_arr)
    cal_full_metrics = _compute_quantile_metrics(y_arr, p50_arr, p90_cal_full)

    calib_summary = {
        "method": "split_conformal_cqr_upper_quantile",
        "selected_space": chosen_mode,
        "target_coverage": float(alpha),
        "conformal_q_hat": float(q_hat_full),
        "honest_split_eval_q_hat": float(eval_q_hat),
        "n_oot_total": int(n),
        "raw_metrics": raw_eval_metrics,
        "calibrated_metrics": eval_cal_metrics,
        "oot_full_raw_metrics": raw_full_metrics,
        "oot_full_calibrated_metrics": cal_full_metrics,
        "non_crossing_enforced": True,
    }
    return float(q_hat_full), p90_cal_full, calib_summary


def compute_group_breakdowns(
    val_df_meta: pd.DataFrame,
    y_val: np.ndarray,
    p50_val: np.ndarray,
    p90_val: np.ndarray,
    test_df: pd.DataFrame,
    p50_test: np.ndarray,
    p90_test: np.ndarray,
    seen_mitra: set[str],
) -> dict[str, Any]:
    """Compute segment error breakdowns on OOT validation and test-set unseen-courier summaries."""
    val_work = pd.DataFrame(
        {
            "layanan": val_df_meta["layanan"].astype(str).to_numpy(),
            "pulau_tujuan": val_df_meta["pulau_tujuan"].astype(str).to_numpy(),
            "mitra_kurir": val_df_meta["mitra_kurir"].astype(str).to_numpy(),
            "y": np.asarray(y_val, dtype=np.float64),
            "p50": np.asarray(p50_val, dtype=np.float64),
            "p90": np.asarray(p90_val, dtype=np.float64),
        }
    )

    def _summarize_val_group(col: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for grp_name, sub in val_work.groupby(col, sort=True):
            if len(sub) == 0:
                continue
            y_g = sub["y"].to_numpy(dtype=np.float64)
            p50_g = sub["p50"].to_numpy(dtype=np.float64)
            p90_g = sub["p90"].to_numpy(dtype=np.float64)
            cov_rate = float(np.mean(y_g <= p90_g))
            rows.append(
                {
                    "group": str(grp_name),
                    "n": int(len(sub)),
                    "mean_actual": float(np.mean(y_g)),
                    "median_actual": float(np.median(y_g)),
                    "mean_p50": float(np.mean(p50_g)),
                    "mean_p90": float(np.mean(p90_g)),
                    "pinball_p50": float(mean_pinball_loss(y_g, p50_g, alpha=0.5)),
                    "pinball_p90": float(mean_pinball_loss(y_g, p90_g, alpha=0.9)),
                    "mae": float(mean_absolute_error(y_g, p50_g)),
                    "rmse": float(root_mean_squared_error(y_g, p50_g)),
                    "p90_hit_rate": cov_rate,
                    "mean_interval_width": float(np.mean(p90_g - p50_g)),
                }
            )
        return rows

    test_work = pd.DataFrame(
        {
            "layanan": test_df["layanan"].astype(str).to_numpy(),
            "pulau_tujuan": test_df["pulau_tujuan"].astype(str).to_numpy(),
            "mitra_kurir": test_df["mitra_kurir"].astype(str).to_numpy(),
            "p50": np.asarray(p50_test, dtype=np.float64),
            "p90": np.asarray(p90_test, dtype=np.float64),
        }
    )

    def _summarize_test_group(col: str, flag_unseen: bool = False) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for grp_name, sub in test_work.groupby(col, sort=True):
            p50_g = sub["p50"].to_numpy(dtype=np.float64)
            p90_g = sub["p90"].to_numpy(dtype=np.float64)
            row: dict[str, Any] = {
                "group": str(grp_name),
                "n": int(len(sub)),
                "mean_p50": float(np.mean(p50_g)),
                "median_p50": float(np.median(p50_g)),
                "mean_p90": float(np.mean(p90_g)),
                "median_p90": float(np.median(p90_g)),
                "mean_interval_width": float(np.mean(p90_g - p50_g)),
            }
            if flag_unseen:
                row["is_unseen_in_train"] = bool(str(grp_name) not in seen_mitra)
            rows.append(row)
        return rows

    return {
        "by_layanan": _summarize_val_group("layanan"),
        "by_pulau_tujuan": _summarize_val_group("pulau_tujuan"),
        "by_mitra_kurir": _summarize_val_group("mitra_kurir"),
        "test_mitra_kurir_summary": _summarize_test_group("mitra_kurir", flag_unseen=True),
        "test_layanan_summary": _summarize_test_group("layanan"),
        "test_pulau_tujuan_summary": _summarize_test_group("pulau_tujuan"),
    }


def _save_diagnostic_artifacts(
    artifacts_dir: Path,
    cv_summary: dict[str, Any],
    model_results: dict[str, Any],
    calib_summary: dict[str, Any],
    oot_y: np.ndarray,
    oot_p50: np.ndarray,
    oot_p90_raw: np.ndarray,
    oot_p90_cal: np.ndarray,
    breakdowns: dict[str, Any],
    shap_values_2d: np.ndarray | None,
    X_shap_df: pd.DataFrame | None,
    pdp_estimator: Any,
    X_pdp_df: pd.DataFrame | None,
    top_pdp_cols: list[str],
) -> dict[str, str]:
    ensure_dir(artifacts_dir)
    sns.set_theme(style="whitegrid")
    saved: dict[str, str] = {}

    # 1. CV & Model Comparison
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    time_models = pd.DataFrame(cv_summary["time_cv"]["models"])
    rand_models = pd.DataFrame(cv_summary["random_cv"]["models"])
    cv_df = pd.concat(
        [
            time_models.assign(scheme="Time-Based CV"),
            rand_models.assign(scheme="Random K-Fold CV"),
        ],
        ignore_index=True,
    )
    sns.barplot(data=cv_df, x="model", y="pinball_avg", hue="scheme", ax=axes[0])
    axes[0].set_title("Time-Based CV vs Random K-Fold CV (Pinball Avg)")
    axes[0].tick_params(axis="x", rotation=15)

    comp_df = pd.DataFrame(model_results["comparison_table"])
    comp_melt = comp_df.melt(
        id_vars=["model"],
        value_vars=["pinball_p50", "pinball_p90"],
        var_name="quantile",
        value_name="pinball_loss",
    )
    sns.barplot(data=comp_melt, x="model", y="pinball_loss", hue="quantile", ax=axes[1])
    axes[1].set_title("Perbandingan Model Kuantil (Time-CV)")
    axes[1].tick_params(axis="x", rotation=20)
    plt.tight_layout()
    p1 = artifacts_dir / "cv_and_model_comparison.png"
    fig.savefig(p1, dpi=140, bbox_inches="tight")
    plt.close(fig)
    saved["cv_and_model_comparison"] = str(p1)

    # 2. Conformal Coverage Diagnostics
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    raw_cov = calib_summary["raw_metrics"]["p90_coverage"]
    cal_cov = calib_summary["calibrated_metrics"]["p90_coverage"]
    axes[0].bar(["Raw P90", "Conformal P90"], [raw_cov, cal_cov], color=["#5c7cfa", "#20c997"])
    axes[0].axhline(calib_summary["target_coverage"], color="crimson", linestyle="--", label="Target 0.90")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_title("Cakupan Empiris P90 Sebelum vs Sesudah Kalibrasi")
    axes[0].legend()

    sample_n = min(400, len(oot_y))
    order = np.argsort(oot_p50[:sample_n])
    axes[1].scatter(oot_p50[:sample_n][order], oot_y[:sample_n][order], s=12, alpha=0.5, label="Aktual")
    axes[1].plot(oot_p50[:sample_n][order], oot_p90_raw[:sample_n][order], color="orange", alpha=0.7, label="P90 Raw")
    axes[1].plot(oot_p50[:sample_n][order], oot_p90_cal[:sample_n][order], color="crimson", alpha=0.8, label="P90 Calibrated")
    axes[1].set_xlabel("Prediksi P50 (jam)")
    axes[1].set_ylabel("Durasi / Batas P90 (jam)")
    axes[1].set_title("Prediksi P50 vs Batas P90 Conformal")
    axes[1].legend()
    plt.tight_layout()
    p2 = artifacts_dir / "conformal_coverage_diagnostics.png"
    fig.savefig(p2, dpi=140, bbox_inches="tight")
    plt.close(fig)
    saved["conformal_coverage_diagnostics"] = str(p2)

    # 3. Error Breakdown by Layanan & Pulau Tujuan
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    df_lay = pd.DataFrame(breakdowns["by_layanan"])
    df_pul = pd.DataFrame(breakdowns["by_pulau_tujuan"])
    if len(df_lay) > 0:
        sns.barplot(data=df_lay, x="group", y="mae", ax=axes[0], color="#4dabf7")
        axes[0].set_title("MAE OOT per Layanan")
        axes[0].tick_params(axis="x", rotation=15)
    if len(df_pul) > 0:
        sns.barplot(data=df_pul, x="group", y="p90_hit_rate", ax=axes[1], color="#69db7c")
        axes[1].axhline(0.90, color="crimson", linestyle="--")
        axes[1].set_ylim(0.0, 1.05)
        axes[1].set_title("Cakupan P90 OOT per Pulau Tujuan")
        axes[1].tick_params(axis="x", rotation=25)
    plt.tight_layout()
    p3 = artifacts_dir / "error_breakdown_groups.png"
    fig.savefig(p3, dpi=140, bbox_inches="tight")
    plt.close(fig)
    saved["error_breakdown_groups"] = str(p3)

    # 4. SHAP Summary Plot
    if shap_values_2d is not None and X_shap_df is not None and len(X_shap_df) > 0:
        fig = plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values_2d,
            X_shap_df,
            plot_type="bar",
            max_display=15,
            show=False,
            rng=np.random.default_rng(42),
        )
        plt.title("SHAP Top Drivers (Model P50)")
        plt.tight_layout()
        p4 = artifacts_dir / "shap_summary_p50.png"
        fig.savefig(p4, dpi=140, bbox_inches="tight")
        plt.close(fig)
        saved["shap_summary_p50"] = str(p4)

    # 5. Partial Dependence Plot
    if pdp_estimator is not None and X_pdp_df is not None and len(top_pdp_cols) > 0:
        n_cols = len(top_pdp_cols)
        fig, axes = plt.subplots(1, n_cols, figsize=(4.2 * n_cols, 4.2))
        PartialDependenceDisplay.from_estimator(
            pdp_estimator,
            X_pdp_df,
            features=top_pdp_cols,
            grid_resolution=12,
            ax=axes,
        )
        plt.suptitle("Partial Dependence Plot (durasi_jam)")
        plt.tight_layout()
        p5 = artifacts_dir / "pdp_top_features.png"
        fig.savefig(p5, dpi=140, bbox_inches="tight")
        plt.close(fig)
        saved["pdp_top_features"] = str(p5)

    return saved


def compute_interpretability(
    best_model_bundle: dict[str, Any],
    seed: int = SEED,
    smoke_cfg: SmokeTestConfig | None = None,
    cv_summary: dict[str, Any] | None = None,
    model_results: dict[str, Any] | None = None,
    calib_summary: dict[str, Any] | None = None,
    oot_p90_cal: np.ndarray | None = None,
    breakdowns: dict[str, Any] | None = None,
    artifacts_dir: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Compute SHAP TreeExplainer feature importance & Partial Dependence on OOT validation."""
    is_smoke = bool(smoke_cfg is not None and smoke_cfg.enabled)
    oot_bundle = best_model_bundle["oot_bundle"]
    oot_views: PreprocessedViews = best_model_bundle["oot_views"]
    family = str(oot_bundle["family"])

    if oot_bundle["view"] == "hgbr":
        X_full = oot_views.hgbr_df
    elif oot_bundle["view"] == "native":
        X_full = oot_views.native_df
    else:
        X_full = pd.DataFrame(oot_views.linear)

    rng = np.random.default_rng(seed + 101)
    n_avail = len(X_full)
    n_shap = min(100 if is_smoke else 500, n_avail)
    n_pdp = min(150 if is_smoke else 800, n_avail)

    shap_idx = rng.choice(n_avail, size=n_shap, replace=False) if n_avail > n_shap else np.arange(n_avail)
    pdp_idx = rng.choice(n_avail, size=n_pdp, replace=False) if n_avail > n_pdp else np.arange(n_avail)
    X_shap = X_full.iloc[shap_idx]
    X_pdp = X_full.iloc[pdp_idx]

    shap_values_2d: np.ndarray | None = None
    shap_method = "shap_tree_explainer"
    m50 = oot_bundle["m50"]

    if family in ("LightGBM", "HistGradientBoosting", "XGBoost"):
        explainer = shap.TreeExplainer(m50)
        sv_raw = explainer.shap_values(X_shap)
        sv_arr = np.asarray(sv_raw, dtype=np.float64)
        if sv_arr.ndim == 3:
            # Multi-quantile XGBRegressor returns (n_samples, n_features, 2); index 0 is p50
            sv_arr = sv_arr[:, :, 0]
        shap_values_2d = sv_arr
    elif family == "Baseline_RidgeLogQuantile":
        shap_method = "linear_coef_fallback"
        coefs = np.abs(np.asarray(m50.coef_, dtype=np.float64))
        shap_values_2d = np.tile(coefs, (len(X_shap), 1))
    else:
        shap_method = "dummy_fallback"
        shap_values_2d = np.zeros((len(X_shap), X_shap.shape[1]), dtype=np.float64)

    mean_abs = np.mean(np.abs(shap_values_2d), axis=0)
    feat_names = [str(c) for c in X_shap.columns]
    ranked_pairs = sorted(
        zip(feat_names, mean_abs.tolist()),
        key=lambda item: item[1],
        reverse=True,
    )
    top_drivers = [
        {"feature": name, "mean_abs_shap": float(val)}
        for name, val in ranked_pairs[:15]
    ]

    # Select top continuous numeric features for Partial Dependence
    candidate_pdp_cols = [
        d["feature"]
        for d in top_drivers
        if d["feature"] in X_pdp.columns
        and pd.api.types.is_numeric_dtype(X_pdp[d["feature"]])
        and int(X_pdp[d["feature"]].nunique(dropna=True)) > 5
    ]
    for fallback_col in ("jarak_km", "antrean_hub_asal", "berat_kg", "alamat_skor"):
        if len(candidate_pdp_cols) >= 4:
            break
        if (
            fallback_col in X_pdp.columns
            and fallback_col not in candidate_pdp_cols
            and pd.api.types.is_numeric_dtype(X_pdp[fallback_col])
            and int(X_pdp[fallback_col].nunique(dropna=True)) > 3
        ):
            candidate_pdp_cols.append(fallback_col)
    top_pdp_cols = candidate_pdp_cols[:4]

    pdp_estimator = _LogExpPredictor(m50, mode="multi" if family == "XGBoost" else "single", col_idx=0)
    pdp_summary: list[dict[str, Any]] = []
    grid_res = 12 if is_smoke else 20
    if oot_bundle["view"] in ("hgbr", "native"):
        for col in top_pdp_cols:
            pd_res = partial_dependence(
                pdp_estimator,
                X_pdp,
                features=[col],
                grid_resolution=grid_res,
                kind="average",
            )
            grid_vals = np.asarray(pd_res["grid_values"][0], dtype=np.float64)
            avg_vals = np.asarray(pd_res["average"][0], dtype=np.float64)
            pdp_summary.append(
                {
                    "feature": str(col),
                    "grid_min": float(grid_vals.min()),
                    "grid_max": float(grid_vals.max()),
                    "pdp_min_hours": float(avg_vals.min()),
                    "pdp_max_hours": float(avg_vals.max()),
                    "pdp_range_hours": float(avg_vals.max() - avg_vals.min()),
                }
            )

    saved_artifacts: dict[str, str] = {}
    if (
        artifacts_dir is not None
        and cv_summary is not None
        and model_results is not None
        and calib_summary is not None
        and oot_p90_cal is not None
        and breakdowns is not None
    ):
        saved_artifacts = _save_diagnostic_artifacts(
            artifacts_dir=Path(artifacts_dir),
            cv_summary=cv_summary,
            model_results=model_results,
            calib_summary=calib_summary,
            oot_y=best_model_bundle["oot_y"],
            oot_p50=best_model_bundle["oot_p50_raw"],
            oot_p90_raw=best_model_bundle["oot_p90_raw"],
            oot_p90_cal=oot_p90_cal,
            breakdowns=breakdowns,
            shap_values_2d=shap_values_2d,
            X_shap_df=X_shap if oot_bundle["view"] in ("hgbr", "native") else None,
            pdp_estimator=pdp_estimator if oot_bundle["view"] in ("hgbr", "native") else None,
            X_pdp_df=X_pdp if oot_bundle["view"] in ("hgbr", "native") else None,
            top_pdp_cols=top_pdp_cols,
        )

    interp_dict = {
        "method": f"{shap_method}_and_partial_dependence",
        "explained_model": family,
        "shap_sample_n": int(len(X_shap)),
        "pdp_sample_n": int(len(X_pdp)),
        "shap_target_space": "log1p(durasi_jam)",
        "pdp_target_space": "durasi_jam_hours",
        "top_drivers": top_drivers,
        "pdp_summary": pdp_summary,
    }
    return interp_dict, saved_artifacts


def export_regresi_csv(
    ids: pd.Series | np.ndarray,
    p50: np.ndarray,
    p90: np.ndarray,
    output_csv: str | Path,
) -> Path:
    """Validate invariants and write ``prediksi_regresi.csv`` with exact header ``id_paket,p50,p90``."""
    p50_arr = np.asarray(p50, dtype=np.float64)
    p90_arr = np.asarray(p90, dtype=np.float64)
    if not (np.isfinite(p50_arr).all() and np.isfinite(p90_arr).all()):
        raise ValueError("Predictions contain NaN or infinite values.")
    p50_arr = np.clip(p50_arr, 1e-3, None)
    p90_arr = np.maximum(p90_arr, p50_arr)

    out_path = Path(output_csv).expanduser().resolve()
    ensure_dir(out_path.parent)
    out_df = pd.DataFrame(
        {
            "id_paket": pd.Series(ids).astype(str).to_numpy(),
            "p50": np.round(p50_arr, 4),
            "p90": np.round(p90_arr, 4),
        }
    )
    # Re-enforce after rounding so rounding never creates p90 < p50
    out_df["p50"] = np.maximum(out_df["p50"].to_numpy(dtype=np.float64), 1e-3)
    out_df["p90"] = np.maximum(
        out_df["p90"].to_numpy(dtype=np.float64),
        out_df["p50"].to_numpy(dtype=np.float64),
    )
    out_df.to_csv(out_path, index=False, encoding="utf-8")
    return out_path


def run_regresi_pipeline(
    data_dir: str | Path | None = None,
    output_csv: str | Path | None = None,
    artifacts_dir: str | Path | None = None,
    smoke_test: bool = False,
    sample_n: int | None = None,
    seed: int = SEED,
) -> dict[str, Any]:
    """Run the end-to-end Task 1 regression pipeline and return a JSON-safe summary dict."""
    t0 = time.perf_counter()
    reconfigure_stdio_utf8()
    set_global_seed(seed)

    smoke_cfg = get_smoke_config(smoke_test=smoke_test, sample_n=sample_n, seed=seed)
    resolved_data_dir = get_data_dir(data_dir, must_exist=True)
    resolved_output_csv = resolve_output_path(output_csv, DEFAULT_REGRESI_CSV.name)
    resolved_artifacts_dir = Path(artifacts_dir).expanduser().resolve() if artifacts_dir else None

    lib_versions = get_library_versions()
    LOGGER.info(
        "Starting Task 1 Regresi | seed=%s | smoke_test=%s | libs=%s",
        seed,
        smoke_cfg.enabled,
        format_library_versions(lib_versions),
    )

    LOGGER.info("[Step 1/6] Loading regresi_train.csv.gz & regresi_test.csv.gz...")
    train_raw, test_raw = load_regresi_datasets(resolved_data_dir, smoke_cfg=smoke_cfg)
    LOGGER.info(
        "[Step 2/6] Loaded train=%d rows, test=%d rows. Running data quality & leakage audit...",
        len(train_raw),
        len(test_raw),
    )
    train_clean, test_clean, audit_summary = audit_and_clean_regresi(
        train_raw, test_raw, seed=seed, smoke_cfg=smoke_cfg
    )
    cv_summary = evaluate_time_vs_random_cv(train_clean, seed=seed, smoke_cfg=smoke_cfg)
    model_results, best_model_bundle = train_and_tune_quantile_models(
        train_clean, seed=seed, smoke_cfg=smoke_cfg
    )

    oot_val_idx = best_model_bundle["oot_val_idx"]
    q_hat, oot_p90_cal, calib_summary = calibrate_conformal_intervals(
        y_cal=best_model_bundle["oot_y"],
        p50_cal=best_model_bundle["oot_p50_raw"],
        p90_cal_raw=best_model_bundle["oot_p90_raw"],
        tgl_cal=train_clean.iloc[oot_val_idx][DATE_COL].to_numpy(),
        alpha=0.90,
    )

    # Predict test set using full-train fitted model + conformal calibration
    full_prep: RegresiFeaturePreprocessor = best_model_bundle["full_preprocessor"]
    test_views = full_prep.transform(test_clean)
    test_p50_raw, test_p90_raw = _predict_with_bundle(best_model_bundle["full_bundle"], test_views)
    test_p50, test_p90 = apply_conformal_calibration(
        test_p50_raw,
        test_p90_raw,
        q_hat=q_hat,
        mode=str(calib_summary["selected_space"]),
    )

    LOGGER.info("[Step 6/6] Computing group error breakdowns, SHAP/PDP interpretability, and exporting CSV...")

    seen_mitra = set(train_clean["mitra_kurir"].astype(str).unique())
    breakdowns = compute_group_breakdowns(
        val_df_meta=train_clean.iloc[oot_val_idx],
        y_val=best_model_bundle["oot_y"],
        p50_val=best_model_bundle["oot_p50_raw"],
        p90_val=oot_p90_cal,
        test_df=test_clean,
        p50_test=test_p50,
        p90_test=test_p90,
        seen_mitra=seen_mitra,
    )

    interp_summary, saved_artifacts = compute_interpretability(
        best_model_bundle=best_model_bundle,
        seed=seed,
        smoke_cfg=smoke_cfg,
        cv_summary=cv_summary,
        model_results=model_results,
        calib_summary=calib_summary,
        oot_p90_cal=oot_p90_cal,
        breakdowns=breakdowns,
        artifacts_dir=resolved_artifacts_dir,
    )

    written_csv = export_regresi_csv(
        ids=test_clean[ID_COL],
        p50=test_p50,
        p90=test_p90,
        output_csv=resolved_output_csv,
    )

    elapsed = float(time.perf_counter() - t0)
    result: dict[str, Any] = {
        "metadata": {
            "task": "regresi",
            "seed": int(seed),
            "smoke_test": bool(smoke_cfg.enabled),
            "sample_n": smoke_cfg.sample_n if smoke_cfg.enabled else None,
            "data_dir": str(resolved_data_dir),
            "output_csv": str(written_csv),
            "artifacts_dir": str(resolved_artifacts_dir) if resolved_artifacts_dir else None,
            "library_versions": lib_versions,
            "n_train_used": int(len(train_clean)),
            "n_test_used": int(len(test_clean)),
            "elapsed_seconds": elapsed,
        },
        "audit": audit_summary,
        "cv_comparison": cv_summary,
        "models": model_results,
        "calibration": calib_summary,
        "breakdowns": breakdowns,
        "interpretability": interp_summary,
        "output": {
            "csv_path": str(written_csv),
            "n_rows": int(len(test_clean)),
            "columns": [ID_COL, "p50", "p90"],
            "p50_min": float(np.min(test_p50)),
            "p50_median": float(np.median(test_p50)),
            "p50_max": float(np.max(test_p50)),
            "p90_min": float(np.min(test_p90)),
            "p90_median": float(np.median(test_p90)),
            "p90_max": float(np.max(test_p90)),
            "invariant_p90_ge_p50": bool(np.all(test_p90 >= test_p50)),
            "invariant_p50_gt_0": bool(np.all(test_p50 > 0.0)),
        },
        "artifacts": saved_artifacts,
    }

    if resolved_artifacts_dir is not None:
        summary_json_path = save_json(result, resolved_artifacts_dir / "regresi_summary.json")
        result["artifacts"]["summary_json"] = str(summary_json_path)

    LOGGER.info(
        "Finished Task 1 Regresi in %.2fs | selected=%s | q_hat=%.4f | output=%s",
        elapsed,
        model_results["selected_model"],
        q_hat,
        written_csv,
    )
    return to_json_compatible(result)


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="UTS ML 2026 Task 1 Regresi Pipeline")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to dataset folder")
    parser.add_argument("--output-csv", type=str, default=None, help="Path for prediksi_regresi.csv")
    parser.add_argument("--artifacts-dir", type=str, default=None, help="Optional folder for diagnostic plots/JSON")
    parser.add_argument("--smoke-test", action="store_true", help="Run fast stratified subsample dry-run")
    parser.add_argument("--sample-n", type=int, default=None, help="Smoke-test train sample size")
    parser.add_argument("--seed", type=int, default=SEED, help="Global random seed")
    return parser


def main() -> None:
    parser = _build_cli_parser()
    args = parser.parse_args()
    run_regresi_pipeline(
        data_dir=args.data_dir,
        output_csv=args.output_csv,
        artifacts_dir=args.artifacts_dir,
        smoke_test=args.smoke_test,
        sample_n=args.sample_n,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
