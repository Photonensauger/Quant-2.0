"""Feature engineering pipeline that orchestrates technical, time, and
changepoint features into train-ready numpy arrays.

Responsibilities:

* Call ``compute_technical_features`` and ``compute_time_features``.
* Run the ``BayesianChangePointDetector`` over log-returns.
* Drop all-NaN columns, apply a pairwise-correlation filter, and
  normalise features with a rolling z-score.
* Produce aligned ``(features, targets)`` arrays with NaN rows removed.
* Expose ``get_state`` / ``load_state`` so the pipeline can be
  serialised and resumed on new data.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from quant.config.settings import FeatureConfig
from quant.features.changepoint import BayesianChangePointDetector
from quant.features.technical import compute_technical_features
from quant.features.time_features import compute_time_features


class FeaturePipeline:
    """End-to-end feature engineering pipeline.

    Parameters
    ----------
    config : FeatureConfig, optional
        Pipeline hyper-parameters.  Defaults to ``FeatureConfig()``.
    """

    def __init__(self, config: FeatureConfig | None = None) -> None:
        self.cfg = config or FeatureConfig()

        # State populated during fit_transform, reused in transform
        self._feature_names: list[str] = []
        self._dropped_features: list[str] = []
        self._rolling_mean: pd.Series | None = None  # per-feature
        self._rolling_std: pd.Series | None = None  # per-feature
        self._fitted: bool = False

        # Changepoint detector (kept as part of pipeline state)
        self._cpd = BayesianChangePointDetector(self.cfg)

        # Columns that belong to the original OHLCV input (not features)
        self._ohlcv_cols = {"open", "high", "low", "close", "volume"}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Compute all features, fit normalisation stats, return arrays.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame with DatetimeIndex.

        Returns
        -------
        features : np.ndarray, shape ``[T', n_features]``
            Z-score normalised feature matrix (NaN warmup rows removed).
        targets : np.ndarray, shape ``[T']``
            Forward log-returns (un-normalised).  ``T' < T`` because the
            first warmup rows and any remaining NaN rows are dropped.
        """
        logger.info("FeaturePipeline.fit_transform  |  input rows: {}", len(df))

        # 1. Compute raw features
        feat_df = self._build_raw_features(df)

        # 2. Build target *before* dropping columns so index stays aligned
        targets = self._compute_targets(df)

        # 3. Separate feature columns from pass-through OHLCV
        feature_cols = [
            c for c in feat_df.columns if c.lower() not in self._ohlcv_cols
        ]
        feat_only = feat_df[feature_cols]

        # 4. Drop all-NaN columns
        feat_only, dropped_nan = self._drop_all_nan_columns(feat_only)

        # 5. Correlation filter (fit)
        feat_only, dropped_corr = self._correlation_filter(feat_only)
        self._dropped_features = dropped_nan + dropped_corr

        # 6. Store feature names
        self._feature_names = list(feat_only.columns)

        # 7. Rolling z-score normalisation (fit: compute stats, then apply)
        feat_norm = self._rolling_zscore_fit(feat_only)

        # 8. Align and drop NaN rows
        features_arr, targets_arr = self._align_and_clean(feat_norm, targets)

        self._fitted = True
        logger.info(
            "fit_transform complete  |  features: {} x {}  |  dropped: {}",
            features_arr.shape[0],
            features_arr.shape[1],
            len(self._dropped_features),
        )
        return features_arr, targets_arr

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Apply the *already-fitted* pipeline to new data.

        Parameters
        ----------
        df : pd.DataFrame
            New OHLCV data (may be as short as a single bar).

        Returns
        -------
        features : np.ndarray, shape ``[T', n_features]``
        targets : np.ndarray, shape ``[T']``
        """
        if not self._fitted:
            raise RuntimeError(
                "Pipeline has not been fitted.  Call fit_transform first."
            )

        logger.debug("FeaturePipeline.transform  |  input rows: {}", len(df))

        # 1. Build raw features
        feat_df = self._build_raw_features(df)

        # 2. Targets
        targets = self._compute_targets(df)

        # 3. Keep only fitted feature names (silently ignore new ones)
        available = [c for c in self._feature_names if c in feat_df.columns]
        missing = set(self._feature_names) - set(available)
        if missing:
            logger.warning("Missing {} features in transform input", len(missing))

        feat_only = feat_df[available]

        # Re-add any missing columns as NaN so shape is consistent
        for col in self._feature_names:
            if col not in feat_only.columns:
                feat_only[col] = np.nan
        feat_only = feat_only[self._feature_names]

        # 4. Rolling z-score normalisation (transform: use saved stats)
        feat_norm = self._rolling_zscore_transform(feat_only)

        # 5. Align and clean
        features_arr, targets_arr = self._align_and_clean(feat_norm, targets)

        logger.debug(
            "transform complete  |  features: {} x {}",
            features_arr.shape[0],
            features_arr.shape[1] if features_arr.ndim == 2 else 0,
        )
        return features_arr, targets_arr

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def get_state(self) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot of the pipeline state."""
        return {
            "feature_names": self._feature_names,
            "dropped_features": self._dropped_features,
            "rolling_mean": (
                self._rolling_mean.to_dict() if self._rolling_mean is not None else None
            ),
            "rolling_std": (
                self._rolling_std.to_dict() if self._rolling_std is not None else None
            ),
            "fitted": self._fitted,
            "cpd_state": self._cpd.get_state(),
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Restore pipeline state from a previous :meth:`get_state` call."""
        self._feature_names = state["feature_names"]
        self._dropped_features = state["dropped_features"]
        self._rolling_mean = (
            pd.Series(state["rolling_mean"]) if state["rolling_mean"] is not None else None
        )
        self._rolling_std = (
            pd.Series(state["rolling_std"]) if state["rolling_std"] is not None else None
        )
        self._fitted = state["fitted"]
        self._cpd.load_state(state["cpd_state"])
        logger.debug("FeaturePipeline state restored ({} features)", len(self._feature_names))

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def feature_names(self) -> list[str]:
        """Names of the features that survived filtering."""
        return list(self._feature_names)

    @property
    def n_features(self) -> int:
        """Number of features after filtering."""
        return len(self._feature_names)

    @property
    def dropped_features(self) -> list[str]:
        """Feature names dropped by NaN / correlation filters."""
        return list(self._dropped_features)

    # ------------------------------------------------------------------
    # Internal: feature construction
    # ------------------------------------------------------------------

    def _build_raw_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Chain all feature generators and append changepoint columns."""
        # Technical
        feat_df = compute_technical_features(df)

        # Time
        feat_df = compute_time_features(feat_df)

        # Changepoint features (run over log-returns)
        feat_df = self._add_changepoint_features(feat_df)

        return feat_df

    def _add_changepoint_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run BOCPD on log-returns and append cp_score / cp_severity."""
        col_lower = {c.lower(): c for c in df.columns}
        log_ret_col = col_lower.get("log_return")

        if log_ret_col is None:
            logger.warning("log_return column not found; skipping changepoint features")
            df["cp_score"] = 0.0
            df["cp_severity"] = 0.0
            return df

        log_rets = df[log_ret_col].fillna(0.0).values
        scores = np.zeros(len(log_rets))
        severities = np.zeros(len(log_rets))

        for i, lr in enumerate(log_rets):
            s, sev = self._cpd.update(float(lr))
            scores[i] = s
            severities[i] = sev

        df = df.copy()
        df["cp_score"] = scores
        df["cp_severity"] = severities
        return df

    def _compute_targets(self, df: pd.DataFrame) -> pd.Series:
        """Forward log-returns over ``forecast_horizon`` bars."""
        df_lower = df.copy()
        df_lower.columns = [c.lower() for c in df_lower.columns]
        close = df_lower["close"]
        horizon = self.cfg.forecast_horizon
        targets = np.log(close.shift(-horizon) / close)
        targets.name = "target"
        return targets

    # ------------------------------------------------------------------
    # Internal: cleaning / filtering
    # ------------------------------------------------------------------

    @staticmethod
    def _drop_all_nan_columns(
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, list[str]]:
        """Drop columns that are *entirely* NaN and log a warning."""
        all_nan_mask = df.isna().all()
        dropped = list(df.columns[all_nan_mask])
        if dropped:
            logger.warning("Dropping {} all-NaN columns: {}", len(dropped), dropped)
        return df.loc[:, ~all_nan_mask], dropped

    def _correlation_filter(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[str]]:
        """Remove features whose pairwise |correlation| exceeds threshold.

        For each highly-correlated pair the *second* column (in order) is
        dropped, so the process is stable across runs.
        """
        threshold = self.cfg.correlation_threshold
        # Use only non-NaN rows for correlation computation
        corr = df.dropna().corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))

        to_drop: list[str] = []
        for col in upper.columns:
            if col in to_drop:
                continue
            correlated = upper.index[upper[col] > threshold].tolist()
            to_drop.extend([c for c in correlated if c not in to_drop])

        if to_drop:
            logger.info(
                "Correlation filter dropping {} features (threshold={}): {}",
                len(to_drop),
                threshold,
                to_drop,
            )
        return df.drop(columns=to_drop), to_drop

    # ------------------------------------------------------------------
    # Internal: normalisation
    # ------------------------------------------------------------------

    def _rolling_zscore_fit(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute and store rolling mean/std, then normalise."""
        window = self.cfg.rolling_window
        eps = self.cfg.epsilon

        roll_mean = df.rolling(window=window, min_periods=1).mean()
        roll_std = df.rolling(window=window, min_periods=1).std().fillna(0.0)

        # Store the *last* row of rolling stats for transform re-use
        self._rolling_mean = roll_mean.iloc[-1].copy()
        self._rolling_std = roll_std.iloc[-1].copy()

        normed = (df - roll_mean) / (roll_std + eps)
        return normed

    def _rolling_zscore_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalise using saved rolling stats for short / single-bar inputs.

        For longer inputs we still compute a rolling window; the saved stats
        serve as the seed for the first ``rolling_window - 1`` rows.
        """
        eps = self.cfg.epsilon

        if len(df) >= self.cfg.rolling_window:
            # Enough data: compute fresh rolling stats
            roll_mean = df.rolling(
                window=self.cfg.rolling_window, min_periods=1
            ).mean()
            roll_std = (
                df.rolling(window=self.cfg.rolling_window, min_periods=1)
                .std()
                .fillna(0.0)
            )
            # Update saved stats with last row
            self._rolling_mean = roll_mean.iloc[-1].copy()
            self._rolling_std = roll_std.iloc[-1].copy()
            normed = (df - roll_mean) / (roll_std + eps)
        else:
            # Short input: use saved stats directly
            mean = self._rolling_mean
            std = self._rolling_std
            normed = (df - mean) / (std + eps)

        return normed

    # ------------------------------------------------------------------
    # Internal: alignment
    # ------------------------------------------------------------------

    @staticmethod
    def _align_and_clean(
        features_df: pd.DataFrame, targets: pd.Series
    ) -> tuple[np.ndarray, np.ndarray]:
        """Drop rows where *any* feature or the target is NaN, return arrays.

        Returns empty arrays (shape ``[0, n_features]`` and ``[0]``) when no
        valid rows remain so that the caller can simply check ``.shape[0]``.
        """
        combined = features_df.copy()
        combined["__target__"] = targets

        valid_mask = combined.notna().all(axis=1)
        combined = combined.loc[valid_mask]

        if combined.empty:
            n_feat = features_df.shape[1]
            logger.warning(
                "No valid rows after NaN removal (features={}, target rows={})",
                features_df.shape[0],
                targets.shape[0],
            )
            return np.empty((0, n_feat), dtype=np.float64), np.empty(0, dtype=np.float64)

        targets_arr = combined["__target__"].values.astype(np.float64)
        features_arr = combined.drop(columns="__target__").values.astype(np.float64)

        return features_arr, targets_arr
