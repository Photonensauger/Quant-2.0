"""Technical indicator computation using only pandas and numpy.

Produces ~40 features covering trend, momentum, volatility, volume, and
price-channel families.  The first ``warmup_period`` rows (default 50) will
contain NaN values that must be handled downstream.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


# ── helpers ────────────────────────────────────────────────────────────────

def _ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=span, adjust=False).mean()


def _sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(window=window).mean()


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """True Range (used by ATR, ADX, etc.)."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


# ── public API ─────────────────────────────────────────────────────────────

def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute ~40 technical indicator columns from an OHLCV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns ``open, high, low, close, volume`` (case-
        insensitive).  The index should be sorted chronologically.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with ~40 additional indicator columns appended.
        The first ~50 rows will contain NaN due to lookback requirements.
    """
    # Normalise column names to lowercase
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {missing}")

    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]

    features = pd.DataFrame(index=df.index)

    # ── 1. Log returns ────────────────────────────────────────────────────
    features["log_return"] = np.log(c / c.shift(1))

    # ── 2-3. SMA (20, 50) distance ───────────────────────────────────────
    sma20 = _sma(c, 20)
    sma50 = _sma(c, 50)
    features["sma20_dist"] = (c - sma20) / (sma20 + 1e-8)
    features["sma50_dist"] = (c - sma50) / (sma50 + 1e-8)

    # ── 4-5. EMA (12, 26) distance ───────────────────────────────────────
    ema12 = _ema(c, 12)
    ema26 = _ema(c, 26)
    features["ema12_dist"] = (c - ema12) / (ema12 + 1e-8)
    features["ema26_dist"] = (c - ema26) / (ema26 + 1e-8)

    # ── 6-8. MACD, signal, histogram ──────────────────────────────────────
    macd_line = ema12 - ema26
    macd_signal = _ema(macd_line, 9)
    features["macd"] = macd_line
    features["macd_signal"] = macd_signal
    features["macd_hist"] = macd_line - macd_signal

    # ── 9-10. RSI (14, 28) ────────────────────────────────────────────────
    for period in (14, 28):
        delta = c.diff()
        gain = delta.clip(lower=0)
        loss = (-delta.clip(upper=0))
        avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
        avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-8)
        features[f"rsi_{period}"] = 100.0 - 100.0 / (1.0 + rs)

    # ── 11-13. Bollinger Bands (20, 2σ) ──────────────────────────────────
    bb_mid = sma20
    bb_std = c.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    features["bb_width"] = (bb_upper - bb_lower) / (bb_mid + 1e-8)
    features["bb_pct"] = (c - bb_lower) / (bb_upper - bb_lower + 1e-8)
    features["bb_mid_dist"] = (c - bb_mid) / (bb_mid + 1e-8)

    # ── 14-15. ATR (14, 28) ──────────────────────────────────────────────
    tr = _true_range(h, l, c)
    for period in (14, 28):
        atr = tr.ewm(span=period, adjust=False).mean()
        features[f"atr_{period}"] = atr / (c + 1e-8)  # normalised

    # ── 16. OBV (on-balance volume) ──────────────────────────────────────
    obv_sign = np.sign(c.diff()).fillna(0)
    features["obv"] = (obv_sign * v).cumsum()
    # normalise OBV by rolling z-score to keep it bounded
    obv_mean = features["obv"].rolling(50).mean()
    obv_std = features["obv"].rolling(50).std()
    features["obv_zscore"] = (features["obv"] - obv_mean) / (obv_std + 1e-8)
    features.drop(columns="obv", inplace=True)

    # ── 17-19. ADX / +DI / -DI (14) ──────────────────────────────────────
    plus_dm = (h - h.shift(1)).clip(lower=0)
    minus_dm = (l.shift(1) - l).clip(lower=0)
    # zero out whichever is smaller
    plus_dm[plus_dm < minus_dm] = 0
    minus_dm[minus_dm < plus_dm] = 0

    atr14 = tr.ewm(span=14, adjust=False).mean()
    plus_di = 100 * _ema(plus_dm, 14) / (atr14 + 1e-8)
    minus_di = 100 * _ema(minus_dm, 14) / (atr14 + 1e-8)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-8)
    adx = _ema(dx, 14)
    features["adx"] = adx
    features["plus_di"] = plus_di
    features["minus_di"] = minus_di

    # ── 20-22. Stochastic %K, %D, slow %D (14, 3) ────────────────────────
    low14 = l.rolling(14).min()
    high14 = h.rolling(14).max()
    stoch_k = 100 * (c - low14) / (high14 - low14 + 1e-8)
    stoch_d = _sma(stoch_k, 3)
    stoch_slow_d = _sma(stoch_d, 3)
    features["stoch_k"] = stoch_k
    features["stoch_d"] = stoch_d
    features["stoch_slow_d"] = stoch_slow_d

    # ── 23. Williams %R (14) ─────────────────────────────────────────────
    features["williams_r"] = -100 * (high14 - c) / (high14 - low14 + 1e-8)

    # ── 24. CCI (20) ─────────────────────────────────────────────────────
    tp = (h + l + c) / 3.0
    tp_sma = _sma(tp, 20)
    tp_mad = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    features["cci"] = (tp - tp_sma) / (0.015 * tp_mad + 1e-8)

    # ── 25. MFI (14) ─────────────────────────────────────────────────────
    mf_raw = tp * v
    mf_pos = mf_raw.where(tp > tp.shift(1), 0.0)
    mf_neg = mf_raw.where(tp < tp.shift(1), 0.0)
    mf_ratio = mf_pos.rolling(14).sum() / (mf_neg.rolling(14).sum() + 1e-8)
    features["mfi"] = 100.0 - 100.0 / (1.0 + mf_ratio)

    # ── 26-28. ROC (5, 10, 20) ───────────────────────────────────────────
    for period in (5, 10, 20):
        features[f"roc_{period}"] = c.pct_change(period)

    # ── 29-30. Momentum (10, 20) ─────────────────────────────────────────
    for period in (10, 20):
        features[f"momentum_{period}"] = c - c.shift(period)

    # ── 31-32. Volatility (10, 20) realised (annualised-ish) ─────────────
    log_ret = np.log(c / c.shift(1))
    for period in (10, 20):
        features[f"volatility_{period}"] = log_ret.rolling(period).std()

    # ── 33. VWAP distance ────────────────────────────────────────────────
    cum_vwap = (tp * v).rolling(20).sum() / (v.rolling(20).sum() + 1e-8)
    features["vwap_dist"] = (c - cum_vwap) / (cum_vwap + 1e-8)

    # ── 34-35. Volume features ───────────────────────────────────────────
    vol_sma20 = _sma(v, 20)
    features["volume_ratio"] = v / (vol_sma20 + 1e-8)
    features["volume_change"] = v.pct_change()

    # ── 36-37. Donchian Channel (20) ─────────────────────────────────────
    dc_high = h.rolling(20).max()
    dc_low = l.rolling(20).min()
    features["donchian_width"] = (dc_high - dc_low) / (c + 1e-8)
    features["donchian_pct"] = (c - dc_low) / (dc_high - dc_low + 1e-8)

    # ── 38. Keltner Channel width (20, 1.5 ATR) ─────────────────────────
    kc_mid = _ema(c, 20)
    kc_atr = tr.ewm(span=20, adjust=False).mean()
    kc_upper = kc_mid + 1.5 * kc_atr
    kc_lower = kc_mid - 1.5 * kc_atr
    features["keltner_width"] = (kc_upper - kc_lower) / (kc_mid + 1e-8)

    # ── 39. Chaikin Money Flow (20) ──────────────────────────────────────
    clv = ((c - l) - (h - c)) / (h - l + 1e-8)
    features["cmf"] = (clv * v).rolling(20).sum() / (v.rolling(20).sum() + 1e-8)

    # ── 40. Average Directional Movement Rating (ADXR) ───────────────────
    features["adxr"] = (adx + adx.shift(14)) / 2.0

    # ── 41. Price acceleration (second-order momentum) ────────────────────
    features["price_accel"] = log_ret.diff()

    n_features = len(features.columns)
    logger.debug(
        "Computed {} technical features; first valid row ~ {}",
        n_features,
        features.first_valid_index(),
    )

    return pd.concat([df, features], axis=1)
