"""Feature engineering layer for Quant 2.0.

Exports
-------
compute_technical_features : callable
    ~40 OHLCV-derived technical indicators.
compute_time_features : callable
    ~10 sin/cos-encoded calendar features.
compute_information_geometry_features : callable
    7 information-geometry features (Fisher info, Renyi entropy, KL rate).
BayesianChangePointDetector : class
    Online BOCPD with Normal-Inverse-Gamma prior.
FeaturePipeline : class
    Orchestrates all features into normalised numpy arrays.
"""

from quant.features.changepoint import BayesianChangePointDetector
from quant.features.information_geometry import compute_information_geometry_features
from quant.features.pipeline import FeaturePipeline
from quant.features.technical import compute_technical_features
from quant.features.time_features import compute_time_features

__all__ = [
    "compute_technical_features",
    "compute_time_features",
    "compute_information_geometry_features",
    "BayesianChangePointDetector",
    "FeaturePipeline",
]
