"""Confidence loss weights; optional train-only extra emphasis on large rises."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluation import training_weights
from .features import TARGETS


def confidence_factors(frame, high_confidence=0.9, missing_t2dm_factor=0.5):
    if not 0 < high_confidence <= 1 or not 0 <= missing_t2dm_factor <= 1:
        raise ValueError("Confidence threshold must be in (0,1]; missing-confidence factor in [0,1]")
    confidence = pd.to_numeric(frame.get("macro_confidence", pd.Series(np.nan, index=frame.index)), errors="raise")
    known = confidence.notna()
    if not np.isfinite(confidence[known]).all() or not confidence[known].between(0, 1).all():
        raise ValueError("Macro confidence must be a finite value in [0,1], or missing")
    fallback = np.where(frame.source.eq("t2dm"), missing_t2dm_factor, 1.0)
    factors = np.where(known, np.minimum(confidence.to_numpy() / high_confidence, 1.0), fallback)
    return factors


def loss_weights(frame, high_confidence=0.9, missing_t2dm_factor=0.5, large_rise_multiplier=1.0,
                 rise_quantile=0.75, emphasize_rises=False):
    """Outcome-based weighting is allowed only for fitting, never feature inputs."""
    if not np.isfinite(large_rise_multiplier) or large_rise_multiplier < 1 or not 0 < rise_quantile < 1:
        raise ValueError("Large-rise multiplier must be finite and >=1; quantile in (0,1)")
    base = training_weights(frame)
    confidence = confidence_factors(frame, high_confidence, missing_t2dm_factor)
    boost = np.ones(len(frame))
    thresholds = {}
    rise = frame[TARGETS].max(axis=1).to_numpy() - frame.baseline.to_numpy()
    if emphasize_rises and large_rise_multiplier > 1:
        for source in sorted(frame.source.unique()):
            mask = frame.source.eq(source).to_numpy()
            threshold = float(np.quantile(rise[mask], rise_quantile))
            thresholds[source] = threshold
            boost[mask & (rise > threshold) & (rise > 0)] = large_rise_multiplier
    raw = base * confidence * boost
    if raw.sum() <= 0:
        raise ValueError("No positive loss weights after macro-confidence weighting")
    weights = raw * len(raw) / raw.sum()
    confidence_values = pd.to_numeric(frame.get("macro_confidence", pd.Series(np.nan, index=frame.index)))
    detail = frame[["participant", "source", "meal_id", "time"]].copy()
    detail["macro_confidence"] = confidence_values
    detail["base_weight"] = base
    detail["confidence_factor"] = confidence
    detail["observed_peak_rise"] = rise
    detail["large_rise_factor"] = boost
    detail["loss_weight"] = weights
    detail["confidence_origin"] = np.where(confidence_values.notna(), "recorded", np.where(frame.source.eq("cg_macros"), "unscored_normal_weight", "missing_reduced_weight"))
    policy = {"high_confidence_threshold": high_confidence, "missing_t2dm_factor": missing_t2dm_factor,
              "unscored_cg_macros_factor": 1.0, "large_rise_multiplier": large_rise_multiplier,
              "rise_quantile": rise_quantile, "large_rise_thresholds_mg_dl": thresholds,
              "confidence_downweighted_meals": int((confidence < 1).sum()),
              "large_rise_upweighted_meals": int((boost > 1).sum()),
              "confidence_missing_meals": int(confidence_values.isna().sum()),
              "normalization": "Base source/patient weights times confidence and optional rise factor; one final mean-1 rescaling",
              "rise_emphasis_applied": emphasize_rises and large_rise_multiplier > 1}
    return weights, detail, policy
