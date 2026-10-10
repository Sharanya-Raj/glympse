"""Patient-balanced metrics, bootstrap intervals, and reproducible group splits."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import summarize


def split_manifest(examples, seed=42):
    patients = examples[["participant", "source", "cohort"]].drop_duplicates().sort_values("participant")
    rng = np.random.default_rng(seed)
    records = []
    for (source, cohort), group in patients.groupby(["source", "cohort"], sort=True):
        ids = group.participant.to_numpy().copy()
        rng.shuffle(ids)
        # Sparse cohorts stay in training; source-level validation/test counts
        # are checked below rather than splitting rows from the same person.
        count = max(1, int(round(len(ids) * 0.15))) if len(ids) >= 3 else 0
        for i, participant in enumerate(ids):
            partition = "test" if i < count else "validation" if i < 2 * count else "train"
            records.append({"participant": participant, "source": source, "cohort": cohort, "partition": partition})
    return pd.DataFrame(records, columns=["participant", "source", "cohort", "partition"])


def training_weights(frame):
    """Equal source weight, equal patient weight within each source."""
    counts = frame.groupby("participant").participant.transform("size").to_numpy()
    patient_counts = frame.groupby("source").participant.transform("nunique").to_numpy()
    weights = 1 / (counts * patient_counts)
    return weights * len(weights) / weights.sum()


def patient_curve_rmse(frame, truth, prediction):
    squared = ((np.asarray(truth) - np.asarray(prediction)) ** 2).mean(axis=1)
    return float(np.sqrt(pd.Series(squared).groupby(frame.participant.to_numpy()).mean()).mean())


def metric_values(truth, prediction, weights):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    error = prediction - truth
    mean = np.average(truth, weights=weights)
    variance = np.average((truth - mean) ** 2, weights=weights)
    return {"mae": float(np.average(np.abs(error), weights=weights)),
            "rmse": float(np.sqrt(np.average(error ** 2, weights=weights))),
            "r2": float(1 - np.average(error ** 2, weights=weights) / variance) if variance > 0 else None}


def score_group(frame, truth_curve, predicted_curve, bootstraps=300, seed=42):
    ids = frame.participant.to_numpy()
    unique, index, counts = np.unique(ids, return_inverse=True, return_counts=True)
    weights = 1 / counts[index]
    truth = summarize(truth_curve, frame.baseline)
    predicted = summarize(predicted_curve, frame.baseline)
    metrics = {name: metric_values(truth[name], predicted[name], weights) for name in truth}
    metrics["curve"] = {"rmse": patient_curve_rmse(frame, truth_curve, predicted_curve)}
    rng = np.random.default_rng(seed)
    samples = {name: {key: [] for key in item} for name, item in metrics.items()}
    for _ in range(bootstraps):
        multiplicity = np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique))
        boot_weights = weights * multiplicity[index]
        for name in truth:
            for key, value in metric_values(truth[name], predicted[name], boot_weights).items():
                if value is not None and np.isfinite(value):
                    samples[name][key].append(value)
        # Curve RMSE is a mean of each patient's RMSE, not a pooled-row RMSE.
        per_patient = np.sqrt(pd.Series(((truth_curve - predicted_curve) ** 2).mean(axis=1)).groupby(ids).mean()).reindex(unique).to_numpy()
        samples["curve"]["rmse"].append(float(np.average(per_patient, weights=multiplicity)))
    for name in metrics:
        metrics[name]["ci95"] = {key: np.quantile(values, [0.025, 0.975]).tolist() if values else None
                                 for key, values in samples[name].items()}
    return {"patients": len(unique), "meals": len(frame), "metrics": metrics}


def evaluate(frame, truth, prediction, bootstraps=300, seed=42):
    frame = frame.reset_index(drop=True)
    groups = {"overall": np.ones(len(frame), dtype=bool)}
    for column in ["source", "cohort"]:
        for value in sorted(frame[column].unique()):
            groups[f"{column}:{value}"] = frame[column].eq(value).to_numpy()
    days = frame.history_days
    groups.update({"history:<1d": (days < 1).to_numpy(),
                   "history:1-3d": ((days >= 1) & (days < 3)).to_numpy(),
                   "history:3-7d": (days >= 3).to_numpy()})
    return {name: score_group(frame.loc[mask], truth[mask], prediction[mask], bootstraps, seed)
            for name, mask in groups.items() if mask.any()}
