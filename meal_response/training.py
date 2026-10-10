"""Fit ordinary least squares or 24 validation-tuned XGBoost regressors."""
from __future__ import annotations

import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from .evaluation import evaluate, patient_curve_rmse
from .features import META, MINUTES, TARGETS, summarize
from .weighting import loss_weights


def write_json(path, value):
    # Pandas/NumPy missing numeric values are represented as JSON null, not NaN.
    def clean(item):
        if isinstance(item, dict):
            return {str(k): clean(v) for k, v in item.items()}
        if isinstance(item, (list, tuple, np.ndarray)):
            return [clean(v) for v in item]
        if isinstance(item, (float, np.floating)):
            return float(item) if np.isfinite(item) else None
        if isinstance(item, np.integer):
            return int(item)
        if isinstance(item, Path):
            return str(item)
        return item
    Path(path).write_text(json.dumps(clean(value), indent=2, allow_nan=False), encoding="utf-8")


def feature_frame(frame, names):
    result = frame.reindex(columns=names).copy()
    for name in names:
        if name.startswith("med::"):
            result[name] = result[name].fillna(0)
    return result


def predict_bundle(bundle, frame):
    transformed = bundle["preprocessor"].transform(feature_frame(frame, bundle["feature_names"]))
    if bundle["family"] == "linear":
        return bundle["models"].predict(transformed)
    return np.column_stack([model.predict(transformed) for model in bundle["models"]])


def train_run(examples, manifest, dataset, family, features_mode, output, trials=30,
              max_trees=2000, bootstraps=300, seed=42, jobs=1, label_policy="supplied",
              high_confidence=0.9, missing_confidence_factor=0.5, large_rise_multiplier=1.25, rise_quantile=0.75):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    selected = examples if dataset == "combined" else examples.loc[examples.source.eq(dataset)]
    selected = selected.merge(manifest[["participant", "partition"]], on="participant", validate="many_to_one")
    partitions = {name: selected.loc[selected.partition.eq(name)].reset_index(drop=True)
                  for name in ["train", "validation", "test"]}
    if any(frame.empty for frame in partitions.values()):
        raise ValueError(f"{dataset}: need usable meals from distinct train, validation and test patients")
    train, validation, test = [partitions[name] for name in ["train", "validation", "test"]]
    names = [name for name in examples if name not in set(META + TARGETS + ["partition"])]
    # baseline and history_days are both metadata for reporting and causal predictors.
    names += ["baseline", "history_days"]
    names = sorted(set(names))
    # Medication identities are admitted only if observed in fitting patients.
    names = [name for name in names if not name.startswith("med::") or train[name].fillna(0).ne(0).any()]
    preprocessor = make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True), StandardScaler())
    x_train = preprocessor.fit_transform(feature_frame(train, names))
    x_validation = preprocessor.transform(feature_frame(validation, names))
    y_train, y_validation = train[TARGETS].to_numpy(), validation[TARGETS].to_numpy()
    weights, weight_detail, weight_policy = loss_weights(train, high_confidence, missing_confidence_factor,
                                                        large_rise_multiplier, rise_quantile, emphasize_rises=True)
    validation_weights, validation_detail, validation_policy = loss_weights(validation, high_confidence, missing_confidence_factor)
    weight_detail.to_csv(output / "training_weights.csv", index=False)
    validation_detail.to_csv(output / "validation_weights.csv", index=False)
    write_json(output / "weighting.json", {"training": weight_policy, "validation": validation_policy,
                                          "test_metrics": "Unchanged, patient-balanced; never weighted by confidence or observed rise"})
    tuning = []
    if family == "linear":
        models = LinearRegression().fit(x_train, y_train, sample_weight=weights)
        best = {"estimator": "ordinary_least_squares", "rank": int(models.rank_),
                "transformed_features": x_train.shape[1]}
    else:
        rng = np.random.default_rng(seed)
        best_score, models, best = np.inf, None, None
        for trial in range(trials):
            params = {"max_depth": int(rng.integers(2, 7)), "learning_rate": float(rng.uniform(0.02, 0.1)),
                      "subsample": float(rng.uniform(0.7, 1)), "colsample_bytree": float(rng.uniform(0.7, 1)),
                      "min_child_weight": int(rng.integers(1, 11)), "reg_lambda": float(rng.uniform(1, 10))}
            candidate = []
            for horizon in range(24):
                model = XGBRegressor(**params, objective="reg:squarederror", tree_method="hist",
                                     n_estimators=max_trees, early_stopping_rounds=50,
                                     random_state=seed, n_jobs=jobs)
                model.fit(x_train, y_train[:, horizon], sample_weight=weights,
                          eval_set=[(x_validation, y_validation[:, horizon])],
                          sample_weight_eval_set=[validation_weights], verbose=False)
                candidate.append(model)
            score = patient_curve_rmse(validation, y_validation, np.column_stack([m.predict(x_validation) for m in candidate]))
            tuning.append({"trial": trial + 1, "parameters": params, "validation_patient_curve_rmse": score,
                           "best_iterations": [m.best_iteration for m in candidate]})
            print(f"{dataset}/{family}: trial {trial + 1}/{trials}, validation RMSE={score:.3f}", flush=True)
            if score < best_score:
                best_score, models, best = score, candidate, params
    bundle = {"format_version": 1, "family": family, "dataset": dataset, "features_mode": features_mode,
              "feature_names": names, "preprocessor": preprocessor, "models": models,
              "minutes": MINUTES.tolist(), "history_limit_days": 7, "label_policy": label_policy,
              "weighting": weight_policy,
              "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__, "xgboost": xgboost.__version__}}
    joblib.dump(bundle, output / "model.joblib")
    prediction = predict_bundle(bundle, test)
    reloaded = predict_bundle(joblib.load(output / "model.joblib"), test)
    np.testing.assert_allclose(prediction, reloaded, rtol=0, atol=0)
    truth = test[TARGETS].to_numpy()
    persistence = np.repeat(test.baseline.to_numpy()[:, None], 24, axis=1)
    report = {"dataset": dataset, "model": family, "features": features_mode, "label_policy": label_policy,
              "protocol": "Patient-held-out; patient-balanced metrics; at most 7 days of prediction-time history",
              "partitions": {name: {"patients": frame.participant.nunique(), "meals": len(frame)} for name, frame in partitions.items()},
              "configuration": best, "trials": tuning,
              "weighting": weight_policy,
              "validation_patient_curve_rmse": patient_curve_rmse(validation, y_validation, predict_bundle(bundle, validation)),
              "test": evaluate(test, truth, prediction, bootstraps, seed),
              "persistence": evaluate(test, truth, persistence, bootstraps, seed),
              "test_flagged_interpolated_label_fraction": float(test.supplied_label_fraction.mean()),
              "limitations": ["CGMacros original-observation flags unavailable; dense supplied curves may contain interpolation.",
                              "T2DM labels include flagged interpolation unless observed-only policy selected.",
                              "Unlogged food or medications cannot be ruled out.",
                              "Actual Amount Consumed is excluded from meal-time inputs; planned meal may differ from intake."]}
    write_json(output / "metrics.json", report)
    write_json(output / "feature_schema.json", {"ordered_features": names, "versions": bundle["versions"],
                                              "units": {"glucose": "mg/dL", "macros": "g", "calories": "kcal", "time": "minutes", "iauc": "mg/dL*minutes"}})
    write_json(output / "config.json", {"dataset": dataset, "model": family, "features": features_mode,
                                      "trials": trials, "max_trees": max_trees, "seed": seed, "bootstraps": bootstraps,
                                      "jobs": jobs, "label_policy": label_policy,
                                      "high_confidence": high_confidence, "missing_confidence_factor": missing_confidence_factor,
                                      "large_rise_multiplier": large_rise_multiplier, "rise_quantile": rise_quantile})
    results = test.reindex(columns=META).copy()
    for i, target in enumerate(TARGETS):
        results[f"true_{target}"] = truth[:, i]
        results[f"predicted_{target}"] = prediction[:, i]
        results[f"persistence_{target}"] = persistence[:, i]
    for prefix, values in [("true", truth), ("predicted", prediction), ("persistence", persistence)]:
        for name, column in summarize(values, test.baseline).items():
            results[f"{prefix}_{name}"] = column.to_numpy()
    results.to_csv(output / "test_predictions.csv", index=False)
    plot_results(output, test, truth, prediction, dataset, family)
    return report


def plot_results(output, test, truth, prediction, dataset, family):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for label, curve in [("Model", prediction), ("Persistence", np.repeat(test.baseline.to_numpy()[:, None], 24, axis=1))]:
        mae = pd.DataFrame(np.abs(truth - curve)).groupby(test.participant.to_numpy()).mean().mean().to_numpy()
        axes[0].plot(MINUTES, mae, label=label)
    axes[0].set(xlabel="Minutes after meal", ylabel="Patient-balanced MAE (mg/dL)")
    axes[0].legend()
    for i in range(min(4, len(test))):
        color = plt.get_cmap("tab10")(i)
        axes[1].plot(MINUTES, truth[i], color=color, alpha=0.7, label=f"Meal {i + 1} actual")
        axes[1].plot(MINUTES, prediction[i], color=color, linestyle="--", alpha=0.7, label=f"Meal {i + 1} predicted")
    axes[1].set(xlabel="Minutes after meal", ylabel="Glucose (mg/dL)")
    axes[1].legend(fontsize=7)
    fig.suptitle(f"{dataset}: {family} — held-out patients")
    fig.tight_layout()
    fig.savefig(output / "evaluation.png", dpi=150)
    plt.close(fig)
