"""Build an executed backend/meal_response comparison without retraining on test data.

Run: uv run python -m meal_response.comparison_notebook
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

import joblib
import numpy as np
import pandas as pd
from xgboost import DMatrix

from backend.features import FEATURES as BACKEND_FEATURES
from backend.model import make_regressor
from meal_response.data import fingerprint
from meal_response.features import TARGETS, summarize
from meal_response.training import feature_frame, predict_bundle, train_run, write_json
from meal_response.verify_artifacts import verify

ROOT = Path(__file__).resolve().parents[1]
ENDPOINTS = ["glucose_30", "glucose_60", "glucose_90", "glucose_120"]
COMMON_TARGETS = ENDPOINTS + ["peak", "time_to_peak", "peak_rise"]
LABELS = {"backend": "Backend adapted", "linear": "Meal response linear", "xgboost": "Meal response XGBoost"}


def backend_inputs(frame):
    """Explicit causal adapter; activity is unavailable in the shared core table."""
    result = pd.DataFrame(index=frame.index)
    for name in BACKEND_FEATURES[:5]:
        result[name] = frame[name.lower()]
    for name in ["baseline", "hour_sin", "hour_cos"]:
        result[name] = frame[name]
    result["premeal_slope"] = (frame.lag_0 - frame.lag_30) / 30
    result["history_mean"] = frame.history_7d_mean
    result["history_std"] = frame.history_7d_std
    result["recent_hr"] = np.nan
    result["recent_activity"] = np.nan
    return result[BACKEND_FEATURES]


def prepare(source_run, output):
    output.mkdir(parents=True, exist_ok=False)
    print("Verifying saved models against current raw inputs and feature construction", flush=True)
    verified = verify(source_run, ROOT / "data/T2DM_5min_with_macros.csv", ROOT / "data/cg_macros")
    for filename in ["core_examples.csv.gz", "participant_splits.csv", "data_metadata.json"]:
        shutil.copy2(source_run / filename, output / filename)
    examples = pd.read_csv(output / "core_examples.csv.gz", parse_dates=["time"])
    manifest = pd.read_csv(output / "participant_splits.csv")
    selected = examples.merge(manifest[["participant", "partition"]], on="participant", validate="many_to_one")
    for mode in ["cg_macros", "t2dm", "combined"]:
        frame = selected if mode == "combined" else selected.loc[selected.source.eq(mode)]
        train = frame.loc[frame.partition.eq("train")]
        test = frame.loc[frame.partition.eq("test")].reset_index(drop=True)
        truth = summarize(train[TARGETS].to_numpy(), train.baseline)[COMMON_TARGETS]
        model = make_regressor(multioutput=True)
        model.fit(backend_inputs(train), truth)
        directory = output / f"{mode}_backend"
        directory.mkdir()
        joblib.dump(model, directory / "model.joblib")
        prediction = model.predict(backend_inputs(test))
        np.testing.assert_allclose(prediction, joblib.load(directory / "model.joblib").predict(backend_inputs(test)))
        result = test[["participant", "source", "time", "meal_id", "baseline"]].copy()
        for i, target in enumerate(COMMON_TARGETS):
            result[f"predicted_{target}"] = prediction[:, i]
        result.to_csv(directory / "test_predictions.csv", index=False)
        print(f"Trained {mode}/backend: {len(train)} train; {len(test)} test meals", flush=True)
        for family in ["linear", "xgboost"]:
            existing = source_run / f"{mode}_{family}_core"
            destination = output / f"{mode}_{family}_core"
            if existing.exists():
                shutil.copytree(existing, destination)
                print(f"Reused verified {mode}/{family}", flush=True)
            else:
                train_run(examples, manifest, mode, family, "core", destination)
    write_json(output / "provenance.json", {
        "source_run": str(source_run.relative_to(ROOT)), "verified_models": verified,
        "source_artifacts": {str(p.relative_to(source_run)): fingerprint(p)
                             for p in source_run.glob("*/model.joblib")},
        "adapter": "13 backend feature slots; 11 available. HR/activity missing for both sources. "
                   "30-minute endpoint slope; 7-day mean and population std from causal shared examples.",
        "evaluation": "Shared core eligibility, supplied labels and patient splits; common 5-120 minute endpoints. "
                      "Backend original calendar-day split replaced for matched comparison. "
                      "Backend direct peak target excludes minute zero to match meal_response. "
                      "No test tuning; source-only meal_response uses default 30 trials/2000 trees/300 bootstraps.",
        "created_utc": datetime.now(timezone.utc).isoformat()})


def load_comparison(run):
    examples = pd.read_csv(run / "core_examples.csv.gz", parse_dates=["time"])
    manifest = pd.read_csv(run / "participant_splits.csv")
    examples = examples.merge(manifest[["participant", "partition"]], on="participant", validate="many_to_one")
    assert not manifest.participant.duplicated().any()
    test = examples.loc[examples.partition.eq("test")].reset_index(drop=True)
    truth = summarize(test[TARGETS].to_numpy(), test.baseline)
    records, models = [], {}
    for mode in ["cg_macros", "t2dm", "combined"]:
        block = test if mode == "combined" else test.loc[test.source.eq(mode)]
        for family in ["backend", "linear", "xgboost", "persistence"]:
            if family == "backend":
                bundle = joblib.load(run / f"{mode}_backend/model.joblib")
                prediction = pd.DataFrame(bundle.predict(backend_inputs(block)), index=block.index, columns=COMMON_TARGETS)
                models[(mode, family)] = bundle
            elif family == "persistence":
                prediction = summarize(np.repeat(block.baseline.to_numpy()[:, None], 24, axis=1), block.baseline)
                prediction.index = block.index
            else:
                bundle = joblib.load(run / f"{mode}_{family}_core/model.joblib")
                curve = predict_bundle(bundle, block)
                prediction = summarize(curve, block.baseline)
                prediction.index = block.index
                models[(mode, family)] = bundle
                saved = pd.read_csv(run / f"{mode}_{family}_core/test_predictions.csv")
                np.testing.assert_array_equal(block.meal_id.to_numpy(), saved.meal_id.to_numpy())
                np.testing.assert_allclose(curve, saved[[f"predicted_{t}" for t in TARGETS]], rtol=1e-7, atol=1e-5)
            for index, row in block.iterrows():
                for target in COMMON_TARGETS:
                    records.append({"training_data": mode, "source": row.source,
                        "model": LABELS.get(family, "Persistence"), "family": family,
                        "participant": row.participant, "meal_id": row.meal_id,
                        "target": target, "truth": truth.loc[index, target],
                        "prediction": prediction.loc[index, target]})
    results = pd.DataFrame(records)
    results["error"] = results.prediction - results.truth
    results["absolute_error"] = results.error.abs()
    results["squared_error"] = results.error ** 2
    patient = results.groupby(["training_data", "source", "model", "family", "target", "participant"]).agg(
        mae=("absolute_error", "mean"), mse=("squared_error", "mean")).reset_index()
    table = patient.groupby(["training_data", "source", "model", "family", "target"]).agg(
        mae=("mae", "mean"), mse=("mse", "mean"), patients=("participant", "nunique")).reset_index()
    table["rmse"] = np.sqrt(table.mse)
    return examples, test, models, results, patient, table


def contributions(bundle, family, frame):
    if family == "backend":
        x = bundle.steps[0][1].transform(backend_inputs(frame))
        estimator = bundle.steps[1][1].estimators_[1]
        names = BACKEND_FEATURES
        values = estimator.get_booster().predict(DMatrix(x), pred_contribs=True)
        prediction = bundle.predict(backend_inputs(frame))[:, 1]
    else:
        x = bundle["preprocessor"].transform(feature_frame(frame, bundle["feature_names"]))
        names = bundle["preprocessor"].get_feature_names_out().tolist()
        if family == "linear":
            estimator = bundle["models"]
            values = np.column_stack([x * estimator.coef_[11], np.full(len(frame), estimator.intercept_[11])])
        else:
            estimator = bundle["models"][11]
            stop = estimator.best_iteration + 1
            values = estimator.get_booster().predict(DMatrix(x), pred_contribs=True, iteration_range=(0, stop))
        prediction = predict_bundle(bundle, frame)[:, 11]
    np.testing.assert_allclose(values.sum(axis=1), prediction, atol=0.002, rtol=1e-5)
    return pd.DataFrame(values[:, :-1], columns=names, index=frame.index), values[:, -1], prediction


def build_notebook(run, path):
    daily = run / "daily_snapshot"
    if not daily.exists():
        daily.mkdir()
        for split in ["train", "validation", "test"]:
            shutil.copy2(ROOT / f"data/sharanya_cg/{split}.csv", daily / f"{split}.csv")
        shutil.copy2(ROOT / "backend/artifacts/regression.joblib", daily / "regression.joblib")
        write_json(daily / "fingerprints.json", {p.name: fingerprint(p) for p in daily.iterdir() if p.is_file()})
    cells = []
    def md(source):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)})
    def code(source):
        cells.append({"cell_type": "code", "metadata": {}, "source": source.splitlines(keepends=True),
                      "execution_count": None, "outputs": []})
    md("""# Backend vs meal_response: CGMacros and Shanghai T2DM

This notebook contains executed results, not just code. It compares the backend's fixed XGBoost design with **both** meal_response model families, on **both** raw-data sources, individually and pooled. The Sharanya daily-forecast result appears separately because a daily average and a post-meal glucose curve are different prediction tasks.

The matched comparison uses identical eligible meals, held-out patients, labels and evaluation weights. The backend is adapted to these common inputs and endpoints; its original day-8 evaluation is **not** used to rank the pipelines. This experiment compares complete designs (features, fitting weights, tuning and target representation), so it cannot attribute a difference to one design choice alone.
""")
    code(f'''from pathlib import Path
import sys, io, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from IPython.display import display, Markdown, Image
root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "meal_response/training.py").exists())
sys.path.insert(0, str(root))
from meal_response.comparison_notebook import load_comparison, contributions, LABELS, ENDPOINTS, COMMON_TARGETS
from meal_response.training import predict_bundle
from meal_response.features import TARGETS, MINUTES
from backend.data import load_forecast_splits
import joblib
run = root / {str(run.relative_to(ROOT))!r}
examples, test, models, results, patient_scores, metrics = load_comparison(run)
plt.rcParams.update({{"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False}})
colors = {{"Backend adapted": "#e07a36", "Meal response linear": "#6b69b2", "Meal response XGBoost": "#168b86", "Persistence": "#7c8790"}}
def show(fig, name):
    fig.tight_layout()
    fig.savefig(run / (name + ".png"), dpi=160, bbox_inches="tight")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    display(Image(data=buf.getvalue()))
    plt.close(fig)
display(Markdown("Artifacts: `" + str(run.relative_to(root)) + "`"))
''')
    md("""## What is different?

| Design | Backend adapted | meal_response XGBoost | meal_response linear |
|---|---|---|---|
| Decision rule | Sum of 150 small regression trees per target | Sum of validation-selected regression trees at each horizon | Intercept plus a weighted sum of input features at each horizon |
| Inputs in this comparison | Meal macros, baseline, 30-minute slope, clock, seven-day glucose mean/std (11 available inputs in 13 slots) | Core meal macros, 13 recent glucose lags, trend/variability, history summaries and logged nutrition/coverage, with missingness indicators | Same core features as meal_response XGBoost |
| Outputs | Seven independently fitted endpoint/peak/rise/time targets | 24 independent glucose predictions at 5-minute intervals; endpoints and peak derived from the curve | Joint linear fit of 24 glucose outputs; derived endpoints |
| Training influence | Every meal has equal loss weight | Patient/source balancing × confidence factor × training-only large-rise emphasis | Same weighting as meal_response XGBoost |
| Selection | Fixed depth 3, 150 trees, learning rate .05, lambda 5 | 30 random trials, up to 2,000 trees, horizon-specific early stopping; validation selects parameters | Ordinary least squares; training-only imputation and scaling |
| Original protocol | Raw CGMacros: train calendar days 1–7, test day 8 for known people. Sharanya: supplied patient splits. | Held-out patients, with source/cohort stratification | Same held-out patients |

**Adaptations made for the matched experiment:** backend inputs come from the same causal core table as meal_response. The slope is `(lag_0 - lag_30) / 30`; history std uses the core population std. HR and activity are unavailable in core and remain missing for both sources. Backend peak is trained over minutes 5–120 (rather than including minute zero), matching meal_response. The original backend's calendar-day split is replaced with meal_response's patient manifest. These changes make this an explicitly adapted backend comparison, not a rerun of its original protocol.

Source-only models learn from their own source; pooled models learn from both sources. Validation and test remain disjoint from training. IDs, source/cohort labels, confidence and future values are not prediction features. Confidence and observed training rises only affect training weights.
""")
    md("## Dataset coverage and verification\nSaved pooled models were checked against current raw-data fingerprints, reconstructed features, saved predictions, weighting and online inference before reuse. Source-only meal_response models were freshly trained with the default full tuning budget.")
    code('''coverage = examples.groupby(["source", "partition"]).agg(patients=("participant", "nunique"), meals=("meal_id", "size"))
display(coverage)
display(pd.DataFrame(json.loads((run / "provenance.json").read_text())["verified_models"]))
display(Markdown("**Label limits:** T2DM supplied outcomes include interpolation; CGMacros original-observation flags are unavailable. Accuracy here is against the supplied curves, not independently observed five-minute measurements. The original test set has already been inspected in earlier experiments; treat this comparison as exploratory."))
''')
    md("## Fitted model settings and full-curve results\nThe meal_response curve RMSE averages each patient's 24-point curve RMSE. Backend predicts discrete endpoints and has no full-curve RMSE. The source breakdowns below use exactly the same test meals as the endpoint comparison. Do not compare a 24-point curve RMSE directly with an endpoint MAE.")
    code('''configuration_rows, curve_rows = [], []
for mode in ["cg_macros", "t2dm", "combined"]:
    for family in ["linear", "xgboost"]:
        report = json.loads((run / f"{mode}_{family}_core/metrics.json").read_text())
        configuration_rows.append({"training_data": mode, "model": LABELS[family], **report["configuration"], "trials": len(report["trials"]), "validation_curve_RMSE": report["validation_patient_curve_rmse"]})
        sources = ["cg_macros", "t2dm"] if mode == "combined" else [mode]
        for source in sources:
            item = report["test"][f"source:{source}"]
            error = item["metrics"]["curve"]
            curve_rows.append({"training_data": mode, "source": source, "model": LABELS[family], "patients": item["patients"], "meals": item["meals"], "curve_RMSE": error["rmse"], "95% interval": error["ci95"]["rmse"], "persistence_curve_RMSE": report["persistence"][f"source:{source}"]["metrics"]["curve"]["rmse"]})
display(pd.DataFrame(configuration_rows))
display(pd.DataFrame(curve_rows).round(2))
pd.DataFrame(curve_rows).to_csv(run / "full_curve_metrics.csv", index=False)
''')
    md("## Matched results: the same 30/60/90/120-minute targets\nAll errors below give each held-out patient equal weight within a source. MAE/RMSE for glucose and peak/rise are in mg/dL; time-to-peak error is in minutes. Lower is better. The daily-mean MAE is deliberately excluded from these rankings.")
    code('''endpoint_table = metrics.loc[metrics.target.isin(ENDPOINTS)].pivot(index=["training_data", "source", "target"], columns="model", values="mae")
display(endpoint_table.round(2))
display(metrics.loc[metrics.target.isin(ENDPOINTS)].pivot(index=["training_data", "source", "target"], columns="model", values="rmse").round(2).rename_axis(columns="Endpoint RMSE (mg/dL)"))
metrics.to_csv(run / "matched_metrics.csv", index=False)
results.to_csv(run / "matched_predictions.csv", index=False)
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
for row, source in enumerate(["cg_macros", "t2dm"]):
    for col, mode in enumerate([source, "combined"]):
        ax = axes[row, col]
        block = metrics.loc[(metrics.source == source) & (metrics.training_data == mode) & metrics.target.isin(ENDPOINTS)]
        for label in colors:
            values = block.loc[block.model == label].set_index("target").reindex(ENDPOINTS)
            ax.plot([30, 60, 90, 120], values.mae, marker="o", color=colors[label], label=label)
        ax.set(title=f"{source} | {'source-only training' if col == 0 else 'pooled training'}", xlabel="Minutes after meal", ylabel="Patient-balanced MAE (mg/dL)")
        ax.legend(fontsize=8)
show(fig, "endpoint_errors")
display(metrics.loc[~metrics.target.isin(ENDPOINTS)].pivot(index=["training_data", "source", "target"], columns="model", values="mae").round(2))
consistency_rows = []
for source in ["cg_macros", "t2dm"]:
    for family in ["backend", "linear", "xgboost"]:
        p = results.loc[(results.source == source) & (results.training_data == "combined") & (results.family == family)].pivot(index="meal_id", columns="target", values="prediction")
        base = test.set_index("meal_id").baseline.reindex(p.index)
        consistency_rows.append({"source":source, "model":LABELS[family], "peak_below_predicted_endpoint_pct":100 * (p.peak < p[ENDPOINTS].max(axis=1) - .001).mean(), "inconsistent_peak_rise_pct":100 * (np.abs(p.peak - base - p.peak_rise) > .001).mean()})
display(Markdown("**Internal consistency:** percentage of pooled test predictions with incompatible endpoints. Separate backend regressors can disagree; curve-derived peak and rise share a common construction."))
display(pd.DataFrame(consistency_rows).round(2))
''')
    md("## Does an apparent improvement hold across patients?\nPaired patient bootstrap: the 60-minute MAE difference is `meal_response − backend`. Negative values favor meal_response. Each resample uses the same patients for both models. Intervals describe patient variation on this inspected test set, not a fresh confirmatory evaluation.")
    code('''rng = np.random.default_rng(42)
paired_rows = []
for source in ["cg_macros", "t2dm"]:
    for mode in [source, "combined"]:
        p = patient_scores.loc[(patient_scores.source == source) & (patient_scores.training_data == mode) & (patient_scores.target == "glucose_60")].pivot(index="participant", columns="model", values="mae")
        for label in [LABELS["linear"], LABELS["xgboost"]]:
            delta = (p[label] - p[LABELS["backend"]]).to_numpy()
            draws = rng.choice(delta, size=(2000, len(delta)), replace=True).mean(axis=1)
            lo, hi = np.quantile(draws, [.025, .975])
            paired_rows.append({"source": source, "training_data": mode, "comparison": label, "patients": len(delta), "MAE difference": delta.mean(), "95% lower": lo, "95% upper": hi})
paired = pd.DataFrame(paired_rows)
display(paired.round(2))
paired.to_csv(run / "paired_patient_bootstrap.csv", index=False)
''')
    md("""## Why the CGMacros source-only linear result is unusually poor

This result is retained as measured. Ordinary least squares has no coefficient shrinkage and no bound on its output. The diagnostics below show the fitted matrix rank, coefficient magnitude and worst held-out predictions. Rank deficiency indicates redundant inputs; it alone does not prove the cause of bad generalization. Large coefficients and extreme predictions here are consistent with an unstable linear extrapolation using correlated inputs. Pooled linear regression is a different fitted model and does not show the same magnitude of failure. Ridge regularization would be a separate experiment and has not been substituted into these results.
""")
    code('''diagnostics = []
for mode in ["cg_macros", "t2dm", "combined"]:
    estimator = models[(mode, "linear")]["models"]
    diagnostics.append({"training_data":mode, "transformed_features":estimator.coef_.shape[1], "matrix_rank":estimator.rank_, "largest_absolute_coefficient":np.abs(estimator.coef_).max()})
display(pd.DataFrame(diagnostics).round(2))
bad = results.loc[(results.training_data == "cg_macros") & (results.family == "linear") & (results.target == "glucose_60")].nlargest(5, "absolute_error")
display(bad[["participant", "meal_id", "truth", "prediction", "absolute_error"]].round(2))
frame = test.loc[test.source == "cg_macros"]
values, reference, prediction = contributions(models[("cg_macros", "linear")], "linear", frame)
index = test.loc[test.meal_id == bad.iloc[0].meal_id].index[0]
local = values.loc[index]
top = local.loc[local.abs().nlargest(8).index]
display(top.rename("signed contribution to worst 60-minute prediction (mg/dL)").to_frame().round(2))
''')
    md("## The same held-out meals, side by side\nFor each source, show a typical and a difficult meal, selected by the pooled XGBoost's median and worst 60-minute absolute error. These are illustrative examples, not independent evidence. Backend dots show its four endpoint predictions; it has no fitted full curve.")
    code('''chosen = {}
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
for r, source in enumerate(["cg_macros", "t2dm"]):
    block = results.loc[(results.source == source) & (results.training_data == "combined") & (results.family == "xgboost") & (results.target == "glucose_60")].sort_values("absolute_error")
    ids = [block.iloc[len(block)//2].meal_id, block.iloc[-1].meal_id]
    chosen[source] = ids[0]
    for c, meal_id in enumerate(ids):
        row = test.loc[test.meal_id == meal_id]
        ax = axes[r, c]
        ax.plot(MINUTES, row[TARGETS].to_numpy()[0], color="black", lw=2, label="Supplied actual")
        ax.axhline(row.baseline.iloc[0], color=colors["Persistence"], ls=":", label="Persistence")
        for family in ["linear", "xgboost"]:
            ax.plot(MINUTES, predict_bundle(models[("combined", family)], row)[0], color=colors[LABELS[family]], label=LABELS[family])
        b = results.loc[(results.training_data == "combined") & (results.family == "backend") & (results.meal_id == meal_id)].set_index("target")
        ax.scatter([30,60,90,120], b.loc[ENDPOINTS].prediction, color=colors[LABELS["backend"]], s=40, label=LABELS["backend"])
        ax.set(title=f"{source}: {'typical' if c == 0 else 'difficult'} meal", xlabel="Minutes after meal", ylabel="Glucose (mg/dL)")
        ax.legend(fontsize=8)
show(fig, "same_meal_predictions")
display(test.loc[test.meal_id.isin(sum(([v] for v in chosen.values()), [])), ["source", "participant", "time", "meal_id", "carbs", "protein", "fat", "calories", "baseline"]])
''')
    md("## How are they making decisions?\nTrees repeatedly split on feature thresholds. Each leaf contributes a correction and the model adds those corrections. Linear regression adds `coefficient × transformed feature value` to its intercept, with no tree thresholds or learned interactions.\n\nThe plots explain the **60-minute prediction** of the pooled models. Tree models use exact native TreeSHAP, limited to the trees actually used by early stopping. The linear model uses exact contributions relative to the zero point after training-fitted scaling. Those reference points differ, so importance magnitudes across families are descriptive, not causal comparisons. All contribution sums are checked against model predictions.\n\nMean absolute contributions show which inputs a model uses most on held-out rows. Correlated glucose lags/history summaries may share or shift attribution; missingness indicators can also influence the prediction. These are explanations of model behavior, not physiological effects.")
    code('''explanations = {}
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
importance_rows = []
for r, source in enumerate(["cg_macros", "t2dm"]):
    frame = test.loc[test.source == source]
    for c, family in enumerate(["backend", "linear", "xgboost"]):
        values, reference, prediction = contributions(models[("combined", family)], family, frame)
        explanations[(source, family)] = (values, reference, prediction)
        importance = values.abs().mean().sort_values(ascending=False)
        for name, value in importance.items():
            importance_rows.append({"source":source, "model":LABELS[family], "feature":name, "mean_abs_contribution":value})
        top = importance.head(10).sort_values()
        axes[r,c].barh(top.index, top.values, color=colors[LABELS[family]])
        axes[r,c].set(title=f"{source} | {LABELS[family]}", xlabel="Mean absolute contribution (mg/dL)")
show(fig, "global_decision_features")
pd.DataFrame(importance_rows).to_csv(run / "feature_contributions_60min.csv", index=False)
''')
    md("## Why this individual prediction went up or down\nPositive contributions push the prediction above the model reference; negative contributions lower it. Each panel uses the same typical held-out meal for that source. The displayed top ten features plus the remaining-feature total reconstruct the prediction.")
    code('''fig, axes = plt.subplots(2, 3, figsize=(18, 10))
local_rows = []
for r, source in enumerate(["cg_macros", "t2dm"]):
    index = test.loc[test.meal_id == chosen[source]].index[0]
    for c, family in enumerate(["backend", "linear", "xgboost"]):
        values, reference, prediction = explanations[(source,family)]
        position = values.index.get_loc(index)
        full = values.loc[index]
        top = full.loc[full.abs().nlargest(10).index].copy()
        top.loc["All other features"] = full.sum() - top.sum()
        top = top.sort_values()
        axes[r,c].barh(top.index, top.values, color=["#d16b55" if v < 0 else "#168b86" for v in top.values])
        axes[r,c].axvline(0, color="black", lw=.7)
        axes[r,c].set(title=f"{source} | {LABELS[family]}\\n{reference[position]:.1f} {full.sum():+.1f} = {prediction[position]:.1f} mg/dL", xlabel="Signed contribution (mg/dL)")
        local_rows.append({"source":source, "model":LABELS[family], "reference":reference[position], "feature_sum":full.sum(), "prediction":prediction[position], "actual_glucose_60":test.loc[index,"glucose_60"]})
show(fig, "individual_decisions")
display(pd.DataFrame(local_rows).round(2))
''')
    md("## The Sharanya daily-mean model, separately\nThis is the backend model from the preceding task. It uses 28 daily glucose summaries plus history food/medication/logging features to predict `target` (daily mean glucose). Its fixed XGBoost settings resemble the backend meal regressors, but its target and inputs differ. A lower daily-average MAE cannot establish that it is a better meal-response model. The prepared CSVs' historical `predicted_glucose_mean` and `absolute_error` columns are excluded from fitting; their original algorithm is not identified here.")
    code('''splits, daily_features = load_forecast_splits(run / "daily_snapshot")
daily_bundle = joblib.load(run / "daily_snapshot/regression.joblib")
assert daily_bundle.get("dataset") == "daily_forecast", "Run python -m backend.model on Sharanya first"
daily_rows = []
for split in ["validation", "test"]:
    frame = splits[split]
    pred = daily_bundle["model"].predict(frame[daily_bundle["features"]])
    for label, values in [("Backend daily XGBoost", pred), ("Last-day mean baseline", frame.day7_glucose_mean.to_numpy())]:
        err = values - frame.target.to_numpy()
        per_patient = pd.DataFrame({"Patient":frame.Patient, "abs":np.abs(err), "sq":err**2}).groupby("Patient").mean()
        daily_rows.append({"split":split, "model":label, "rows":len(frame), "row_MAE":np.abs(err).mean(), "row_RMSE":np.sqrt((err**2).mean()), "patient_MAE":per_patient["abs"].mean(), "patient_RMSE":np.sqrt(per_patient["sq"].mean())})
display(pd.DataFrame(daily_rows).round(2))
frame = splits["test"]
pipeline = daily_bundle["model"]
from backend.explain import tree_contributions
daily_x = pipeline.steps[0][1].transform(frame[daily_bundle["features"]])
daily_importance = pd.Series(np.abs(tree_contributions(pipeline.steps[1][1], daily_x)).mean(axis=0), index=daily_bundle["features"]).nlargest(12).sort_values()
fig, ax = plt.subplots(figsize=(10,5))
ax.barh(daily_importance.index, daily_importance.values, color="#e07a36")
ax.set(title="Sharanya daily model: most influential inputs on test rows", xlabel="Mean absolute SHAP contribution (mg/dL)")
show(fig, "daily_model_decisions")
''')
    md("## Reading the comparison\nUse the matched meal endpoint table to compare model designs; use the daily table only for the daily forecasting task. A model needs to beat persistence on the source and horizon you care about. Source-only versus pooled training answers whether adding the other population helps this held-out cohort. The paired intervals indicate how sensitive differences are to the small number of held-out patients. Independently fitted endpoint regressors can predict inconsistent peak/rise values; meal_response derives those values consistently from its predicted curve, although the curve itself is not constrained to be smooth.\n\nThe attribution panels answer what features influenced a prediction, not why glucose changed biologically. Improvement cannot be assigned solely to XGBoost versus linear: feature availability, balancing, confidence weights, rise emphasis, validation tuning and target design also differ.")
    code('''summary_rows = []
for source in ["cg_macros", "t2dm"]:
    for mode in [source, "combined"]:
        block = metrics.loc[(metrics.source == source) & (metrics.training_data == mode) & metrics.target.isin(ENDPOINTS)]
        average = block.groupby("model").mae.mean().sort_values()
        summary_rows.append({"source":source, "training_data":mode, "best_average_endpoint_MAE":average.index[0], "MAE":average.iloc[0], "backend_MAE":average[LABELS["backend"]], "meal_XGBoost_MAE":average[LABELS["xgboost"]], "linear_MAE":average[LABELS["linear"]], "persistence_MAE":average["Persistence"]})
summary = pd.DataFrame(summary_rows)
display(summary.round(2))
summary.to_csv(run / "summary.csv", index=False)
display(Markdown("Average endpoint MAE is the simple mean of the four patient-balanced endpoint MAEs; it is not the full-curve RMSE in the original meal_response report."))
''')
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python (glyimpse)", "language": "python", "name": "python3"},
                "language_info": {"name": "python", "version": sys.version.split()[0]}}, "nbformat": 4, "nbformat_minor": 5}
    # Execute through IPython and capture rich HTML/PNG outputs using existing env.
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output
    shell = InteractiveShell.instance()
    count = 0
    for index, cell in enumerate(cells):
        cell["id"] = f"comparison-{index:02d}"
        if cell["cell_type"] != "code":
            continue
        count += 1
        print(f"Executing notebook cell {count}", flush=True)
        with capture_output() as captured:
            result = shell.run_cell("".join(cell["source"]), store_history=True)
        if result.error_before_exec or result.error_in_exec:
            raise RuntimeError(f"Notebook cell {count} failed: {captured.stdout} {captured.stderr}") from result.error_in_exec
        cell["execution_count"] = count
        if captured.stdout:
            cell["outputs"].append({"output_type": "stream", "name": "stdout", "text": captured.stdout})
        if captured.stderr:
            cell["outputs"].append({"output_type": "stream", "name": "stderr", "text": captured.stderr})
        for rich in captured.outputs:
            cell["outputs"].append({"output_type": "display_data", "data": rich.data, "metadata": rich.metadata})
    path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"Executed notebook saved: {path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=ROOT / "artifacts/meal_response/20261010T141845959329Z")
    parser.add_argument("--run", type=Path, help="Reuse an already prepared comparison run")
    args = parser.parse_args()
    run = args.run or ROOT / "artifacts/model_comparison" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    if not args.run:
        prepare(args.source_run, run)
    build_notebook(run, ROOT / "meal_response/model_comparison.ipynb")


if __name__ == "__main__":
    main()
