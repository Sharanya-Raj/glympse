"""CLI: python -m meal_response {audit,train,compare,predict}."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .data import NUTRIENTS, Patient, glucose_frame, load_sources
from .evaluation import split_manifest
from .features import build_examples, forecast
from .training import train_run, write_json

ROOT = Path(__file__).resolve().parent.parent


def input_patient(payload):
    source = payload["source"]
    glucose = pd.DataFrame(payload["glucose_history"])
    glucose["time"] = pd.to_datetime(glucose.time, errors="raise")
    if source == "t2dm" and "interpolated" not in glucose:
        raise ValueError("T2DM history requires boolean interpolated flags")
    flags = glucose.get("interpolated", pd.Series(False, index=glucose.index))
    if not flags.map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise ValueError("interpolated flags must be JSON booleans")
    glucose = glucose_frame(glucose, glucose.time, "glucose", flags)
    def history(key, columns):
        frame = pd.DataFrame(payload.get(key, []), columns=columns)
        frame["time"] = pd.to_datetime(frame.time, errors="raise")
        return frame
    meals = history("meal_history", ["time", "carbs", "protein", "fat", "calories", "fiber", "nutrition_complete"])
    for column in NUTRIENTS:
        meals[column] = pd.to_numeric(meals[column], errors="raise")
    meals["nutrition_complete"] = meals.nutrition_complete.fillna(False).astype(bool)
    activity = history("activity_history", ["time", "mets", "hr", "activity_calories"])
    for column in ["mets", "hr", "activity_calories"]:
        activity[column] = pd.to_numeric(activity[column], errors="raise")
    medication = history("medication_history", ["time", "drug", "route", "unit", "dose", "raw", "parsed"])
    medication["dose"] = pd.to_numeric(medication.dose, errors="raise")
    medication["parsed"] = medication.parsed.fillna(False).astype(bool)
    return Patient("inference", source, "unknown", glucose, meals, activity, medication)


def parser():
    result = argparse.ArgumentParser(description="Meal-time two-hour glucose curve forecasting")
    commands = result.add_subparsers(dest="command", required=True)
    for name in ["audit", "train", "compare"]:
        command = commands.add_parser(name)
        command.add_argument("--t2dm", type=Path, default=ROOT / "data/T2DM_5min_with_macros.csv")
        command.add_argument("--cg-root", type=Path, default=ROOT / "data/cg_macros")
        command.add_argument("--output", type=Path, default=ROOT / "artifacts/meal_response")
        command.add_argument("--seed", type=int, default=42)
        command.add_argument("--label-policy", choices=["supplied", "observed"], default="supplied")
        command.add_argument("--features", choices=["core", "enhanced"], default="core")
        if name != "audit":
            command.add_argument("--trials", type=int, default=30)
            command.add_argument("--max-trees", type=int, default=2000)
            command.add_argument("--bootstraps", type=int, default=300)
            command.add_argument("--jobs", type=int, default=1)
            command.add_argument("--high-confidence-threshold", type=float, default=0.9)
            command.add_argument("--missing-confidence-factor", type=float, default=0.5,
                                 help="Weight factor for T2DM meals missing macro confidence")
            command.add_argument("--large-rise-multiplier", type=float, default=1.25,
                                 help="Training-only multiplier for larger observed glucose rises; 1 disables")
            command.add_argument("--rise-quantile", type=float, default=0.75)
        if name == "train":
            command.add_argument("--dataset", choices=["cg_macros", "t2dm", "combined"], required=True)
            command.add_argument("--model", choices=["linear", "xgboost", "both"], default="both")
        if name == "compare":
            command.add_argument("--include-enhanced", action="store_true", help="Also run both families with source-specific enhancements")
    predict = commands.add_parser("predict")
    predict.add_argument("--model-file", type=Path, required=True)
    predict.add_argument("--input", type=Path, required=True, help="JSON with source, meal_time, planned_meal and histories")
    predict.add_argument("--output", type=Path)
    return result


def main(argv=None):
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    if args.command == "predict":
        payload = json.loads(args.input.read_text(encoding="utf-8"))
        bundle = joblib.load(args.model_file)
        if bundle["dataset"] != "combined" and payload["source"] != bundle["dataset"]:
            raise ValueError("Inference source does not match the model's training dataset")
        result = forecast(bundle, input_patient(payload), payload["meal_time"], payload["planned_meal"])
        if args.output:
            write_json(args.output, result)
        else:
            print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    if args.command == "train" and args.dataset == "combined" and args.features == "enhanced":
        argument_parser.error("Enhanced experiments are source-specific; combined uses core features")
    if args.command != "audit" and (args.trials < 1 or args.max_trees < 1 or args.bootstraps < 0 or args.jobs < 1):
        argument_parser.error("trials, max-trees and jobs must be positive; bootstraps must be nonnegative")
    if args.command != "audit" and (not 0 < args.high_confidence_threshold <= 1 or
                                   not 0 <= args.missing_confidence_factor <= 1 or
                                   not np.isfinite(args.large_rise_multiplier) or args.large_rise_multiplier < 1 or
                                   not 0 < args.rise_quantile < 1):
        argument_parser.error("Invalid confidence or large-rise weighting settings")
    patients, metadata = load_sources(args.t2dm, args.cg_root)
    core, audit = build_examples(patients, observed_only=args.label_policy == "observed")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output / run_id
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "data_metadata.json", metadata)
    write_json(output / "audit.json", audit)
    pd.DataFrame(audit).fillna(0).to_csv(output / "audit.csv", index=False)
    if core.empty:
        write_json(output / "status.json", {"status": "unsupported", "reason": "No eligible meals; inspect audit.json"})
        print(f"No eligible meals; audit saved to {output}")
        return 1
    core.to_csv(output / "core_examples.csv.gz", index=False)
    manifest = split_manifest(core, args.seed)
    manifest.to_csv(output / "participant_splits.csv", index=False)
    write_json(output / "participant_splits.json", manifest.to_dict(orient="records"))
    print(core.groupby("source").agg(meals=("participant", "size"), patients=("participant", "nunique")).to_string(), flush=True)
    print(f"Artifacts: {output}", flush=True)
    if args.command == "audit":
        return 0
    enhanced = None
    if args.features == "enhanced" or getattr(args, "include_enhanced", False):
        enhanced, enhanced_audit = build_examples(patients, enhanced=True, observed_only=args.label_policy == "observed")
        enhanced.to_csv(output / "enhanced_examples.csv.gz", index=False)
        write_json(output / "enhanced_audit.json", enhanced_audit)
    datasets = [args.dataset] if args.command == "train" else (["cg_macros", "t2dm", "combined"] if args.features == "core" else ["cg_macros", "t2dm"])
    families = [args.model] if args.command == "train" and args.model != "both" else ["linear", "xgboost"]
    specifications = [(dataset, family, args.features) for dataset in datasets for family in families]
    if getattr(args, "include_enhanced", False) and args.features == "core":
        specifications += [(dataset, family, "enhanced") for dataset in ["cg_macros", "t2dm"] for family in families]
    comparison = []
    for dataset, family, mode in specifications:
        try:
            report = train_run(core if mode == "core" else enhanced, manifest, dataset, family, mode,
                               output / f"{dataset}_{family}_{mode}", args.trials, args.max_trees,
                               args.bootstraps, args.seed, args.jobs, args.label_policy,
                               args.high_confidence_threshold, args.missing_confidence_factor,
                               args.large_rise_multiplier, args.rise_quantile)
            model_metrics = report["test"]["overall"]["metrics"]
            baseline_metrics = report["persistence"]["overall"]["metrics"]
            comparison.append({"dataset": dataset, "model": family, "features": mode, "status": "complete",
                               "curve_rmse": model_metrics["curve"]["rmse"],
                               "persistence_curve_rmse": baseline_metrics["curve"]["rmse"],
                               "beats_persistence": model_metrics["curve"]["rmse"] < baseline_metrics["curve"]["rmse"],
                               **{f"{name}_mae": values["mae"] for name, values in model_metrics.items() if "mae" in values}})
            print(f"Completed {dataset}/{family}/{mode}: test curve RMSE={model_metrics['curve']['rmse']:.3f}", flush=True)
        except ValueError as error:
            comparison.append({"dataset": dataset, "model": family, "features": mode, "status": "failed", "reason": str(error)})
            write_json(output / f"{dataset}_{family}_{mode}" / "failure.json", {"reason": str(error)})
            print(f"Failed {dataset}/{family}/{mode}: {error}", flush=True)
        write_json(output / "comparison.json", comparison)
        pd.DataFrame(comparison).to_csv(output / "comparison.csv", index=False)
    if any(row["status"] == "complete" for row in comparison):
        comparison_plot(output, comparison)
    return int(any(row["status"] != "complete" for row in comparison))


def comparison_plot(output, comparison):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    frame = pd.DataFrame([row for row in comparison if row["status"] == "complete"])
    fig, axis = plt.subplots(figsize=(max(7, len(frame) * 1.2), 4))
    x = np.arange(len(frame))
    axis.bar(x - 0.18, frame.curve_rmse, width=0.36, label="Model")
    axis.bar(x + 0.18, frame.persistence_curve_rmse, width=0.36, label="Persistence")
    axis.set_xticks(x, [f"{r.dataset}\n{r.model}\n{r.features}" for r in frame.itertuples()])
    axis.set_ylabel("Mean patient curve RMSE (mg/dL)")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
