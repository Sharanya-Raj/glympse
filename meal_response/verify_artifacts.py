"""Verify all completed runs, inference equality, and optionally refresh plots."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .data import load_sources
from .features import TARGETS, build_examples, forecast, summarize
from .training import plot_results, predict_bundle, write_json
from .weighting import loss_weights


def verify(root, t2dm, cg_root, refresh_plots=False):
    root = Path(root)
    patients, metadata = load_sources(t2dm, cg_root)
    by_id = {p.participant: p for p in patients}
    previous_metadata = json.loads((root / "data_metadata.json").read_text())
    def hashes(value):
        return {str(Path(item["path"]).resolve()): item["sha256"] for item in value["files"]}
    if hashes(metadata) != hashes(previous_metadata):
        raise ValueError("Input fingerprints differ from the training run")
    core, _ = build_examples(patients)
    enhanced = None
    manifests = pd.read_csv(root / "participant_splits.csv")
    if manifests.participant.duplicated().any():
        raise ValueError("Participant appears in multiple partitions")
    reports = []
    for entry in json.loads((root / "comparison.json").read_text()):
        if entry["status"] != "complete":
            raise ValueError(f"Incomplete experiment: {entry}")
        dataset, family, mode = entry["dataset"], entry["model"], entry["features"]
        if mode == "enhanced" and enhanced is None:
            enhanced, _ = build_examples(patients, enhanced=True)
        examples = core if mode == "core" else enhanced
        saved_examples = pd.read_csv(root / f"{mode}_examples.csv.gz", parse_dates=["time"])
        pd.testing.assert_frame_equal(examples.reindex(columns=saved_examples.columns).reset_index(drop=True),
                                      saved_examples, check_dtype=False, check_exact=False, rtol=1e-10, atol=1e-10)
        test_ids = manifests.loc[manifests.partition.eq("test"), "participant"]
        test = examples.loc[examples.participant.isin(test_ids)]
        if dataset != "combined":
            test = test.loc[test.source.eq(dataset)]
        test = test.reset_index(drop=True)
        directory = root / f"{dataset}_{family}_{mode}"
        bundle = joblib.load(directory / "model.joblib")
        if "weighting" in bundle:
            config = json.loads((directory / "config.json").read_text())
            source_examples = examples if dataset == "combined" else examples.loc[examples.source.eq(dataset)]
            for partition, filename in [("train", "training_weights.csv"), ("validation", "validation_weights.csv")]:
                ids = manifests.loc[manifests.partition.eq(partition), "participant"]
                block = source_examples.loc[source_examples.participant.isin(ids)].reset_index(drop=True)
                _, detail, _ = loss_weights(block, config["high_confidence"], config["missing_confidence_factor"],
                                            config["large_rise_multiplier"], config["rise_quantile"], partition == "train")
                saved_weights = pd.read_csv(directory / filename)
                np.testing.assert_array_equal(detail.meal_id, saved_weights.meal_id)
                np.testing.assert_allclose(detail.loss_weight, saved_weights.loss_weight, rtol=1e-10, atol=1e-10)
        prediction = predict_bundle(bundle, test)
        saved = pd.read_csv(directory / "test_predictions.csv", parse_dates=["time"])
        pd.testing.assert_frame_equal(test[["participant", "time"]], saved[["participant", "time"]], check_dtype=False)
        np.testing.assert_allclose(prediction, saved[[f"predicted_{name}" for name in TARGETS]], rtol=1e-7, atol=1e-5)
        for name, values in summarize(prediction, test.baseline).items():
            np.testing.assert_allclose(values, saved[f"predicted_{name}"], rtol=1e-7, atol=1e-5)
        row = test.iloc[0]
        patient = by_id[row.participant]
        meal = patient.meals.loc[patient.meals.time.eq(row.time)].iloc[0]
        online = forecast(bundle, patient, row.time, meal)
        np.testing.assert_allclose(online["glucose_mg_dl"], prediction[0], rtol=1e-7, atol=1e-5)
        if refresh_plots:
            plot_results(directory, test, test[TARGETS].to_numpy(), prediction, dataset, family)
        reports.append({"dataset": dataset, "model": family, "features": mode, "test_meals": len(test),
                        "saved_predictions_match": True, "online_prediction_matches": True,
                        "outcomes_match": True, "current_examples_match": True})
        if "weighting" in bundle:
            reports[-1]["confidence_and_rise_weights_match"] = True
        print(f"Verified {dataset}/{family}/{mode}", flush=True)
    write_json(root / "verification.json", reports)
    return reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--t2dm", type=Path, default=Path("data/T2DM_5min_with_macros.csv"))
    parser.add_argument("--cg-root", type=Path, default=Path("data/cg_macros"))
    parser.add_argument("--refresh-plots", action="store_true")
    args = parser.parse_args()
    verify(args.run, args.t2dm, args.cg_root, args.refresh_plots)
