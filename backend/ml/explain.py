"""Exact additive glucose contributions and daily top-three rankings."""
import numpy as np
import pandas as pd


def contributions(model, personal_coefficients, features, x):
    """Shape: meals x horizons x features; includes direct pre-meal baseline."""
    z = model[:-1].transform(x[features])
    values = z[:, None, :] * (model[-1].coef_ + personal_coefficients)[None, :, :]
    index = features.index("baseline")
    # Population mean baseline in original units, reversing fitted MinMaxScaler.
    bounds, scaler = model[-3], model[-2]
    reference = (scaler.mean_[index] - bounds.min_[index]) / bounds.scale_[index]
    values[:, :, index] += (x.baseline.to_numpy() - reference)[:, None]
    return values, float(reference)


def daily_top_three(effects):
    """Average |contribution| across meals/horizons; never rank by coefficient alone."""
    grouping = ["Patient", "Recording", "day", "feature"]
    table = effects.assign(absolute_contribution=effects.contribution.abs()).groupby(grouping).agg(
        importance_mg_dl=("absolute_contribution", "mean"),
        signed_contribution_mg_dl=("contribution", "mean"),
        prediction_count=("contribution", "size")).reset_index()
    table = table.sort_values(["Patient", "Recording", "day", "importance_mg_dl", "feature"],
                              ascending=[True, True, True, False, True])
    table["rank"] = table.groupby(["Patient", "Recording", "day"]).cumcount() + 1
    table["direction"] = np.select([table.signed_contribution_mg_dl > 0,
                                     table.signed_contribution_mg_dl < 0],
                                    ["higher", "lower"], default="balanced")
    # Do not invent three influential features if fewer than three contribute.
    return table[(table["rank"] <= 3) & (table.importance_mg_dl > 1e-10)]
