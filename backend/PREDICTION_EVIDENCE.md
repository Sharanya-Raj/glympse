# Prediction evidence and top factors

This is a presentation/audit layer around the unchanged original Ridge forecast, not a confidence model. It adds `uncertainty` and `explanation` to POST /predict, persists them when a meal is confirmed, returns them through GET /meals, and includes them for prediction alternatives.

## Evidence gaps v1

Score = sum of five unweighted Boolean gap flags (0 to 5). No accuracy percentage, calibrated interval, or probability is inferred. The UI calls this an evidence-gap count.

1. Seven-day CGM: fewer than 90% of 2016 expected five-minute slots contain nonmissing positive finite uploaded CGM. The 90% threshold is an explicit product heuristic, not an empirically calibrated error threshold.
2. Recent CGM: fewer than seven occupied five-minute slots in the preceding 30 minutes including meal time. Replay-simulated values may be included, and this is disclosed.
3. Personal response support: fewer than five completed training meal responses, matching the existing correction threshold.
4. Meal portion: no recognizable g/ml quantity in the description (explicit no-food requests are not applicable). This is a coarse syntax check, not confirmation that every ingredient or amount is correct.
5. Medication completeness: unknown with today's event-only schema. This flag remains set even when medications have been logged: event presence does not establish a complete regimen, non-use, or completeness for every day. A later explicit reconciliation/confirmed-none workflow could resolve it; no such assumption is made here.

Seven 24-hour blocks end at the model's personal_cutoff, matching its actual training window. They are not necessarily midnight calendar days. Missing entire days have zero CGM coverage. Future data is excluded. Missing meal descriptions on CGM-only rows are not missing meals or missing nutritional information. Daily meal and medication counts describe logged events only, not completeness. No interpolation or filling is performed by the audit; uploaded CGM provenance cannot be established from the current schema.

Activity is not modeled and is disclosed separately. Statistical prediction confidence remains uncalibrated. The existing preprocessing still turns some blank fields into zero indicators; this layer does not alter or repair the original trained model's behavior.

## Three explanation cards

Choose the highest glucose prediction among 30, 60, 90 and 120 minutes. Take the already calculated additive contributions at that horizon, including the existing personal correction and sugar adjustment. Group features by baseline, recent glucose trend, time of day, meal history, meal type, meal portion, medications and sugar adjustment. Sum signed contributions within each group, rank absolute net group totals, display at most three nonzero groups. Opposing effects within a group cancel intentionally. Missing ranks get neutral placeholders rather than invented influences.

These effects are relative to the model reference inputs, not increases from pre-meal glucose. They are predictive associations, not causal findings, and the top three do not sum to the total glucose because other groups, intercepts and the reference remain. Sugar adjustment is labeled as an existing post-model rule. Activity is never invented as a driver.

## Verification

python -B -m unittest backend.test_prediction_summary backend.test_api backend.test_aws_phase1 -q

Tests cover full and absent days, blank non-meal rows, exclusion of future glucose, signed grouping, no mutation of predictions, and API persistence/isolation using synthetic patients and mocked AWS. No model artifact or backend/ml source is modified, and no AWS deployment is performed.
