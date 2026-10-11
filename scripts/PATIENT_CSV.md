# Prepare a patient's CSV

Run from the repository root:

```powershell
python scripts/prepare_patient_csv.py
```

Defaults: read `data/T2DM_5min_with_macros.csv`, select Patient `2088`, and save
`data/sharanya_cg/patient_2088.csv`. The actual input extension is `.csv`.
This local script uses only Python's standard library and needs no AWS credentials.

For another patient or destination:

```powershell
python scripts/prepare_patient_csv.py --patient 2088 --source data/T2DM_5min_with_macros.csv --output data/sharanya_cg/patient_2088.csv
```

The output begins with the app's supported columns: `Recording`, `Patient`,
`Date`, `CGM (mg / dl)`, `Dietary intake`, `Insulin dose - s.c.`,
`Non-insulin hypoglycemic agents`, `CSII - bolus insulin (Novolin R, IU)`, and
`CSII - basal insulin (Novolin R, IU / H)`. Remaining source columns, including
macros and interpolation flags, follow in their original order. CBG, blood
ketones, and columns with Chinese characters in their names are excluded.
The app's reader supplies its optional CBG field as blank internally.

Dates become ISO timestamps with seconds, sorted chronologically. Naive source
times keep their clock values; explicit timezone offsets convert to UTC, matching
the app's reader. Other retained cell values keep their original precision/text.
There is no new interpolation, model training, or invented glucose/meal data.
Extra macro fields are preserved in the file; the current app reader uses only
its supported columns for predictions.

The script rejects missing patients, duplicate timestamps, invalid CGM values,
and multiple recordings unless `--recording` selects one. Identical reruns are
safe; a different existing output is refused. Original source and existing
`train.csv`, `validation.csv`, and `test.csv` files remain unchanged.

Verified export: 3,994 rows, 29 columns, 2020-11-23 11:28:00 through
2020-12-07 08:13:00. The backend history validator accepts it, with 288 readings
on each of the first seven days.

This creates a local upload file. Assigning it to `sharanyaraj24@gmail.com` in
DynamoDB requires an AWS credential session; the account's Cognito `sub` is
looked up by the separate import script.
