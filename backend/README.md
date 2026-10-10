# Backend

The maintained ML pipeline lives in `ml/`.
Read `ml/README.md` for account creation, uploads, prediction, daily feature
rankings, tests and benchmarks. `app.py` is the authenticated API;
`lambda_function.py` is the existing deployment entry point.

Old CGMacros modules and the superseded daily-average model were removed.
Source data and historical generated reports/artifacts remain intact.
Install ML dependencies with `python -m pip install -r backend/requirements.txt`.
API runtime and test dependencies are included in this list.
