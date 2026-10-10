# Backend

The maintained ML pipeline lives in `regression-ml-glucose/`.
Read that folder's README for commands, benchmarks, and the model explanation.
`app.py` and `lambda_function.py` are existing API/deployment scaffolding;
the `/predict` endpoint is still a placeholder, not connected to the trained model.

Old CGMacros modules and the superseded daily-average model were removed.
Source data and historical generated reports/artifacts remain intact.
Install ML dependencies with `python -m pip install -r backend/requirements.txt`.
FastAPI/Mangum dependencies for deployment are separate from this ML list.
