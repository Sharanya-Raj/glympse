"""Authenticated account -> upload -> day-8 prediction and daily explanations."""
import io
import os
from pathlib import Path
import sqlite3

import pandas as pd
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from .accounts import Accounts
from .ml.explain import daily_top_three
from .ml.service import forecast_meal, load_population, readiness, validate_history

ROOT = Path(__file__).resolve().parent


class Credentials(BaseModel):
    username: str = Field(min_length=3, max_length=80, pattern=r"^[a-zA-Z0-9_.@-]+$")
    password: str = Field(min_length=12, max_length=128)


class Reading(BaseModel):
    timestamp: str
    glucose: float = Field(gt=0, allow_inf_nan=False)


class MealRequest(BaseModel):
    meal_time: str
    dietary_intake: str = Field(min_length=1, max_length=4000)
    recent_cgm: list[Reading] = Field(min_length=4, max_length=30)


def create_app(state_dir=None, model_path=None):
    app = FastAPI(title="Glyimpse", version="1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    state = Path(state_dir or os.environ.get("GLYIMPSE_STATE_DIR", ROOT / "runtime"))
    artifact = Path(model_path or os.environ.get("GLYIMPSE_MODEL_PATH", ROOT / "artifacts/population.joblib"))
    accounts = Accounts(state / "accounts.sqlite3")
    bearer = HTTPBearer(auto_error=False)

    def user(credentials: HTTPAuthorizationCredentials = Depends(bearer)):
        user_id = accounts.authenticate(credentials.credentials) if credentials else None
        if not user_id:
            raise HTTPException(401, "Sign in with a valid bearer token")
        return user_id

    def history(user_id):
        saved, revision = accounts.history(user_id)
        if not saved:
            raise HTTPException(409, "Upload glucose history first")
        return validate_history(pd.read_csv(io.StringIO(saved), dtype="string")), revision

    @app.get("/health")
    def health():
        return {"status": "ok", "shared_model_available": artifact.is_file()}

    @app.post("/accounts", status_code=201)
    def signup(payload: Credentials):
        try:
            user_id = accounts.signup(payload.username, payload.password)
        except sqlite3.IntegrityError:
            raise HTTPException(409, "Username unavailable")
        return {"user_id": user_id, "access_token": accounts.login(payload.username, payload.password),
                "token_type": "bearer", "expires_in": 86400}

    @app.post("/sessions")
    def signin(payload: Credentials):
        token = accounts.login(payload.username, payload.password)
        if not token:
            raise HTTPException(401, "Invalid credentials")
        return {"access_token": token, "token_type": "bearer", "expires_in": 86400}

    @app.get("/history")
    def status(user_id=Depends(user)):
        raw, revision = history(user_id)
        return {"rows": len(raw), "revision": revision, **readiness(raw)}

    @app.post("/history")
    async def upload(file: UploadFile = File(...), user_id=Depends(user)):
        content = await file.read(10 * 1024 * 1024 + 1)
        if len(content) > 10 * 1024 * 1024:
            raise HTTPException(413, "CSV must be at most 10 MB")
        try:
            raw = validate_history(pd.read_csv(io.BytesIO(content), dtype="string"))
            saved, revision = accounts.history(user_id)
            if saved:
                old = validate_history(pd.read_csv(io.StringIO(saved), dtype="string"))
                raw = validate_history(pd.concat([old, raw]).drop_duplicates("Date", keep="last"))
            accounts.save_history(user_id, raw.to_csv(index=False), revision)
            return {"rows": len(raw), "revision": revision + 1, **readiness(raw)}
        except (ValueError, UnicodeError) as exc:
            raise HTTPException(422, str(exc))

    @app.post("/predict")
    def prediction(payload: MealRequest, user_id=Depends(user)):
        raw, revision = history(user_id)
        if not artifact.is_file():
            raise HTTPException(503, "Shared model missing; run python -m backend.ml.train")
        try:
            bundle = load_population(artifact)
            result = forecast_meal(raw, payload.meal_time, payload.dietary_intake,
                [reading.model_dump() for reading in payload.recent_cgm], bundle)
            accounts.save_forecast(user_id, result, revision)
            return result
        except ValueError as exc:
            raise HTTPException(422, str(exc))

    @app.get("/explanations/daily")
    def daily(user_id=Depends(user)):
        if not artifact.is_file():
            raise HTTPException(503, "Shared model unavailable")
        saved = accounts.forecasts(user_id, load_population(artifact)["version"])
        rows = [{"Patient": user_id, "Recording": "active", "day": forecast["day"],
                 **effect} for forecast in saved for effect in forecast["feature_contributions"]]
        if not rows:
            return {"days": [], "message": "Make a meal prediction first"}
        ranked = daily_top_three(pd.DataFrame(rows))
        days = [{"day": int(day), "top_features": group.drop(columns=["Patient", "Recording", "day"]).to_dict("records")}
                for day, group in ranked.groupby("day")]
        return {"model_version": saved[0]["model_version"], "days": days,
                "meaning": "Average absolute additive contribution across saved meal/horizon predictions; not causal effects"}

    return app


app = create_app()
