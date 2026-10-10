"""Authenticated account -> upload -> day-8 prediction and daily explanations."""
import io
import os
from pathlib import Path
import sqlite3
import time
from typing import Literal

import pandas as pd
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .accounts import Accounts
from .food.routes import build_food_router
from .ml.explain import daily_top_three
from .ml.service import (forecast_meal, load_population, readiness, recent_from_history,
                         validate_history)
from .ml import personal
from .ml.catalog import INSULINS, TABLETS, catalog, insulin_text, tablet_text
from .ml.spike import assess, variants
from .seed import seed_patients

ROOT = Path(__file__).resolve().parent
SPEEDS = [300, 10, 3, 1]      # real seconds per 5-minute CGM reading; 300 = real time
DEFAULT_SPEED = 3
TREATMENTS = ["Insulin dose - s.c.", "Non-insulin hypoglycemic agents",
              "CSII - bolus insulin (Novolin R, IU)", "CSII - basal insulin (Novolin R, IU / H)"]


def load_env_file(path):
    """Read KEY=VALUE lines from a local, gitignored .env; real environment variables win."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and key and not key.startswith("#"):
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def env_flag(name):
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


class Credentials(BaseModel):
    username: str = Field(min_length=3, max_length=80, pattern=r"^[a-zA-Z0-9_.@-]+$")
    password: str = Field(min_length=12, max_length=128)


class Login(BaseModel):
    # Length rules apply when passwords are set; sign-in only checks the stored hash.
    username: str = Field(min_length=3, max_length=80, pattern=r"^[a-zA-Z0-9_.@-]+$")
    password: str = Field(min_length=1, max_length=128)


class Reading(BaseModel):
    timestamp: str
    glucose: float = Field(gt=0, allow_inf_nan=False)


class Dose(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    dose: float = Field(gt=0, le=5000, allow_inf_nan=False)


class FoodItem(BaseModel):
    # USDA numbers from the meal builder for one food in dietary_intake (matched by name).
    name: str = Field(min_length=1, max_length=200)
    grams: float = Field(gt=0, le=5000, allow_inf_nan=False)
    carbs_g: float | None = Field(default=None, ge=0, le=5000, allow_inf_nan=False)
    sugar_g: float | None = Field(default=None, ge=0, le=5000, allow_inf_nan=False)


class MealRequest(BaseModel):
    # Omit to use the live clock's current time; later times are rejected (no peeking ahead).
    meal_time: str | None = None
    dietary_intake: str = Field(min_length=1, max_length=4000)
    foods: list[FoodItem] | None = Field(default=None, max_length=30)
    # Omit to use the latest uploaded CGM readings in the 30 minutes before the meal.
    recent_cgm: list[Reading] | None = Field(default=None, min_length=4, max_length=30)
    insulin: Dose | None = None
    tablet: Dose | None = None
    # False = what-if only. True = the user ate it: save the forecast to compare with what happens.
    confirm: bool = False


class Control(BaseModel):
    action: Literal["play", "pause", "restart", "speed", "seek"]
    seconds_per_reading: float | None = None
    minutes: float | None = Field(default=None, ge=0)  # seek: minutes after the day-8 start


def create_app(state_dir=None, model_path=None, allow_signup=None, admin=None,
               allowed_origins=None, frontend_dir=None, dev=None, evaluation_path=None,
               seed_recordings=None, seed_password=None, data_dir=None, food_cache=None):
    app = FastAPI(title="Glyimpse", version="1.1")

    origins = allowed_origins or [o.strip() for o in
        os.environ.get("GLYIMPSE_ALLOWED_ORIGINS", "*").split(",") if o.strip()]
    # Bearer tokens travel in headers, not cookies, so credentials stay off (also required for "*").
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Authorization", "Content-Type"],
    )

    state = Path(state_dir or os.environ.get("GLYIMPSE_STATE_DIR", ROOT / "runtime"))
    artifact = Path(model_path or os.environ.get("GLYIMPSE_MODEL_PATH", ROOT / "artifacts/population.joblib"))
    accounts = Accounts(state / "accounts.sqlite3")
    bearer = HTTPBearer(auto_error=False)
    signup_open = env_flag("GLYIMPSE_ALLOW_SIGNUP") if allow_signup is None else allow_signup
    dev = os.environ.get("GLYIMPSE_ENV", "").strip().lower() == "dev" if dev is None else dev
    evaluation = Path(evaluation_path or os.environ.get(
        "GLYIMPSE_EVALUATION_PATH", ROOT.parent / "output_v.1/day8_actual_vs_predicted.csv"))

    if admin is None and os.environ.get("GLYIMPSE_ADMIN_USERNAME"):
        admin = (os.environ["GLYIMPSE_ADMIN_USERNAME"], os.environ.get("GLYIMPSE_ADMIN_PASSWORD", ""))
    if admin:
        # Outside GLYIMPSE_ENV=dev a short admin password is a startup error, never a silent default.
        configured = (Login if dev else Credentials)(username=admin[0], password=admin[1])
        accounts.ensure_user(configured.username, configured.password)

    example_users = []
    if seed_recordings is None:
        seed_recordings = [r.strip() for r in os.environ.get("GLYIMPSE_SEED_RECORDINGS", "").split(",") if r.strip()]
    if seed_recordings:
        if not dev:
            raise RuntimeError("GLYIMPSE_SEED_RECORDINGS is for local development only (set GLYIMPSE_ENV=dev)")
        default = seed_password or os.environ.get("GLYIMPSE_SEED_PASSWORD", "")
        pairs = []
        for entry in seed_recordings:
            recording, _, own = entry.partition(":")
            pairs.append((recording.strip(), Login(username="seed", password=own.strip() or default).password))
        example_users = seed_patients(accounts, Path(data_dir or os.environ.get(
            "GLYIMPSE_DATA_DIR", ROOT.parent / "data/frontend_test")), pairs)

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

    profiles, bundles = {}, {}

    def profile(user_id, raw, revision):
        """This patient's spike thresholds and carb response from their first seven days."""
        key = (user_id, revision)
        if key not in profiles:
            if len(profiles) > 100:
                profiles.clear()
            profiles[key] = personal.profile(raw)
        return profiles[key]

    def summary(raw, revision, user_id=None):
        return {"rows": len(raw), "revision": revision,
                "first_reading": raw.Date.min().isoformat(), "last_reading": raw.Date.max().isoformat(),
                **readiness(raw), "personal": profile(user_id, raw, revision)}

    @app.get("/health")
    def health():
        return {"status": "ok", "shared_model_available": artifact.is_file(), "signup_open": signup_open,
                "example_users": example_users}

    @app.post("/accounts", status_code=201)
    def signup(payload: Credentials):
        if not signup_open:
            raise HTTPException(403, "Sign-up is disabled; ask the administrator for access")
        try:
            user_id = accounts.signup(payload.username, payload.password)
        except sqlite3.IntegrityError:
            raise HTTPException(409, "Username unavailable")
        return {"user_id": user_id, "access_token": accounts.login(payload.username, payload.password),
                "token_type": "bearer", "expires_in": 86400}

    @app.post("/sessions")
    def signin(payload: Login):
        token = accounts.login(payload.username, payload.password)
        if not token:
            raise HTTPException(401, "Invalid credentials")
        return {"access_token": token, "token_type": "bearer", "expires_in": 86400}

    @app.delete("/sessions", status_code=204)
    def signout(credentials: HTTPAuthorizationCredentials = Depends(bearer), user_id=Depends(user)):
        accounts.logout(credentials.credentials)

    @app.get("/me")
    def me(user_id=Depends(user)):
        username = accounts.username(user_id)
        return {"username": username, "admin": bool(admin) and username == admin[0].casefold()}

    @app.get("/history")
    def status(user_id=Depends(user)):
        raw, revision = history(user_id)
        return summary(raw, revision, user_id)

    def timeline(rows):
        rows = rows.reset_index(drop=True)
        stamps = rows.Date.dt.strftime("%Y-%m-%dT%H:%M:%S")
        glucose = pd.to_numeric(rows["CGM (mg / dl)"], errors="coerce")
        recorded = rows["recorded"] if "recorded" in rows else glucose
        cgm = []
        for t, g, r in zip(stamps[glucose.notna()], glucose.dropna(), recorded[glucose.notna()]):
            reading = {"timestamp": t, "glucose": float(g)}
            if abs(float(g) - float(r)) >= 0.05:
                reading["recorded"] = float(r)   # simulated: a confirmed meal's effect is added
            cgm.append(reading)
        text = rows[["Dietary intake", *TREATMENTS]].fillna("").astype(str).apply(lambda c: c.str.strip())
        events = []
        for i in text.index[text.ne("").any(axis=1)]:
            entry = {"timestamp": stamps[i]}
            if text.at[i, "Dietary intake"]:
                entry["dietary_intake"] = text.at[i, "Dietary intake"]
            treatments = [f"{col}: {text.at[i, col]}" for col in TREATMENTS if text.at[i, col]]
            if treatments:
                entry["treatment"] = "; ".join(treatments)
            events.append(entry)
        return cgm, events

    def clock(user_id, raw):
        """Replay clock: days 1-7 are history, then the recording plays back from the day-8 start."""
        start, end = raw.Date.min() + pd.Timedelta(days=7), raw.Date.max()
        sim = accounts.simulation(user_id)
        if sim is None:
            sim = {"base_minutes": 0.0, "anchor": time.time(), "seconds_per_reading": DEFAULT_SPEED, "paused": False}
            accounts.save_simulation(user_id, **sim)
        minutes = sim["base_minutes"]
        if not sim["paused"]:
            minutes += (time.time() - sim["anchor"]) / sim["seconds_per_reading"] * 5
        minutes = min(max(minutes, 0.0), max(0.0, (end - start).total_seconds() / 60))
        now = start + pd.Timedelta(minutes=5 * (minutes // 5))
        return {"now": now, "minutes": minutes, "start": start, "end": end, **sim}

    def confirmed(user_id):
        """Saved confirmed meals for the current model; none when the model is missing."""
        if not artifact.is_file():
            return []
        return accounts.forecasts(user_id, model_bundle()["version"])

    def live_state(raw, c, revision, since=None, meals=None):
        sim = personal.simulate(raw, meals or [])
        visible = sim[sim.Date <= c["now"]]
        if since:
            visible = visible[visible.Date > pd.to_datetime(since, utc=True).tz_convert(None)]
        cgm, events = timeline(visible)
        return {"now": c["now"].isoformat(), "day": int((c["now"] - raw.Date.min()).total_seconds() // 86400) + 1,
                "first_reading": raw.Date.min().isoformat(), "day8_start": c["start"].isoformat(),
                "end": c["end"].isoformat(), "finished": c["now"] >= c["end"], "paused": c["paused"],
                "seconds_per_reading": c["seconds_per_reading"], "speeds": SPEEDS, "revision": revision,
                "readings": cgm, "events": events}

    @app.get("/history/readings")
    def readings(user_id=Depends(user)):
        raw, revision = history(user_id)
        sim = personal.simulate(raw, confirmed(user_id))
        cgm, events = timeline(sim[sim.Date <= clock(user_id, raw)["now"]])
        return {"revision": revision, "readings": cgm, "events": events}

    @app.get("/live")
    def live(since: str | None = None, user_id=Depends(user)):
        """Everything up to the replay clock's current time (only new rows when `since` is given).
        Readings after a confirmed meal include its simulated effect; `recorded` keeps the real value."""
        raw, revision = history(user_id)
        return live_state(raw, clock(user_id, raw), revision, since, confirmed(user_id))

    @app.post("/live/control")
    def control(payload: Control, user_id=Depends(user)):
        raw, revision = history(user_id)
        c = clock(user_id, raw)
        minutes, speed, paused = c["minutes"], c["seconds_per_reading"], c["paused"]
        if payload.action == "pause":
            paused = True
        elif payload.action == "play":
            paused = False
        elif payload.action == "restart":
            minutes = 0.0
            accounts.clear_forecasts(user_id)
        elif payload.action == "speed":
            if payload.seconds_per_reading not in SPEEDS:
                raise HTTPException(422, f"seconds_per_reading must be one of {SPEEDS}")
            speed = payload.seconds_per_reading
        elif payload.minutes is None:
            raise HTTPException(422, "seek needs minutes")
        else:
            minutes = payload.minutes
        accounts.save_simulation(user_id, minutes, time.time(), speed, paused)
        return live_state(raw, clock(user_id, raw), revision, meals=confirmed(user_id))

    @app.get("/catalog")
    def choices():
        return catalog()

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
            return summary(raw, revision + 1, user_id)
        except (ValueError, UnicodeError) as exc:
            raise HTTPException(422, str(exc))

    def meal_inputs(payload, user_id, raw, sim):
        """Meal time (default: live clock now), recent CGM (simulated after confirmed meals), and medication."""
        now = clock(user_id, raw)["now"]
        meal_time = pd.to_datetime(payload.meal_time, utc=True).tz_convert(None) if payload.meal_time else now
        if meal_time > now:
            raise HTTPException(422, f"The live clock is at {now:%b %d %H:%M}; forecasts can only start at or before now")
        stamp = meal_time.isoformat()
        recent = ([reading.model_dump() for reading in payload.recent_cgm] if payload.recent_cgm
                  else recent_from_history(sim, stamp))
        treatments = {}
        if payload.insulin:
            if payload.insulin.name not in INSULINS:
                raise HTTPException(422, f"Unknown insulin: {payload.insulin.name}")
            treatments["Insulin dose - s.c."] = insulin_text(payload.insulin.name, payload.insulin.dose)
        if payload.tablet:
            if payload.tablet.name not in [name for name, _ in TABLETS]:
                raise HTTPException(422, f"Unknown tablet: {payload.tablet.name}")
            treatments["Non-insulin hypoglycemic agents"] = tablet_text(payload.tablet.name, payload.tablet.dose)
        return stamp, recent, treatments

    def model_bundle():
        if not artifact.is_file():
            raise HTTPException(503, "Shared model missing; run python -m backend.ml.train")
        stamp = artifact.stat().st_mtime_ns
        if bundles.get("stamp") != stamp:
            bundles.update(stamp=stamp, bundle=load_population(artifact))
        return bundles["bundle"]

    def meal_context(payload, user_id):
        """Everything a meal forecast needs: recorded history, the simulated replay, and this patient's profile."""
        raw, revision = history(user_id)
        bundle = model_bundle()
        meals = accounts.forecasts(user_id, bundle["version"])
        sim = personal.simulate(raw, meals)
        stamp, recent, treatments = meal_inputs(payload, user_id, raw, sim)
        foods = [food.model_dump() for food in payload.foods or []]
        return {"raw": raw, "revision": revision, "bundle": bundle, "model_raw": personal.with_meals(sim, meals),
                "stamp": stamp, "recent": recent, "treatments": treatments, "foods": foods,
                "me": profile(user_id, raw, revision)}

    def forecast(ctx, text):
        """Model forecast plus the patient's sugar response (which the model can't see)."""
        result = forecast_meal(ctx["model_raw"], ctx["stamp"], text, ctx["recent"], ctx["bundle"],
                               ctx["treatments"], history=ctx["raw"])
        return personal.add_sugar(result, text, ctx["foods"], ctx["me"])

    @app.post("/predict")
    def prediction(payload: MealRequest, user_id=Depends(user)):
        try:
            ctx = meal_context(payload, user_id)
            text = payload.dietary_intake
            result = forecast(ctx, text)
            spike = {**assess(result, text, ctx["me"]["thresholds"]), "sweet_items": variants(text)["sweet_items"]}
            meal = {**result, "dietary_intake": text, "medication": ctx["treatments"], "spike": spike}
            if payload.confirm:
                without = forecast_meal(ctx["model_raw"], ctx["stamp"], "No food ingested", ctx["recent"],
                                        ctx["bundle"], ctx["treatments"], history=ctx["raw"])
                meal["meal_effect"] = personal.meal_effect(result, without)
                accounts.save_forecast(user_id, meal, ctx["revision"])
            return {**meal, "recent_cgm": ctx["recent"], "confirmed": payload.confirm}
        except ValueError as exc:
            raise HTTPException(422, str(exc))

    @app.get("/meals")
    def confirmed_meals(user_id=Depends(user)):
        """Meals the user confirmed eating, each with the forecast made at that time."""
        keys = ["meal_time", "day", "dietary_intake", "medication", "baseline_glucose", "personalized", "predictions",
                "spike", "meal_effect", "sugar_adjustment"]
        saved = accounts.forecasts(user_id, model_bundle()["version"])
        return {"meals": sorted(({k: m.get(k) for k in keys} for m in saved), key=lambda m: m["meal_time"])}

    @app.post("/predict/alternatives")
    def alternatives(payload: MealRequest, user_id=Depends(user)):
        """What-if forecasts for changed versions of the planned meal. Nothing is saved."""
        try:
            ctx = meal_context(payload, user_id)
            limits = ctx["me"]["thresholds"]
            original = assess(forecast(ctx, payload.dietary_intake), payload.dietary_intake, limits)
            options = variants(payload.dietary_intake)
            rows = []
            for option in options["variants"]:
                result = forecast(ctx, option["dietary_intake"])
                spike = assess(result, option["dietary_intake"], limits)
                rows.append({**option, "spike": spike, "predictions": result["predictions"],
                             "peak_change_mg_dl": spike["peak_glucose_mg_dl"] - original["peak_glucose_mg_dl"]})
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        rows.sort(key=lambda row: row["spike"]["peak_glucose_mg_dl"])
        return {"original": {"dietary_intake": payload.dietary_intake, "spike": original},
                "sweet_items": options["sweet_items"], "alternatives": rows,
                "meaning": "Model forecasts for the same meal time and starting glucose; not medical advice"}

    @app.get("/model/accuracy")
    def accuracy(user_id=Depends(user)):
        """Offline day-8 evaluation (actual vs predicted) for the shared model, if available."""
        if not evaluation.is_file():
            raise HTTPException(404, "No offline evaluation results on this server")
        results = pd.read_csv(evaluation)

        def scores(group):
            error = group.absolute_error_mg_dl
            return {"forecasts": len(group), "patients": int(group.Patient.nunique()),
                    "mae_mg_dl": float(error.mean()), "rmse_mg_dl": float((error ** 2).mean() ** 0.5),
                    "within_20_mg_dl_percent": float(100 * group.within_20_mg_dl.mean()),
                    "within_15_percent_percent": float(100 * group.within_15_percent.mean())}

        return {"source": evaluation.name, "overall": scores(results),
                "horizons": [{"horizon_minutes": int(h), **scores(g)} for h, g in results.groupby("horizon_minutes")],
                "points": results[["horizon_minutes", "actual_glucose", "predicted_glucose"]].round(1).values.tolist(),
                "meaning": "Held-out patients' day-8 meals; offline results, not a guarantee for a new user"}

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

    # Photo/typed food -> USDA nutrients & standard portions. Authenticated like the meal endpoints;
    # the feed into /predict reuses the same dietary-intake text the forecaster already reads.
    app.include_router(build_food_router(user, cache=food_cache))

    # Local dev: serve the static frontend from the same origin. Set GLYIMPSE_FRONTEND_DIR= (empty) to disable.
    static = frontend_dir if frontend_dir is not None else os.environ.get(
        "GLYIMPSE_FRONTEND_DIR", str(ROOT.parent / "frontend"))
    if static and Path(static).is_dir():
        app.mount("/", StaticFiles(directory=static, html=True), name="frontend")

    return app


load_env_file(ROOT / ".env")
load_env_file(ROOT.parent / ".env")
app = create_app()
