"""
NEM Price Forecaster sidecar — FastAPI application.

Endpoints:
    GET  /health                — liveness + readiness
    GET  /price_forecast        — cached price forecast (raw + resampled)
    GET  /load_forecast         — cached load forecast (raw + resampled)
    POST /calibration/import    — feed a (predicted, actual import) observation
    POST /calibration/export    — feed a (predicted, actual export) observation
    POST /load_observation      — add a 30-min load measurement
    POST /trigger/price         — manually trigger a price predict cycle
    POST /trigger/train         — manually trigger a load train cycle

DESIGN PRINCIPLE: GET endpoints NEVER compute — they read from ForecastCache.
POST /trigger/* endpoints enqueue work in the background scheduler.

ESCALATION NOTE (for Opus parent):
The current price_model=darts implementation uses the isotonic calibrator for
per-slot calibration even in "darts" mode, because the Darts price model
requires a historical series of ACTUAL NEM prices which the sidecar does not
yet collect autonomously (it relies on the calibration POST endpoint to receive
actuals from the HA automation).  The Darts price model would be trained on
those actuals and predict the full-horizon in one shot.

Two options for completing this:
  A) Add a NEMWeb TRADINGPRICE client to autonomously download actuals (simple,
     requires another NEMWeb URL for the 5-min dispatch prices aggregated to 30min).
  B) Keep the current architecture where the HA integration or an automation
     POSTs actuals as they arrive from a realtime price sensor (Amber, etc).

Option B is already wired — POSTing via /calibration/import accumulates actuals;
the Darts model can be trained on those accumulated observations via
/trigger/train.  This is a design decision for Opus to resolve.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks, status
from pydantic import BaseModel, Field

# Ensure sidecar app/ is on the path (for Docker and local tests)
sys.path.insert(0, os.path.dirname(__file__))

from config import load_config, SidecarConfig, _VALID_PRICE_MODELS, _VALID_CALIBRATORS
from forecast_cache import ForecastCache, ForecastMetadata
from observation_store import ObservationStore
from price_engine import PriceEngine
from load_engine import LoadEngine
from scheduler import build_scheduler, _price_predict_job, _load_predict_job, _load_train_job

_LOGGER = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

# ---------------------------------------------------------------------------
# Application state (module-level singletons, initialised in lifespan)
# ---------------------------------------------------------------------------

_sidecar_config: Optional[SidecarConfig] = None
_forecast_cache: Optional[ForecastCache] = None
_observation_store: Optional[ObservationStore] = None
_price_engine: Optional[PriceEngine] = None
_load_engine: Optional[LoadEngine] = None
_scheduler = None

# Guards the runtime price-model / calibrator swap (POST /config).  The predict
# cycle runs on the APScheduler thread, so mutating _sidecar_config + rebuilding
# the calibrators must be serialised against a concurrent predict.
_config_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan — start background services on startup,
    flush observations on shutdown.
    """
    global _sidecar_config, _forecast_cache, _observation_store
    global _price_engine, _load_engine, _scheduler

    # --- Startup ---
    _sidecar_config = load_config()
    _LOGGER.info(
        "NEM Price Forecaster sidecar starting: region=%s price_model=%s",
        _sidecar_config.region,
        _sidecar_config.price_model,
    )

    os.makedirs(_sidecar_config.data_dir, exist_ok=True)

    _forecast_cache = ForecastCache()
    _forecast_cache.update_metadata(
        region=_sidecar_config.region,
        price_model=_sidecar_config.price_model,
        forecast_horizon_hours=_sidecar_config.forecast_horizon_hours,
        forecast_period_minutes=_sidecar_config.forecast_period_minutes,
    )

    _observation_store = ObservationStore(_sidecar_config.data_dir, _sidecar_config.region)
    _observation_store.load_from_disk()

    _price_engine = PriceEngine(_sidecar_config, _forecast_cache, _observation_store)
    _price_engine.restore_calibration_from_store()

    _load_engine = None
    if _sidecar_config.load_forecaster_enabled:
        _load_engine = LoadEngine(_sidecar_config, _forecast_cache, _observation_store)
        _load_engine.restore_model_from_disk()

    # Build and start scheduler
    _scheduler = build_scheduler(
        _sidecar_config,
        _price_engine,
        _load_engine if _load_engine else _create_noop_load_engine(),
        _observation_store,
    )
    _scheduler.start()
    _LOGGER.info("Background scheduler started")

    # Kick off an immediate price predict on startup (don't wait for first interval)
    await asyncio.get_event_loop().run_in_executor(
        None, lambda: _price_predict_job(_price_engine)
    )
    if _load_engine is not None:
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: _load_predict_job(_load_engine)
        )

    yield  # Application runs

    # --- Shutdown ---
    _LOGGER.info("Sidecar shutting down — flushing observations")
    if _scheduler:
        _scheduler.shutdown(wait=False)
    if _observation_store:
        _observation_store.flush_to_disk()
    _LOGGER.info("Sidecar shutdown complete")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="NEM Price Forecaster Sidecar",
    description=(
        "ML price + load forecaster sidecar for the NEM Price Forecaster HA integration. "
        "Forecasts are cached in memory; endpoints return sub-millisecond responses."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    region: str
    price_model: str
    price_forecast_ready: bool
    load_forecast_ready: bool
    price_computed_at: Optional[str] = None
    load_computed_at: Optional[str] = None
    price_calibration_active: bool
    price_calibration_observations: int
    load_model_trained: bool
    load_training_observations: int


class PriceForecastResponse(BaseModel):
    region: str
    price_model: str
    pd7day_run_datetime: Optional[str] = None
    computed_at: Optional[str] = None
    calibration_active: bool
    calibration_observations: int
    forecast_horizon_hours: int
    forecast_period_minutes: int
    forecast: list[dict[str, Any]]      # resampled at configured period
    raw_forecast: list[dict[str, Any]]  # native 30-min PD7DAY slots


class LoadForecastResponse(BaseModel):
    region: str
    model_trained: bool
    training_observations: int
    computed_at: Optional[str] = None
    forecast_horizon_hours: int
    forecast_period_minutes: int
    forecast: list[dict[str, Any]]      # resampled
    raw_forecast: list[dict[str, Any]]  # native 30-min slots


class CalibrationObservationRequest(BaseModel):
    predicted_rrp_per_mwh: float = Field(..., description="PD7DAY raw RRP $/MWh")
    actual_rrp_per_mwh: float = Field(..., description="Realised spot price $/MWh")
    hour_of_day: int = Field(..., ge=0, le=23, description="NEM hour of day (0-23)")
    observed_at: Optional[str] = Field(
        default=None,
        description="ISO-8601 UTC datetime (defaults to now)",
    )


class LoadObservationRequest(BaseModel):
    interval_start_utc: str = Field(..., description="ISO-8601 UTC start of the 30-min interval")
    load_watts: float = Field(..., ge=0, le=50000, description="30-min average house load (W)")


class TriggerResponse(BaseModel):
    triggered: bool
    message: str


class ConfigResponse(BaseModel):
    """Effective runtime forecast configuration."""
    price_model: str
    calibrator: str
    region: str


class ConfigUpdateRequest(BaseModel):
    """
    Runtime config change.  Either field may be omitted to leave it unchanged;
    at least one must be supplied.  Values are validated against the same
    allow-lists used at startup (see config.py).
    """
    price_model: Optional[str] = Field(
        default=None,
        description="One of: darts_naive_blend | isotonic | darts | hybrid",
    )
    calibrator: Optional[str] = Field(
        default=None,
        description="One of: isotonic | monotone_gbm",
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness + readiness check. Always returns 200."""
    _assert_ready()
    metadata = _forecast_cache.get_metadata()
    return HealthResponse(
        status="ok",
        region=metadata.region,
        price_model=metadata.price_model,
        price_forecast_ready=_forecast_cache.has_price_forecast,
        load_forecast_ready=_forecast_cache.has_load_forecast,
        price_computed_at=metadata.price_computed_at,
        load_computed_at=metadata.load_computed_at,
        price_calibration_active=metadata.price_calibration_active,
        price_calibration_observations=metadata.price_calibration_observations,
        load_model_trained=metadata.load_model_trained,
        load_training_observations=metadata.load_training_observations,
    )


@app.get("/price_forecast", response_model=PriceForecastResponse)
async def get_price_forecast() -> PriceForecastResponse:
    """
    Return the cached price forecast.

    Response includes:
      - forecast[]: resampled at configured period + horizon (for EMHASS / automations)
      - raw_forecast[]: native 30-min PD7DAY slots with raw + calibrated + retail prices
    """
    _assert_ready()

    raw_slots, resampled_slots, metadata = _forecast_cache.get_price_forecast()

    if not raw_slots and not resampled_slots:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Price forecast not yet available — sidecar is still warming up. "
                "Check /health for status."
            ),
        )

    return PriceForecastResponse(
        region=metadata.region,
        price_model=metadata.price_model,
        pd7day_run_datetime=metadata.pd7day_run_datetime,
        computed_at=metadata.price_computed_at,
        calibration_active=metadata.price_calibration_active,
        calibration_observations=metadata.price_calibration_observations,
        forecast_horizon_hours=metadata.forecast_horizon_hours,
        forecast_period_minutes=metadata.forecast_period_minutes,
        forecast=resampled_slots,
        raw_forecast=[slot.as_dict() for slot in raw_slots],
    )


@app.get("/load_forecast", response_model=LoadForecastResponse)
async def get_load_forecast() -> LoadForecastResponse:
    """
    Return the cached load forecast.

    Returns 503 if load forecaster is disabled or not yet trained.
    """
    _assert_ready()

    if _sidecar_config and not _sidecar_config.load_forecaster_enabled:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Load forecaster is disabled (set SIDECAR_LOAD_FORECASTER_ENABLED=true)",
        )

    raw_slots, resampled_slots, metadata = _forecast_cache.get_load_forecast()

    if not raw_slots and not resampled_slots:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Load forecast not yet available — need more observations or still training. "
                "Check /health for status."
            ),
        )

    return LoadForecastResponse(
        region=metadata.region,
        model_trained=metadata.load_model_trained,
        training_observations=metadata.load_training_observations,
        computed_at=metadata.load_computed_at,
        forecast_horizon_hours=metadata.forecast_horizon_hours,
        forecast_period_minutes=metadata.forecast_period_minutes,
        forecast=resampled_slots,
        raw_forecast=[slot.as_dict() for slot in raw_slots],
    )


@app.post("/calibration/import", status_code=status.HTTP_202_ACCEPTED)
async def post_import_calibration(request: CalibrationObservationRequest) -> dict:
    """
    Feed a (predicted PD7DAY RRP, actual import price) observation to the
    isotonic calibrator.  Call this from an HA automation or integration
    that has access to realised import prices (Amber, AEMO TRADINGPRICE, etc).
    """
    _assert_ready()
    observed_at = _parse_iso_or_now(request.observed_at)
    _price_engine.add_import_calibration_observation(
        request.predicted_rrp_per_mwh,
        request.actual_rrp_per_mwh,
        request.hour_of_day,
        observed_at,
    )
    return {"accepted": True, "hour_of_day": request.hour_of_day}


@app.post("/calibration/export", status_code=status.HTTP_202_ACCEPTED)
async def post_export_calibration(request: CalibrationObservationRequest) -> dict:
    """
    Feed a (predicted PD7DAY RRP, actual EXPORT/feed-in price) observation.
    Must use export prices, NOT import prices (separate calibrators).
    """
    _assert_ready()
    observed_at = _parse_iso_or_now(request.observed_at)
    _price_engine.add_export_calibration_observation(
        request.predicted_rrp_per_mwh,
        request.actual_rrp_per_mwh,
        request.hour_of_day,
        observed_at,
    )
    return {"accepted": True, "hour_of_day": request.hour_of_day}


@app.post("/load_observation", status_code=status.HTTP_202_ACCEPTED)
async def post_load_observation(request: LoadObservationRequest) -> dict:
    """
    Add a 30-min average house load measurement (watts).

    Call this from an HA automation at each 30-min boundary to feed the
    load forecaster.  Once enough observations accumulate (≥288 = ~3 days),
    the nightly train cycle will fit the Darts model.
    """
    _assert_ready()
    try:
        interval_start = datetime.fromisoformat(
            request.interval_start_utc.replace("Z", "+00:00")
        )
    except ValueError as parse_error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid interval_start_utc: {parse_error}",
        ) from parse_error

    _observation_store.add_load_observation(interval_start, request.load_watts)
    return {"accepted": True}


@app.post("/trigger/price", response_model=TriggerResponse)
async def trigger_price_predict(background_tasks: BackgroundTasks) -> TriggerResponse:
    """
    Manually trigger an immediate price predict cycle (force=True bypasses change detection).
    Runs in the background; returns immediately.
    """
    _assert_ready()
    # The shared scheduled job (_price_predict_job) calls run_predict_cycle()
    # WITHOUT force=True, so it can early-out via change detection — meaning
    # /trigger/price was a no-op whenever the engine considered the cache
    # fresh.  The docstring promised force=True; honour it by invoking the
    # engine directly.
    background_tasks.add_task(_price_engine.run_predict_cycle, True)
    return TriggerResponse(triggered=True, message="Price predict cycle enqueued (force=True)")


@app.post("/trigger/train", response_model=TriggerResponse)
async def trigger_load_train(background_tasks: BackgroundTasks) -> TriggerResponse:
    """
    Manually trigger a load model training cycle.  CPU-heavy; runs in background.
    """
    _assert_ready()
    if _load_engine is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Load forecaster disabled",
        )
    background_tasks.add_task(_load_train_job, _load_engine)
    return TriggerResponse(triggered=True, message="Load train cycle enqueued")


@app.get("/config", response_model=ConfigResponse)
async def get_config() -> ConfigResponse:
    """
    Return the effective runtime forecast config (price_model + calibrator + region).

    The HA integration calls this to default its picker to the live sidecar state.
    """
    _assert_ready()
    return ConfigResponse(
        price_model=_sidecar_config.price_model,
        calibrator=_sidecar_config.calibrator,
        region=_sidecar_config.region,
    )


@app.post("/config", response_model=ConfigResponse)
async def post_config(request: ConfigUpdateRequest) -> ConfigResponse:
    """
    Change the price model and/or calibrator at runtime, then re-predict.

    This is the missing link that makes the HA integration's method picker real:
    the sidecar reads SIDECAR_PRICE_MODEL / SIDECAR_CALIBRATOR from env only at
    startup, so without this endpoint a UI choice could never reach the engine.

    On a valid request we (under a lock, serialised against the scheduler's
    predict cycle):
      1. mutate the live SidecarConfig (price_model / calibrator),
      2. rebuild both calibrators for the new backend,
      3. re-seed them from the persisted observation store (so a calibrator
         switch never throws away accumulated calibration),
      4. lazily initialise the Darts price model if switching into a Darts mode,
      5. update the cache metadata, then
      6. force an immediate re-predict (off the event loop).

    Returns the new effective {price_model, calibrator, region}.
    Rejects unknown values with HTTP 400.
    """
    _assert_ready()

    if request.price_model is None and request.calibrator is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide at least one of price_model / calibrator",
        )
    if (
        request.price_model is not None
        and request.price_model not in _VALID_PRICE_MODELS
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"price_model must be one of {sorted(_VALID_PRICE_MODELS)}, "
                f"got: {request.price_model!r}"
            ),
        )
    if (
        request.calibrator is not None
        and request.calibrator not in _VALID_CALIBRATORS
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"calibrator must be one of {sorted(_VALID_CALIBRATORS)}, "
                f"got: {request.calibrator!r}"
            ),
        )

    # Apply the mutation + calibrator rebuild on the executor under the lock,
    # then force a re-predict (mirrors the startup _price_predict_job pattern).
    await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: _apply_runtime_config(request.price_model, request.calibrator),
    )

    return ConfigResponse(
        price_model=_sidecar_config.price_model,
        calibrator=_sidecar_config.calibrator,
        region=_sidecar_config.region,
    )


def _apply_runtime_config(
    price_model: Optional[str],
    calibrator: Optional[str],
) -> None:
    """
    Mutate config + rebuild calibrators + re-predict.  Runs on a worker thread.

    Guarded by _config_lock so it cannot race the scheduler's predict cycle.
    """
    with _config_lock:
        if price_model is not None:
            _sidecar_config.price_model = price_model
            _price_engine._config.price_model = price_model
        if calibrator is not None:
            _sidecar_config.calibrator = calibrator
            _price_engine._config.calibrator = calibrator

        # Rebuild both calibrators for the (possibly) new backend, then re-seed
        # them from the persisted observation store so a switch never discards
        # the accumulated calibration corpus.
        _price_engine._import_calibrator = _price_engine._build_calibrator()
        _price_engine._export_calibrator = _price_engine._build_calibrator()
        _price_engine.restore_calibration_from_store()

        # If we just switched into a Darts-backed mode and the model object was
        # never created (started in a non-Darts mode), build + load it now so the
        # new mode actually predicts with Darts rather than silently falling back.
        if (
            _price_engine._config.price_model in ("darts", "hybrid", "darts_naive_blend")
            and _price_engine._darts_price_model is None
        ):
            _price_engine._initialise_darts_price_model()
            try:
                _price_engine._darts_price_model.load_model_with_bundled_fallback(
                    _sidecar_config.data_dir, _sidecar_config.region
                )
            except Exception as load_error:  # pragma: no cover - defensive
                _LOGGER.warning(
                    "Runtime config: Darts model load failed (will self-train): %s",
                    load_error,
                )

        _forecast_cache.update_metadata(price_model=_sidecar_config.price_model)

        _LOGGER.info(
            "Runtime config applied: price_model=%s calibrator=%s",
            _sidecar_config.price_model,
            _sidecar_config.calibrator,
        )

    # Force an immediate re-predict so the new config takes effect right away.
    try:
        _price_engine.run_predict_cycle(force=True)
    except Exception as predict_error:  # pragma: no cover - defensive
        _LOGGER.error("Runtime config: forced re-predict failed: %s", predict_error)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert_ready() -> None:
    """Raise 503 if the sidecar is not yet initialised (startup race)."""
    if _forecast_cache is None or _price_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Sidecar still initialising",
        )


def _parse_iso_or_now(iso_string: Optional[str]) -> datetime:
    if iso_string is None:
        return datetime.now(timezone.utc)
    try:
        parsed = datetime.fromisoformat(iso_string.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    except ValueError:
        return datetime.now(timezone.utc)


def _create_noop_load_engine():
    """Return a minimal stub when load forecaster is disabled."""
    class _NoopLoadEngine:
        def run_predict_cycle(self, force: bool = False) -> bool:
            return False
        def run_train_cycle(self) -> bool:
            return False
    return _NoopLoadEngine()
