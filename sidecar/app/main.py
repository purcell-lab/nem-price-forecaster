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
import shutil
import sys
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, List, Optional

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


# ---------------------------------------------------------------------------
# Data-dir auto-migration
# ---------------------------------------------------------------------------
#
# Background: in the HA add-on, the Supervisor maps /share onto
# /nem_forecaster_data inside the container.  Historically the sidecar wrote
# directly under that mount, which co-mingled NPF files with other add-ons
# that share /share (e.g. EMHASS).
#
# To isolate NPF state we now point SIDECAR_DATA_DIR at a SUBDIRECTORY of the
# mount (e.g. /nem_forecaster_data/nem_forecaster_data).  On first start after
# the upgrade, NPF-pattern files at the legacy parent path are moved into the
# new subdir so we preserve observations / calibration / archive across the
# rename.  The function is idempotent: if the new subdir already contains a
# given file, the legacy copy is left alone (operator can clean up manually).
#
# Only files matching known NPF patterns are touched.  EMHASS pickles, weather
# caches, etc. at the parent are never moved.
# ---------------------------------------------------------------------------

# Filename patterns owned by NPF.  Anything under the parent that does NOT
# match one of these patterns is left untouched.
_NPF_FILE_PATTERNS = (
    "calibration_",      # calibration_<region>.json
    "load_obs_",         # load_obs_<region>.json
    "price_darts_model", # price_darts_model.pkl + .meta.json
    "load_darts_model",  # load_darts_model.pkl + .meta.json
)
_NPF_DIR_NAMES = ("aemo_archive",)


def _is_npf_owned(name: str) -> bool:
    """True if a file/dir basename belongs to NPF (and may be migrated)."""
    if name in _NPF_DIR_NAMES:
        return True
    return any(name.startswith(p) for p in _NPF_FILE_PATTERNS)


def _migrate_legacy_data_dir(data_dir: str) -> List[str]:
    """Move NPF files from the legacy parent path into ``data_dir``.

    Only invoked when ``data_dir`` is a subdirectory of a mount root that
    might still contain legacy NPF files at the parent level.  Returns the
    list of basenames that were moved (for logging / introspection).

    Idempotent: safe to call on every startup.
    """
    parent = os.path.dirname(os.path.normpath(data_dir))
    if not parent or parent == "/" or not os.path.isdir(parent):
        return []
    # Guard: only migrate when the parent looks like the addon share mount.
    # We refuse to migrate from "/" or from arbitrary user paths to avoid
    # surprising behaviour for standalone deployments.
    if not parent.startswith("/nem_forecaster_data"):
        return []

    moved: List[str] = []
    try:
        entries = os.listdir(parent)
    except OSError as exc:
        _LOGGER.warning("data_dir migration: cannot list %s: %s", parent, exc)
        return []

    for name in entries:
        # Skip the new subdir itself.
        if name == os.path.basename(os.path.normpath(data_dir)):
            continue
        if not _is_npf_owned(name):
            continue
        src = os.path.join(parent, name)
        dst = os.path.join(data_dir, name)
        if os.path.exists(dst):
            _LOGGER.info(
                "data_dir migration: %s already exists at new path, leaving legacy copy at %s",
                name,
                src,
            )
            continue
        try:
            shutil.move(src, dst)
            moved.append(name)
            _LOGGER.info("data_dir migration: moved %s -> %s", src, dst)
        except OSError as exc:
            _LOGGER.warning(
                "data_dir migration: failed to move %s -> %s: %s", src, dst, exc
            )

    if moved:
        _LOGGER.info(
            "data_dir migration: moved %d NPF entr%s from %s into %s",
            len(moved),
            "y" if len(moved) == 1 else "ies",
            parent,
            data_dir,
        )
    return moved


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

    # Migrate any legacy NPF files from the parent share path into the new
    # subdirectory layout.  Idempotent; only moves NPF-owned filenames.
    try:
        _migrate_legacy_data_dir(_sidecar_config.data_dir)
    except Exception as exc:  # never let migration kill startup
        _LOGGER.exception("data_dir migration raised; continuing: %s", exc)

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


class PersistedFileInfo(BaseModel):
    """On-disk inventory entry for one file under SIDECAR_DATA_DIR."""
    path: str
    size_bytes: int
    modified_utc: Optional[str] = None
    note: Optional[str] = None


class DataInventory(BaseModel):
    """
    Observation/row counts for the on-disk datasets the sidecar keeps under
    SIDECAR_DATA_DIR.  Counts answer 'how much history actually survived the
    last Supervisor action?' at a glance, without parsing the raw files.

    Fields are best-effort: a missing file reports 0; a corrupt file reports
    None and includes a brief note.  All counts are derived by inspecting
    files on disk at request time, NOT from in-memory state, so they reflect
    what would be re-loaded on the next container restart.
    """
    calibration_import_observations: Optional[int] = Field(
        default=None,
        description="Rows in calibration_<region>.json -> import_observations",
    )
    calibration_export_observations: Optional[int] = Field(
        default=None,
        description="Rows in calibration_<region>.json -> export_observations",
    )
    load_observations: Optional[int] = Field(
        default=None,
        description="Rows in load_obs_<region>.json -> load_observations",
    )
    aemo_rrp_days_cached: int = Field(
        default=0,
        description="Days cached in aemo_archive/ (one JSON file per day)",
    )
    aemo_rrp_intervals_total: int = Field(
        default=0,
        description="Total 30-min RRP rows across all aemo_archive day files",
    )
    aemo_rrp_earliest_utc: Optional[str] = None
    aemo_rrp_latest_utc: Optional[str] = None
    price_darts_model_present: bool = Field(
        default=False,
        description="price_darts_model.pkl exists on disk",
    )
    load_darts_model_present: bool = Field(
        default=False,
        description="load_darts_model.pkl exists on disk",
    )
    pd7day_filename: Optional[str] = Field(
        default=None,
        description="NEMWeb PD7DAY ZIP filename currently held in memory",
    )
    pd7day_run_datetime_utc: Optional[str] = Field(
        default=None,
        description="PD7DAY predispatch run datetime (UTC)",
    )
    pd7day_region: Optional[str] = Field(
        default=None,
        description="Region the cached PD7DAY forecast was loaded for",
    )
    pd7day_slots_in_memory: Optional[int] = Field(
        default=None,
        description="Number of 30-min RRP slots cached in PD7DAY memory",
    )
    pd7day_earliest_utc: Optional[str] = Field(
        default=None,
        description="Interval-start UTC of the first PD7DAY slot in memory",
    )
    pd7day_latest_utc: Optional[str] = Field(
        default=None,
        description="Interval-start UTC of the last PD7DAY slot in memory",
    )
    notes: list[str] = Field(
        default_factory=list,
        description="Per-file warnings (corrupt JSON, permission denied, etc.)",
    )


class VersionResponse(BaseModel):
    """
    Build + runtime identity of the sidecar.

    Lets operators answer "which commit is the container ACTUALLY running?"
    without shelling into the supervisor.  Git SHA / build time / addon
    version are baked at image-build time from Dockerfile ARGs; everything
    else is observed at request time.
    """
    git_sha: str
    git_branch: Optional[str] = None
    build_time_utc: str
    addon_version: Optional[str] = None
    build_arch: Optional[str] = None
    api_version: str
    python_version: str
    data_dir: str
    region: str
    persisted_files: list[PersistedFileInfo]
    data_inventory: DataInventory


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


@app.get("/version", response_model=VersionResponse)
async def version() -> VersionResponse:
    """
    Report build + runtime identity.

    Answers questions an upgrade operator routinely needs:
      * Which commit is the live container running?  (git_sha)
      * When was the image built?                    (build_time_utc)
      * Which add-on version did Supervisor install? (addon_version)
      * Where is observation state actually written? (data_dir)
      * What is currently persisted on disk?         (persisted_files)

    Useful for verifying that a Supervisor `update` / `rebuild` / `restart`
    actually rolled forward to the expected commit, and for confirming that
    the calibration / load-observation JSON files survived the operation
    (the data_dir is a host bind-mount; this endpoint shows it from the
    container's point of view).

    Build identity (git_sha, build_time_utc, addon_version, build_arch) is
    baked at image build time via Dockerfile ARGs.  Values are 'unknown'
    when the image is built locally without those args.  All other fields
    are observed at request time.
    """
    import platform

    git_sha = os.environ.get("SIDECAR_GIT_SHA", "unknown")
    git_branch = os.environ.get("SIDECAR_GIT_BRANCH") or None
    build_time = os.environ.get("SIDECAR_BUILD_TIME", "unknown")
    addon_version = os.environ.get("SIDECAR_ADDON_VERSION") or None
    build_arch = os.environ.get("SIDECAR_BUILD_ARCH") or None

    data_dir = (
        _sidecar_config.data_dir if _sidecar_config is not None
        else os.environ.get("SIDECAR_DATA_DIR", "/data")
    )
    region = _sidecar_config.region if _sidecar_config is not None else "unknown"

    persisted: list[PersistedFileInfo] = []
    if os.path.isdir(data_dir):
        # Top-level inventory (calibration, load_obs, model pickles, meta sidecars).
        try:
            entries = sorted(os.listdir(data_dir))
        except OSError as listdir_error:
            persisted.append(PersistedFileInfo(
                path=data_dir,
                size_bytes=0,
                note=f"listdir failed: {listdir_error}",
            ))
            entries = []
        for entry in entries:
            full_path = os.path.join(data_dir, entry)
            try:
                stat_result = os.stat(full_path)
            except OSError:
                continue
            if not os.path.isfile(full_path):
                continue
            persisted.append(PersistedFileInfo(
                path=entry,
                size_bytes=int(stat_result.st_size),
                modified_utc=datetime.fromtimestamp(
                    stat_result.st_mtime, tz=timezone.utc,
                ).isoformat(),
            ))
    else:
        persisted.append(PersistedFileInfo(
            path=data_dir,
            size_bytes=0,
            note="data_dir does not exist",
        ))

    inventory = _inventory_data_dir(data_dir, region)
    _attach_pd7day_inventory(inventory)

    return VersionResponse(
        git_sha=git_sha,
        git_branch=git_branch,
        build_time_utc=build_time,
        addon_version=addon_version,
        build_arch=build_arch,
        api_version=app.version,
        python_version=platform.python_version(),
        data_dir=data_dir,
        region=region,
        persisted_files=persisted,
        data_inventory=inventory,
    )


def _inventory_data_dir(data_dir: str, region: str) -> "DataInventory":
    """
    Walk SIDECAR_DATA_DIR and count rows in each known dataset.

    Best-effort by design: a missing file reports 0/None, a corrupt or
    permission-denied file reports None and adds a note.  Never raises.
    """
    import json as _json

    notes: list[str] = []
    calibration_import_count: Optional[int] = None
    calibration_export_count: Optional[int] = None
    load_obs_count: Optional[int] = None
    aemo_days = 0
    aemo_intervals = 0
    aemo_earliest: Optional[str] = None
    aemo_latest: Optional[str] = None
    price_model_present = False
    load_model_present = False

    if not os.path.isdir(data_dir):
        return DataInventory(notes=[f"data_dir does not exist: {data_dir}"])

    # The on-disk filename conventions in this codebase are inconsistent:
    #   observation_store.py lowercases the region (calibration_qld1.json)
    #   aemo_historical_client.py uppercases it (price_QLD1_YYYYMMDD.json)
    # Cope with both so the inventory works regardless of which writer ran
    # last and regardless of any future normalisation change.
    region_lower = region.lower()
    region_upper = region.upper()

    calibration_path = os.path.join(data_dir, f"calibration_{region_lower}.json")
    if not os.path.exists(calibration_path):
        # Fallback: tolerate uppercase if the writer convention ever flips.
        alt = os.path.join(data_dir, f"calibration_{region_upper}.json")
        if os.path.exists(alt):
            calibration_path = alt
    if os.path.exists(calibration_path):
        try:
            with open(calibration_path, encoding="utf-8") as calibration_file:
                payload = _json.load(calibration_file)
            calibration_import_count = len(payload.get("import_observations", []))
            calibration_export_count = len(payload.get("export_observations", []))
        except (OSError, _json.JSONDecodeError, TypeError) as calibration_error:
            notes.append(
                f"calibration_{region}.json unreadable: {calibration_error}"
            )
    else:
        calibration_import_count = 0
        calibration_export_count = 0

    load_obs_path = os.path.join(data_dir, f"load_obs_{region_lower}.json")
    if not os.path.exists(load_obs_path):
        alt = os.path.join(data_dir, f"load_obs_{region_upper}.json")
        if os.path.exists(alt):
            load_obs_path = alt
    if os.path.exists(load_obs_path):
        try:
            with open(load_obs_path, encoding="utf-8") as load_obs_file:
                payload = _json.load(load_obs_file)
            load_obs_count = len(payload.get("load_observations", []))
        except (OSError, _json.JSONDecodeError, TypeError) as load_obs_error:
            notes.append(
                f"load_obs_{region}.json unreadable: {load_obs_error}"
            )
    else:
        load_obs_count = 0

    aemo_archive_dir = os.path.join(data_dir, "aemo_archive")
    if os.path.isdir(aemo_archive_dir):
        try:
            aemo_files = sorted(
                f for f in os.listdir(aemo_archive_dir)
                if (
                    f.startswith(f"price_{region_upper}_")
                    or f.startswith(f"price_{region_lower}_")
                )
                and f.endswith(".json")
            )
        except OSError as aemo_listdir_error:
            notes.append(f"aemo_archive listdir failed: {aemo_listdir_error}")
            aemo_files = []
        aemo_days = len(aemo_files)
        for aemo_filename in aemo_files:
            full_path = os.path.join(aemo_archive_dir, aemo_filename)
            try:
                with open(full_path, encoding="utf-8") as aemo_file:
                    slots = _json.load(aemo_file)
                if not isinstance(slots, list) or not slots:
                    continue
                aemo_intervals += len(slots)
                first_iso = slots[0].get("interval_start_utc")
                last_iso = slots[-1].get("interval_start_utc")
                if first_iso and (aemo_earliest is None or first_iso < aemo_earliest):
                    aemo_earliest = first_iso
                if last_iso and (aemo_latest is None or last_iso > aemo_latest):
                    aemo_latest = last_iso
            except (OSError, _json.JSONDecodeError, TypeError, AttributeError) as aemo_error:
                notes.append(f"{aemo_filename} unreadable: {aemo_error}")

    price_model_present = os.path.isfile(
        os.path.join(data_dir, "price_darts_model.pkl")
    )
    load_model_present = os.path.isfile(
        os.path.join(data_dir, "load_darts_model.pkl")
    )

    return DataInventory(
        calibration_import_observations=calibration_import_count,
        calibration_export_observations=calibration_export_count,
        load_observations=load_obs_count,
        aemo_rrp_days_cached=aemo_days,
        aemo_rrp_intervals_total=aemo_intervals,
        aemo_rrp_earliest_utc=aemo_earliest,
        aemo_rrp_latest_utc=aemo_latest,
        price_darts_model_present=price_model_present,
        load_darts_model_present=load_model_present,
        notes=notes,
    )


def _attach_pd7day_inventory(inventory: "DataInventory") -> None:
    """
    Populate PD7DAY in-memory fields on the inventory from the live
    PriceEngine, if it has been initialised.

    PD7DAY is in-memory only — the predispatch ZIP is not persisted to
    SIDECAR_DATA_DIR — so these fields complement the on-disk inventory
    with what would be LOST on a container restart (until the next
    predict cycle re-fetches PD7DAY from NEMWeb, typically within minutes).
    Best-effort: any failure to introspect the engine adds a note and
    leaves the PD7DAY fields unset.
    """
    if _price_engine is None:
        return
    try:
        client = getattr(_price_engine, "_pd7day_client", None)
        if client is None:
            return
        forecast = getattr(client, "_cached_forecast", None)
        filename = getattr(client, "_cached_filename", None)
        if filename is not None:
            inventory.pd7day_filename = str(filename)
        if forecast is None:
            return
        inventory.pd7day_region = getattr(forecast, "region", None)
        run_dt = getattr(forecast, "run_datetime_utc", None)
        if run_dt is not None:
            inventory.pd7day_run_datetime_utc = run_dt.isoformat()
        slots = list(getattr(forecast, "slots", []) or [])
        inventory.pd7day_slots_in_memory = len(slots)
        if slots:
            first_dt = getattr(slots[0], "interval_start_utc", None)
            last_dt = getattr(slots[-1], "interval_start_utc", None)
            if first_dt is not None:
                inventory.pd7day_earliest_utc = first_dt.isoformat()
            if last_dt is not None:
                inventory.pd7day_latest_utc = last_dt.isoformat()
    except Exception as pd7day_error:  # pragma: no cover - defensive
        inventory.notes.append(f"pd7day introspection failed: {pd7day_error}")


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
