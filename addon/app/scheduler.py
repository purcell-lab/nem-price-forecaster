"""
Background scheduler — drives compute cycles without blocking the HTTP server.

Uses APScheduler (BackgroundScheduler) which runs jobs in a ThreadPoolExecutor
separate from the FastAPI/uvicorn event loop.

Scheduled jobs:
  1. price_predict    — every predict_interval_seconds (default 300)
                        also triggers immediately on startup
  2. load_predict     — every predict_interval_seconds
  3. load_train       — nightly cron (default "0 2 * * *")
  4. pd7day_poll      — every 5 min: cheap index check, triggers price_predict
                        on new file (change-detection via PriceEngine)
  5. obs_flush        — every 30 min: persist observations to disk

PD7DAY polling strategy:
  - The NEMWeb index page is ~10 KB; we fetch it to discover the latest filename.
  - If the filename changed → price_predict is called immediately (event-driven).
  - This means most cycles are no-ops (cache-hit skip inside PriceEngine).

Thread safety:
  - APScheduler serialises same-id jobs by default (misfire_grace_time / max_instances=1).
  - PriceEngine.run_predict_cycle() is idempotent.
  - LoadEngine.run_train_cycle() is non-reentrant (max_instances=1 prevents overlap).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.schedulers.background import BackgroundScheduler

if TYPE_CHECKING:
    from config import SidecarConfig
    from price_engine import PriceEngine
    from load_engine import LoadEngine
    from observation_store import ObservationStore

_LOGGER = logging.getLogger(__name__)

_OBS_FLUSH_INTERVAL_SECONDS = 1800  # 30 min


def build_scheduler(
    config: "SidecarConfig",
    price_engine: "PriceEngine",
    load_engine: "LoadEngine",
    store: "ObservationStore",
) -> BackgroundScheduler:
    """
    Build and return a configured APScheduler BackgroundScheduler.

    Caller is responsible for calling scheduler.start() and scheduler.shutdown().
    """
    executors = {
        "default": ThreadPoolExecutor(max_workers=2),
        # Heavy training jobs get their own thread so they don't block predict
        "train": ThreadPoolExecutor(max_workers=1),
    }

    job_defaults = {
        "coalesce": True,        # merge multiple missed firings into one
        "max_instances": 1,      # never run the same job concurrently
        "misfire_grace_time": 60,
    }

    scheduler = BackgroundScheduler(
        executors=executors,
        job_defaults=job_defaults,
        timezone="UTC",
    )

    # ------------------------------------------------------------------
    # Price predict — runs every predict_interval_seconds
    # ------------------------------------------------------------------
    scheduler.add_job(
        _price_predict_job,
        trigger="interval",
        seconds=config.predict_interval_seconds,
        id="price_predict",
        args=[price_engine],
    )

    # ------------------------------------------------------------------
    # Load predict — runs every predict_interval_seconds
    # ------------------------------------------------------------------
    if config.load_forecaster_enabled:
        scheduler.add_job(
            _load_predict_job,
            trigger="interval",
            seconds=config.predict_interval_seconds,
            id="load_predict",
            args=[load_engine],
        )

        # ------------------------------------------------------------------
        # Load train — nightly cron
        # ------------------------------------------------------------------
        cron_parts = _parse_cron(config.train_cron)
        scheduler.add_job(
            _load_train_job,
            trigger="cron",
            id="load_train",
            executor="train",
            args=[load_engine],
            **cron_parts,
        )

    # ------------------------------------------------------------------
    # Observation flush — every 30 min
    # ------------------------------------------------------------------
    scheduler.add_job(
        _obs_flush_job,
        trigger="interval",
        seconds=_OBS_FLUSH_INTERVAL_SECONDS,
        id="obs_flush",
        args=[store],
    )

    return scheduler


# ---------------------------------------------------------------------------
# Job functions (run in background threads)
# ---------------------------------------------------------------------------

def _price_predict_job(price_engine: "PriceEngine") -> None:
    try:
        updated = price_engine.run_predict_cycle()
        if updated:
            _LOGGER.debug("Scheduled price predict: cache updated")
    except Exception as job_error:
        _LOGGER.error("Price predict job failed: %s", job_error)


def _load_predict_job(load_engine: "LoadEngine") -> None:
    try:
        updated = load_engine.run_predict_cycle()
        if updated:
            _LOGGER.debug("Scheduled load predict: cache updated")
    except Exception as job_error:
        _LOGGER.error("Load predict job failed: %s", job_error)


def _load_train_job(load_engine: "LoadEngine") -> None:
    _LOGGER.info("Nightly load train starting")
    try:
        success = load_engine.run_train_cycle()
        _LOGGER.info("Nightly load train finished: success=%s", success)
    except Exception as job_error:
        _LOGGER.error("Load train job failed: %s", job_error)


def _obs_flush_job(store: "ObservationStore") -> None:
    try:
        store.flush_to_disk()
    except Exception as flush_error:
        _LOGGER.error("Observation flush failed: %s", flush_error)


# ---------------------------------------------------------------------------
# Cron string parser (handles standard 5-field cron expressions)
# ---------------------------------------------------------------------------

def _parse_cron(cron_expression: str) -> dict:
    """
    Parse a 5-field cron expression into APScheduler CronTrigger kwargs.

    Fields: minute hour day month day_of_week
    """
    parts = cron_expression.strip().split()
    if len(parts) != 5:
        _LOGGER.warning(
            "Invalid cron expression %r (need 5 fields); defaulting to 02:00 daily",
            cron_expression,
        )
        return {"hour": 2, "minute": 0}

    cron_field_names = ["minute", "hour", "day", "month", "day_of_week"]
    result = {}
    for field_name, field_value in zip(cron_field_names, parts):
        if field_value != "*":
            result[field_name] = field_value
    return result
