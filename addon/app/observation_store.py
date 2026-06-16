"""
Observation store — persists price calibration and load training data.

This is the sidecar's equivalent of calibration_store.py in the HA component.
Data is written to JSON in SIDECAR_DATA_DIR so it survives container restarts.

Two JSON files per region:
  <data_dir>/calibration_<region>.json  — price calibration (import + export)
  <data_dir>/load_obs_<region>.json     — load training observations

Writes are atomic (temp-file rename) to avoid corruption on restart.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

from isotonic_calibrator import CalibrationObservation
from load_forecaster import LoadObservation

_LOGGER = logging.getLogger(__name__)

_CALIBRATION_MAX_AGE_DAYS = 120
_LOAD_MAX_AGE_DAYS = 90


class ObservationStore:
    """
    Thread-safe in-memory + on-disk store for calibration and load observations.

    Background engines call add_import_observation / add_load_observation as
    new data arrives.  Periodic flush() writes to disk.
    """

    def __init__(self, data_dir: str, region: str) -> None:
        self._data_dir = data_dir
        self._region = region.lower()
        self._lock = threading.Lock()

        self._import_observations: list[CalibrationObservation] = []
        self._export_observations: list[CalibrationObservation] = []
        self._load_observations: list[LoadObservation] = []

        # Lazy prune: only scan the full load-observation list every N adds to
        # avoid O(n) list comprehension on each /load_observation POST.
        self._load_observations_since_prune: int = 0
        _LOAD_PRUNE_EVERY_N_ADDS = 48  # once per ~day of 30-min observations

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_from_disk(self) -> None:
        """Load persisted observations from JSON files on startup.

        Calibration load order:
          1. The user's own persisted file (<data_dir>/calibration_<region>.json),
             written by flush_to_disk() as the user's /calibration POSTs accumulate.
          2. If that file is absent or carries no import observations, fall back to
             the BUNDLED pre-fit seed (app/seed/seed_calibration_<REGION>.json) so a
             fresh install ships with a calibrated price model on day 1 instead of
             passing raw PD7DAY straight through until pairs accumulate.  The seed is
             region-matched market data (PD7DAY predispatch joined to realised RRP);
             regions with no bundled seed simply start empty (self-fit as before).
        Once the user accumulates their own pairs, flush_to_disk() writes the user
        file and that takes precedence on the next start — the seed is a starting
        point, never a ceiling.
        """
        calibration_path = self._calibration_path()
        load_obs_path = self._load_obs_path()

        import_obs: list[CalibrationObservation] = []
        export_obs: list[CalibrationObservation] = []
        loaded_source: Optional[str] = None

        if os.path.exists(calibration_path):
            try:
                with open(calibration_path, encoding="utf-8") as calibration_file:
                    calibration_data = json.load(calibration_file)
                import_obs = _deserialise_calibration_list(
                    calibration_data.get("import_observations", [])
                )
                export_obs = _deserialise_calibration_list(
                    calibration_data.get("export_observations", [])
                )
                loaded_source = calibration_path
            except (OSError, json.JSONDecodeError, KeyError, ValueError) as load_error:
                _LOGGER.warning(
                    "Could not load calibration from %s: %s — will try bundled seed",
                    calibration_path,
                    load_error,
                )

        # Fall back to the bundled pre-fit seed when the user has no calibration yet.
        if not import_obs:
            seed_import, seed_export = self._load_bundled_seed()
            if seed_import:
                import_obs, export_obs = seed_import, seed_export
                loaded_source = self._seed_path()

        if loaded_source is not None:
            with self._lock:
                self._import_observations = import_obs
                self._export_observations = export_obs
            _LOGGER.info(
                "Observation store: loaded %d import + %d export calibration "
                "observations from %s%s",
                len(import_obs),
                len(export_obs),
                loaded_source,
                " (BUNDLED PRE-FIT SEED — fresh install)"
                if loaded_source == self._seed_path()
                else "",
            )

        if os.path.exists(load_obs_path):
            try:
                with open(load_obs_path, encoding="utf-8") as load_obs_file:
                    load_data = json.load(load_obs_file)
                load_obs = _deserialise_load_list(load_data.get("load_observations", []))
                with self._lock:
                    self._load_observations = load_obs
                _LOGGER.info(
                    "Observation store: loaded %d load observations", len(load_obs)
                )
            except (OSError, json.JSONDecodeError, KeyError, ValueError) as load_error:
                _LOGGER.warning(
                    "Could not load load observations from %s: %s — starting fresh",
                    load_obs_path,
                    load_error,
                )

    def flush_to_disk(self) -> None:
        """Write current observations to JSON files (atomic)."""
        cutoff_utc = datetime.now(timezone.utc) - timedelta(days=_CALIBRATION_MAX_AGE_DAYS)
        load_cutoff_utc = datetime.now(timezone.utc) - timedelta(days=_LOAD_MAX_AGE_DAYS)

        with self._lock:
            import_obs_copy = [
                obs for obs in self._import_observations
                if obs.observed_at >= cutoff_utc
            ]
            export_obs_copy = [
                obs for obs in self._export_observations
                if obs.observed_at >= cutoff_utc
            ]
            load_obs_copy = [
                obs for obs in self._load_observations
                if obs.interval_start_utc >= load_cutoff_utc
            ]

        calibration_payload = {
            "version": 1,
            "region": self._region,
            "import_observations": [_serialise_calibration(obs) for obs in import_obs_copy],
            "export_observations": [_serialise_calibration(obs) for obs in export_obs_copy],
        }
        _atomic_write_json(self._calibration_path(), calibration_payload)

        load_payload = {
            "version": 1,
            "region": self._region,
            "load_observations": [_serialise_load(obs) for obs in load_obs_copy],
        }
        _atomic_write_json(self._load_obs_path(), load_payload)

        _LOGGER.debug(
            "Observation store flushed: import=%d, export=%d, load=%d",
            len(import_obs_copy),
            len(export_obs_copy),
            len(load_obs_copy),
        )

    def add_import_observation(
        self,
        predicted_rrp_per_mwh: float,
        actual_rrp_per_mwh: float,
        hour_of_day: int,
        observed_at: datetime,
    ) -> None:
        with self._lock:
            self._import_observations.append(
                CalibrationObservation(
                    predicted_rrp_per_mwh=float(predicted_rrp_per_mwh),
                    actual_rrp_per_mwh=float(actual_rrp_per_mwh),
                    hour_of_day=int(hour_of_day),
                    observed_at=observed_at,
                )
            )

    def add_export_observation(
        self,
        predicted_rrp_per_mwh: float,
        actual_rrp_per_mwh: float,
        hour_of_day: int,
        observed_at: datetime,
    ) -> None:
        with self._lock:
            self._export_observations.append(
                CalibrationObservation(
                    predicted_rrp_per_mwh=float(predicted_rrp_per_mwh),
                    actual_rrp_per_mwh=float(actual_rrp_per_mwh),
                    hour_of_day=int(hour_of_day),
                    observed_at=observed_at,
                )
            )

    def add_load_observation(
        self,
        interval_start_utc: datetime,
        load_watts: float,
    ) -> None:
        """Add a new 30-min load observation (thread-safe).

        Pruning is deferred: the full list scan fires every 48 adds (~1 day of
        30-min observations) rather than on every POST, so the hot path is O(1).
        """
        with self._lock:
            self._load_observations.append(
                LoadObservation(
                    interval_start_utc=interval_start_utc,
                    load_watts=float(load_watts),
                )
            )
            self._load_observations_since_prune += 1
            if self._load_observations_since_prune >= 48:
                cutoff = datetime.now(timezone.utc) - timedelta(days=_LOAD_MAX_AGE_DAYS)
                self._load_observations = [
                    obs for obs in self._load_observations
                    if obs.interval_start_utc >= cutoff
                ]
                self._load_observations_since_prune = 0

    def get_import_observations(self) -> list[CalibrationObservation]:
        with self._lock:
            return list(self._import_observations)

    def get_export_observations(self) -> list[CalibrationObservation]:
        with self._lock:
            return list(self._export_observations)

    def get_load_observations(self) -> list[LoadObservation]:
        """Return load observations, applying age-based pruning on each read.

        Pruning on read (not just on write) ensures stale observations never
        appear in the returned list regardless of how many writes triggered the
        deferred-prune threshold.
        """
        with self._lock:
            cutoff = datetime.now(timezone.utc) - timedelta(days=_LOAD_MAX_AGE_DAYS)
            self._load_observations = [
                obs for obs in self._load_observations
                if obs.interval_start_utc >= cutoff
            ]
            self._load_observations_since_prune = 0
            return list(self._load_observations)

    # ------------------------------------------------------------------
    # Internal path helpers
    # ------------------------------------------------------------------

    def _calibration_path(self) -> str:
        return os.path.join(self._data_dir, f"calibration_{self._region}.json")

    def _load_obs_path(self) -> str:
        return os.path.join(self._data_dir, f"load_obs_{self._region}.json")

    def _seed_path(self) -> str:
        """Path to the bundled pre-fit calibration seed for this region.

        Lives next to the app code (app/seed/) so it is shipped inside the Docker
        image (the Dockerfiles COPY app/ wholesale) and is read-only — distinct
        from the writable <data_dir> the user's own calibration accumulates in.
        """
        return os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "seed",
            f"seed_calibration_{self._region.upper()}.json",
        )

    def _load_bundled_seed(
        self,
    ) -> tuple[list[CalibrationObservation], list[CalibrationObservation]]:
        """Load the bundled pre-fit seed observations, or ([], []) if none exists."""
        seed_path = self._seed_path()
        if not os.path.exists(seed_path):
            return [], []
        try:
            with open(seed_path, encoding="utf-8") as seed_file:
                seed_data = json.load(seed_file)
            seed_import = _deserialise_calibration_list(
                seed_data.get("import_observations", [])
            )
            seed_export = _deserialise_calibration_list(
                seed_data.get("export_observations", [])
            )
            return seed_import, seed_export
        except (OSError, json.JSONDecodeError, KeyError, ValueError) as seed_error:
            _LOGGER.warning(
                "Could not load bundled calibration seed %s: %s — starting empty",
                seed_path,
                seed_error,
            )
            return [], []


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _serialise_calibration(obs: CalibrationObservation) -> dict[str, Any]:
    return {
        "predicted_rrp_per_mwh": obs.predicted_rrp_per_mwh,
        "actual_rrp_per_mwh": obs.actual_rrp_per_mwh,
        "hour_of_day": obs.hour_of_day,
        "observed_at": obs.observed_at.isoformat(),
    }


def _deserialise_calibration_list(raw_list: list[Any]) -> list[CalibrationObservation]:
    result: list[CalibrationObservation] = []
    for raw_item in raw_list:
        try:
            result.append(
                CalibrationObservation(
                    predicted_rrp_per_mwh=float(raw_item["predicted_rrp_per_mwh"]),
                    actual_rrp_per_mwh=float(raw_item["actual_rrp_per_mwh"]),
                    hour_of_day=int(raw_item["hour_of_day"]),
                    observed_at=datetime.fromisoformat(raw_item["observed_at"]),
                )
            )
        except (KeyError, ValueError, TypeError):
            pass
    return result


def _serialise_load(obs: LoadObservation) -> dict[str, Any]:
    return {
        "interval_start_utc": obs.interval_start_utc.isoformat(),
        "load_watts": obs.load_watts,
    }


def _deserialise_load_list(raw_list: list[Any]) -> list[LoadObservation]:
    result: list[LoadObservation] = []
    for raw_item in raw_list:
        try:
            result.append(
                LoadObservation(
                    interval_start_utc=datetime.fromisoformat(raw_item["interval_start_utc"]),
                    load_watts=float(raw_item["load_watts"]),
                )
            )
        except (KeyError, ValueError, TypeError):
            pass
    return result


def _atomic_write_json(file_path: str, payload: dict[str, Any]) -> None:
    parent_dir = os.path.dirname(file_path)
    os.makedirs(parent_dir, exist_ok=True)
    temp_file_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=parent_dir,
            delete=False,
            suffix=".tmp",
        ) as temp_file:
            json.dump(payload, temp_file, indent=2)
            temp_file_path = temp_file.name
        os.replace(temp_file_path, file_path)
    except OSError as write_error:
        _LOGGER.error("Failed to write %s: %s", file_path, write_error)
        if temp_file_path is not None:
            try:
                os.unlink(temp_file_path)
            except OSError:
                pass
