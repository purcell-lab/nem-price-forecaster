"""
Sidecar configuration — loaded from environment variables and/or a JSON file.

Priority: env vars > config file > defaults.

Environment variables (all prefixed SIDECAR_):
    SIDECAR_REGION              NEM region, e.g. QLD1 (default NSW1)
    SIDECAR_PORT                HTTP port (default 8765)
    SIDECAR_DATA_DIR            Directory for model + calibration persistence (default /data)
    SIDECAR_CONFIG_FILE         Path to a JSON config file (default /data/config.json)
    SIDECAR_PRICE_MODEL         darts_naive_blend | isotonic | darts | hybrid
                                (default darts_naive_blend)
                                - darts_naive_blend: 50/50 mixture of Darts LightGBM and
                                  seasonal-naive (same-hour-same-DOW one-week-ago).
                                  Blending reduces worst-case regime-transition error
                                  while matching seasonal-naive's average accuracy.
                                  No prediction cache required.
                                  The blend weight is tunable via SIDECAR_NAIVE_BLEND_WEIGHT.
                                  MAX REACH: theoretically unbounded (Darts recurses, naive
                                  uses 7-days-ago observation); in practice validated to ≤7d.
                                  Beyond the chain seam, this model is the chain continuation.
                                - isotonic: PD7DAY → per-hour PAV isotonic calibration.
                                  Best when AEMO predispatch is accurate; catastrophic when
                                  PD7DAY diverges from settlement (e.g. May-Jun 2026 +2025).
                                  MAX REACH: bounded by PD7DAY (~7d); chain-extended beyond
                                  SIDECAR_CHAIN_SEAM_DAYS using darts_naive_blend continuation.
                                - darts: Darts LightGBM model trained on actual FiT history.
                                  Consistent but carries a positive bias; no PD7DAY dependency.
                                  MAX REACH: theoretically unbounded (recursive prediction);
                                  validated accuracy is ≤7d.
                                - hybrid: isotonic for slots ≤ SIDECAR_HYBRID_CROSSOVER_HOURS
                                          lead time, Darts for slots beyond that.
                                          Falls back to isotonic everywhere if Darts is
                                          unavailable (insufficient history or training error).
                                          Falls back to Darts everywhere if isotonic
                                          calibrator has insufficient observations.
                                  MAX REACH: bounded by PD7DAY for isotonic leg (~7d);
                                  chain-extended beyond SIDECAR_CHAIN_SEAM_DAYS using
                                  darts_naive_blend continuation.
                                UPGRADE PATH: when ≥14 days of live predictions are banked,
                                an adaptive rolling-regime-selector can be enabled
                                (not yet implemented; see TODO in price_engine.py).
    SIDECAR_NAIVE_BLEND_WEIGHT  Float 0.0–1.0: weight for seasonal-naive in the
                                darts_naive_blend.  0.5 = equal blend (default, evidence-
                                based).  0.0 = pure Darts; 1.0 = pure seasonal-naive.
    SIDECAR_HYBRID_CROSSOVER_HOURS  Lead-time threshold (hours) at which the hybrid model
                                    switches from isotonic to Darts (default 120).
                                    Slots with lead_time <= crossover → isotonic.
                                    Slots with lead_time > crossover → Darts.
                                    Tuned for QLD1 — different markets/regions
                                    may need different values; re-tune on your
                                    own data to find yours.
    SIDECAR_HYBRID_BLEND_ENABLED    Enable smooth linear blend across a ±blend_window around
                                    the crossover to avoid a step discontinuity (default true).
                                    When enabled, slots within [crossover−blend_window,
                                    crossover+blend_window] use a linearly interpolated
                                    mixture of isotonic and Darts.
    SIDECAR_HYBRID_BLEND_WINDOW_HOURS  Half-width of the smooth blend band in hours (default 12).
                                    Slots within crossover ± blend_window transition linearly
                                    between isotonic and Darts.  Set to 0 for a hard switch.
                                    Tuned for QLD1 — re-tune for other markets.

    --- Forecast horizon ---

    SIDECAR_FORECAST_HORIZON_DAYS    Forecast horizon in days as a float (default 7.0 = the
                                    validated best; 7 days = PD7DAY's natural reach).
                                    HONESTY NOTE: validated accuracy is ≤7 days — that is the
                                    window covered by the backtests.
                                    Beyond 7 days the chain-resolver keeps the lights on by
                                    extending with darts_naive_blend; accuracy is UNVALIDATED
                                    convenience, not a measured claim.
                                    Accepts any float ≥0.5 and ≤14.0.
                                    Examples: 1.0 = 24h, 7.0 = 168h (default), 10.0 = 240h.
                                    Setting horizon > chain_seam_days activates the chain.
                                    SIDECAR_FORECAST_HORIZON_HOURS still accepted for backward
                                    compatibility; if both are set the DAYS value wins.

    SIDECAR_FORECAST_HORIZON_HOURS   Horizon in integer hours (legacy knob; use DAYS instead).
                                    Default 168 (7 days).  Ignored if FORECAST_HORIZON_DAYS set.

    --- Chain resolver (horizon-aware model chaining) ---

    The chain resolver activates automatically when the configured forecast horizon
    exceeds the active model's validated reach (SIDECAR_CHAIN_SEAM_DAYS).  When the
    horizon is exactly at or below the seam, all slots are served by the primary model
    (byte-identical to the pre-chain behaviour).  When it exceeds the seam, the primary
    model covers 0→seam, and darts_naive_blend covers seam→horizon_end.  A linear blend
    window around the seam avoids a price discontinuity at the handoff.

    Each chain slot carries model_segment="chain:darts_naive_blend" in the API response
    so consumers can distinguish validated from chained output.

    SIDECAR_CHAIN_SEAM_DAYS     Days at which the primary model hands to the chain
                                continuation (default 7.0 = PD7DAY's natural reach).
                                Slots with lead_time_days <= seam → primary model.
                                Slots with lead_time_days >  seam → chain continuation.
                                Tuned for QLD1 — re-tune on your own data to find yours.
    SIDECAR_CHAIN_BLEND_WINDOW_HOURS  Half-width of the linear blend band (hours) around
                                the chain seam to avoid a price step discontinuity
                                (default 12).  The primary model's price and the chain
                                model's price are linearly interpolated across
                                [seam - blend_window, seam + blend_window].
                                Set to 0 for a hard handoff at exactly the seam.

    --- Calibrator adjacency blend ---

    SIDECAR_CALIBRATOR_ADJACENCY_ALPHA  Float 0.0–1.0.  Default 0.5.
                                    When the isotonic calibrator calibrates a price, it
                                    blends the target hour's curve with the adjacent hours'
                                    curves using this weight:
                                      blended = (target + alpha*prev + alpha*next) / (1 + 2*alpha)
                                    alpha=0.0 → pure per-hour curve (maximum raw-MAE accuracy;
                                    +26% more hour-boundary jumps vs default; worsens EMHASS plan
                                    stability).
                                    alpha=0.5 → default; the stability-adjusted optimum.
                                    alpha=0.0 wins marginally on raw MAE but produces more
                                    hour-boundary jumps; alpha=0.5 trades a small amount of raw
                                    accuracy for smoother, more stable forecasts (fewer
                                    hour-boundary jumps, better forecast persistence).
                                    Set alpha=0.0 only if raw MAE is your sole objective and you
                                    measure this on your own corpus first.

    --- Calibrator backend (isotonic vs monotone-GBM) ---

    SIDECAR_CALIBRATOR          isotonic | monotone_gbm  (default isotonic)
                                Selects HOW raw PD7DAY RRP is calibrated to a
                                realised price (applies to the isotonic / hybrid
                                paths; the Darts price models do their own thing).
                                - isotonic (DEFAULT, opt-out): 24 per-hour PAV
                                  curves.  Dependency-free, the shipped baseline,
                                  human-gated default.
                                - monotone_gbm (OPT-IN): a single LightGBM
                                  regressor with a monotone constraint on the raw
                                  forecast price + cyclic hour/day-of-week
                                  features.  It trains on the runtime-available
                                  data (raw price + hour + day-of-week — the SAME
                                  data isotonic uses) and keeps a never-lose runtime
                                  fallback: if the GBM does not beat isotonic on a
                                  held-out tail (or LightGBM is missing / the fit
                                  fails) it transparently serves isotonic output.
                                  So selecting it can only match or beat isotonic
                                  on your corpus.  Isotonic remains the DEFAULT —
                                  this is an opt-in, human-gated change.

    SIDECAR_FORECAST_PERIOD_MINUTES  5 | 15 | 30 | 60 (default 30)
    SIDECAR_TRAIN_CRON          Cron expression for nightly train (default "0 2 * * *")
    SIDECAR_PREDICT_INTERVAL_SECONDS  Background predict interval (default 300)
    SIDECAR_GST_RATE            default 0.10
    SIDECAR_FIXED_ADDER_PER_KWH default 0.0
    SIDECAR_FEED_IN_IS_WHOLESALE default true
    SIDECAR_TOU_BANDS_JSON      JSON array of ToU band dicts (default "[]")
    SIDECAR_LOAD_FORECASTER_ENABLED  true | false (default true)

Weather covariate configuration (Open-Meteo, free, no API key):
    SIDECAR_WEATHER_ENABLED     true | false (default true)
                                Enable Open-Meteo weather covariates for both models.
    SIDECAR_LATITUDE            Decimal latitude for weather fetch.
                                Defaults to the representative capital city for the
                                configured NEM region (e.g. -27.47 for QLD1/Brisbane).
    SIDECAR_LONGITUDE           Decimal longitude for weather fetch.

Historical AEMO price data:
    SIDECAR_AEMO_HISTORY_DAYS   Days of AEMO NEMWeb archive price history to download
                                for seeding the Darts price model training set (default 365).
                                Set to 0 to disable historical seeding (use only
                                accumulated /calibration/import observations).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

_LOGGER = logging.getLogger(__name__)

_NEM_REGIONS = {"QLD1", "NSW1", "VIC1", "SA1", "TAS1"}
_VALID_PRICE_MODELS = {"darts_naive_blend", "isotonic", "darts", "hybrid"}
_VALID_CALIBRATORS = {"isotonic", "monotone_gbm"}
_VALID_PERIODS = {5, 15, 30, 60}


@dataclass
class SidecarConfig:
    """Validated sidecar configuration."""
    region: str = "NSW1"
    port: int = 8765
    data_dir: str = "/data"
    # Shipped default: 50/50 Darts + seasonal-naive blend.
    # Blending reduces worst-case regime-transition error vs the Darts model
    # alone while matching seasonal-naive's average accuracy.
    price_model: str = "hybrid"

    # Hybrid price model parameters (used only when price_model="hybrid")
    # Crossover lead time: slots with lead_time <= hybrid_crossover_hours use isotonic;
    # slots with lead_time > hybrid_crossover_hours use Darts.
    # Tuned for QLD1 — different markets/regions may need different values;
    # re-tune on your own data to find yours.
    hybrid_crossover_hours: float = 120.0
    # When True, blend linearly over a ±hybrid_blend_window_hours band around the crossover
    # so there is no step discontinuity at the boundary.  weight_isotonic = 1 − blend_fraction,
    # weight_darts = blend_fraction where blend_fraction ramps 0→1 across the band.
    hybrid_blend_enabled: bool = True
    # Half-width of the smooth blend band in hours (active only when hybrid_blend_enabled=True).
    # Tuned for QLD1 — re-tune for other markets.
    hybrid_blend_window_hours: float = 12.0

    # Darts+naive blend weight (used when price_model="darts_naive_blend").
    # 0.5 = equal blend (evidence-based default).
    # 0.0 = pure Darts; 1.0 = pure seasonal-naive.
    # Tune once 30+ days of live predictions are banked — an adaptive
    # rolling-regime-selector upgrade path will eventually make this adaptive.
    naive_blend_weight: float = 0.5

    # Forecast horizon — configurable as float days or legacy integer hours.
    # forecast_horizon_days is the canonical knob; forecast_horizon_hours is
    # derived from it for backward compatibility (rounded to nearest integer).
    # Default 7.0 = validated best (all published backtests are ≤7 days).
    # Beyond 7 days: the chain-resolver extends with darts_naive_blend;
    # accuracy is UNVALIDATED convenience, not a measured claim.
    forecast_horizon_days: float = 7.0
    forecast_horizon_hours: int = 168       # derived from forecast_horizon_days
    forecast_period_minutes: int = 30

    # Calibrator backend — how raw PD7DAY RRP is mapped to a realised price for
    # the isotonic / hybrid paths.  DEFAULT = "isotonic" (the shipped, human-gated
    # baseline).  "monotone_gbm" is an OPT-IN LightGBM calibrator with a monotone
    # constraint on the raw price + cyclic hour/dow features; it keeps a runtime
    # never-lose fallback to isotonic (see monotone_gbm_calibrator.py).
    # Isotonic stays the default deliberately — switching is
    # an opt-in, human-gated change.
    calibrator: str = "monotone_gbm"

    # Chain resolver — horizon-aware model chaining.
    # When forecast_horizon_days > chain_seam_days, the primary model covers
    # 0→seam and darts_naive_blend covers seam→horizon.  Set seam=horizon to
    # disable chaining (primary model covers entire horizon).
    # Default 7.0 = PD7DAY's natural reach (validated for all published models).
    # Tuned for QLD1 — re-tune on your own data to find yours.
    chain_seam_days: float = 7.0
    # Half-width of the linear blend band (hours) around the chain seam.
    # Eliminates price discontinuity at the handoff between primary and chain model.
    # Default 12h — same window as the hybrid model's internal blend.
    # Set to 0 for a hard handoff at exactly the seam.
    chain_blend_window_hours: float = 12.0
    train_cron: str = "0 2 * * *"          # nightly 02:00
    predict_interval_seconds: int = 300     # 5 min background predict floor

    # Tariff
    gst_rate: float = 0.10
    fixed_adder_per_kwh: float = 0.0
    feed_in_is_wholesale: bool = True
    tou_bands: list[dict[str, Any]] = field(default_factory=list)

    # Load forecaster
    load_forecaster_enabled: bool = True

    # Weather covariates (Open-Meteo — free, no API key)
    # When enabled, both the Darts price model and the Darts load forecaster
    # receive weather as a future covariate.  The client falls back gracefully
    # (logs a WARNING and proceeds without weather) if Open-Meteo is unreachable.
    weather_enabled: bool = True
    # Lat/lon for weather fetch.  None means "derive from region" (default).
    # Set to override with the user's actual household location.
    latitude: Optional[float] = None
    longitude: Optional[float] = None

    # AEMO historical price seeding
    # Number of days of NEMWeb DISPATCHIS archive data to download as the
    # training base for the Darts price model.  This supplements the short
    # online-accumulated calibration window (Recipe D) with years of history
    # so the model sees full seasonal + structural variation from day 1.
    # Set to 0 to disable (use only /calibration/import observations).
    aemo_history_days: int = 365

    # Calibration
    calibration_window_days: int = 90
    calibration_min_observations: int = 14
    plausibility_cap_dollars_per_kwh: float = 5.0

    # Darts price model — minimum days of accumulated actual wholesale-price
    # history (from calibration POST observations) before the Darts model will
    # attempt to train.  14 days gives the LightGBM model enough weekday+weekend
    # coverage to learn time-of-day patterns; below this threshold the engine
    # falls back to isotonic calibration with a logged WARNING.
    darts_price_min_training_days: int = 14

    # Isotonic calibrator adjacency-blend alpha.
    # Blends the target hour's PAV curve with the adjacent hours' curves to
    # suppress per-hour boundary discontinuities in the isotonic output.
    # Formula: blended = (target + alpha*prev + alpha*next) / (1 + 2*alpha).
    # 0.0 = pure per-hour (marginally better raw MAE, more boundary jumps).
    # 0.5 = default (stability-adjusted optimum: fewer boundary jumps and
    # better forecast persistence for a small raw-accuracy cost).
    calibrator_adjacency_alpha: float = 0.5


def load_config() -> SidecarConfig:
    """
    Build a SidecarConfig from env vars + optional JSON file.

    The JSON config file can contain any subset of the SidecarConfig fields.
    Env vars always override the file.
    """
    config = SidecarConfig()

    # Load JSON file first (lowest priority)
    config_file_path = os.environ.get("SIDECAR_CONFIG_FILE", os.path.join(
        os.environ.get("SIDECAR_DATA_DIR", "/data"), "config.json"
    ))
    if os.path.exists(config_file_path):
        try:
            with open(config_file_path, encoding="utf-8") as config_file_handle:
                file_data = json.load(config_file_handle)
            _apply_dict_to_config(config, file_data)
            _LOGGER.info("Sidecar config loaded from %s", config_file_path)
        except (OSError, json.JSONDecodeError, ValueError) as file_error:
            _LOGGER.warning("Could not read config file %s: %s", config_file_path, file_error)

    # Apply env vars (highest priority)
    _apply_env_vars(config)

    # Validate
    _validate_config(config)

    _LOGGER.info(
        "Sidecar config: region=%s price_model=%s calibrator=%s horizon=%.1fd (%dh) "
        "chain_seam=%.1fd blend_window=%.0fh period=%dmin "
        "calibrator_adjacency_alpha=%.2f "
        "load=%s weather=%s aemo_history_days=%d",
        config.region,
        config.price_model,
        config.calibrator,
        config.forecast_horizon_days,
        config.forecast_horizon_hours,
        config.chain_seam_days,
        config.chain_blend_window_hours,
        config.forecast_period_minutes,
        config.calibrator_adjacency_alpha,
        config.load_forecaster_enabled,
        config.weather_enabled,
        config.aemo_history_days,
    )
    return config


def _apply_env_vars(config: SidecarConfig) -> None:
    """Apply SIDECAR_* environment variables to *config* in place."""
    env_str = os.environ.get("SIDECAR_REGION")
    if env_str:
        config.region = env_str.strip().upper()

    env_str = os.environ.get("SIDECAR_PORT")
    if env_str:
        config.port = int(env_str)

    env_str = os.environ.get("SIDECAR_DATA_DIR")
    if env_str:
        config.data_dir = env_str.strip()

    env_str = os.environ.get("SIDECAR_PRICE_MODEL")
    if env_str:
        config.price_model = env_str.strip().lower()

    env_str = os.environ.get("SIDECAR_CALIBRATOR")
    if env_str:
        config.calibrator = env_str.strip().lower()

    # SIDECAR_FORECAST_HORIZON_DAYS (canonical) takes priority over _HOURS (legacy).
    # If only _HOURS is set, derive _DAYS from it.  If both are set, _DAYS wins.
    horizon_days_env = os.environ.get("SIDECAR_FORECAST_HORIZON_DAYS")
    horizon_hours_env = os.environ.get("SIDECAR_FORECAST_HORIZON_HOURS")
    if horizon_days_env:
        try:
            config.forecast_horizon_days = float(horizon_days_env.strip())
            config.forecast_horizon_hours = max(1, round(config.forecast_horizon_days * 24))
        except ValueError as horizon_days_error:
            _LOGGER.warning(
                "SIDECAR_FORECAST_HORIZON_DAYS is not a valid float: %s", horizon_days_error
            )
    elif horizon_hours_env:
        try:
            config.forecast_horizon_hours = int(horizon_hours_env.strip())
            config.forecast_horizon_days = config.forecast_horizon_hours / 24.0
        except ValueError as horizon_hours_error:
            _LOGGER.warning(
                "SIDECAR_FORECAST_HORIZON_HOURS is not a valid integer: %s", horizon_hours_error
            )

    env_str = os.environ.get("SIDECAR_CHAIN_SEAM_DAYS")
    if env_str:
        try:
            config.chain_seam_days = float(env_str.strip())
        except ValueError as chain_seam_error:
            _LOGGER.warning("SIDECAR_CHAIN_SEAM_DAYS is not a valid float: %s", chain_seam_error)

    env_str = os.environ.get("SIDECAR_CHAIN_BLEND_WINDOW_HOURS")
    if env_str:
        try:
            config.chain_blend_window_hours = float(env_str.strip())
        except ValueError as chain_blend_error:
            _LOGGER.warning(
                "SIDECAR_CHAIN_BLEND_WINDOW_HOURS is not a valid float: %s", chain_blend_error
            )

    env_str = os.environ.get("SIDECAR_FORECAST_PERIOD_MINUTES")
    if env_str:
        config.forecast_period_minutes = int(env_str)

    env_str = os.environ.get("SIDECAR_TRAIN_CRON")
    if env_str:
        config.train_cron = env_str.strip()

    env_str = os.environ.get("SIDECAR_PREDICT_INTERVAL_SECONDS")
    if env_str:
        config.predict_interval_seconds = int(env_str)

    env_str = os.environ.get("SIDECAR_GST_RATE")
    if env_str:
        config.gst_rate = float(env_str)

    env_str = os.environ.get("SIDECAR_FIXED_ADDER_PER_KWH")
    if env_str:
        config.fixed_adder_per_kwh = float(env_str)

    env_str = os.environ.get("SIDECAR_FEED_IN_IS_WHOLESALE")
    if env_str:
        config.feed_in_is_wholesale = env_str.strip().lower() not in ("0", "false", "no")

    env_str = os.environ.get("SIDECAR_TOU_BANDS_JSON")
    if env_str:
        try:
            parsed = json.loads(env_str)
            if isinstance(parsed, list):
                config.tou_bands = parsed
        except json.JSONDecodeError as json_error:
            _LOGGER.warning("SIDECAR_TOU_BANDS_JSON is invalid JSON: %s", json_error)

    if not config.tou_bands:
        _LOGGER.warning(
            "No ToU bands configured (SIDECAR_TOU_BANDS_JSON is empty) — import "
            "forecasts will be WHOLESALE-ONLY (+GST), far below a real retail bill. "
            "Set your distributor's bands; see README 'Network tariff bands' for a "
            "worked QLD1/Energex example."
        )

    env_str = os.environ.get("SIDECAR_LOAD_FORECASTER_ENABLED")
    if env_str:
        config.load_forecaster_enabled = env_str.strip().lower() not in ("0", "false", "no")

    env_str = os.environ.get("SIDECAR_WEATHER_ENABLED")
    if env_str:
        config.weather_enabled = env_str.strip().lower() not in ("0", "false", "no")

    env_str = os.environ.get("SIDECAR_LATITUDE")
    if env_str:
        try:
            config.latitude = float(env_str.strip())
        except ValueError as latitude_error:
            _LOGGER.warning("SIDECAR_LATITUDE is not a valid float: %s", latitude_error)

    env_str = os.environ.get("SIDECAR_LONGITUDE")
    if env_str:
        try:
            config.longitude = float(env_str.strip())
        except ValueError as longitude_error:
            _LOGGER.warning("SIDECAR_LONGITUDE is not a valid float: %s", longitude_error)

    env_str = os.environ.get("SIDECAR_AEMO_HISTORY_DAYS")
    if env_str:
        try:
            config.aemo_history_days = int(env_str.strip())
        except ValueError as days_error:
            _LOGGER.warning("SIDECAR_AEMO_HISTORY_DAYS is not a valid integer: %s", days_error)

    env_str = os.environ.get("SIDECAR_HYBRID_CROSSOVER_HOURS")
    if env_str:
        try:
            config.hybrid_crossover_hours = float(env_str.strip())
        except ValueError as crossover_error:
            _LOGGER.warning(
                "SIDECAR_HYBRID_CROSSOVER_HOURS is not a valid float: %s", crossover_error
            )

    env_str = os.environ.get("SIDECAR_HYBRID_BLEND_ENABLED")
    if env_str:
        config.hybrid_blend_enabled = env_str.strip().lower() not in ("0", "false", "no")

    env_str = os.environ.get("SIDECAR_HYBRID_BLEND_WINDOW_HOURS")
    if env_str:
        try:
            config.hybrid_blend_window_hours = float(env_str.strip())
        except ValueError as blend_window_error:
            _LOGGER.warning(
                "SIDECAR_HYBRID_BLEND_WINDOW_HOURS is not a valid float: %s", blend_window_error
            )

    env_str = os.environ.get("SIDECAR_NAIVE_BLEND_WEIGHT")
    if env_str:
        try:
            config.naive_blend_weight = float(env_str.strip())
        except ValueError as blend_weight_error:
            _LOGGER.warning(
                "SIDECAR_NAIVE_BLEND_WEIGHT is not a valid float: %s", blend_weight_error
            )

    env_str = os.environ.get("SIDECAR_CALIBRATOR_ADJACENCY_ALPHA")
    if env_str:
        try:
            config.calibrator_adjacency_alpha = float(env_str.strip())
        except ValueError as adjacency_alpha_error:
            _LOGGER.warning(
                "SIDECAR_CALIBRATOR_ADJACENCY_ALPHA is not a valid float: %s",
                adjacency_alpha_error,
            )


def _apply_dict_to_config(config: SidecarConfig, data: dict[str, Any]) -> None:
    """Apply a dict of config values to *config* in place (used for JSON file loading).

    When a JSON file sets the legacy 'forecast_horizon_hours' without also setting
    'forecast_horizon_days', the days value is derived automatically to keep them
    consistent.  When 'forecast_horizon_days' is present it takes priority.
    """
    for field_name, field_value in data.items():
        if hasattr(config, field_name):
            setattr(config, field_name, field_value)
    # Derive the other field if only one was set in the JSON file
    if "forecast_horizon_days" in data and "forecast_horizon_hours" not in data:
        config.forecast_horizon_hours = max(1, round(config.forecast_horizon_days * 24))
    elif "forecast_horizon_hours" in data and "forecast_horizon_days" not in data:
        config.forecast_horizon_days = config.forecast_horizon_hours / 24.0


def _validate_config(config: SidecarConfig) -> None:
    """Raise ValueError if any config value is out of range."""
    if config.region not in _NEM_REGIONS:
        raise ValueError(
            f"SIDECAR_REGION must be one of {sorted(_NEM_REGIONS)}, got: {config.region!r}"
        )
    if config.price_model not in _VALID_PRICE_MODELS:
        raise ValueError(
            f"SIDECAR_PRICE_MODEL must be one of {sorted(_VALID_PRICE_MODELS)}, "
            f"got: {config.price_model!r}"
        )
    if config.calibrator not in _VALID_CALIBRATORS:
        raise ValueError(
            f"SIDECAR_CALIBRATOR must be one of {sorted(_VALID_CALIBRATORS)}, "
            f"got: {config.calibrator!r}"
        )
    if config.forecast_period_minutes not in _VALID_PERIODS:
        raise ValueError(
            f"SIDECAR_FORECAST_PERIOD_MINUTES must be one of {sorted(_VALID_PERIODS)}, "
            f"got: {config.forecast_period_minutes}"
        )
    if not (0.5 <= config.forecast_horizon_days <= 14.0):
        raise ValueError(
            f"SIDECAR_FORECAST_HORIZON_DAYS must be 0.5..14.0 days, "
            f"got: {config.forecast_horizon_days}"
        )
    if not (1 <= config.forecast_horizon_hours <= 336):
        raise ValueError(
            f"SIDECAR_FORECAST_HORIZON_HOURS must be 1..336, got: {config.forecast_horizon_hours}"
        )
    if not (0.5 <= config.chain_seam_days <= 14.0):
        raise ValueError(
            f"SIDECAR_CHAIN_SEAM_DAYS must be 0.5..14.0, got: {config.chain_seam_days}"
        )
    if not (0.0 <= config.chain_blend_window_hours <= 48.0):
        raise ValueError(
            f"SIDECAR_CHAIN_BLEND_WINDOW_HOURS must be 0..48, "
            f"got: {config.chain_blend_window_hours}"
        )
    if not (10 <= config.predict_interval_seconds <= 3600):
        raise ValueError(
            f"SIDECAR_PREDICT_INTERVAL_SECONDS must be 10..3600, "
            f"got: {config.predict_interval_seconds}"
        )
    if config.calibration_min_observations < 1:
        raise ValueError(
            f"calibration_min_observations must be >= 1, "
            f"got: {config.calibration_min_observations}"
        )
    if config.aemo_history_days < 0:
        raise ValueError(
            f"aemo_history_days must be >= 0, got: {config.aemo_history_days}"
        )
    if config.darts_price_min_training_days < 1:
        raise ValueError(
            f"darts_price_min_training_days must be >= 1, "
            f"got: {config.darts_price_min_training_days}"
        )
    if config.price_model == "hybrid" and not (
        1.0 <= config.hybrid_crossover_hours <= 168.0
    ):
        raise ValueError(
            f"hybrid_crossover_hours must be 1..168 when price_model=hybrid, "
            f"got: {config.hybrid_crossover_hours}"
        )
    if config.price_model == "hybrid" and not (
        0.0 <= config.hybrid_blend_window_hours <= 48.0
    ):
        raise ValueError(
            f"hybrid_blend_window_hours must be 0..48 when price_model=hybrid, "
            f"got: {config.hybrid_blend_window_hours}"
        )
    if not (0.0 <= config.naive_blend_weight <= 1.0):
        raise ValueError(
            f"naive_blend_weight must be 0.0..1.0, got: {config.naive_blend_weight}"
        )
    if not (0.0 <= config.calibrator_adjacency_alpha <= 1.0):
        raise ValueError(
            f"SIDECAR_CALIBRATOR_ADJACENCY_ALPHA must be 0.0..1.0, "
            f"got: {config.calibrator_adjacency_alpha}"
        )
