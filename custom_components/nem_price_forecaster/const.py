"""Constants for the NEM Price Forecaster integration."""

DOMAIN = "nem_price_forecaster"
PLATFORMS = ["sensor"]

# NEMWeb PD7DAY endpoint — public, no auth required
NEMWEB_PD7DAY_INDEX_URL = "https://nemweb.com.au/Reports/CURRENT/PD7Day/"

# NEM dispatch regions
NEM_REGIONS = ["QLD1", "NSW1", "VIC1", "SA1", "TAS1"]

# Config entry keys
CONF_REGION = "region"
CONF_GST_RATE = "gst_rate"
CONF_FIXED_ADDER_PER_KWH = "fixed_adder_per_kwh"
CONF_TOU_BANDS = "tou_bands"
CONF_FEED_IN_IS_WHOLESALE = "feed_in_is_wholesale"
CONF_CALIBRATION_WINDOW_DAYS = "calibration_window_days"
CONF_CALIBRATION_MIN_OBSERVATIONS = "calibration_min_observations"
CONF_PLAUSIBILITY_CAP_DOLLARS_PER_KWH = "plausibility_cap_dollars_per_kwh"
# Note: demand charges ($/kVA peak) cannot be expressed as per-kWh slot forecasts
# and are therefore out of scope for this integration.

# ToU band config keys (each band is a dict)
CONF_BAND_NAME = "name"
CONF_BAND_RATE_PER_KWH = "rate_per_kwh"
CONF_BAND_WINDOWS = "windows"  # list of {"days": [0..6], "start": "HH:MM", "end": "HH:MM"}

# Defaults
DEFAULT_GST_RATE = 0.10
DEFAULT_FIXED_ADDER_PER_KWH = 0.0
DEFAULT_FEED_IN_IS_WHOLESALE = True
DEFAULT_CALIBRATION_WINDOW_DAYS = 90
DEFAULT_CALIBRATION_MIN_OBSERVATIONS = 14  # days before calibration activates
DEFAULT_PLAUSIBILITY_CAP_DOLLARS_PER_KWH = 5.0  # above MPC but sanity guard

# Calibration
PAV_HOUR_BUCKETS = 24  # one isotonic curve per hour-of-day
CALIBRATION_RECENCY_HALF_LIFE_DAYS = 30  # exponential decay for observation weighting

# Coordinator update interval
# The sidecar runs its price-predict job every 5 minutes (apscheduler), so the
# HA companion needs to poll meaningfully more often than the PD7DAY 3×/day
# cadence to surface fresh isotonic-calibrated and (eventually) sub-PD7DAY
# updates.  15 minutes is a good balance — three companion polls per sidecar
# refresh cycle without hammering it.  Configurable via the integration options
# flow (CONF_UPDATE_INTERVAL_MINUTES).
DEFAULT_UPDATE_INTERVAL_MINUTES = 15
CONF_UPDATE_INTERVAL_MINUTES = "update_interval_minutes"
# Kept for backward compatibility with existing config entries / external code.
UPDATE_INTERVAL_HOURS = DEFAULT_UPDATE_INTERVAL_MINUTES / 60

# Sensor attributes
ATTR_FORECAST = "forecast"  # list of {interval_start, import_price, export_price, raw_rrp}
ATTR_RUN_DATETIME = "run_datetime"
ATTR_REGION = "region"
ATTR_CALIBRATION_STATUS = "calibration_status"
ATTR_CALIBRATION_OBSERVATIONS = "calibration_observations"
ATTR_NEXT_UPDATE = "next_update"

# Sensor unique id suffixes
SENSOR_IMPORT_PRICE = "import_price"
SENSOR_EXPORT_PRICE = "export_price"
SENSOR_LOAD_FORECAST = "load_forecast"

# Units
UNIT_DOLLARS_PER_KWH = "AUD/kWh"
UNIT_WATTS = "W"

# Forecast horizon + period (configurable — applies to both price and load output)
CONF_FORECAST_HORIZON_HOURS = "forecast_horizon_hours"
CONF_FORECAST_PERIOD_MINUTES = "forecast_period_minutes"

# Allowed period values (minutes) presented to the user
FORECAST_PERIOD_OPTIONS = [5, 15, 30, 60]

# Defaults
# 30-min is PD7DAY's native resolution; use it as the default so no resampling needed
DEFAULT_FORECAST_HORIZON_HOURS = 168   # 7 days
DEFAULT_FORECAST_PERIOD_MINUTES = 30   # 30-min native (matches PD7DAY)

# Load forecaster config keys
CONF_LOAD_FORECASTER_ENABLED = "load_forecaster_enabled"
CONF_LOAD_ENTITY_ID = "load_entity_id"
CONF_LOAD_FORECAST_HORIZON_HOURS = "load_forecast_horizon_hours"
# Note: CONF_LOAD_USE_LIGHTGBM is no longer used — the load forecaster is Darts-only.
# Kept here as a deprecated constant for backward compatibility with stored config entries.
CONF_LOAD_USE_LIGHTGBM = "load_use_lightgbm"

# Load forecaster defaults
DEFAULT_LOAD_FORECASTER_ENABLED = False  # opt-in (requires HA recorder access + darts package)
DEFAULT_LOAD_FORECAST_HORIZON_HOURS = 144  # 6 days = 288 slots (single-shot Darts output)
DEFAULT_LOAD_USE_LIGHTGBM = True  # deprecated; Darts is always LightGBM

# Load forecaster: how many days of recorder history to use for training
LOAD_TRAINING_HISTORY_DAYS = 90   # 3 months for seasonal coverage
# How many recent observations to pass as the lag input window for prediction
# (must cover at least lags × 30 min = 96 × 30 min = 2 days = 96 half-hours)
LOAD_LAG_BUFFER_HOURS = 52  # slight headroom above the 48h (96 slot) lag window

# Load forecast sensor attributes
ATTR_LOAD_FORECAST = "forecast"
ATTR_LOAD_MODEL_NAME = "model_name"
ATTR_LOAD_TRAINING_OBSERVATIONS = "training_observations"
ATTR_LOAD_IS_TRAINED = "model_trained"
ATTR_LOAD_NEXT_TRAIN = "next_train"

# Calibration persistence: save interval (how often to write to .storage/)
CALIBRATION_SAVE_INTERVAL_MINUTES = 30

# Sidecar URL (new in sidecar architecture)
CONF_SIDECAR_URL = "sidecar_url"
DEFAULT_SIDECAR_URL = "http://localhost:8765"

# ---------------------------------------------------------------------------
# Price model + calibrator selection
#
# These ARE forwarded to the sidecar at runtime via POST /config (see
# sidecar_client.async_post_config), so the picker is now live: changing it in
# the config flow / options flow reconfigures the running sidecar and triggers a
# re-predict.  (The previous dead selector that never reached the sidecar has
# been replaced with this working wiring.)
# ---------------------------------------------------------------------------
CONF_PRICE_MODEL = "price_model"
CONF_CALIBRATOR = "calibrator"

# Price models — must match the sidecar's _VALID_PRICE_MODELS.
PRICE_MODEL_DARTS_NAIVE_BLEND = "darts_naive_blend"
PRICE_MODEL_ISOTONIC = "isotonic"
PRICE_MODEL_DARTS = "darts"
PRICE_MODEL_HYBRID = "hybrid"
PRICE_MODEL_OPTIONS = [
    PRICE_MODEL_ISOTONIC,
    PRICE_MODEL_DARTS_NAIVE_BLEND,
    PRICE_MODEL_DARTS,
    PRICE_MODEL_HYBRID,
]

# Calibrators — must match the sidecar's _VALID_CALIBRATORS.
CALIBRATOR_ISOTONIC = "isotonic"
CALIBRATOR_MONOTONE_GBM = "monotone_gbm"
CALIBRATOR_OPTIONS = [
    CALIBRATOR_MONOTONE_GBM,
    CALIBRATOR_ISOTONIC,
]

# Defaults mirror the sidecar's SidecarConfig defaults.
DEFAULT_PRICE_MODEL = PRICE_MODEL_ISOTONIC
DEFAULT_CALIBRATOR = CALIBRATOR_MONOTONE_GBM
