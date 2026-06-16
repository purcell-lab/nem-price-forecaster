# NEM Price Forecaster for Home Assistant

7-day-ahead forecasts of Australian **NEM wholesale electricity prices**, exposed
as Home Assistant sensors. The integration ships with a **pre-trained QLD1 model**,
so it produces useful, calibrated price forecasts from day one — no weeks of data
collection required before it becomes useful.

Forecasts are published as `sensor` entities (with the full forecast horizon as
state attributes), ready to drive automations, dashboards, and energy optimisers
such as [EMHASS](https://github.com/davidusb-geek/emhass).

## How it works

The integration is split into two parts:

- **The Home Assistant integration** (`custom_components/nem_price_forecaster/`) —
  a config-flow integration that polls a forecasting **sidecar** over HTTP and
  exposes the results as sensors.
- **The forecasting sidecar / engine** (`sidecar/`, also packaged as an
  `addon/`) — a small FastAPI service that fetches AEMO PD7DAY predispatch data,
  calibrates it against realised prices, optionally runs a Darts LightGBM model,
  and serves the resulting price forecast.

Splitting the engine into a sidecar keeps the heavy ML dependencies (Darts,
LightGBM) out of the Home Assistant Python environment and lets the engine run
on any host.

### Sensors

Each configured region creates a device, **NEM Price Forecaster (REGION)**, with:

| Sensor | Description |
| --- | --- |
| **Import Price** | Forecast import price ($/kWh). The full horizon is available as a forecast attribute. |
| **Export Price** | Forecast export (feed-in) price ($/kWh). |
| **Load Forecast** | Optional household load forecast (opt-in; requires HA recorder history). |

## Installation

You need two things: the **sidecar engine** running somewhere reachable from
Home Assistant, and the **HA integration** installed and pointed at it.

### 1. Install the integration (HACS)

1. In HACS, go to **⋮ → Custom repositories**.
2. Add this repository's URL as an **Integration**.
3. Install **NEM Price Forecaster** and restart Home Assistant.
4. Go to **Settings → Devices & Services → Add Integration**, search for
   **NEM Price Forecaster**, and follow the config flow. You'll be asked for the
   **Sidecar URL** (see below) and your **region**.

(You can also install manually by copying
`custom_components/nem_price_forecaster/` into your HA `config/custom_components/`
directory and restarting.)

### 2. Run the sidecar engine

Pick **one** of the following.

#### Option A — Docker / docker-compose (any host)

```bash
cp .env.example .env      # edit SIDECAR_REGION and any options
docker compose up -d
```

The sidecar listens on port `8765`. In the HA config flow set the Sidecar URL to
`http://<host>:8765` (or `http://host.docker.internal:8765` if HA itself runs in
Docker on the same host).

#### Option B — Home Assistant add-on (HAOS / Supervised)

1. **Settings → Add-ons → Add-on Store → ⋮ → Repositories**, and add this
   repository's URL.
2. Install and start the **NEM Price Forecaster** add-on. Configure the region
   and options on the add-on's **Configuration** tab.
3. In the HA integration config flow, set the Sidecar URL to
   `http://localhost:8765`.

Both install methods ship the same engine code and the same bundled QLD1 model.

## Configuration

The sidecar is configured via `SIDECAR_*` environment variables (Docker) or the
add-on options. The full reference lives in
[`sidecar/app/config.py`](sidecar/app/config.py); the most common options:

| Option | Default | Notes |
| --- | --- | --- |
| `region` | `NSW1` | One of `QLD1`, `NSW1`, `VIC1`, `SA1`, `TAS1`. **QLD1 ships pre-trained.** |
| `price_model` | `hybrid` | `darts_naive_blend`, `isotonic`, `darts`, or `hybrid` (see below). |
| `calibrator` | `monotone_gbm` | How raw PD7DAY RRP is mapped to a realised price: `isotonic` (default) or opt-in `monotone_gbm`. |
| `forecast_horizon_days` | `7.0` | Forecast horizon (0.5–14 days). Validated accuracy is ≤ 7 days. |
| `naive_blend_weight` | `0.5` | Blend weight for `darts_naive_blend` (0 = pure Darts, 1 = pure seasonal-naive). |
| `gst_rate` | `0.10` | Applied to import prices. |
| `feed_in_is_wholesale` | `true` | Treat the export price as the calibrated wholesale RRP. |
| `weather_enabled` | `true` | Use free Open-Meteo weather covariates (no API key). |
| `latitude` / `longitude` | region default | Override the weather location for your household. |

### Price models

- **`darts_naive_blend`** (default) — a 50/50 blend of a Darts LightGBM model and
  a seasonal-naive estimate (same hour, same day-of-week, one week ago). Blending
  reduces worst-case error during market regime transitions while matching
  seasonal-naive's average accuracy. Works immediately.
- **`isotonic`** — per-hour isotonic (PAV) calibration of AEMO's PD7DAY
  predispatch. Best when predispatch tracks settlement closely; weaker when
  PD7DAY diverges.
- **`darts`** — Darts LightGBM only. Needs accumulated history to train.
- **`hybrid`** — isotonic for the near horizon, Darts beyond a configurable
  crossover (experimental).

### Calibrator backend

- **`isotonic`** (default) — 24 dependency-free per-hour PAV curves.
- **`monotone_gbm`** (opt-in) — a single LightGBM regressor with a monotone
  constraint on the raw forecast price plus cyclic hour/day-of-week features. It
  keeps a built-in never-lose fallback to isotonic: if it doesn't beat isotonic
  on a held-out tail (or LightGBM is unavailable), it transparently serves
  isotonic output, so selecting it can only match or beat the default.

## Pre-trained model

The repository bundles a pre-trained **QLD1** Darts price model
(`sidecar/app/models/qld1/`) plus a calibration seed
(`sidecar/app/seed/`). These ship inside the Docker image, so a fresh QLD1
install produces a model-backed forecast on day one.

Other regions self-train once enough live price history has accumulated, and
fall back to the seasonal-naive / isotonic path in the meantime. The bundled
model contains only learned price patterns — no personal data, coordinates, or
credentials.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
