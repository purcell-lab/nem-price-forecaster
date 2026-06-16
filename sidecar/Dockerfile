# NEM Price Forecaster sidecar — FastAPI + Darts LightGBM
#
# python:3.12-slim chosen for:
#   - Full binary wheel availability for numpy, scikit-learn, lightgbm
#   - No torch dependency: darts[notorch] installs the tabular models only
#   - Slim base keeps the image under ~600 MB
#
# Multi-stage build: we install deps in a builder layer so the final image
# has no dangling build artefacts.

FROM python:3.12-slim AS builder

WORKDIR /build

# System deps for lightgbm (libgomp) and general health
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install into a prefix we can COPY wholesale to the final stage
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


FROM python:3.12-slim AS final

# Runtime libgomp (LightGBM needs it)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy pre-built packages from builder
COPY --from=builder /install /usr/local

WORKDIR /app
COPY app/ /app/

# Persist models + calibration cache across container restarts
VOLUME ["/data"]

ENV PYTHONUNBUFFERED=1
ENV SIDECAR_DATA_DIR=/data

EXPOSE 8765

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8765", "--workers", "1"]
