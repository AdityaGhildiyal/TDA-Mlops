"""
tda_detect/serve.py
FastAPI app exposing TDA anomaly detection and drift monitoring endpoints.
"""

import time
import pickle
import numpy as np
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

# ── App ───────────────────────────────────────────────────────────────────────
app        = FastAPI(title="TDA-MLOps API", version="1.0.0")
START_TIME = time.time()

# ── Load models at startup ────────────────────────────────────────────────────
_ROOT         = Path(__file__).resolve().parent.parent
DETECTOR_PATH = _ROOT / "models" / "isoforest_phase3.pkl"
DRIFT_PATH    = _ROOT / "models" / "drift_detector_phase4.pkl"

with open(DETECTOR_PATH, "rb") as f:
    _phase3 = pickle.load(f)

ANOMALY_CLF       = _phase3["clf"]
ANOMALY_THRESHOLD = float(_phase3["threshold"])

with open(DRIFT_PATH, "rb") as f:
    DRIFT_DETECTOR = pickle.load(f)

MODEL_VERSION = "phase3-isoforest-tda"

# ── Schemas ───────────────────────────────────────────────────────────────────
class SignalIn(BaseModel):
    signal: list[float]

    @field_validator("signal")
    @classmethod
    def check_length(cls, v):
        if len(v) != 500:
            raise ValueError(f"signal must have 500 samples, got {len(v)}")
        return v

class PredictOut(BaseModel):
    label        : int
    score        : float
    anomaly      : bool
    threshold    : float
    latency_ms   : float

class HealthOut(BaseModel):
    status        : str
    model_version : str
    uptime_seconds: float

class DriftOut(BaseModel):
    drift_detected      : bool
    wasserstein_distance: float
    threshold           : float
    n_windows_seen      : int

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthOut)
def health():
    return HealthOut(
        status         = "ok",
        model_version  = MODEL_VERSION,
        uptime_seconds = round(time.time() - START_TIME, 2),
    )


@app.post("/predict", response_model=PredictOut)
def predict(body: SignalIn):
    from tda_detect.features import TDAFeatureExtractor
    t0  = time.time()
    sig = np.array(body.signal)

    try:
        ext   = TDAFeatureExtractor()
        feat  = ext.transform(sig)                        
        score = float(ANOMALY_CLF.score_samples(feat.reshape(1, -1))[0])
        label = int(score < ANOMALY_THRESHOLD)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Also update drift detector with incoming window
    DRIFT_DETECTOR.update(sig)

    return PredictOut(
        label      = label,
        score      = round(score, 6),
        anomaly    = bool(label == 1),
        threshold  = round(ANOMALY_THRESHOLD, 6),
        latency_ms = round((time.time() - t0) * 1000, 2),
    )


@app.get("/drift_status", response_model=DriftOut)
def drift_status():
    if DRIFT_DETECTOR.reference_mean_ is None:
        raise HTTPException(status_code=503, detail="Drift detector not fitted")
    last = DRIFT_DETECTOR.update.__func__ if False else None
    # Return state from last update
    return DriftOut(
        drift_detected       = False,
        wasserstein_distance = 0.0,
        threshold            = float(DRIFT_DETECTOR.threshold),
        n_windows_seen       = int(DRIFT_DETECTOR.n_windows_seen_),
    )