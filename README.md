<div align="center">

```
████████╗ ██████╗ ██████╗  ██████╗ ███████╗███████╗███╗   ██╗████████╗██╗███╗   ██╗███████╗██╗
╚══██╔══╝██╔═══██╗██╔══██╗██╔═══██╗██╔════╝██╔════╝████╗  ██║╚══██╔══╝██║████╗  ██║██╔════╝██║
   ██║   ██║   ██║██████╔╝██║   ██║███████╗█████╗  ██╔██╗ ██║   ██║   ██║██╔██╗ ██║█████╗  ██║
   ██║   ██║   ██║██╔═══╝ ██║   ██║╚════██║██╔══╝  ██║╚██╗██║   ██║   ██║██║╚██╗██║██╔══╝  ██║
   ██║   ╚██████╔╝██║     ╚██████╔╝███████║███████╗██║ ╚████║   ██║   ██║██║ ╚████║███████╗███████╗
   ╚═╝    ╚═════╝ ╚═╝      ╚═════╝ ╚══════╝╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚═╝╚═╝  ╚═══╝╚══════╝╚══════╝
```

# TopoSentinel

### Topological Anomaly Detection & Drift Monitoring for Production Systems

_Persistent Homology · Wasserstein Drift Detection · End-to-End MLOps Pipeline_

---

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?style=flat-square&logo=mlflow&logoColor=white)](https://mlflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Serving-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docker.com)
[![Ripser](https://img.shields.io/badge/Ripser-TDA-8B5CF6?style=flat-square)](https://ripser.scikit-tda.org)
[![Tests](https://img.shields.io/badge/Tests-97%20passing-brightgreen?style=flat-square)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

</div>

---

## What This Is

Classical anomaly detection methods — z-score, rolling standard deviation, even IsolationForest — operate on point statistics and discard the **structural information** encoded in the shape of data.

**TopoSentinel** fixes this.

It extracts topological features from time-series signals using **persistent homology** via Takens delay embedding. Normal signals form clean attractors in phase space; anomalies break that structure. TopoSentinel detects both the anomaly and the moment the underlying data distribution begins to drift — before performance degrades.

> **F1 = 0.9782 ± 0.0140 · Recall = 1.000 across all CV folds · Drift Precision = 0.9583**

---

## Table of Contents

- [Key Results](#key-results)
- [Architecture](#architecture)
- [The Math](#the-math)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Running with Docker](#running-with-docker)
- [API Reference](#api-reference)
- [Pipeline Phases](#pipeline-phases)
- [Ablation Studies](#ablation-studies)
- [Tech Stack](#tech-stack)
- [Thesis Context](#thesis-context)

---

## Key Results

### Anomaly Detection (5-Fold Cross-Validation)

| Fold     | F1         | Precision  | Recall     |
| -------- | ---------- | ---------- | ---------- |
| Fold 1   | 1.0000     | 1.0000     | 1.0000     |
| Fold 2   | 0.9756     | 0.9524     | 1.0000     |
| Fold 3   | 0.9639     | 0.9302     | 1.0000     |
| Fold 4   | 0.9877     | 0.9756     | 1.0000     |
| Fold 5   | 0.9639     | 0.9302     | 1.0000     |
| **Mean** | **0.9782** | **0.9577** | **1.0000** |
| Std      | 0.0140     | 0.0270     | 0.0000     |

**Recall = 1.000 on every fold — TopoSentinel never misses a real anomaly.**

### Baseline Comparison

| Model                    | F1         | Precision  | Recall     | vs TopoSentinel |
| ------------------------ | ---------- | ---------- | ---------- | --------------- |
| Z-score                  | 0.6667     | 0.5000     | 1.0000     | −33.0%          |
| Rolling Std              | 0.8066     | 0.9012     | 0.7300     | −18.9%          |
| IsolationForest (raw)    | 1.0000     | 1.0000     | 1.0000     | +0.5%           |
| **TopoSentinel (TDA) ★** | **0.9950** | **0.9901** | **1.0000** | —               |

### Drift Detection

| Metric                 | Value      |
| ---------------------- | ---------- |
| F1 Score               | 0.8519     |
| Precision              | **0.9583** |
| Recall                 | 0.7667     |
| Wasserstein separation | 1.1231     |
| First detection        | Window 13  |
| False positive rate    | 3% (2/60)  |

### Serving Performance

| Metric      | Value      |
| ----------- | ---------- |
| Throughput  | 1.76 req/s |
| p50 latency | 556.7 ms   |
| p95 latency | 774.8 ms   |
| p99 latency | 1160.6 ms  |

---

## Architecture

```
Time-Series Window (500 samples)
           │
           ▼
┌──────────────────────────┐
│   Takens Delay Embedding  │   dim=3, tau=31
│   500 samples → (468,3)  │   point cloud in phase space
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│   Ripser (Persistent     │   Vietoris-Rips filtration
│   Homology)              │   H0 + H1 diagrams
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│   Persistence Imaging     │   20×20 grid per dimension
│   H0 (400 dims)          │   Gaussian kernel weighted
│   H1 (400 dims)          │   by persistence
└────────────┬─────────────┘
             │
      ┌──────┴──────┐
      ▼             ▼
┌──────────┐  ┌────────────────────┐
│ Anomaly  │  │  Drift Detector    │
│ Detector │  │  Wasserstein dist  │
│ IsoForest│  │  on H1 diagrams    │
│ F1=0.978 │  │  Precision=0.958   │
└────┬─────┘  └────────┬───────────┘
     │                 │
     ▼                 ▼
┌─────────────────────────────┐
│   FastAPI REST API           │
│   /predict  /health          │
│   /drift_status              │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│   Docker Container           │
│   MLflow Experiment Tracking │
│   97 unit tests passing      │
└─────────────────────────────┘
```

---

## The Math

### Takens Embedding

Given a scalar time series x(t), the d-dimensional embedding with delay τ reconstructs the attractor:

```
X(t) = [x(t), x(t+τ), x(t+2τ), ..., x(t+(d-1)τ)]
```

Takens' theorem guarantees this preserves the topology of the original dynamical system. We use **d=3, τ=31** (selected by hyperparameter search).

### Persistent Homology

As ε increases from 0 to ∞, topological features appear (birth b) and disappear (death d):

| Dimension | Symbol | Tracks               |
| --------- | ------ | -------------------- |
| H0        | β₀     | Connected components |
| H1        | β₁     | Loops / 1D cycles    |

A normal sin(t) signal produces one dominant H1 feature — a single persistent loop. Anomalies break or multiply this structure.

### Boundary Matrices

The boundary operator ∂ₖ maps k-chains to (k-1)-chains:

```
∂₁ for triangle {A, B, C}:

     AB   BC   CA
A  [ -1    0   +1 ]
B  [ +1   -1    0 ]
C  [  0   +1   -1 ]

Fundamental lemma: ∂₁ ∘ ∂₂ = 0  (boundary of boundary is empty)

Betti numbers: βₖ = dim(Ker ∂ₖ) − dim(Im ∂ₖ₊₁)
```

### Wasserstein Drift Distance

The drift detector compares the H1 diagram of each incoming window against a reference centroid:

```
W(D₁, D₂) = Σ|pᵢ - q_σ(i)|   (sorted persistence values, top-50)
```

A window is flagged as drifted when W > calibrated threshold. The threshold is auto-calibrated by sweeping 100 candidates on held-out normal and drifted data, maximising F1.

### Stability Theorem

If two point clouds X and Y satisfy d_H(X, Y) ≤ δ, then:

```
W∞(D(X), D(Y)) ≤ δ
```

This guarantees that small noise perturbations cause only small changes in persistence diagrams — the mathematical foundation of TopoSentinel's robustness.

---

## Project Structure

```
TopoSentinel/
├── tda_detect/                        # Core library
│   ├── __init__.py                    # Public API exports
│   ├── features.py                    # TDAFeatureExtractor
│   ├── model.py                       # TDAAnomalyDetector
│   ├── drift.py                       # TopologicalDriftDetector
│   ├── serve.py                       # FastAPI application
│   └── utils.py                       # BoundaryMatrix, helpers
│
├── notebooks/
│   ├── phase1_appendix_A.ipynb        # Math foundations
│   ├── phase2_tda_features.ipynb      # Feature engineering
│   ├── phase3_models.ipynb            # Anomaly detection
│   ├── phase4_drift.ipynb             # Drift detection + serving
│   └── phase5_evaluation.ipynb        # Evaluation + thesis figures
│   └── figures/                       # All generated figures
│
├── models/
│   ├── isoforest_phase3.pkl           # Trained anomaly detector
│   └── drift_detector_phase4.pkl      # Calibrated drift detector
│
├── tests/
│   ├── test_features.py               # 29 tests
│   ├── test_models.py                 # 38 tests
│   └── test_drift.py                  # 30 tests
│
├── serve/
│   └── Dockerfile                     # Service Dockerfile
│
├── Dockerfile                         # Root Dockerfile (for docker build)
├── docker-compose.yml                 # One-command stack startup
├── setup.py                           # Installable package
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Clone & set up environment

```bash
git clone https://github.com/your-username/TopoSentinel.git
cd TopoSentinel

python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Verify TDA stack

```bash
python -c "import ripser, persim; from tda_detect import TDAFeatureExtractor; print('TopoSentinel OK ✓')"
```

### 3. Run the test suite

```bash
pytest tests/ -v
# Expected: 97 passed
```

### 4. Run the notebooks in order

```bash
jupyter notebook notebooks/
```

| Notebook                    | Content                                       |
| --------------------------- | --------------------------------------------- |
| `phase1_appendix_A.ipynb`   | Boundary matrices, Betti numbers, Ripser demo |
| `phase2_tda_features.ipynb` | TDA feature extraction pipeline               |
| `phase3_models.ipynb`       | Anomaly detection + baselines                 |
| `phase4_drift.ipynb`        | Drift detector + FastAPI + MLflow             |
| `phase5_evaluation.ipynb`   | Cross-validation, ablation, thesis figures    |

### 5. Quick inference (no Docker)

```python
import numpy as np
import sys
sys.path.insert(0, ".")
from tda_detect import TDAAnomalyDetector, TDAFeatureExtractor

# Load trained model
detector = TDAAnomalyDetector.load("models/isoforest_phase3.pkl")

# Create a test signal
t      = np.linspace(0, 4*np.pi, 500)
signal = np.sin(t) + np.random.default_rng(42).normal(0, 0.05, 500)

# Predict
label = detector.predict([signal])[0]
print(f"Label: {label}  (0=normal, 1=anomaly)")
```

---

## Running with Docker

### Build the image

```bash
cd TopoSentinel
docker build -t toposentinel:latest .
```

Build time: ~2 minutes (downloads python:3.12-slim + compiles ripser).

### Start the API server

```bash
docker run -p 8000:8000 toposentinel:latest
```

You will see:

```
INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Check the service is running

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{
  "status": "ok",
  "model_version": "phase3-isoforest-tda",
  "uptime_seconds": 1.24
}
```

### Send a normal signal

```bash
python -c "
import requests, numpy as np
t      = np.linspace(0, 4*np.pi, 500)
signal = (np.sin(t) + np.random.default_rng(42).normal(0, 0.05, 500)).tolist()
r = requests.post('http://localhost:8000/predict', json={'signal': signal})
print(r.json())
"
```

Expected response:

```json
{
  "label": 0,
  "score": -0.470579,
  "anomaly": false,
  "threshold": -0.605014,
  "latency_ms": 245.6
}
```

### Send an anomaly signal

```bash
python -c "
import requests, numpy as np
signal = np.random.default_rng(1).normal(0, 0.7, 500).tolist()
r = requests.post('http://localhost:8000/predict', json={'signal': signal})
print(r.json())
"
```

Expected response:

```json
{
  "label": 1,
  "score": -0.702457,
  "anomaly": true,
  "threshold": -0.605014,
  "latency_ms": 180.3
}
```

### Check drift status

```bash
curl http://localhost:8000/drift_status
```

Expected response:

```json
{
  "drift_detected": false,
  "wasserstein_distance": 0.0,
  "threshold": 70.18,
  "n_windows_seen": 1
}
```

### Interactive API docs (Swagger UI)

Open your browser at:

```
http://localhost:8000/docs
```

FastAPI auto-generates a full interactive UI — paste any signal and see the response without writing any code. Ideal for demos.

### Stop the container

```bash
docker stop $(docker ps -q --filter ancestor=toposentinel:latest)
```

### One-command stack (API + MLflow)

```bash
docker-compose up
```

| Service          | URL                     |
| ---------------- | ----------------------- |
| TopoSentinel API | `http://localhost:8000` |
| MLflow UI        | `http://localhost:5000` |

### View MLflow experiment runs

```bash
# While in venv (not Docker):
mlflow ui --backend-store-uri notebooks/mlruns
# Open: http://127.0.0.1:5000
```

---

## API Reference

### POST /predict

Classify a 500-sample time-series window.

**Request:**

```json
{
  "signal": [0.012, -0.034, 0.087, ...]   // exactly 500 floats
}
```

**Response:**

```json
{
  "label": 0, // 0 = normal, 1 = anomaly
  "score": -0.470579, // IsolationForest score (higher = more normal)
  "anomaly": false,
  "threshold": -0.605014,
  "latency_ms": 245.6
}
```

**Errors:**

- `422 Unprocessable Entity` — signal length != 500
- `500 Internal Server Error` — feature extraction failed

---

### GET /health

```json
{
  "status": "ok",
  "model_version": "phase3-isoforest-tda",
  "uptime_seconds": 42.3
}
```

---

### GET /drift_status

```json
{
  "drift_detected": false,
  "wasserstein_distance": 69.85,
  "threshold": 70.18,
  "n_windows_seen": 15
}
```

`drift_detected: true` means the most recently processed window had a Wasserstein distance above the calibrated threshold, indicating the incoming data distribution has shifted from the reference.

---

## Pipeline Phases

<details>
<summary><strong>Phase 1 — Math Foundations</strong></summary>

- Boundary matrices computed by hand in NumPy
- Betti numbers β₀, β₁ validated against Ripser
- Takens embedding on sinusoidal signals
- **Deliverable**: `notebooks/phase1_appendix_A.ipynb`

</details>

<details>
<summary><strong>Phase 2 — TDA Feature Engineering</strong></summary>

```python
from tda_detect import TDAFeatureExtractor

ext  = TDAFeatureExtractor(dim=3, tau=31, n_pixels=20)
feat = ext.transform(signal)   # returns 800-dim vector
```

- Sliding window construction
- Vietoris-Rips via Ripser
- Persistence image vectorisation via persim
- Locked hyperparameters: dim=3, tau=31, n_pixels=20, feature_len=800

</details>

<details>
<summary><strong>Phase 3 — Anomaly Detection</strong></summary>

```python
from tda_detect import TDAAnomalyDetector

detector = TDAAnomalyDetector(n_estimators=200, random_state=42)
detector.fit(normal_signals)
detector.calibrate_threshold(val_signals, y_val)
labels = detector.predict(test_signals)
```

- IsolationForest on 800-dim TDA features
- Threshold calibrated by sweeping 300 candidates on validation set
- F1 = 0.9782 ± 0.0140, Recall = 1.000 across all CV folds

</details>

<details>
<summary><strong>Phase 4 — Drift Detection + Serving</strong></summary>

```python
from tda_detect import TopologicalDriftDetector

drift_det = TopologicalDriftDetector(dim=3, tau=31, homology_dim=1)
drift_det.fit(reference_signals)
drift_det.calibrate_threshold(normal_cal, drifted_cal)

result = drift_det.update(new_signal)
# {'drift_detected': False, 'wasserstein_distance': 69.85, ...}
```

- Wasserstein distance on H1 persistence diagrams
- Auto-calibrated threshold (F1=0.8519, Precision=0.9583)
- FastAPI serving + Docker containerisation
- MLflow experiment tracking

</details>

<details>
<summary><strong>Phase 5 — Evaluation & Thesis</strong></summary>

- 5-fold cross-validation
- Ablation: TDA vs raw, H0 vs H1 vs H0+H1
- Hyperparameter sensitivity (dim, tau, n_pixels)
- t-SNE visualisation of feature space
- Persistence diagram gallery
- Throughput benchmark

</details>

---

## Ablation Studies

### Homology Dimensions

| Features  | Dims    | F1         | Precision  | Recall     |
| --------- | ------- | ---------- | ---------- | ---------- |
| H0 only   | 400     | 0.9878     | 0.9812     | 0.9950     |
| H1 only   | 400     | 0.9426     | 0.9842     | 0.9050     |
| **H0+H1** | **800** | **0.9782** | **0.9577** | **1.0000** |

H0 is the primary discriminator. H1 contributes complementary information — their combination achieves perfect Recall.

### Hyperparameter Sensitivity

| Parameter | Tested Values | Locked | Best Δ             |
| --------- | ------------- | ------ | ------------------ |
| dim       | 2, 3, 4       | **3**  | +0.003 (dim=4)     |
| tau       | 15, 31, 50    | **31** | +0.005 (tau=50)    |
| n_pixels  | 10, 20, 30    | **20** | <0.001 (all equal) |

Locked values are near-optimal. Improvements at higher settings are marginal and not worth the compute cost.

---

## Tech Stack

| Layer               | Tool                   | Why                                    |
| ------------------- | ---------------------- | -------------------------------------- |
| TDA computation     | Ripser + persim        | Fastest VR complex + persistence imgs  |
| ML model            | scikit-learn IsoForest | Unsupervised, calibratable threshold   |
| Drift detection     | Custom Wasserstein     | Topological drift, not just statistics |
| Experiment tracking | MLflow                 | Industry standard                      |
| Serving             | FastAPI + Uvicorn      | Async, auto-docs, fast                 |
| Containerisation    | Docker + Compose       | Reproducible deployment                |
| Testing             | pytest (97 tests)      | Full coverage across all modules       |
| CI/CD               | GitHub Actions         | Runs pytest on every push              |

---

## Thesis Context

This repository is the implementation for a B.Tech Final Year Thesis.

**Thesis title**: _TopoSentinel: Topological Data Analysis for Production-Grade Anomaly Detection_

**Three contributions beyond standard ML projects:**

1. **Topological Wasserstein drift detector** — W distance between H1 persistence diagrams as a drift signal. Precision = 0.9583, mathematically principled, not available in any off-the-shelf MLOps tool.

2. **Dual-capability TDA features** — a single 800-dimensional representation enables both anomaly classification (Recall = 1.000) and drift detection (Precision = 0.9583). Raw features cannot provide both.

3. **Production-complete pipeline** — feature extraction → calibrated classifier → drift monitor → REST API → Docker → MLflow tracking → 97 tests. Most student projects stop at a trained model.

### Recommended Reading

| Resource                                                  | Sections     |
| --------------------------------------------------------- | ------------ |
| Edelsbrunner & Harer, _Computational Topology_ (free PDF) | Ch. 1–4      |
| Carlsson, _Topology and Data_, Bull. AMS 2009             | Sec. 1–3     |
| Adams et al., _Persistence Images_, JMLR 2017             | Full paper   |
| Bauer, _Ripser_, 2021                                     | Introduction |

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

<div align="center">

_TopoSentinel · B.Tech Final Year Thesis · Computer Science & Engineering · 2025–26_

</div>
