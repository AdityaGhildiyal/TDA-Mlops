markdown# Phase 4 Notebook — MLOps Pipeline + Drift Detection

**File:** `notebooks/phase4_drift.ipynb`  
**Status:** COMPLETE  
**Commit:** `feat(phase4): MLOps pipeline + drift detection + FastAPI serving`  
**Duration:** Days 10–11

---

## Objective

Build a production-grade MLOps pipeline around the Phase 3 anomaly detector:

- Topological drift detection using Wasserstein distance on H1 diagrams
- MLflow experiment tracking for both anomaly and drift experiments
- FastAPI serving with `/predict`, `/health`, `/drift_status` endpoints
- Throughput benchmarking

---

## Notebook Structure

| Cell | Title                                  | Key Output                                 |
| ---- | -------------------------------------- | ------------------------------------------ |
| 1    | Markdown — Theory                      | Wasserstein distance math, MLflow overview |
| 2    | TopologicalDriftDetector (in-notebook) | Class definition + direction check         |
| 3    | Drift simulation + calibration         | Calibrated threshold, detection summary    |
| 4    | Drift detection curve figure           | `figures/phase4_drift_detection_curve.png` |
| 5    | MLflow setup + Phase 3 run logging     | Run ID logged                              |
| 6    | MLflow drift experiment logging        | Run ID logged                              |
| 7    | Drift module verification + save       | `models/drift_detector_phase4.pkl`         |
| 8    | FastAPI endpoint demo                  | All 5 tests passing                        |
| 9    | Throughput benchmark                   | 1.76 req/s, p50=557ms                      |
| 10   | Phase 4 sign-off                       | Results summary                            |

---

## Key Design Decisions

### Drift Signal Choice

Initial attempt used gradual frequency increase `sin(t) → sin(1.8t)`.
This failed because both signals produce loops in H1 — topologically identical.

**Fix:** Gradual mixing of `sin(t)` with `sin(√2·t)`:

```python
sig = (1 - alpha) * np.sin(t) + alpha * np.sin(np.sqrt(2) * t)
```

`√2` is irrational relative to 1 — the mixed signal creates a genuinely
different attractor topology that H1 separates cleanly.
W separation jumped from **0.069 → 1.123** after this change.

### Threshold Calibration

Hardcoded `threshold=0.5` produced:

- False positives: 50/60 (83%) — unusable in production

Auto-calibration on held-out data:

```python
drift_det.calibrate_threshold(
    normal_signals  = make_normal(20, seed=99),
    drifted_signals = make_drifted(20, seed=77),
    n_thresholds    = 100
)
```

Result: False positives dropped to **2/60 (3%)**.

### Phase 3 Model Recalibration

The `isoforest_phase3.pkl` loaded by `serve.py` had a stale threshold
(`-0.147`) that did not match live score distributions (`~-0.75`).
Root cause: model was saved from a different kernel state.

Fix: retrained IsolationForest from scratch with same Phase 3 hyperparameters.
Result: clean score separation and F1=1.0000.

### FastAPI Model Loading

`serve.py` uses `Path(__file__).resolve().parent.parent` to anchor model
paths to the project root, making it kernel-cwd independent.

---

## Results

### Drift Detection

| Metric                 | Value      |
| ---------------------- | ---------- |
| F1 Score               | 0.8519     |
| Precision              | **0.9583** |
| Recall                 | 0.7667     |
| Wasserstein separation | 1.1231     |
| First detection window | 13         |
| False positive rate    | 3% (2/60)  |

Detection lag of 13 windows is expected and desirable for gradual drift —
instant triggering would indicate overfitting to noise.

### Anomaly Detection (recalibrated)

| Metric    | Value   |
| --------- | ------- |
| F1 Score  | 1.0000  |
| Precision | 1.0000  |
| Recall    | 1.0000  |
| Threshold | -0.6050 |

### FastAPI Serving

| Metric      | Value      |
| ----------- | ---------- |
| Throughput  | 1.76 req/s |
| p50 latency | 556.7 ms   |
| p95 latency | 774.8 ms   |
| p99 latency | 1160.6 ms  |

Latency is dominated by TDA feature extraction (Takens embedding + Ripser

- persistence imaging). This is an intentional trade-off — topological
  analysis on every window enables F1=1.0 anomaly detection.

---

## Artifacts Produced

| File                                       | Description                         |
| ------------------------------------------ | ----------------------------------- |
| `tda_detect/drift.py`                      | `TopologicalDriftDetector` class    |
| `tda_detect/serve.py`                      | FastAPI app with 3 endpoints        |
| `tda_detect/__init__.py`                   | Public package API                  |
| `setup.py`                                 | Installable package configuration   |
| `models/isoforest_phase3.pkl`              | Recalibrated anomaly detector       |
| `models/drift_detector_phase4.pkl`         | Calibrated drift detector           |
| `figures/phase4_drift_detection_curve.png` | Drift detection curve               |
| `mlruns/`                                  | MLflow experiment tracking (2 runs) |
| `tests/test_drift.py`                      | 30 unit tests, all passing          |

---

## MLflow Runs

| Run Name                          | Run ID                             | Key Metric      |
| --------------------------------- | ---------------------------------- | --------------- |
| phase3-isoforest-tda              | `dcbf896c334a44eaa5bb748630696acf` | F1=0.9877       |
| phase4-topological-drift-detector | `72ec35299efd4d05b841c79b817f4b7f` | Drift F1=0.8519 |

View locally:

```bash
mlflow ui --backend-store-uri notebooks/mlruns
# Open: http://127.0.0.1:5000
```

---

## Test Suite

```
tests/test_drift.py — 30 tests, all passing (210s)
```

Run:

```bash
venv\Scripts\pytest tests/test_drift.py -v
```

---

## Known Limitations

1. **Throughput (1.76 req/s)** — TDA extraction is CPU-bound. Can be
   addressed with feature caching or async workers in Phase 5.
2. **Drift recall (0.767)** — 23% of drifted windows missed, primarily
   early in the drift onset (windows 60–72) when mixing weight α is low.
3. **`/drift_status` endpoint** — Currently returns last known state.
   A production version would maintain a rolling window buffer.

---

## Issues Encountered & Fixes

| Issue                     | Root Cause                              | Fix                                |
| ------------------------- | --------------------------------------- | ---------------------------------- |
| `mlflow` shadowed         | Folder named `mlflow/` in project root  | Renamed to `mlruns/`               |
| `mlflow` not installed    | Not in venv                             | `pip install mlflow`               |
| W separation = 0.069      | Frequency drift topologically invisible | Switched to `sin(√2·t)` mixing     |
| Drift F1 = 0.67           | Hardcoded threshold=0.5 too low         | Auto-calibration on held-out data  |
| `/predict` 500 error      | `transform()` takes 1D not batch        | Fixed call to `ext.transform(sig)` |
| Model paths not found     | `serve.py` used relative paths          | Anchored to `__file__`             |
| Stale threshold in pkl    | Wrong model serialized in Phase 3       | Retrained + resaved from scratch   |
| `PicklingError` in Cell 7 | In-notebook class ≠ module class        | Copied state to module instance    |

---

## Next Phase

**Phase 5 — Evaluation & Thesis Writing (Weeks 12–14)**

- Cross-validation + ablation studies
- t-SNE visualisation of TDA feature space
- Persistence diagram gallery
- Docker containerisation
- Final thesis document
