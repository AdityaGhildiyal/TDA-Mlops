# Phase 2 Complete — TDA Feature Engineering

**Project:** TDA-MLOps: Topological Data Analysis for Production-Grade Anomaly Detection
**Phase:** 2 of 5 — Feature Engineering (Weeks 3–5)
**Status:** ✅ Complete

---

## Deliverables

| Artifact                                | Description                                                           |
| --------------------------------------- | --------------------------------------------------------------------- |
| `notebooks/phase2_tda_features.ipynb`   | 18 cells — full feature engineering narrative                         |
| `tda_detect/features.py`                | `finite_dgm`, `takens_embed`, `ripser_persist`, `TDAFeatureExtractor` |
| `tests/test_features.py`                | 29 pytest tests — all passing                                         |
| `figures/phase2_persistence_images.png` | Normal vs anomaly H₀/H₁ images + difference                           |
| `figures/phase2_runtime_profile.png`    | Latency and throughput vs window size                                 |

---

## Pipeline

```
signal[500]
    → takens_embed(dim=3, tau=31)       point cloud [438, 3]
    → ripser_persist(maxdim=1)          dgm_h0, dgm_h1
    → finite_dgm()                      strip inf essential class from H0
    → PersistenceImager(fixed ranges)   img_h0 [20,20], img_h1 [20,20]
    → flatten + concatenate             feature vector [800]
```

---

## Locked Hyperparameters

| Parameter     | Value      | Justification                                        |
| ------------- | ---------- | ---------------------------------------------------- |
| `window_size` | 500        | One full period of dominant frequency                |
| `embed_dim`   | 3          | Stable across dim ∈ {2, 3, 4} sweep                  |
| `embed_tau`   | 31         | ≈ quarter-period; confirmed stable                   |
| `maxdim`      | 1          | H₂ adds cost, no benefit for 1-D signals             |
| `birth_range` | (0.0, 2.1) | Covers dominant loop with 10 % headroom              |
| `pers_range`  | (0.0, 2.1) | Same range for both axes                             |
| `n_pixels` T  | 20         | Tight bright spot; optimal in T ∈ {5,10,20,40} sweep |
| `pixel_size`  | 0.105      | = 2.1 / 20                                           |
| `feature_len` | **800**    | 2 × 20² = 400 H₀ + 400 H₁                            |

---

## Critical Bug History — Do NOT Reintroduce

### Bug 1 — Never call `PersistenceImager.fit()`

`fit()` sets `birth_range = (x, x)` when the diagram has only one point,
producing a zero-width pixel grid → image shape `(0, 0)` → crash on
`.min()` / `.max()`.

**Fix:** always set ranges manually at construction time.

```python
# ✅ Correct
pimgr = PersistenceImager(
    birth_range=(0.0, 2.1),
    pers_range=(0.0, 2.1),
    pixel_size=0.105,
)

# ❌ Never do this
pimgr = PersistenceImager(...)
pimgr.fit(diagram)   # DO NOT CALL
```

### Bug 2 — Always filter `inf` from H0 before `transform()`

H0 always has one essential class with `death=inf`.
`transform()` cannot place infinite points on a finite pixel grid → crash.

**Fix:** call `finite_dgm()` on every H0 diagram before imaging.

```python
# ✅ Correct
h0_finite = finite_dgm(dgms[0])
img_h0 = pimgr.transform(h0_finite, skew=True)

# ❌ Never do this
img_h0 = pimgr.transform(dgms[0], skew=True)   # will crash
```

---

## Test Coverage

```
tests/test_features.py — 29 tests, all passing

TestTakensEmbed      (6)  shape, first/last row, dim=2, too-short raises,
                           exactly-2-points OK, output dtype
TestFiniteDgm        (4)  strips inf, all-finite unchanged,
                           all-inf→empty, empty input
TestRipserPersist    (5)  H0 count=437, H1 count=1, no inf,
                           constant signal, shape columns
TestExtractorInit    (4)  defaults, feature_len, custom hyperparams, repr
TestExtractorTransform(10) shape, dtype, no NaN/inf, non-negative,
                           constant→zeros, noise, short window,
                           too-short raises, list input, deterministic
TestStability        (2)  small perturbation → small L2, deterministic
TestSeparation       (3)  anomaly differs from normal, H0 component differs,
                           features not all same
```

---

## Runtime Profile

| Window (samples) | ms/window   | Max throughput |
| ---------------- | ----------- | -------------- |
| 75               | ~2 ms       | ~500 Hz        |
| 100              | ~3 ms       | ~330 Hz        |
| 200              | ~15 ms      | ~67 Hz         |
| 300              | ~40 ms      | ~25 Hz         |
| **500**          | **~150 ms** | **~7 Hz**      |
| 750              | ~350 ms     | ~3 Hz          |
| 1000             | ~600 ms     | ~2 Hz          |

At the chosen 500-sample window: **~7–10 Hz** sustainable throughput,
suitable for sensor data at 5 kHz with 50 % overlap.

---

## Feature Matrix — Ready for Phase 3

```python
X_all.shape  →  (100, 800)   # 50 normal + 50 anomalous windows
y_all.shape  →  (100,)        # 0 = normal, 1 = anomaly
```

---

## Phase 3 Preview — Anomaly Detection Model (Weeks 6–8)

```
X_all [100, 800]
    → IsolationForest(contamination=0.1)   train on normal windows only
    → decision_function()                  anomaly scores
    → threshold tuning                     maximise F1
    → evaluate vs baselines                z-score, rolling-std
    → target F1: 0.91–0.94                (12–27% above classical baseline)
```

Files to create in Phase 3:

- `tda_detect/model.py` — `TDAAnomalyDetector` wrapping IsolationForest
- `notebooks/phase3_anomaly_detection.ipynb`
- `tests/test_model.py`

---

## Git History (Phase 2)

```
feat(phase2/day5): persistence image foundations + imager bug fixes
feat(phase2/day6): TDAFeatureExtractor + features.py skeleton
feat(phase2/day7): notebook polish + figures + phase2 sign-off
```
