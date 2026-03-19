"""
tests/test_drift.py
Unit tests for TopologicalDriftDetector.
"""

import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tda_detect.drift import TopologicalDriftDetector


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture
def t():
    return np.linspace(0, 4*np.pi, 500)

@pytest.fixture
def normal_signals(t):
    rng = np.random.default_rng(42)
    return [np.sin(t) + rng.normal(0, 0.05, 500) for _ in range(20)]

@pytest.fixture
def drifted_signals():
    """Multi-frequency signal — richer H1 topology than sin(t) reference."""
    rng = np.random.default_rng(99)
    t   = np.linspace(0, 4*np.pi, 500)
    return [
        np.sin(t) + np.sin(3*t) + np.sin(5*t) + rng.normal(0, 0.05, 500)
        for _ in range(20)
    ]

@pytest.fixture
def fitted_detector(normal_signals):
    det = TopologicalDriftDetector(threshold=0.5)
    det.fit(normal_signals)
    return det


# ── Construction ──────────────────────────────────────────────────────────────
class TestInit:
    def test_default_params(self):
        det = TopologicalDriftDetector()
        assert det.threshold    == 0.5
        assert det.dim          == 3
        assert det.tau          == 31
        assert det.homology_dim == 1

    def test_custom_params(self):
        det = TopologicalDriftDetector(threshold=1.0, dim=2, tau=15,
                                       homology_dim=0)
        assert det.threshold    == 1.0
        assert det.dim          == 2
        assert det.tau          == 15
        assert det.homology_dim == 0

    def test_unfitted_state(self):
        det = TopologicalDriftDetector()
        assert det.reference_diagrams_ is None
        assert det.reference_mean_     is None
        assert det.n_windows_seen_     == 0


# ── Fitting ───────────────────────────────────────────────────────────────────
class TestFit:
    def test_fit_sets_reference(self, normal_signals):
        det = TopologicalDriftDetector()
        det.fit(normal_signals)
        assert det.reference_diagrams_ is not None
        assert det.reference_mean_     is not None

    def test_fit_reference_count(self, normal_signals):
        det = TopologicalDriftDetector()
        det.fit(normal_signals)
        assert len(det.reference_diagrams_) == len(normal_signals)

    def test_fit_returns_self(self, normal_signals):
        det = TopologicalDriftDetector()
        result = det.fit(normal_signals)
        assert result is det

    def test_fit_resets_counter(self, normal_signals):
        det = TopologicalDriftDetector()
        det.fit(normal_signals)
        assert det.n_windows_seen_ == 0

    def test_reference_mean_shape(self, fitted_detector):
        assert fitted_detector.reference_mean_.ndim == 2
        assert fitted_detector.reference_mean_.shape[1] == 2


# ── Update ────────────────────────────────────────────────────────────────────
class TestUpdate:
    def test_update_returns_dict(self, fitted_detector, normal_signals):
        result = fitted_detector.update(normal_signals[0])
        assert isinstance(result, dict)

    def test_update_keys(self, fitted_detector, normal_signals):
        result = fitted_detector.update(normal_signals[0])
        assert "drift_detected"       in result
        assert "wasserstein_distance" in result
        assert "threshold"            in result
        assert "n_windows_seen"       in result

    def test_update_increments_counter(self, fitted_detector, normal_signals):
        before = fitted_detector.n_windows_seen_
        fitted_detector.update(normal_signals[0])
        assert fitted_detector.n_windows_seen_ == before + 1

    def test_update_distance_positive(self, fitted_detector, normal_signals):
        result = fitted_detector.update(normal_signals[0])
        assert result["wasserstein_distance"] >= 0

    def test_update_drift_is_bool(self, fitted_detector, normal_signals):
        result = fitted_detector.update(normal_signals[0])
        assert isinstance(result["drift_detected"], bool)

    def test_normal_no_drift(self, normal_signals):
        det = TopologicalDriftDetector(threshold=999.0)
        det.fit(normal_signals)
        result = det.update(normal_signals[0])
        assert result["drift_detected"] is False

    def test_drifted_triggers_drift(self, normal_signals, drifted_signals):
        det = TopologicalDriftDetector(threshold=0.0)
        det.fit(normal_signals)
        result = det.update(drifted_signals[0])
        assert result["drift_detected"] is True


# ── Update batch ──────────────────────────────────────────────────────────────
class TestUpdateBatch:
    def test_batch_length(self, fitted_detector, normal_signals):
        results = fitted_detector.update_batch(normal_signals)
        assert len(results) == len(normal_signals)

    def test_batch_all_dicts(self, fitted_detector, normal_signals):
        results = fitted_detector.update_batch(normal_signals)
        assert all(isinstance(r, dict) for r in results)

    def test_batch_counter(self, fitted_detector, normal_signals):
        before = fitted_detector.n_windows_seen_
        fitted_detector.update_batch(normal_signals)
        assert fitted_detector.n_windows_seen_ == before + len(normal_signals)


# ── Calibration ───────────────────────────────────────────────────────────────
class TestCalibration:
    def test_calibration_returns_self(self, fitted_detector,
                                      normal_signals, drifted_signals):
        result = fitted_detector.calibrate_threshold(normal_signals,
                                                      drifted_signals)
        assert result is fitted_detector

    def test_calibration_sets_threshold(self, fitted_detector,
                                        normal_signals, drifted_signals):
        old_thr = fitted_detector.threshold
        fitted_detector.calibrate_threshold(normal_signals, drifted_signals)
        assert fitted_detector.threshold != old_thr

    def test_calibration_threshold_positive(self, fitted_detector,
                                            normal_signals, drifted_signals):
        fitted_detector.calibrate_threshold(normal_signals, drifted_signals)
        assert fitted_detector.threshold > 0

    def test_calibration_direction(self, fitted_detector,
                                   normal_signals, drifted_signals):
        """Calibration should achieve F1 > 0.5 on held-out data."""
        from sklearn.metrics import f1_score

        fitted_detector.calibrate_threshold(normal_signals, drifted_signals)

        all_signals = normal_signals + drifted_signals
        labels      = [0]*len(normal_signals) + [1]*len(drifted_signals)
        preds       = [int(fitted_detector.update(s)["drift_detected"])
                       for s in all_signals]
        f1 = f1_score(labels, preds, zero_division=0)
        assert f1 > 0.5, f"Calibration F1={f1:.3f} too low"


# ── Reset ─────────────────────────────────────────────────────────────────────
class TestReset:
    def test_reset_clears_counter(self, fitted_detector, normal_signals):
        fitted_detector.update_batch(normal_signals)
        fitted_detector.reset()
        assert fitted_detector.n_windows_seen_ == 0

    def test_reset_with_new_signals(self, fitted_detector, normal_signals):
        fitted_detector.reset(new_signals=normal_signals)
        assert fitted_detector.reference_diagrams_ is not None
        assert fitted_detector.n_windows_seen_ == 0


# ── Persistence ───────────────────────────────────────────────────────────────
class TestPersistence:
    def test_save_load_roundtrip(self, fitted_detector, tmp_path):
        path = str(tmp_path / "detector.pkl")
        fitted_detector.save(path)
        loaded = TopologicalDriftDetector.load(path)
        assert loaded.threshold == fitted_detector.threshold
        assert len(loaded.reference_mean_) == len(fitted_detector.reference_mean_)

    def test_loaded_detector_predicts(self, fitted_detector,
                                      normal_signals, tmp_path):
        path = str(tmp_path / "detector.pkl")
        fitted_detector.save(path)
        loaded = TopologicalDriftDetector.load(path)
        result = loaded.update(normal_signals[0])
        assert "drift_detected" in result


# ── Wasserstein helper ────────────────────────────────────────────────────────
class TestWasserstein:
    def test_empty_dgm1(self, fitted_detector):
        dgm1 = np.empty((0, 2))
        dgm2 = np.array([[0.1, 0.5], [0.2, 0.8]])
        dist = fitted_detector._wasserstein(dgm1, dgm2)
        assert dist >= 0

    def test_empty_dgm2(self, fitted_detector):
        dgm1 = np.array([[0.1, 0.5], [0.2, 0.8]])
        dgm2 = np.empty((0, 2))
        dist = fitted_detector._wasserstein(dgm1, dgm2)
        assert dist >= 0

    def test_both_empty(self, fitted_detector):
        dgm1 = np.empty((0, 2))
        dgm2 = np.empty((0, 2))
        dist = fitted_detector._wasserstein(dgm1, dgm2)
        assert dist == 0.0

    def test_identical_diagrams(self, fitted_detector):
        dgm = np.array([[0.1, 0.5], [0.2, 0.8], [0.3, 1.0]])
        dist = fitted_detector._wasserstein(dgm, dgm)
        assert dist == 0.0