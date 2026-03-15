"""
tests/test_model.py
===================
pytest suite for tda_detect.model.TDAAnomalyDetector

Run from the repo root:
    pytest tests/test_model.py -v

Coverage
--------
1.  __init__          — default + custom params, attributes present
2.  fit               — trains without error, sets is_fitted_
3.  score             — output shape, dtype, direction (normal > anomaly)
4.  predict           — output shape, dtype, values in {0,1}
5.  calibrate_threshold — returns float, stores on self, improves F1
6.  evaluate          — returns dict with correct keys, F1 ≥ 0.0
7.  save / load       — round-trip pickle, scores match
8.  error handling    — score/predict before fit raises RuntimeError
9.  end-to-end        — fit → calibrate → evaluate, F1 ≥ 0.80
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest

from tda_detect.model import TDAAnomalyDetector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_NORMAL  = 80    # training windows
N_TEST    = 40    # test windows (20 normal + 20 anomaly)
WIN_SIZE  = 500


def _make_normal(n, seed=0):
    rng = np.random.default_rng(seed)
    t   = np.linspace(0, 4 * np.pi, WIN_SIZE)
    return [np.sin(t) + rng.normal(0, 0.02, WIN_SIZE) for _ in range(n)]


def _make_anomaly(n, seed=1):
    rng = np.random.default_rng(seed)
    t   = np.linspace(0, 4 * np.pi, WIN_SIZE)
    signals = []
    for i in range(n):
        sig      = np.sin(t) + rng.normal(0, 0.02, WIN_SIZE)
        break_pt = int(rng.integers(150, 350))
        sig[break_pt:] = (np.sin(t[break_pt:] + np.pi * 0.7)
                          + rng.normal(0, 0.02, WIN_SIZE - break_pt))
        signals.append(sig)
    return signals


@pytest.fixture(scope="module")
def normal_train():
    return _make_normal(N_NORMAL, seed=42)


@pytest.fixture(scope="module")
def test_signals():
    normal  = _make_normal(N_TEST // 2,  seed=99)
    anomaly = _make_anomaly(N_TEST // 2, seed=100)
    signals = normal + anomaly
    labels  = np.array([0] * (N_TEST // 2) + [1] * (N_TEST // 2))
    return signals, labels


@pytest.fixture(scope="module")
def fitted_detector(normal_train):
    det = TDAAnomalyDetector(n_estimators=100, random_state=42)
    det.fit(normal_train)
    return det


@pytest.fixture(scope="module")
def calibrated_detector(fitted_detector, test_signals):
    signals, labels = test_signals
    fitted_detector.calibrate_threshold(signals, labels)
    return fitted_detector


# ---------------------------------------------------------------------------
# 1. __init__
# ---------------------------------------------------------------------------

class TestInit:

    def test_default_params(self):
        det = TDAAnomalyDetector()
        assert det.n_estimators  == 200
        assert det.contamination == "auto"
        assert det.random_state  == 42

    def test_custom_params(self):
        det = TDAAnomalyDetector(n_estimators=50, random_state=0)
        assert det.n_estimators == 50
        assert det.random_state == 0

    def test_extractor_created(self):
        det = TDAAnomalyDetector()
        assert det.extractor_ is not None
        assert det.extractor_.feature_len == 800

    def test_not_fitted_initially(self):
        det = TDAAnomalyDetector()
        assert det.is_fitted_ is False

    def test_default_threshold(self):
        det = TDAAnomalyDetector()
        assert det.threshold_ == 0.0

    def test_repr(self):
        det = TDAAnomalyDetector()
        r   = repr(det)
        assert "TDAAnomalyDetector" in r
        assert "n_estimators" in r


# ---------------------------------------------------------------------------
# 2. fit
# ---------------------------------------------------------------------------

class TestFit:

    def test_fit_sets_is_fitted(self, normal_train):
        det = TDAAnomalyDetector(n_estimators=50, random_state=0)
        det.fit(normal_train)
        assert det.is_fitted_ is True

    def test_fit_returns_self(self, normal_train):
        det = TDAAnomalyDetector(n_estimators=50, random_state=0)
        ret = det.fit(normal_train)
        assert ret is det

    def test_fit_creates_clf(self, normal_train):
        det = TDAAnomalyDetector(n_estimators=50, random_state=0)
        det.fit(normal_train)
        assert det.clf_ is not None

    def test_fit_accepts_ndarray(self, normal_train):
        det = TDAAnomalyDetector(n_estimators=50, random_state=0)
        arr = np.array(normal_train)
        det.fit(arr)
        assert det.is_fitted_ is True


# ---------------------------------------------------------------------------
# 3. score
# ---------------------------------------------------------------------------

class TestScore:

    def test_output_shape(self, fitted_detector, test_signals):
        signals, _ = test_signals
        scores = fitted_detector.score(signals)
        assert scores.shape == (len(signals),)

    def test_output_dtype(self, fitted_detector, test_signals):
        signals, _ = test_signals
        scores = fitted_detector.score(signals)
        assert scores.dtype == float

    def test_no_nan(self, fitted_detector, test_signals):
        signals, _ = test_signals
        scores = fitted_detector.score(signals)
        assert not np.any(np.isnan(scores))

    def test_score_direction(self, fitted_detector, test_signals):
        """Normal windows should score higher (more normal) than anomalies."""
        signals, labels = test_signals
        scores = fitted_detector.score(signals)
        mean_normal  = scores[labels == 0].mean()
        mean_anomaly = scores[labels == 1].mean()
        assert mean_normal > mean_anomaly, (
            f"Score direction wrong: normal mean {mean_normal:.4f} "
            f"≤ anomaly mean {mean_anomaly:.4f}"
        )

    def test_score_before_fit_raises(self):
        det = TDAAnomalyDetector()
        with pytest.raises(RuntimeError, match="not fitted"):
            det.score([np.ones(500)])


# ---------------------------------------------------------------------------
# 4. predict
# ---------------------------------------------------------------------------

class TestPredict:

    def test_output_shape(self, fitted_detector, test_signals):
        signals, _ = test_signals
        labels = fitted_detector.predict(signals, threshold=-0.05)
        assert labels.shape == (len(signals),)

    def test_values_binary(self, fitted_detector, test_signals):
        signals, _ = test_signals
        labels = fitted_detector.predict(signals, threshold=-0.05)
        assert set(np.unique(labels)).issubset({0, 1})

    def test_dtype_int(self, fitted_detector, test_signals):
        signals, _ = test_signals
        labels = fitted_detector.predict(signals, threshold=-0.05)
        assert labels.dtype in (np.int32, np.int64, int)

    def test_uses_stored_threshold(self, calibrated_detector, test_signals):
        """predict() with no threshold arg should use self.threshold_."""
        signals, _ = test_signals
        thr    = calibrated_detector.threshold_
        pred1  = calibrated_detector.predict(signals)
        pred2  = calibrated_detector.predict(signals, threshold=thr)
        np.testing.assert_array_equal(pred1, pred2)

    def test_predict_before_fit_raises(self):
        det = TDAAnomalyDetector()
        with pytest.raises(RuntimeError, match="not fitted"):
            det.predict([np.ones(500)])


# ---------------------------------------------------------------------------
# 5. calibrate_threshold
# ---------------------------------------------------------------------------

class TestCalibrateThreshold:

    def test_returns_float(self, fitted_detector, test_signals):
        det = TDAAnomalyDetector(n_estimators=50, random_state=0)
        det.fit(_make_normal(N_NORMAL, seed=42))
        signals, labels = test_signals
        thr = det.calibrate_threshold(signals, labels)
        assert isinstance(thr, float)

    def test_stores_on_self(self, fitted_detector, test_signals):
        det = TDAAnomalyDetector(n_estimators=50, random_state=0)
        det.fit(_make_normal(N_NORMAL, seed=42))
        signals, labels = test_signals
        thr = det.calibrate_threshold(signals, labels)
        assert det.threshold_ == thr

    def test_threshold_in_score_range(self, fitted_detector, test_signals):
        det = TDAAnomalyDetector(n_estimators=50, random_state=0)
        det.fit(_make_normal(N_NORMAL, seed=42))
        signals, labels = test_signals
        scores = det.score(signals)
        thr    = det.calibrate_threshold(signals, labels)
        assert scores.min() <= thr <= scores.max()


# ---------------------------------------------------------------------------
# 6. evaluate
# ---------------------------------------------------------------------------

class TestEvaluate:

    def test_returns_dict(self, calibrated_detector, test_signals):
        signals, labels = test_signals
        result = calibrated_detector.evaluate(signals, labels, verbose=False)
        assert isinstance(result, dict)

    def test_dict_keys(self, calibrated_detector, test_signals):
        signals, labels = test_signals
        result = calibrated_detector.evaluate(signals, labels, verbose=False)
        assert set(result.keys()) == {"f1", "precision", "recall", "threshold"}

    def test_f1_in_range(self, calibrated_detector, test_signals):
        signals, labels = test_signals
        result = calibrated_detector.evaluate(signals, labels, verbose=False)
        assert 0.0 <= result["f1"] <= 1.0

    def test_precision_in_range(self, calibrated_detector, test_signals):
        signals, labels = test_signals
        result = calibrated_detector.evaluate(signals, labels, verbose=False)
        assert 0.0 <= result["precision"] <= 1.0

    def test_recall_in_range(self, calibrated_detector, test_signals):
        signals, labels = test_signals
        result = calibrated_detector.evaluate(signals, labels, verbose=False)
        assert 0.0 <= result["recall"] <= 1.0

    def test_custom_threshold_used(self, calibrated_detector, test_signals):
        signals, labels = test_signals
        r1 = calibrated_detector.evaluate(signals, labels,
                                           threshold=-999.0, verbose=False)
        r2 = calibrated_detector.evaluate(signals, labels,
                                           threshold=999.0,  verbose=False)
        # Extreme thresholds produce different results
        assert r1["f1"] != r2["f1"] or r1["threshold"] != r2["threshold"]


# ---------------------------------------------------------------------------
# 7. save / load
# ---------------------------------------------------------------------------

class TestSaveLoad:

    def test_save_creates_file(self, calibrated_detector):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "detector.pkl"
            calibrated_detector.save(path)
            assert path.exists()

    def test_load_returns_detector(self, calibrated_detector):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "detector.pkl"
            calibrated_detector.save(path)
            loaded = TDAAnomalyDetector.load(path)
            assert isinstance(loaded, TDAAnomalyDetector)

    def test_round_trip_scores_match(self, calibrated_detector, test_signals):
        signals, _ = test_signals
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "detector.pkl"
            calibrated_detector.save(path)
            loaded = TDAAnomalyDetector.load(path)
            scores_orig   = calibrated_detector.score(signals)
            scores_loaded = loaded.score(signals)
            np.testing.assert_array_almost_equal(scores_orig, scores_loaded)

    def test_round_trip_threshold_preserved(self, calibrated_detector):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "detector.pkl"
            calibrated_detector.save(path)
            loaded = TDAAnomalyDetector.load(path)
            assert loaded.threshold_ == calibrated_detector.threshold_

    def test_load_wrong_type_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.pkl"
            with open(path, "wb") as f:
                pickle.dump({"not": "a detector"}, f)
            with pytest.raises(TypeError):
                TDAAnomalyDetector.load(path)


# ---------------------------------------------------------------------------
# 8. Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:

    def test_score_before_fit(self):
        det = TDAAnomalyDetector()
        with pytest.raises(RuntimeError, match="not fitted"):
            det.score([np.ones(500)])

    def test_predict_before_fit(self):
        det = TDAAnomalyDetector()
        with pytest.raises(RuntimeError, match="not fitted"):
            det.predict([np.ones(500)])


# ---------------------------------------------------------------------------
# 9. End-to-end
# ---------------------------------------------------------------------------

class TestEndToEnd:

    def test_fit_calibrate_evaluate(self, test_signals):
        """Full pipeline: fit on normal, calibrate on mixed, evaluate."""
        normal_signals  = _make_normal(N_NORMAL, seed=7)
        signals, labels = test_signals

        det = TDAAnomalyDetector(n_estimators=100, random_state=42)
        det.fit(normal_signals)
        det.calibrate_threshold(signals, labels)
        result = det.evaluate(signals, labels, verbose=False)

        assert result["f1"] >= 0.80, (
            f"End-to-end F1 {result['f1']:.4f} < 0.80 — pipeline may be broken."
        )

    def test_ndarray_signals_accepted(self, test_signals):
        """fit() and score() must accept numpy arrays, not just lists."""
        normal_arr = np.array(_make_normal(40, seed=10))
        det = TDAAnomalyDetector(n_estimators=50, random_state=0)
        det.fit(normal_arr)
        signals, _ = test_signals
        scores = det.score(np.array(signals))
        assert scores.shape == (len(signals),)