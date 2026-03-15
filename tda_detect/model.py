"""
tda_detect/model.py
===================
TDAAnomalyDetector — end-to-end anomaly detection pipeline.

Wraps TDAFeatureExtractor + IsolationForest into a single
sklearn-compatible estimator.

Usage
-----
    from tda_detect.model import TDAAnomalyDetector

    # Train on normal windows only (semi-supervised)
    detector = TDAAnomalyDetector()
    detector.fit(normal_signals)

    # Score new windows (higher = more normal, lower = more anomalous)
    scores = detector.score(signals)

    # Predict labels (1 = anomaly, 0 = normal)
    labels = detector.predict(signals)

    # Evaluate against ground truth
    report = detector.evaluate(signals, y_true)

Pipeline
--------
    raw signal [N]
        → TDAFeatureExtractor.transform()    feature vector [800]
        → IsolationForest.decision_function() anomaly score  [1]
        → threshold comparison               label {0, 1}

Design decisions
----------------
- Semi-supervised: fit() accepts normal windows only.
- Threshold is learned from a calibration set via calibrate_threshold().
  Default threshold = 0.0 (IsolationForest convention) until calibrated.
- All feature extraction hyperparameters are fixed at Phase 2 values.
- n_estimators=200 for stable scores; random_state=42 for reproducibility.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from tda_detect.features import TDAFeatureExtractor


# ---------------------------------------------------------------------------
# TDAAnomalyDetector
# ---------------------------------------------------------------------------

class TDAAnomalyDetector:
    """End-to-end TDA anomaly detector.

    Parameters
    ----------
    n_estimators : int
        Number of trees in the IsolationForest.  Default 200.
    contamination : float or 'auto'
        Expected proportion of anomalies in the training set.
        Use 'auto' (default) — threshold is set by calibrate_threshold().
    random_state : int
        Random seed for reproducibility.  Default 42.
    extractor_kwargs : dict or None
        Keyword arguments forwarded to TDAFeatureExtractor.
        None → use Phase 2 defaults (dim=3, tau=31, n_pixels=20).

    Attributes
    ----------
    extractor_ : TDAFeatureExtractor
        Feature extractor instance (set after __init__).
    clf_ : IsolationForest
        Trained IsolationForest (set after fit()).
    threshold_ : float
        Decision threshold (set after calibrate_threshold() or fit()).
        Predict anomaly when score < threshold_.
    is_fitted_ : bool
        True after fit() has been called.

    Examples
    --------
    >>> import numpy as np
    >>> from tda_detect.model import TDAAnomalyDetector
    >>> t = np.linspace(0, 4*np.pi, 500)
    >>> normal_signals = [np.sin(t) + np.random.normal(0, 0.02, 500)
    ...                   for _ in range(160)]
    >>> detector = TDAAnomalyDetector()
    >>> detector.fit(normal_signals)
    TDAAnomalyDetector(n_estimators=200, threshold=0.0)
    >>> scores = detector.score(normal_signals[:5])
    >>> scores.shape
    (5,)
    """

    def __init__(
        self,
        n_estimators: int = 200,
        contamination: Union[float, str] = "auto",
        random_state: int = 42,
        extractor_kwargs: Optional[dict] = None,
    ) -> None:
        self.n_estimators    = n_estimators
        self.contamination   = contamination
        self.random_state    = random_state
        self.extractor_kwargs = extractor_kwargs or {}

        self.extractor_: TDAFeatureExtractor = TDAFeatureExtractor(
            **self.extractor_kwargs
        )
        self.clf_: Optional[IsolationForest] = None
        self.threshold_: float = 0.0
        self.is_fitted_: bool  = False

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def fit(self, signals) -> "TDAAnomalyDetector":
        """Extract features from normal signals and train IsolationForest.

        Parameters
        ----------
        signals : list or ndarray of shape (n_windows, N)
            Normal (non-anomalous) time-series windows.
            Each window must have at least 64 samples (dim=3, tau=31 default).

        Returns
        -------
        self
        """
        X = self._extract(signals)

        self.clf_ = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.clf_.fit(X)
        self.is_fitted_ = True
        return self

    def score(self, signals) -> np.ndarray:
        """Return anomaly scores for each window.

        Higher score = more normal.
        Lower score  = more anomalous.

        Parameters
        ----------
        signals : list or ndarray of shape (n_windows, N)
            Time-series windows to score.

        Returns
        -------
        scores : ndarray, shape (n_windows,)
            IsolationForest decision_function output.
        """
        self._check_fitted()
        X = self._extract(signals)
        return self.clf_.decision_function(X)

    def predict(
        self,
        signals,
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        """Predict anomaly labels for each window.

        Parameters
        ----------
        signals : list or ndarray of shape (n_windows, N)
            Time-series windows to classify.
        threshold : float or None
            Decision threshold.  score < threshold → anomaly (1).
            None → use self.threshold_ (set by calibrate_threshold()).

        Returns
        -------
        labels : ndarray of int, shape (n_windows,)
            1 = anomaly, 0 = normal.
        """
        thr    = threshold if threshold is not None else self.threshold_
        scores = self.score(signals)
        return (scores < thr).astype(int)

    def calibrate_threshold(
        self,
        signals,
        y_true: np.ndarray,
        n_thresholds: int = 300,
    ) -> float:
        """Find the threshold that maximises F1 on a labelled calibration set.

        Parameters
        ----------
        signals : list or ndarray of shape (n_windows, N)
            Calibration windows (mix of normal and anomalous).
        y_true : ndarray of int, shape (n_windows,)
            Ground-truth labels (0 = normal, 1 = anomaly).
        n_thresholds : int
            Number of threshold candidates to sweep.  Default 300.

        Returns
        -------
        best_threshold : float
            Threshold value that maximised F1.  Also stored as self.threshold_.
        """
        scores     = self.score(signals)
        thresholds = np.linspace(scores.min(), scores.max(), n_thresholds)

        best_f1  = -1.0
        best_thr = thresholds[0]

        for thr in thresholds:
            y_pred = (scores < thr).astype(int)
            f1     = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1  = f1
                best_thr = thr

        self.threshold_ = best_thr
        return best_thr

    def evaluate(
        self,
        signals,
        y_true: np.ndarray,
        threshold: Optional[float] = None,
        verbose: bool = True,
    ) -> dict:
        """Evaluate detector against ground-truth labels.

        Parameters
        ----------
        signals : list or ndarray of shape (n_windows, N)
            Test windows.
        y_true : ndarray of int, shape (n_windows,)
            Ground-truth labels (0 = normal, 1 = anomaly).
        threshold : float or None
            Decision threshold.  None → use self.threshold_.
        verbose : bool
            If True, print classification report.

        Returns
        -------
        metrics : dict
            Keys: 'f1', 'precision', 'recall', 'threshold'.
        """
        y_pred = self.predict(signals, threshold=threshold)
        thr    = threshold if threshold is not None else self.threshold_

        metrics = {
            "f1":        f1_score(y_true, y_pred, zero_division=0),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall":    recall_score(y_true, y_pred, zero_division=0),
            "threshold": thr,
        }

        if verbose:
            print("TDAAnomalyDetector — evaluation report")
            print("=" * 45)
            print(classification_report(
                y_true, y_pred,
                target_names=["Normal", "Anomaly"],
                digits=4,
            ))
            print(f"  F1        : {metrics['f1']:.4f}")
            print(f"  Precision : {metrics['precision']:.4f}")
            print(f"  Recall    : {metrics['recall']:.4f}")
            print(f"  Threshold : {metrics['threshold']:.6f}")

        return metrics

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """Serialise detector to a pickle file.

        Parameters
        ----------
        path : str or Path
            Destination file path (e.g. 'models/detector.pkl').
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Saved detector → {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TDAAnomalyDetector":
        """Load a serialised detector from a pickle file.

        Parameters
        ----------
        path : str or Path
            Source file path.

        Returns
        -------
        TDAAnomalyDetector
        """
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected TDAAnomalyDetector, got {type(obj)}")
        return obj

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract(self, signals) -> np.ndarray:
        """Extract TDA features from a list/array of signals."""
        if isinstance(signals, np.ndarray) and signals.ndim == 2:
            return np.array([self.extractor_.transform(s) for s in signals])
        return np.array([self.extractor_.transform(s) for s in signals])

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError(
                "Detector is not fitted. Call fit() before score() or predict()."
            )

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"TDAAnomalyDetector("
            f"n_estimators={self.n_estimators}, "
            f"threshold={self.threshold_:.4f})"
        )