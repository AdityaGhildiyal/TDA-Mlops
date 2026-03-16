"""
tda_detect/drift.py
Topological drift detector using Wasserstein distance on H1 persistence diagrams.
"""

import numpy as np
import pickle
from pathlib import Path
from tda_detect.features import takens_embed, finite_dgm
from ripser import ripser


class TopologicalDriftDetector:
    """
    Detects distribution drift by comparing H1 persistence diagrams
    of incoming windows against a reference distribution using
    Wasserstein distance.

    Parameters
    ----------
    threshold : float
        Wasserstein distance above which drift is declared.
    dim : int
        Takens embedding dimension.
    tau : int
        Takens embedding delay.
    homology_dim : int
        Homology dimension to track (1 = H1, loops).
    """

    def __init__(self, threshold=0.5, dim=3, tau=31, homology_dim=1):
        self.threshold    = threshold
        self.dim          = dim
        self.tau          = tau
        self.homology_dim = homology_dim

        self.reference_diagrams_ = None
        self.reference_mean_     = None
        self.n_windows_seen_     = 0

    # ── Fitting ───────────────────────────────────────────────────────────────
    def fit(self, signals):
        """Compute reference diagrams from normal signals."""
        print(f"Computing reference diagrams from {len(signals)} windows...")
        self.reference_diagrams_ = [self._get_diagram(s) for s in signals]
        all_points = np.vstack([d for d in self.reference_diagrams_ if len(d) > 0])
        persistence = all_points[:, 1] - all_points[:, 0]
        threshold   = np.percentile(persistence, 50)
        self.reference_mean_ = all_points[persistence >= threshold]
        self.n_windows_seen_ = 0
        print(f"  Reference set         : {len(self.reference_diagrams_)} diagrams")
        print(f"  Reference centroid pts: {len(self.reference_mean_)}")
        return self

    def reset(self, new_signals=None):
        """Reset window counter; optionally refit reference."""
        self.n_windows_seen_ = 0
        if new_signals is not None:
            self.fit(new_signals)
        return self

    # ── Threshold calibration ─────────────────────────────────────────────────
    def calibrate_threshold(self, normal_signals, drifted_signals,
                            n_thresholds=100):
        """
        Sweep thresholds and pick the one maximising F1 on held-out data.
        normal_signals  → label 0
        drifted_signals → label 1
        """
        from sklearn.metrics import f1_score

        w_normal  = [self._wasserstein(self._get_diagram(s),
                                       self.reference_mean_)
                     for s in normal_signals]
        w_drifted = [self._wasserstein(self._get_diagram(s),
                                       self.reference_mean_)
                     for s in drifted_signals]

        all_w  = np.array(w_normal + w_drifted)
        labels = np.array([0]*len(w_normal) + [1]*len(w_drifted))
        candidates = np.linspace(all_w.min(), all_w.max(), n_thresholds)

        best_f1, best_thr = -1, candidates[0]
        for thr in candidates:
            preds = (all_w > thr).astype(int)
            f = f1_score(labels, preds, zero_division=0)
            if f > best_f1:
                best_f1, best_thr = f, thr

        self.threshold = float(best_thr)
        print(f"Calibrated threshold: {self.threshold:.4f}  (F1={best_f1:.4f})")
        return self

    # ── Inference ─────────────────────────────────────────────────────────────
    def update(self, signal):
        """Process one window; return drift result dict."""
        self.n_windows_seen_ += 1
        dgm  = self._get_diagram(signal)
        dist = self._wasserstein(dgm, self.reference_mean_)
        return {
            "drift_detected"      : bool(dist > self.threshold),
            "wasserstein_distance": float(dist),
            "threshold"           : float(self.threshold),
            "n_windows_seen"      : self.n_windows_seen_,
        }

    def update_batch(self, signals):
        """Process a list of windows; return list of result dicts."""
        return [self.update(s) for s in signals]

    # ── Persistence diagram helpers ───────────────────────────────────────────
    def _get_diagram(self, signal):
        emb  = takens_embed(signal, dim=self.dim, tau=self.tau)
        dgms = ripser(emb, maxdim=self.homology_dim)["dgms"]
        return finite_dgm(dgms[self.homology_dim])

    def _wasserstein(self, dgm1, dgm2):
        """Sliced Wasserstein approximation; handles empty diagrams."""
        if len(dgm1) == 0 or len(dgm2) == 0:
            return float(max(
                np.sum(dgm1[:, 1] - dgm1[:, 0]) if len(dgm1) > 0 else 0,
                np.sum(dgm2[:, 1] - dgm2[:, 0]) if len(dgm2) > 0 else 0,
            ))
        p1 = dgm1[:, 1] - dgm1[:, 0]
        p2 = dgm2[:, 1] - dgm2[:, 0]
        p1 = np.sort(p1)[::-1][:min(50, len(p1))]
        p2 = np.sort(p2)[::-1][:min(50, len(p2))]
        n  = max(len(p1), len(p2))
        p1 = np.pad(p1, (0, n - len(p1)))
        p2 = np.pad(p2, (0, n - len(p2)))
        return float(np.sum(np.abs(p1 - p2)))

    # ── Persistence ───────────────────────────────────────────────────────────
    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Saved drift detector → {path}")

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)