"""
tda_detect/features.py
======================
TDA feature-engineering pipeline for anomaly detection.

Pipeline per 500-sample window
--------------------------------
    signal[N]
        → takens_embed(dim=3, tau=31)          shape [N-(dim-1)*tau, dim]
        → ripser_persist(maxdim=1)             dgm_h0, dgm_h1
        → PersistenceImager(fixed ranges)      img_h0 [T,T], img_h1 [T,T]
        → flatten + concatenate                feature vector [2*T²]

Key design decisions (locked in Day 5 sensitivity sweeps)
----------------------------------------------------------
- NEVER call PersistenceImager.fit().  fit() sets birth_range = (x,x) when the
  diagram has a single point, producing a zero-width pixel grid that crashes
  on .min()/.max().  Always pass birth_range / pers_range explicitly.
- ALWAYS call finite_dgm() on H0 before transform().  H0 always contains one
  essential class with death=inf; PersistenceImager cannot place infinite
  points on a finite pixel grid.
- Fixed pixel grid (birth_range, pers_range) must be the same across every
  window so that pixel (i, j) encodes the same topological region everywhere.
- T=20 resolution: tight bright spot for the loop, noise stays dim, fast enough.
- dim=3, tau=31: confirmed stable across a 3×3 sensitivity sweep.
"""

from __future__ import annotations

import numpy as np
import ripser
from persim import PersistenceImager


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def finite_dgm(dgm: np.ndarray) -> np.ndarray:
    """Strip points with death=inf (essential classes) before imaging.

    Parameters
    ----------
    dgm : ndarray, shape (k, 2)
        Persistence diagram — rows are (birth, death) pairs.

    Returns
    -------
    ndarray, shape (k', 2)
        Diagram with all rows where death==inf removed.

    Notes
    -----
    H0 always has exactly one essential class (the last connected component,
    death=inf).  H1 may also have essential classes for unusual signals.
    This function handles both cases uniformly.
    """
    return dgm[np.isfinite(dgm[:, 1])]


def takens_embed(signal: np.ndarray, dim: int = 3, tau: int = 31) -> np.ndarray:
    """Takens delay embedding: 1-D time series → point cloud in R^dim.

    Parameters
    ----------
    signal : array-like, shape (N,)
        Raw time-series window.
    dim : int
        Embedding dimension.  Default 3 (captures H0 and H1, still visualisable).
    tau : int
        Delay in samples.  Default 31 (≈ quarter-period for ~125-sample period).

    Returns
    -------
    X : ndarray, shape (N - (dim-1)*tau, dim)
        Each row is [x(t), x(t+τ), x(t+2τ), ...].

    Raises
    ------
    ValueError
        If the signal is too short to produce at least 2 embedding points.

    Examples
    --------
    >>> import numpy as np
    >>> from tda_detect.features import takens_embed
    >>> t = np.linspace(0, 4*np.pi, 500)
    >>> X = takens_embed(np.sin(t), dim=3, tau=31)
    >>> X.shape
    (438, 3)
    """
    signal = np.asarray(signal, dtype=float)
    N = len(signal)
    n_points = N - (dim - 1) * tau

    if n_points < 2:
        raise ValueError(
            f"Signal too short: {N} samples with dim={dim}, tau={tau} "
            f"yields only {n_points} embedding points (minimum 2 required)."
        )

    X = np.zeros((n_points, dim))
    for i in range(dim):
        X[:, i] = signal[i * tau : i * tau + n_points]
    return X


def ripser_persist(X: np.ndarray, maxdim: int = 1):
    """Thin wrapper around ripser.ripser returning finite persistence diagrams.

    Parameters
    ----------
    X : ndarray, shape (N, d)
        Point cloud (output of takens_embed).
    maxdim : int
        Maximum homology dimension to compute.  Default 1 (H0 and H1).

    Returns
    -------
    dgm_h0 : ndarray, shape (k0, 2)
        Finite H0 persistence pairs (birth, death).  Essential class stripped.
    dgm_h1 : ndarray, shape (k1, 2)
        Finite H1 persistence pairs (birth, death).

    Notes
    -----
    - H0 always has exactly one essential class (death=inf).  It is stripped
      because PersistenceImager cannot place infinite points on a finite grid.
    - H1 may also have essential classes for unusual signals; finite_dgm()
      handles both cases uniformly.
    """
    result = ripser.ripser(X, maxdim=maxdim)
    dgm_h0 = finite_dgm(result["dgms"][0])
    dgm_h1 = finite_dgm(result["dgms"][1])
    return dgm_h0, dgm_h1


# ---------------------------------------------------------------------------
# Main feature extractor
# ---------------------------------------------------------------------------

class TDAFeatureExtractor:
    """Full TDA feature pipeline: signal → fixed-length feature vector.

    Pipeline per window
    -------------------
        takens_embed(dim, tau)
        → ripser_persist(maxdim=1)
        → PersistenceImager(birth_range, pers_range, pixel_size)
        → flatten H0 image + flatten H1 image
        → concatenate  →  feature vector of length 2 * n_pixels²

    Parameters
    ----------
    dim : int
        Takens embedding dimension.  Default 3.
    tau : int
        Takens delay (samples).  Default 31.
    birth_range : tuple(float, float) or None
        (min, max) birth range for the persistence image pixel grid.
        None → use class default (0.0, 2.1).
    pers_range : tuple(float, float) or None
        (min, max) persistence range for the pixel grid.
        None → use class default (0.0, 2.1).
    n_pixels : int or None
        Grid resolution T (produces T×T image per diagram).  Default 20.

    Attributes
    ----------
    feature_len : int
        Length of the output feature vector = 2 * n_pixels².

    Notes
    -----
    Design decisions locked in Day 5:

    1. **Never call PersistenceImager.fit().**
       fit() can produce a zero-width pixel grid when a diagram has only one
       point, causing downstream crashes.  Ranges are set at construction time
       and never updated.

    2. **Always call finite_dgm() on H0 before transform().**
       H0 always has one essential class (death=inf) that cannot be placed on
       a finite pixel grid.

    3. **pixel_size = (birth_range[1] - birth_range[0]) / n_pixels.**
       Derived automatically from the other hyperparameters.

    Examples
    --------
    >>> import numpy as np
    >>> from tda_detect.features import TDAFeatureExtractor
    >>> t = np.linspace(0, 4*np.pi, 500)
    >>> extractor = TDAFeatureExtractor()
    >>> feat = extractor.transform(np.sin(t))
    >>> feat.shape
    (800,)
    """

    # Class-level defaults (locked in Day 5 sensitivity sweeps)
    BIRTH_RANGE: tuple = (0.0, 2.1)
    PERS_RANGE:  tuple = (0.0, 2.1)
    N_PIXELS:    int   = 20

    def __init__(
        self,
        dim: int = 3,
        tau: int = 31,
        birth_range: tuple | None = None,
        pers_range:  tuple | None = None,
        n_pixels:    int   | None = None,
    ) -> None:
        self.dim         = dim
        self.tau         = tau
        self.birth_range = birth_range if birth_range is not None else self.BIRTH_RANGE
        self.pers_range  = pers_range  if pers_range  is not None else self.PERS_RANGE
        self.n_pixels    = n_pixels    if n_pixels    is not None else self.N_PIXELS

        pixel_size = (self.birth_range[1] - self.birth_range[0]) / self.n_pixels

        # Build imager once — NEVER call .fit() on it
        self._pimgr = PersistenceImager(
            birth_range=self.birth_range,
            pers_range=self.pers_range,
            pixel_size=pixel_size,
        )

        self.feature_len: int = 2 * self.n_pixels ** 2

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transform(self, signal) -> np.ndarray:
        """Map a 1-D signal window to a fixed-length TDA feature vector.

        Parameters
        ----------
        signal : array-like, shape (N,)
            Raw time-series window.

        Returns
        -------
        features : ndarray, shape (feature_len,)
            Concatenation of flattened H0 and H1 persistence images.
            feature_len = 2 * n_pixels² = 800 at default settings.

        Raises
        ------
        ValueError
            If the signal is too short to produce at least 2 embedding points
            (i.e. N < 2 + (dim-1)*tau).
        """
        X = takens_embed(signal, dim=self.dim, tau=self.tau)
        dgm_h0, dgm_h1 = ripser_persist(X, maxdim=1)

        img_h0 = self._pimgr.transform(dgm_h0, skew=True)
        img_h1 = self._pimgr.transform(dgm_h1, skew=True)

        return np.concatenate([img_h0.flatten(), img_h1.flatten()])

    # ------------------------------------------------------------------
    # Convenience / repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"TDAFeatureExtractor("
            f"dim={self.dim}, tau={self.tau}, "
            f"birth_range={self.birth_range}, pers_range={self.pers_range}, "
            f"n_pixels={self.n_pixels}, feature_len={self.feature_len})"
        )