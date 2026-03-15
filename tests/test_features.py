"""
tests/test_features.py
======================
pytest suite for tda_detect.features

Run from the repo root:
    pytest tests/test_features.py -v

Coverage
--------
1.  takens_embed — output shape, first/last row, ValueError on short signal
2.  finite_dgm   — strips inf, passes finite pairs unchanged, empty diagram
3.  ripser_persist — H0 finite count, H1 count, output dtypes
4.  TDAFeatureExtractor.__init__ — default + custom hyperparams, feature_len
5.  TDAFeatureExtractor.transform — output shape, normal signal, constant signal,
                                    pure noise, short-but-valid window,
                                    ValueError on too-short signal
6.  Stability property — small perturbation → small feature change (L2 norm)
7.  Separation property — anomalous window separates from normal in L2
"""

import numpy as np
import pytest

# Adjust import path if running outside the installed package
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tda_detect.features import (
    TDAFeatureExtractor,
    finite_dgm,
    ripser_persist,
    takens_embed,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def normal_signal():
    """Clean sine wave — the canonical normal window."""
    t = np.linspace(0, 4 * np.pi, 500)
    return np.sin(t)


@pytest.fixture(scope="module")
def anomaly_signal(normal_signal):
    """Sine wave with three spike pairs — the canonical anomaly."""
    sig = normal_signal.copy()
    for pos in [100, 200, 310]:
        sig[pos]     =  3.5
        sig[pos + 1] = -3.5
    return sig


@pytest.fixture(scope="module")
def extractor():
    return TDAFeatureExtractor()


# ---------------------------------------------------------------------------
# 1. takens_embed
# ---------------------------------------------------------------------------

class TestTakensEmbed:

    def test_shape_default(self, normal_signal):
        X = takens_embed(normal_signal, dim=3, tau=31)
        expected_n = len(normal_signal) - (3 - 1) * 31   # = 438
        assert X.shape == (expected_n, 3)

    def test_first_row(self):
        sig = np.arange(1, 13, dtype=float)   # length 12
        X = takens_embed(sig, dim=3, tau=1)
        assert np.all(X[0] == [1.0, 2.0, 3.0]), f"First row wrong: {X[0]}"

    def test_last_row(self):
        sig = np.arange(1, 13, dtype=float)
        X = takens_embed(sig, dim=3, tau=1)
        assert np.all(X[-1] == [10.0, 11.0, 12.0]), f"Last row wrong: {X[-1]}"

    def test_dim2(self, normal_signal):
        X = takens_embed(normal_signal, dim=2, tau=31)
        expected_n = len(normal_signal) - (2 - 1) * 31   # = 469
        assert X.shape == (expected_n, 2)

    def test_too_short_raises(self):
        with pytest.raises(ValueError, match="too short"):
            takens_embed(np.ones(3), dim=3, tau=31)

    def test_exactly_2_points_ok(self):
        # N - (dim-1)*tau == 2 should NOT raise
        dim, tau = 3, 31
        N = 2 + (dim - 1) * tau   # = 64
        X = takens_embed(np.ones(N), dim=dim, tau=tau)
        assert X.shape == (2, dim)

    def test_output_dtype(self, normal_signal):
        X = takens_embed(normal_signal)
        assert X.dtype == float


# ---------------------------------------------------------------------------
# 2. finite_dgm
# ---------------------------------------------------------------------------

class TestFiniteDgm:

    def test_strips_inf(self):
        dgm = np.array([[0.0, 1.0], [0.1, 2.0], [0.0, np.inf]])
        result = finite_dgm(dgm)
        assert result.shape == (2, 2)
        assert not np.any(np.isinf(result))

    def test_all_finite_unchanged(self):
        dgm = np.array([[0.0, 1.0], [0.2, 0.5]])
        result = finite_dgm(dgm)
        np.testing.assert_array_equal(result, dgm)

    def test_all_inf_returns_empty(self):
        dgm = np.array([[0.0, np.inf], [0.1, np.inf]])
        result = finite_dgm(dgm)
        assert result.shape == (0, 2)

    def test_empty_input(self):
        dgm = np.empty((0, 2))
        result = finite_dgm(dgm)
        assert result.shape == (0, 2)


# ---------------------------------------------------------------------------
# 3. ripser_persist
# ---------------------------------------------------------------------------

class TestRipserPersist:

    def test_h0_finite_count(self, normal_signal):
        X = takens_embed(normal_signal, dim=3, tau=31)
        dgm_h0, _ = ripser_persist(X, maxdim=1)
        # 438 vertices → 437 finite H0 pairs (1 essential stripped)
        assert dgm_h0.shape[0] == 437

    def test_h1_count_normal(self, normal_signal):
        X = takens_embed(normal_signal, dim=3, tau=31)
        _, dgm_h1 = ripser_persist(X, maxdim=1)
        assert dgm_h1.shape[0] == 1   # one dominant loop

    def test_no_inf_in_output(self, normal_signal):
        X = takens_embed(normal_signal, dim=3, tau=31)
        dgm_h0, dgm_h1 = ripser_persist(X, maxdim=1)
        assert not np.any(np.isinf(dgm_h0))
        assert not np.any(np.isinf(dgm_h1))

    def test_constant_signal_no_crash(self):
        X = takens_embed(np.ones(500), dim=3, tau=31)
        dgm_h0, dgm_h1 = ripser_persist(X, maxdim=1)
        assert dgm_h0.shape[1] == 2
        assert dgm_h1.shape[1] == 2

    def test_output_shape_columns(self, normal_signal):
        X = takens_embed(normal_signal, dim=3, tau=31)
        dgm_h0, dgm_h1 = ripser_persist(X, maxdim=1)
        assert dgm_h0.shape[1] == 2
        assert dgm_h1.shape[1] == 2


# ---------------------------------------------------------------------------
# 4. TDAFeatureExtractor.__init__
# ---------------------------------------------------------------------------

class TestExtractorInit:

    def test_default_hyperparams(self, extractor):
        assert extractor.dim         == 3
        assert extractor.tau         == 31
        assert extractor.birth_range == (0.0, 2.1)
        assert extractor.pers_range  == (0.0, 2.1)
        assert extractor.n_pixels    == 20

    def test_default_feature_len(self, extractor):
        assert extractor.feature_len == 800   # 2 * 20^2

    def test_custom_hyperparams(self):
        ext = TDAFeatureExtractor(dim=2, tau=15, n_pixels=10)
        assert ext.dim         == 2
        assert ext.tau         == 15
        assert ext.n_pixels    == 10
        assert ext.feature_len == 200   # 2 * 10^2

    def test_repr_contains_key_info(self, extractor):
        r = repr(extractor)
        assert "dim=3"       in r
        assert "tau=31"      in r
        assert "feature_len" in r


# ---------------------------------------------------------------------------
# 5. TDAFeatureExtractor.transform
# ---------------------------------------------------------------------------

class TestExtractorTransform:

    def test_output_shape_normal(self, extractor, normal_signal):
        feat = extractor.transform(normal_signal)
        assert feat.shape == (800,)

    def test_output_dtype(self, extractor, normal_signal):
        feat = extractor.transform(normal_signal)
        assert feat.dtype == float

    def test_no_nan_or_inf(self, extractor, normal_signal):
        feat = extractor.transform(normal_signal)
        assert not np.any(np.isnan(feat))
        assert not np.any(np.isinf(feat))

    def test_non_negative(self, extractor, normal_signal):
        feat = extractor.transform(normal_signal)
        assert np.all(feat >= 0)

    def test_constant_signal_all_zeros(self, extractor):
        feat = extractor.transform(np.ones(500))
        assert np.all(feat == 0)

    def test_constant_signal_shape(self, extractor):
        feat = extractor.transform(np.ones(500))
        assert feat.shape == (800,)

    def test_pure_noise_nonzero(self, extractor):
        rng = np.random.default_rng(42)
        feat = extractor.transform(rng.standard_normal(500))
        assert np.count_nonzero(feat) > 0

    def test_short_but_valid_window(self, extractor):
        """100-sample window: n_points = 100 - 62 = 38 ≥ 2, should succeed."""
        t = np.linspace(0, 4 * np.pi, 100)
        feat = extractor.transform(np.sin(t))
        assert feat.shape == (800,)

    def test_too_short_raises(self, extractor):
        with pytest.raises(ValueError, match="too short"):
            extractor.transform(np.ones(10))

    def test_list_input_accepted(self, extractor, normal_signal):
        """transform() must accept Python lists, not just ndarrays."""
        feat = extractor.transform(list(normal_signal))
        assert feat.shape == (800,)


# ---------------------------------------------------------------------------
# 6. Stability property
# ---------------------------------------------------------------------------

class TestStability:
    """Small input perturbations → small feature-space changes."""

    def test_small_perturbation(self, extractor, normal_signal):
        rng = np.random.default_rng(0)
        perturbed = normal_signal + rng.normal(0, 0.01, size=len(normal_signal))

        feat_orig = extractor.transform(normal_signal)
        feat_pert = extractor.transform(perturbed)

        l2 = np.linalg.norm(feat_orig - feat_pert)
        # Empirically the L2 distance is < 0.5 for σ=0.01 noise
        assert l2 < 0.5, (
            f"Stability test failed: L2 distance {l2:.4f} is unexpectedly large "
            f"for σ=0.01 perturbation."
        )

    def test_deterministic(self, extractor, normal_signal):
        """Same input must yield identical output (no randomness)."""
        feat1 = extractor.transform(normal_signal)
        feat2 = extractor.transform(normal_signal)
        np.testing.assert_array_equal(feat1, feat2)


# ---------------------------------------------------------------------------
# 7. Separation property
# ---------------------------------------------------------------------------

class TestSeparation:
    """Anomalous window must be separable from normal window in feature space."""

    def test_anomaly_differs_from_normal(self, extractor, normal_signal, anomaly_signal):
        feat_normal  = extractor.transform(normal_signal)
        feat_anomaly = extractor.transform(anomaly_signal)

        l2 = np.linalg.norm(feat_normal - feat_anomaly)
        assert l2 > 0.001, (
            f"Separation test failed: L2 distance {l2:.6f} — features are "
            f"nearly identical for normal vs anomalous signals."
        )

    def test_h0_component_differs(self, extractor, normal_signal, anomaly_signal):
        """H0 sub-vector must differ (spikes fragment components)."""
        T = extractor.n_pixels
        feat_n = extractor.transform(normal_signal)
        feat_a = extractor.transform(anomaly_signal)

        l2_h0 = np.linalg.norm(feat_n[:T*T] - feat_a[:T*T])
        assert l2_h0 > 0.001

    def test_features_not_all_same(self, extractor, normal_signal, anomaly_signal):
        feat_n = extractor.transform(normal_signal)
        feat_a = extractor.transform(anomaly_signal)
        assert not np.array_equal(feat_n, feat_a)