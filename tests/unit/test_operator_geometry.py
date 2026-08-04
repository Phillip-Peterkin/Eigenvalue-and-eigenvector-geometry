"""Synthetic unit tests for operator-geometry metric math.

These tests verify formulas on constructed inputs with known answers.
They do not read manuscript JSON artifacts; that role belongs to the
manuscript audit suite.
"""
from __future__ import annotations

import numpy as np
import pytest

from cmcc.analysis.dynamical_systems import (
    JacobianResult,
    detect_exceptional_points,
    estimate_jacobian,
)
from cmcc.features.branching import compute_branching_ratio
from cmcc.features.operator_geometry import (
    PROXIMITY_SCORE_EPSILON,
    compute_nd_score,
    effective_rank,
    eigenvector_overlap,
    geometry_proximity_score,
    minimum_eigenvalue_gap,
    participation_ratio,
    spectral_radius_from_eigenvalues,
)


class TestSpectralRadius:
    def test_known_diagonal_spectrum(self) -> None:
        evals = np.array([0.2, -0.9, 0.5], dtype=complex)
        assert spectral_radius_from_eigenvalues(evals) == pytest.approx(0.9)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            spectral_radius_from_eigenvalues(np.array([]))


class TestMinimumEigenvalueGap:
    def test_known_pair_gap(self) -> None:
        evals = np.array([1.0 + 0j, 0.5 + 0j, 0.51 + 0j])
        gap, i, j = minimum_eigenvalue_gap(evals)
        assert gap == pytest.approx(0.01)
        assert {i, j} == {1, 2}

    def test_identical_eigenvalues_zero_gap(self) -> None:
        evals = np.array([0.7 + 0.1j, 0.7 + 0.1j, 0.2 + 0j])
        gap, _, _ = minimum_eigenvalue_gap(evals)
        assert gap == pytest.approx(0.0)

    def test_single_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="at least two"):
            minimum_eigenvalue_gap(np.array([1.0 + 0j]))


class TestEigenvectorOverlap:
    def test_identical_vectors_unit_overlap(self) -> None:
        v = np.array([1.0, 2.0, 3.0], dtype=complex)
        assert eigenvector_overlap(v, v) == pytest.approx(1.0)

    def test_orthogonal_real_vectors_zero_overlap(self) -> None:
        v_i = np.array([1.0, 0.0, 0.0])
        v_j = np.array([0.0, 1.0, 0.0])
        assert eigenvector_overlap(v_i, v_j) == pytest.approx(0.0)

    def test_zero_norm_returns_zero(self) -> None:
        assert eigenvector_overlap(np.zeros(3), np.ones(3)) == 0.0


class TestGeometryProximityScore:
    """Legacy JSON ``ep_score`` = overlap / (gap + epsilon)."""

    def test_matches_definition(self) -> None:
        overlap = 0.5
        gap = 0.01
        expected = overlap / (gap + PROXIMITY_SCORE_EPSILON)
        assert geometry_proximity_score(overlap, gap) == pytest.approx(expected)

    def test_small_gap_raises_score(self) -> None:
        low = geometry_proximity_score(0.8, gap=0.1)
        high = geometry_proximity_score(0.8, gap=1e-6)
        assert high > low

    def test_zero_overlap_zero_score(self) -> None:
        assert geometry_proximity_score(0.0, gap=0.01) == pytest.approx(0.0)


class TestManuscriptNdScore:
    """Manuscript ND = 0.5 * [z(-log10(gap)) + z(log10(kappa))]."""

    def test_constant_inputs_zero_score(self) -> None:
        gaps = np.full(8, 0.01)
        kappas = np.full(8, 10.0)
        nd = compute_nd_score(gaps, kappas)
        assert nd.shape == (8,)
        assert np.allclose(nd, 0.0)

    def test_hand_computed_two_window_case(self) -> None:
        # Two windows: crowding and non-orthogonality move together.
        gaps = np.array([0.1, 0.001])
        kappas = np.array([2.0, 200.0])
        eps = 1e-12
        crowding = -np.log10(gaps + eps)
        nonorth = np.log10(kappas + eps)
        z_c = (crowding - crowding.mean()) / crowding.std()
        z_n = (nonorth - nonorth.mean()) / nonorth.std()
        expected = 0.5 * (z_c + z_n)
        got = compute_nd_score(gaps, kappas, epsilon=eps)
        assert got == pytest.approx(expected)

    def test_smaller_gap_increases_relative_nd(self) -> None:
        gaps = np.array([0.1, 0.01, 0.001, 0.0001])
        kappas = np.ones(4) * 10.0
        nd = compute_nd_score(gaps, kappas)
        # With fixed kappa, ND tracks -log10(gap); smallest gap -> largest ND.
        assert nd[-1] == pytest.approx(np.max(nd))
        assert nd[0] == pytest.approx(np.min(nd))

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="share shape"):
            compute_nd_score(np.array([0.1, 0.2]), np.array([1.0]))


class TestParticipationRatioAndEffectiveRank:
    """Singular-spectrum concentration metrics (Gini-like summaries)."""

    def test_uniform_spectrum_fully_delocalized(self) -> None:
        sigma = np.ones(5)
        assert participation_ratio(sigma) == pytest.approx(5.0)
        assert effective_rank(sigma) == pytest.approx(5.0)

    def test_rank_one_fully_concentrated(self) -> None:
        sigma = np.array([3.0, 0.0, 0.0, 0.0])
        assert participation_ratio(sigma) == pytest.approx(1.0)
        assert effective_rank(sigma) == pytest.approx(1.0)

    def test_concentration_ordering(self) -> None:
        uniform = np.ones(4)
        mild = np.array([4.0, 3.0, 2.0, 1.0])
        extreme = np.array([10.0, 0.1, 0.1, 0.1])
        assert participation_ratio(extreme) < participation_ratio(mild) < participation_ratio(uniform)

    def test_all_zero_returns_zero(self) -> None:
        assert participation_ratio(np.zeros(3)) == 0.0
        assert effective_rank(np.zeros(3)) == 0.0


class TestDetectExceptionalPointsUsesHelpers:
    """Production detector must emit the proximity-score definition."""

    def _jac_from_spectra(
        self,
        eigenvalues: np.ndarray,
        eigenvectors: np.ndarray,
        jacobian: np.ndarray | None = None,
    ) -> JacobianResult:
        n_ch = eigenvalues.shape[0]
        if jacobian is None:
            jacobian = np.eye(n_ch)
        return JacobianResult(
            jacobians=jacobian[None, ...],
            eigenvalues=eigenvalues[None, ...],
            eigenvectors=eigenvectors[None, ...],
            window_centers=np.array([0.0]),
            spectral_radius=np.array([spectral_radius_from_eigenvalues(eigenvalues)]),
            condition_numbers=np.array([np.linalg.cond(eigenvectors)]),
            residual_variance=np.array([0.0]),
            regularization=1e-4,
        )

    def test_orthogonal_pair_with_known_gap(self) -> None:
        evals = np.array([0.9 + 0j, 0.8 + 0j, 0.1 + 0j])
        evecs = np.eye(3, dtype=complex)
        result = detect_exceptional_points(self._jac_from_spectra(evals, evecs))
        assert result.min_eigenvalue_gaps[0] == pytest.approx(0.1)
        assert result.eigenvector_overlaps[0] == pytest.approx(0.0)
        assert result.ep_scores[0] == pytest.approx(0.0)

    def test_near_coalescence_high_score(self) -> None:
        evals = np.array([0.5 + 0j, 0.5 + 1e-6 + 0j])
        # Nearly parallel eigenvectors.
        evecs = np.array(
            [[1.0, 1.0], [0.0, 1e-3]],
            dtype=complex,
        )
        result = detect_exceptional_points(self._jac_from_spectra(evals, evecs))
        expected = geometry_proximity_score(
            result.eigenvector_overlaps[0],
            result.min_eigenvalue_gaps[0],
        )
        assert result.ep_scores[0] == pytest.approx(expected)
        assert result.ep_scores[0] > 1e4


class TestEstimateJacobianKnownLinearSystem:
    def test_recovers_spectral_radius_of_scaled_rotation(self) -> None:
        # Driven VAR(1): x(t+1) = A x(t) + noise, with known spectral radius 0.8.
        theta = 0.3
        rho = 0.8
        a = rho * np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        rng = np.random.default_rng(0)
        n_samples = 4000
        x = np.zeros((2, n_samples))
        x[:, 0] = rng.normal(size=2)
        for t in range(1, n_samples):
            x[:, t] = a @ x[:, t - 1] + 0.2 * rng.normal(size=2)

        jac = estimate_jacobian(x, window_size=200, step_size=100, regularization=1e-8)
        assert np.mean(jac.spectral_radius) == pytest.approx(rho, abs=0.08)


class TestBranchingRatioSynthetic:
    def test_dying_activity_subcritical(self) -> None:
        # Each active site produces fewer descendants on average.
        binary = np.array(
            [
                [1, 1, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=int,
        )
        result = compute_branching_ratio(binary)
        assert result.sigma < 1.0

    def test_constant_activity_critical(self) -> None:
        binary = np.ones((4, 20), dtype=int)
        result = compute_branching_ratio(binary)
        assert result.sigma == pytest.approx(1.0)
