"""Synthetic unit tests for operator-geometry metric mathematics.

These tests use constructed inputs with known behavior and do not depend on
research datasets or checked-in empirical result files.
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
    """Historical JSON `ep_score` equals overlap divided by gap plus epsilon."""

    def test_matches_definition(self) -> None:
        overlap = 0.5
        gap = 0.01
        expected = overlap / (gap + PROXIMITY_SCORE_EPSILON)
        assert geometry_proximity_score(overlap, gap) == pytest.approx(expected)

    def test_small_gap_raises_score(self) -> None:
        assert geometry_proximity_score(0.8, 1e-6) > geometry_proximity_score(0.8, 0.1)

    def test_zero_overlap_zero_score(self) -> None:
        assert geometry_proximity_score(0.0, gap=0.01) == pytest.approx(0.0)


class TestManuscriptNdScore:
    """Current ND is a sign-normalized first-principal-component projection."""

    def test_constant_inputs_zero_score(self) -> None:
        nd = compute_nd_score(np.full(8, 0.01), np.full(8, 10.0))
        assert nd.shape == (8,)
        assert np.allclose(nd, 0.0)

    def test_perfectly_correlated_features_follow_unit_norm_pc1(self) -> None:
        gaps = np.array([0.1, 0.01, 0.001, 0.0001])
        kappas = np.array([2.0, 20.0, 200.0, 2000.0])
        eps = 1e-12
        crowding = -np.log10(gaps + eps)
        nonorth = np.log10(kappas + eps)
        z_c = (crowding - crowding.mean()) / crowding.std()
        z_k = (nonorth - nonorth.mean()) / nonorth.std()
        expected = (z_c + z_k) / np.sqrt(2.0)
        got = compute_nd_score(gaps, kappas, epsilon=eps)
        assert got == pytest.approx(expected)

    def test_component_orientation_tracks_joint_crowding_and_nonorthogonality(self) -> None:
        gaps = np.array([0.2, 0.08, 0.03, 0.01, 0.003])
        kappas = np.array([2.0, 3.0, 5.0, 12.0, 30.0])
        nd = compute_nd_score(gaps, kappas)
        assert nd[-1] > nd[0]

    def test_within_array_mean_is_zero_by_construction(self) -> None:
        gaps = np.array([0.2, 0.1, 0.03, 0.02, 0.004, 0.001])
        kappas = np.array([2.0, 4.0, 3.0, 9.0, 15.0, 40.0])
        nd = compute_nd_score(gaps, kappas)
        assert np.nanmean(nd) == pytest.approx(0.0, abs=1e-12)

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="share shape"):
            compute_nd_score(np.array([0.1, 0.2]), np.array([1.0]))

    def test_non_vector_input_raises(self) -> None:
        with pytest.raises(ValueError, match="one-dimensional"):
            compute_nd_score(np.ones((2, 2)), np.ones((2, 2)))


class TestParticipationRatioAndEffectiveRank:
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


class TestDetectExceptionalPointsUsesHistoricalScore:
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

    def test_near_coalescence_high_historical_score(self) -> None:
        evals = np.array([0.5 + 0j, 0.5 + 1e-6 + 0j])
        evecs = np.array([[1.0, 1.0], [0.0, 1e-3]], dtype=complex)
        result = detect_exceptional_points(self._jac_from_spectra(evals, evecs))
        expected = geometry_proximity_score(
            result.eigenvector_overlaps[0], result.min_eigenvalue_gaps[0]
        )
        assert result.ep_scores[0] == pytest.approx(expected)
        assert result.ep_scores[0] > 1e4


class TestEstimateJacobianKnownLinearSystem:
    def test_recovers_spectral_radius_of_scaled_rotation(self) -> None:
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
        binary = np.array(
            [[1, 1, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=int
        )
        assert compute_branching_ratio(binary).sigma < 1.0

    def test_constant_activity_critical(self) -> None:
        binary = np.ones((4, 20), dtype=int)
        assert compute_branching_ratio(binary).sigma == pytest.approx(1.0)
