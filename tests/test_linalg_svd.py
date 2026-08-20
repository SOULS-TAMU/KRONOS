"""The algorithm-box step: SVD with the Moore-Penrose pseudoinverse.

These test the *published* formulation
"""
import numpy as np
import pytest

from kronos.linalg import lsqminnorm, min_norm_solve
from kronos.linalg_svd import minnorm_svd, pinv, svd_rank_tolerance


def test_pinv_satisfies_the_moore_penrose_conditions():
    rng = np.random.RandomState(0)
    A = rng.randn(9, 4) @ rng.randn(4, 7)          # 9x7, rank 4
    P = pinv(A)
    assert np.allclose(A @ P @ A, A, atol=1e-8)
    assert np.allclose(P @ A @ P, P, atol=1e-8)
    assert np.allclose((A @ P).T, A @ P, atol=1e-8)
    assert np.allclose((P @ A).T, P @ A, atol=1e-8)


def test_pinv_matches_numpy_on_well_conditioned_input():
    rng = np.random.RandomState(1)
    A = rng.randn(6, 6)
    assert np.allclose(pinv(A), np.linalg.pinv(A), atol=1e-10)


def test_minnorm_svd_equals_pinv_times_b():
    rng = np.random.RandomState(2)
    A = rng.randn(10, 5) @ rng.randn(5, 8)
    b = rng.randn(10)
    assert np.allclose(minnorm_svd(A, b), pinv(A) @ b, atol=1e-10)


def test_minnorm_svd_is_a_minimum_norm_least_squares_solution():
    rng = np.random.RandomState(3)
    A = rng.randn(8, 3) @ rng.randn(3, 8)
    b = rng.randn(8)
    x = minnorm_svd(A, b)
    res = np.linalg.norm(A @ x - b)
    _, _, Vt = np.linalg.svd(A)
    Z = Vt[3:].T                                    # null space
    for _ in range(20):
        y = x + Z @ rng.randn(Z.shape[1])
        assert np.linalg.norm(A @ y - b) >= res - 1e-8
        assert np.linalg.norm(y) >= np.linalg.norm(x) - 1e-10


def test_agrees_with_cod_when_the_system_is_well_conditioned():
    """The two formulations coincide when nothing is near-singular."""
    rng = np.random.RandomState(4)
    A = rng.randn(7, 7)
    b = rng.randn(7)
    assert np.allclose(minnorm_svd(A, b), lsqminnorm(A, b), atol=1e-9)


def test_diverges_from_cod_on_a_rank_deficient_badly_scaled_system():
    """...and demonstrably does not, when they are.  This is the whole point."""
    rng = np.random.RandomState(5)
    A = np.diag(np.logspace(8, -14, 8)) @ rng.randn(8, 8)
    b = rng.randn(8)
    x_svd = minnorm_svd(A, b)
    x_cod = lsqminnorm(A, b)
    rel = np.linalg.norm(x_svd - x_cod) / max(np.linalg.norm(x_cod), 1e-300)
    assert rel > 1e-12


@pytest.mark.parametrize("rule", ["matlab", "numpy", "exact"])
def test_tolerance_rules_are_ordered(rule):
    s = np.array([1.0, 1e-8, 1e-18])
    tol = svd_rank_tolerance((3, 3), s, rule)
    assert tol >= 0.0
    if rule == "exact":
        assert tol == 0.0


def test_exact_rule_inverts_every_nonzero_singular_value():
    """The literal Moore-Penrose definition, and why a cutoff exists."""
    rng = np.random.RandomState(6)
    A = rng.randn(8, 3) @ rng.randn(3, 8)
    b = rng.randn(8)
    x_tol = minnorm_svd(A, b, rule="matlab")
    x_raw = minnorm_svd(A, b, rule="exact")
    assert np.linalg.norm(x_raw) > 1e3 * np.linalg.norm(x_tol)


def test_non_finite_input_propagates_rather_than_raising():
    A = np.array([[1.0, np.nan], [0.0, 1.0]])
    assert np.all(np.isnan(minnorm_svd(A, np.ones(2))))
    assert np.all(np.isnan(pinv(A)))


def test_empty_dimensions():
    assert minnorm_svd(np.zeros((0, 3)), np.zeros(0)).shape == (3,)
    assert minnorm_svd(np.zeros((3, 0)), np.zeros(3)).shape == (0,)


def test_selectable_through_min_norm_solve():
    rng = np.random.RandomState(7)
    A = rng.randn(6, 6)
    b = rng.randn(6)
    assert np.allclose(min_norm_solve(A, b, method="pinv"), minnorm_svd(A, b), atol=1e-12)
