"""The minimum-norm least-squares kernel is the contribution; test it hardest."""
import numpy as np
import pytest

from kronos.linalg import lsqminnorm, min_norm_solve, null_space


def test_full_rank_matches_direct_solve():
    rng = np.random.RandomState(0)
    A = rng.randn(6, 6)
    b = rng.randn(6)
    assert np.allclose(lsqminnorm(A, b), np.linalg.solve(A, b), atol=1e-10)


def test_overdetermined_matches_least_squares():
    rng = np.random.RandomState(1)
    A = rng.randn(20, 5)
    b = rng.randn(20)
    assert np.allclose(lsqminnorm(A, b), np.linalg.lstsq(A, b, rcond=None)[0], atol=1e-10)


def test_rank_deficient_returns_minimum_norm_solution():
    rng = np.random.RandomState(2)
    A = rng.randn(8, 3) @ rng.randn(3, 8)      # rank 3
    b = rng.randn(8)
    x = lsqminnorm(A, b)
    res = np.linalg.norm(A @ x - b)
    # residual is optimal ...
    assert res <= np.linalg.norm(A @ np.linalg.pinv(A) @ b - b) + 1e-8
    # ... and no other optimal solution is shorter
    Z = null_space(A)
    for _ in range(20):
        y = x + Z @ rng.randn(Z.shape[1])
        assert np.linalg.norm(A @ y - b) >= res - 1e-8
        assert np.linalg.norm(y) >= np.linalg.norm(x) - 1e-10


def test_underdetermined_is_in_row_space():
    rng = np.random.RandomState(3)
    A = rng.randn(3, 9)
    b = rng.randn(3)
    x = lsqminnorm(A, b)
    assert np.allclose(A @ x, b, atol=1e-10)
    # minimum-norm solution lies in the row space of A
    assert np.allclose(x, A.T @ np.linalg.lstsq(A @ A.T, b, rcond=None)[0], atol=1e-8)


def test_zero_matrix_gives_zero():
    assert np.allclose(lsqminnorm(np.zeros((4, 3)), np.ones(4)), np.zeros(3))


def test_multiple_right_hand_sides():
    rng = np.random.RandomState(4)
    A = rng.randn(7, 4)
    B = rng.randn(7, 3)
    X = lsqminnorm(A, B)
    assert X.shape == (4, 3)
    for j in range(3):
        assert np.allclose(X[:, j], lsqminnorm(A, B[:, j]), atol=1e-12)


def test_empty_dimensions():
    assert lsqminnorm(np.zeros((0, 3)), np.zeros(0)).shape == (3,)
    assert lsqminnorm(np.zeros((3, 0)), np.zeros(3)).shape == (0,)


@pytest.mark.parametrize("method", ["pinv", "cod", "lstsq", "tikhonov", "backslash"])
def test_all_step_methods_run(method):
    rng = np.random.RandomState(5)
    A = rng.randn(6, 6)
    b = rng.randn(6)
    x = min_norm_solve(A, b, method=method)
    assert x.shape == (6,)
    assert np.all(np.isfinite(x))


def test_cod_and_pinv_agree_closely_on_a_badly_scaled_system():
    """The two formulations are different computations that nearly agree.

    On this matrix (cond ~ 2e23) both make the same rank decision (5) and reach
    essentially the same least-squares residual -- they differ by ~7e-7
    relative, with COD marginally smaller.  That gap is real, not float noise,
    but it is far too small to matter once the solver's line search is wrapped
    around it, which is why the two step methods score the same across the
    whole benchmark.  Compared relatively, since an absolute tolerance at this
    magnitude would sit below the noise floor.
    """
    rng = np.random.RandomState(6)
    A = np.diag(np.logspace(8, -14, 8)) @ rng.randn(8, 8)
    b = rng.randn(8)
    x_cod = min_norm_solve(A, b, method="cod")
    x_pinv = min_norm_solve(A, b, method="pinv")

    r_cod = np.linalg.norm(A @ x_cod - b)
    r_pinv = np.linalg.norm(A @ x_pinv - b)
    assert abs(r_cod - r_pinv) <= 1e-5 * max(r_cod, r_pinv, 1.0)

    # ...but they are not the same computation, and must not be conflated
    assert not np.allclose(x_cod, x_pinv, rtol=1e-12, atol=0)

def test_null_space_basis_is_orthonormal_and_annihilates():
    A = np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    Z = null_space(A)
    assert Z.shape == (3, 1)
    assert np.allclose(A @ Z, 0, atol=1e-12)
    assert np.allclose(Z.T @ Z, np.eye(Z.shape[1]), atol=1e-12)


def test_null_space_of_full_rank_is_empty():
    assert null_space(np.eye(4)).shape[1] == 0
