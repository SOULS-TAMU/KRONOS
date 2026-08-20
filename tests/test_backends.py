"""Every backend must produce the same derivatives."""
import numpy as np
import pytest

from kronos.backends import available_backends, get_backend
from kronos.problem import Problem


@pytest.fixture
def problem():
    return Problem.build(
        "mixed", ["x1", "x2", "x3"],
        objective="exp(x1*x2) + log(1 + x3**2) + sin(x1) * sqrt(4 + x2**2)",
        equalities=["x1 + x2 + x3 - 1", "x1**2 - x2"],
        inequalities=["x1*x3 - 2"])


def test_sympy_backend_shapes(problem):
    b = get_backend(problem, "sympy")
    th = np.linspace(0.2, 0.7, problem.n)
    lam = np.linspace(-0.3, 0.5, problem.m)
    f, g, h, J, H = b.kkt(th, lam)
    assert np.isscalar(f) or np.ndim(f) == 0
    assert g.shape == (problem.n,)
    assert h.shape == (problem.m,)
    assert J.shape == (problem.m, problem.n)
    assert H.shape == (problem.n, problem.n)
    assert np.allclose(H, H.T, atol=1e-10)


def test_gradient_matches_finite_differences(problem):
    b = get_backend(problem, "sympy")
    th = np.array([0.3, 0.11, 0.42, 0.9, 1.1, 0.5])[: problem.n]
    g = b.grad_f(th)
    eps = 1e-6
    for i in range(problem.n):
        e = np.zeros(problem.n); e[i] = eps
        fd = (b.f(th + e) - b.f(th - e)) / (2 * eps)
        assert abs(fd - g[i]) < 1e-5 * max(1.0, abs(g[i]))


def test_constraint_jacobian_matches_finite_differences(problem):
    b = get_backend(problem, "sympy")
    th = np.linspace(0.25, 0.8, problem.n)
    J = b.Jh(th)
    eps = 1e-6
    for i in range(problem.n):
        e = np.zeros(problem.n); e[i] = eps
        fd = (b.h(th + e) - b.h(th - e)) / (2 * eps)
        assert np.allclose(fd, J[:, i], atol=1e-5, rtol=1e-4)


@pytest.mark.skipif("casadi" not in available_backends(), reason="casadi not installed")
def test_casadi_agrees_with_sympy(problem):
    th = np.linspace(0.2, 0.7, problem.n)
    lam = np.linspace(-0.3, 0.5, problem.m)
    a = get_backend(problem, "sympy").kkt(th, lam)
    c = get_backend(problem, "casadi").kkt(th, lam)
    for x, y in zip(a, c):
        assert np.allclose(np.asarray(x, float), np.asarray(y, float), atol=1e-11)


def test_objective_scale_zeroes_only_objective_terms(problem):
    b = get_backend(problem, "sympy")
    th = np.linspace(0.2, 0.7, problem.n)
    lam = np.linspace(-0.3, 0.5, problem.m)
    full = b.kkt(th, lam, 1.0)
    zero = b.kkt(th, lam, 0.0)
    assert zero[0] == 0.0
    assert np.allclose(zero[1], 0.0)
    assert np.allclose(zero[2], full[2])       # h unchanged
    assert np.allclose(zero[3], full[3])       # Jh unchanged


def test_auto_routing_follows_the_switch_threshold():
    small = Problem.build("s", 3, objective="x1**2 + x2**2 + x3**2")
    assert get_backend(small, "auto", switch_n=20).name == "sympy"
    if "casadi" in available_backends():
        big = Problem.build("b", 25, objective="+".join(f"x{i+1}**2" for i in range(25)))
        assert get_backend(big, "auto", switch_n=20).name == "casadi"


@pytest.mark.skipif("jax" not in available_backends(), reason="jax not installed")
def test_jax_agrees_with_sympy(problem):
    th = np.linspace(0.2, 0.7, problem.n)
    lam = np.linspace(-0.3, 0.5, problem.m)
    a = get_backend(problem, "sympy").kkt(th, lam)
    j = get_backend(problem, "jax").kkt(th, lam)
    for x, y in zip(a, j):
        assert np.allclose(np.asarray(x, float), np.asarray(y, float), atol=1e-9)


def test_nan_inputs_propagate_rather_than_raise():
    """MATLAB returns NaN where LAPACK would raise; the iteration depends on it."""
    from kronos.linalg import lsqminnorm, min_norm_solve
    A = np.array([[1.0, np.nan], [0.0, 1.0]])
    b = np.array([1.0, 1.0])
    assert np.all(np.isnan(lsqminnorm(A, b)))
    assert np.all(np.isnan(min_norm_solve(A, b, method="pinv")))


def test_second_derivative_of_abs_does_not_break_lambdify():
    """SymPy emits DiracDelta for d2|x|/dx2; the numpy namespace has no such
    name, so a Hessian containing it used to raise NameError and silently fail
    every problem built from abs()."""
    p = Problem.build("absy", ["x1", "x2"],
                      objective="Abs(x1) + (x2 - 1)**2 + x1**2")
    b = get_backend(p, "sympy")
    H = b.hess_lag(np.array([0.7, 0.3]), np.zeros(0))
    assert np.all(np.isfinite(H))
    assert H.shape == (2, 2)


@pytest.mark.skipif("casadi" not in available_backends(), reason="casadi not installed")
def test_abs_hessian_agrees_between_backends_away_from_the_kink():
    p = Problem.build("absy", ["x1", "x2"],
                      objective="Abs(x1) + (x2 - 1)**2 + x1**2")
    th = np.array([0.7, 0.3])
    a = get_backend(p, "sympy").kkt(th, np.zeros(0))
    c = get_backend(p, "casadi").kkt(th, np.zeros(0))
    for x, y in zip(a, c):
        assert np.allclose(np.asarray(x, float), np.asarray(y, float), atol=1e-10)


def test_auto_routing_warns_instead_of_silently_downgrading(monkeypatch):
    """If CasADi is missing, auto-routing must say so.

    SymPy builds the Hessian element-wise; at n=1000 that is 35s versus 0.2s.
    Falling back without a word would look like the solver is simply slow.
    """
    import kronos.backends as B
    monkeypatch.setattr(B, "available_backends", lambda: ["sympy"])
    big = Problem.build("big", 25, objective="+".join(f"x{i+1}**2" for i in range(25)))
    with pytest.warns(RuntimeWarning, match="CasADi is not installed"):
        b = B.get_backend(big, "auto", switch_n=20)
    assert b.name == "sympy"


def test_no_warning_when_casadi_is_available_or_not_needed():
    import warnings as _w
    small = Problem.build("small", 3, objective="x1**2 + x2**2 + x3**2")
    with _w.catch_warnings():
        _w.simplefilter("error")            # any warning becomes a failure
        assert get_backend(small, "auto", switch_n=20).name == "sympy"
