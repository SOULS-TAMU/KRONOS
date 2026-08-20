"""End-to-end solves against problems with known answers."""
import numpy as np
import pytest

from kronos import Options, Problem, solve
from kronos.library import load_problem, problem_names


def test_rosenbrock_reaches_the_global_minimum():
    p = Problem.build("rosen", ["x1", "x2"],
                      objective="100*(x2 - x1**2)**2 + (1 - x1)**2", x0=[-1.2, 1.0])
    r = solve(p, Options(maxIter=800))
    assert r.converged
    assert r.fval < 1e-10
    assert np.allclose(r.theta, [1.0, 1.0], atol=1e-5)


def test_hs71_reaches_its_documented_optimum():
    p = Problem.build("hs71", ["x1", "x2", "x3", "x4"],
                      objective="x1*x4*(x1 + x2 + x3) + x3",
                      equalities=["x1**2 + x2**2 + x3**2 + x4**2 - 40"],
                      inequalities=["25 - x1*x2*x3*x4"],
                      lb=[1, 1, 1, 1], ub=[5, 5, 5, 5], x0=[1, 5, 5, 1])
    r = solve(p, Options(maxIter=1500))
    assert r.converged
    assert abs(r.fval - 17.0140173) < 1e-4
    assert np.allclose(r.theta[:4], [1.0, 4.7429994, 3.8211503, 1.3794082], atol=1e-3)


def test_constrained_solution_satisfies_its_constraints():
    """From a single start the method finds a KKT *stationary* point.

    min x1 + x2 on the unit circle has two: -sqrt(2) (the minimum) and
    +sqrt(2) (the maximum).  Starting at (0.5, 0.5) it lands on the maximum,
    which is correct behaviour -- and the reason SOSC classification exists.
    """
    p = Problem.build("circle", ["x1", "x2"], objective="x1 + x2",
                      equalities=["x1**2 + x2**2 - 1"], x0=[0.5, 0.5])
    r = solve(p, Options(maxIter=800))
    assert r.converged
    assert abs(r.theta[0] ** 2 + r.theta[1] ** 2 - 1) < 1e-6
    assert abs(abs(r.fval) - np.sqrt(2)) < 1e-5


def test_sosc_rejects_a_constrained_maximum_and_accepts_the_minimum():
    p = Problem.build("circle", ["x1", "x2"], objective="x1 + x2",
                      equalities=["x1**2 + x2**2 - 1"], x0=[0.5, 0.5])
    at_max = solve(p, Options(maxIter=800))
    assert at_max.fval > 0                       # landed on the maximum
    assert not at_max.runs[at_max.best_run].sosc_pass

    p_min = Problem.build("circle", ["x1", "x2"], objective="x1 + x2",
                          equalities=["x1**2 + x2**2 - 1"], x0=[-0.5, -0.5])
    at_min = solve(p_min, Options(maxIter=800))
    assert abs(at_min.fval - (-np.sqrt(2))) < 1e-5
    assert at_min.runs[at_min.best_run].sosc_pass


def test_multistart_returns_one_result_per_run():
    p = Problem.build("sq", ["x1", "x2"], objective="(x1-1)**2 + (x2+2)**2", x0=[0, 0])
    r = solve(p, Options(multi_start=True, ms_num_starts=7, ms_seed=42))
    assert len(r.runs) == 7
    assert r.all_fvals.shape == (7,)
    assert r.n_conv >= 1
    assert r.fval < 1e-8


def test_dual_feasibility_is_certified_at_an_active_bound():
    # minimum of (x-3)^2 subject to x <= 1 is at x = 1, with a positive multiplier
    p = Problem.build("bnd", ["x"], objective="(x - 3)**2", inequalities=["x - 1"], x0=[0.0])
    r = solve(p, Options(maxIter=800, enforce_kkt_sign=True))
    assert r.converged
    assert abs(r.theta[0] - 1.0) < 1e-5
    assert r.runs[r.best_run].kkt_certified


def test_sosc_flags_a_strict_local_minimum():
    p = Problem.build("quad", ["x1", "x2"], objective="x1**2 + 3*x2**2", x0=[1.0, 1.0])
    r = solve(p, Options(maxIter=500))
    assert r.runs[r.best_run].sosc_pass


@pytest.mark.parametrize("step", ["pinv", "cod", "lstsq"])
def test_alternative_step_methods_still_solve_an_easy_problem(step):
    p = Problem.build("sq", ["x1", "x2"], objective="(x1-1)**2 + (x2+2)**2", x0=[0, 0])
    r = solve(p, Options(maxIter=500, step_method=step))
    assert r.converged and r.fval < 1e-8


def test_dummy_variable_can_be_switched_off():
    p = Problem.build("sq", ["x1", "x2"], objective="(x1-1)**2 + (x2+2)**2", x0=[3, 3])
    a = solve(p, Options(maxIter=500, use_dummy_variable=True))
    b = solve(p, Options(maxIter=500, use_dummy_variable=False))
    assert a.converged and b.converged
    assert abs(a.fval - b.fval) < 1e-8


def test_bundled_library_is_complete_and_loadable():
    names = problem_names()
    assert len(names) == 244
    p = load_problem("a09_matyas")
    assert p.n == 2 and p.fstar is not None


def test_solving_a_bundled_problem_hits_its_known_optimum():
    p = load_problem("a09_matyas")
    r = solve(p, Options(multi_start=True, ms_num_starts=3, ms_seed=42,
                         maxIter=500, use_adam_warmup=False))
    assert r.n_conv >= 1
    assert abs(r.fval - p.fstar) < 1e-4


def test_nan_gradient_does_not_abort_the_run():
    """f is finite at the origin but its gradient is 0/0 there.

    d/dx of -200*exp(-|x|/50) involves x/|x|, which is NaN at the origin even
    though f itself is a clean -200.  MATLAB's max() ignores NaN, so
    max(abs(r)) sees only the finite residual entries and the run converges;
    np.max would report NaN and the run could never terminate.
    """
    p = Problem.build("ackley2", ["x1", "x2"],
                      objective="-200*exp(-sqrt(x1**2 + x2**2)/50)",
                      x0=[0.0, 0.0])
    r = solve(p, Options(maxIter=200))
    assert r.n_conv >= 1
    assert np.isfinite(r.fval)
    assert abs(r.fval - (-200.0)) < 1e-6


def test_singular_start_point_does_not_crash_the_run():
    """helix evaluates atan(x2/x1) at x1 = 0, i.e. atan(0/0)."""
    p = load_problem("helix")
    r = solve(p, Options(multi_start=True, ms_num_starts=5, ms_seed=42, maxIter=200,
                         use_adam_warmup=True, adam_mode="C", adam_iters=200,
                         adam_lr=0.05, adam_rho=10))
    assert all(x.error == "" for x in r.runs)
    assert np.isfinite(r.fval)


def test_nan_starting_point_is_sanitised_by_projection():
    """MATLAB's min(max(x, lo), hi) returns lo for NaN, so a NaN start recovers.

    Some reference problems have a NaN starting point (one MATLAB script
    computes zeros ./ sum(zeros)); the solver still handles them, because
    projection replaces the NaN with the bound.  np.clip would propagate it.
    """
    from kronos.core import _clip_matlab
    x = np.array([np.nan, 5.0, -20.0])
    lo, hi = np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0])
    assert np.allclose(_clip_matlab(x, lo, hi), [-1.0, 1.0, -1.0])

    p = Problem.build("nanstart", ["x1", "x2"],
                      objective="(x1 - 2)**2 + (x2 + 1)**2",
                      lb=[-5, -5], ub=[5, 5], x0=[0.0, 0.0])
    p.x0 = np.asarray(p.x0, float).copy()
    p.x0[:2] = np.nan                       # a genuinely NaN starting point
    r = solve(p, Options(maxIter=800))
    assert r.n_conv >= 1
    assert np.isfinite(r.fval)


def test_certified_runs_are_fully_accounted_for():
    """verified minima + stationary-only + untested == KKT-certified.

    'SOSC not measured' is not the same as 'SOSC failed': the Fischer-Burmeister
    fallback does not form a reduced Hessian, so its runs are untested rather
    than rejected.  Reporting them as failures understates the result.
    """
    p = Problem.build("hs71", ["x1", "x2", "x3", "x4"],
                      objective="x1*x4*(x1 + x2 + x3) + x3",
                      equalities=["x1**2 + x2**2 + x3**2 + x4**2 - 40"],
                      inequalities=["25 - x1*x2*x3*x4"],
                      lb=[1, 1, 1, 1], ub=[5, 5, 5, 5], x0=[1, 5, 5, 1])
    r = solve(p, Options(multi_start=True, ms_num_starts=10, ms_seed=42, maxIter=1500))
    assert r.n_local + r.n_stationary + r.n_sosc_unmeasured == r.n_kkt
    for run in r.runs:
        if run.sosc_pass:
            assert run.sosc_measured           # cannot pass a test that never ran
        if not run.sosc_measured:
            assert not np.isfinite(run.lam_min_red) or run.lam_min_red == np.inf


def test_reaching_the_optimum_and_verifying_it_are_different_questions():
    """A run can reach f* without SOSC being verified, and vice versa."""
    p = load_problem("hs030")
    r = solve(p, Options(multi_start=True, ms_num_starts=10, ms_seed=42, maxIter=1500))
    assert r.global_hits(p.fstar) >= 1
    # both counts are well defined and independent
    assert 0 <= r.n_local <= r.n_kkt
    assert 0 <= r.global_hits(p.fstar) <= len(r.runs)


def test_converged_means_certified():
    """n_conv counts certified runs; the looser count is n_residual_conv."""
    p = Problem.build("hs71", ["x1", "x2", "x3", "x4"],
                      objective="x1*x4*(x1 + x2 + x3) + x3",
                      equalities=["x1**2 + x2**2 + x3**2 + x4**2 - 40"],
                      inequalities=["25 - x1*x2*x3*x4"],
                      lb=[1, 1, 1, 1], ub=[5, 5, 5, 5], x0=[1, 5, 5, 1])
    r = solve(p, Options(multi_start=True, ms_num_starts=10, ms_seed=42, maxIter=1500))
    assert r.n_conv == r.n_kkt == int(r.all_kkt.sum())
    assert r.n_residual_conv >= r.n_conv
    assert r.n_conv == sum(1 for x in r.runs if x.kkt_certified)


def test_summary_reports_certified_and_hides_uncertified_by_default():
    p = load_problem("hs030")
    r = solve(p, Options(multi_start=True, ms_num_starts=5, ms_seed=42))
    default = r.summary()
    assert "KKT-certified" in default
    assert "residual-converged" not in default
    assert "residual-converged" in r.summary(show_uncertified=True)


def test_global_lines_appear_only_when_fstar_is_known():
    p = Problem.build("q", ["x1", "x2"], objective="(x1-3)**2 + (x2+1)**2", x0=[0, 0])
    without = solve(p, Options(multi_start=True, ms_num_starts=5, ms_seed=42)).summary()
    assert "reached f*" not in without

    r = solve(p, Options(multi_start=True, ms_num_starts=5, ms_seed=42, fstar=0.0))
    with_fstar = r.summary()
    assert "reached f*" in with_fstar
    assert "f* / converged" in with_fstar
    assert r.global_hits(0.0) >= 1


def test_global_hits_requires_certification():
    """A run at the right objective but with wrong-signed multipliers is not a hit."""
    p = load_problem("hs030")
    r = solve(p, Options(multi_start=True, ms_num_starts=10, ms_seed=42))
    assert r.global_hits(p.fstar) <= r.n_conv
