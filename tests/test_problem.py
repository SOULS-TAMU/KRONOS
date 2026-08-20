"""Problem construction, squared-slack reformulation and serialisation."""
import json

import numpy as np
import pytest
import sympy as sp

from kronos.problem import Problem


def build_hs71():
    return Problem.build(
        "hs71", ["x1", "x2", "x3", "x4"],
        objective="x1*x4*(x1 + x2 + x3) + x3",
        equalities=["x1**2 + x2**2 + x3**2 + x4**2 - 40"],
        inequalities=["25 - x1*x2*x3*x4"],
        lb=[1, 1, 1, 1], ub=[5, 5, 5, 5], x0=[1, 5, 5, 1])


def test_inequality_becomes_squared_slack_row_with_positive_sign():
    p = Problem.build("t", ["x"], objective="x**2", inequalities=["x - 1"])
    assert p.n == 2 and p.m == 1
    assert len(p.slack_rows) == 1
    assert p.slack_rows[0].sign == +1
    s = sp.Symbol(p.var_names[1], real=True)
    assert sp.simplify(p.h[0] - (sp.Symbol("x", real=True) - 1 + s ** 2)) == 0


def test_greater_equal_inequality_gets_negative_sign():
    p = Problem.build("t", ["x"], objective="x**2",
                      inequalities=["x - 1"], inequality_sense=">=")
    assert p.slack_rows[0].sign == -1


def test_bounds_become_two_slack_rows_each():
    p = Problem.build("t", ["x"], objective="x**2", lb=[0], ub=[3])
    assert p.m == 2
    assert [sr.sign for sr in p.slack_rows] == [-1, -1]


def test_bounds_can_stay_as_clamps():
    p = Problem.build("t", ["x"], objective="x**2", lb=[0], ub=[3],
                      bounds_as_slacks=False)
    assert p.m == 0 and p.n == 1
    assert p.lb[0] == 0 and p.ub[0] == 3


def test_hs71_shape_and_slack_bookkeeping():
    p = build_hs71()
    assert (p.n, p.m, len(p.slack_rows)) == (13, 10, 9)
    assert p.ineq_row_mask.sum() == 9
    assert not p.ineq_row_mask[0]          # the true equality row is not a slack row
    assert set(p.slack_var_index[p.ineq_row_mask]) == set(range(4, 13))


def test_slack_detection_recovers_construction():
    p = build_hs71()
    found = p.detect_slack_rows()
    assert [(s.row, s.sign, s.slack_var) for s in found] == \
           [(s.row, s.sign, s.slack_var) for s in p.slack_rows]


def test_slack_initialisation_is_feasible_where_possible():
    p = Problem.build("t", ["x"], objective="x**2", lb=[0], ub=[3], x0=[2])
    # x - 0 - s1^2 = 0 -> s1 = sqrt(2);  3 - x - s2^2 = 0 -> s2 = 1
    assert np.isclose(p.x0[1], np.sqrt(2))
    assert np.isclose(p.x0[2], 1.0)


def test_json_round_trip_preserves_everything():
    p = build_hs71()
    q = Problem.from_dict(json.loads(json.dumps(p.to_dict())))
    assert q.n == p.n and q.m == p.m
    assert sp.simplify(q.f - p.f) == 0
    assert all(sp.simplify(a - b) == 0 for a, b in zip(q.h, p.h))
    assert [s.sign for s in q.slack_rows] == [s.sign for s in p.slack_rows]


def test_matrix_x0_is_accepted_and_flat_input_reshapes():
    p = Problem("t", ["x1", "x2"], sp.sympify("x1**2 + x2**2"),
                x0=np.zeros(2 * 5))
    assert p.x0.shape == (2, 5)


def test_bad_bounds_length_is_rejected():
    with pytest.raises(ValueError):
        Problem("t", ["x1", "x2"], sp.sympify("x1"), lb=[0.0])


def test_every_option_is_accounted_for():
    """Every field is either documented by describe() or explicitly withheld."""
    from kronos.options import Options, _GROUPS, _UNDOCUMENTED
    documented = {f for _, fields in _GROUPS for f, _ in fields}
    actual = set(Options.__dataclass_fields__)
    assert documented.isdisjoint(_UNDOCUMENTED)
    assert documented | _UNDOCUMENTED == actual, (
        f"unaccounted: {sorted(actual - documented - _UNDOCUMENTED)}, "
        f"stale: {sorted((documented | _UNDOCUMENTED) - actual)}")


def test_withheld_options_still_work():
    """Withholding them from the listing must not disable them."""
    from kronos import Options
    o = Options(step_method="cod", rank_rule="dense")
    assert o.step_method == "cod" and o.rank_rule == "dense"


def test_describe_renders_and_filters():
    from kronos import Options
    text = Options.describe()
    assert "tol_r" in text and "converged when" in text
    assert "Convergence" in text and "Certification" in text
    only = Options.describe("Convergence")
    assert "tol_r" in only and "fb_eps" not in only


def test_tolerances_are_actually_honoured():
    """Tightening tol_r must produce a smaller final residual."""
    import numpy as np
    from kronos import Options, Problem, solve
    p = Problem.build("q", ["x1", "x2"], objective="(x1-3)**2 + (x2+1)**4", x0=[0.0, 0.0])
    loose = solve(p, Options(tol_r=1e-3, maxIter=2000))
    tight = solve(p, Options(tol_r=1e-10, maxIter=2000))
    assert loose.converged and tight.converged
    assert tight.runs[tight.best_run].max_r <= loose.runs[loose.best_run].max_r
    assert tight.runs[tight.best_run].max_r < 1e-9
