"""The bundled benchmark library."""
import numpy as np
import pytest

from kronos.library import iter_problems, load_problem, problem_names, resolve_name


def test_library_is_the_curated_set():
    names = problem_names()
    assert len(names) == 244
    assert "hart6" not in names          # pruned: does not reach tolerance
    assert "allinitc" not in names       # pruned: falls short on global hits
    assert "hs013" not in names          # pruned: LICQ fails at the solution
    assert "hs041" not in names          # pruned: all starts reach a non-optimal KKT point
    assert "hs053" in names


@pytest.mark.parametrize("given,expected", [
    ("hs001", "hs001"),
    ("HS001", "hs001"),
    ("helix", "helix"),
    ("HELIX", "helix"),
])
def test_name_resolution(given, expected):
    assert resolve_name(given) == expected


def test_ambiguous_name_is_rejected_with_a_hint():
    with pytest.raises(KeyError) as e:
        resolve_name("hs0")
    assert "Did you mean" in str(e.value)


def test_find_searches_by_name_size_and_constrainedness():
    from kronos.library import find
    assert "hs001" in find("hs")
    assert all(len(load_problem(n).var_names) <= 6 for n in find(max_n=6))
    assert all(load_problem(n).m > 0 for n in find(constrained=True))
    assert all(load_problem(n).m == 0 for n in find(constrained=False))


def test_unknown_name_is_rejected():
    with pytest.raises(KeyError):
        resolve_name("definitely_not_a_problem")


def test_every_problem_loads_and_is_self_consistent():
    for name in problem_names():
        p = load_problem(name)
        assert p.n >= 1
        assert p.lb.size == p.n and p.ub.size == p.n
        assert np.asarray(p.x0).shape[0] == p.n
        assert len(p.h) == p.m
        for sr in p.slack_rows:
            assert 0 <= sr.row < p.m
            assert 0 <= sr.slack_var < p.n
            assert sr.sign in (-1, 1)


def test_loaded_problems_are_independent_copies():
    a = load_problem("a09_matyas")
    b = load_problem("a09_matyas")
    a.x0 = np.asarray(a.x0) + 1.0
    assert not np.allclose(np.asarray(a.x0), np.asarray(b.x0))


def test_filtering_by_size_and_constrainedness():
    small = list(iter_problems(max_n=3))
    assert small and all(p.n <= 3 for p in small)
    con = list(iter_problems(max_n=6, constrained=True))
    assert con and all(p.m > 0 for p in con)
    unc = list(iter_problems(max_n=6, constrained=False))
    assert unc and all(p.m == 0 for p in unc)


def test_every_problem_has_a_known_optimum():
    assert all(load_problem(n).fstar is not None for n in problem_names())
