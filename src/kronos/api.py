"""Top-level solve entry point.
"""

from __future__ import annotations

import sys
from typing import Any, Optional

import numpy as np

from .backends import get_backend
from .core import SolveResult
from .matlab_rng import MatlabRandom
from .options import Options
from .problem import Problem

__all__ = ["solve"]


def solve(
    problem: Problem,
    options: Optional[Options] = None,
    backend: Any = None,
    **overrides: Any,
) -> SolveResult:
    """Solve ``problem`` with KRONOS and certify the result.

    Parameters
    ----------
    problem : Problem
    options : Options, optional
        Defaults to ``Options()``; any keyword argument overrides a field.
    backend : Backend, optional
        Pre-built backend, to avoid recompiling derivatives across repeated
        solves of the same problem.

    Returns
    -------
    SolveResult

    Examples
    --------
    >>> from kronos import Problem, solve
    >>> p = Problem.build("rosen", ["x1", "x2"],
    ...                   objective="100*(x2 - x1**2)**2 + (1 - x1)**2",
    ...                   x0=[-1.2, 1.0])
    >>> r = solve(p, multi_start=True, ms_num_starts=5)
    >>> round(r.fval, 10)
    0.0
    """
    opts = (options or Options()).copy(**overrides) if overrides else (options or Options())
    if opts.fstar is None and problem.fstar is not None:
        opts = opts.copy(fstar=problem.fstar)

    if backend is None:
        backend = get_backend(problem, opts.backend, switch_n=opts.backend_switch_n)

    rng = MatlabRandom(opts.ms_seed if opts.ms_seed >= 0 else 0)

    has_slacks = bool(problem.ineq_row_mask.any())
    if has_slacks and opts.fb_enable:
        from .cascade import solve_cascade
        result = solve_cascade(problem, backend, opts, rng)
    else:
        from .stages import solve_stages
        result = solve_stages(problem, backend, opts, rng)

    result.info.setdefault("problem", problem.name)
    result.info.setdefault("fstar", problem.fstar if opts.fstar is None else opts.fstar)
    result.info.setdefault("n", problem.n)
    result.info.setdefault("m", problem.m)
    result.info.setdefault("has_slack_rows", has_slacks)
    slack = {sr.slack_var for sr in problem.slack_rows}
    result.info.setdefault("user_vars", [i for i in range(problem.n) if i not in slack])
    result.info.setdefault("var_names", list(problem.var_names))
    return result
