"""Per-seed A -> FB cascade.

Stage 2 (squared slacks, "A") runs first on every seed.  Only the seeds where
it fails -- either not converged, or converged to a dual-infeasible multiplier
-- are retried with the Fischer-Burmeister formulation.  Cost is therefore
"A alone, plus FB on the few percent that need it".

The winner per seed is chosen by ``3 * kkt_certified + converged``, ties broken
by the smaller objective; and the reported solution is the best objective among
*certified* seeds, falling back to merely converged ones only if none is
certified.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from .core import RunResult, SolveResult
from .fb import solve_fb
from .matlab_rng import MatlabRandom
from .options import Options
from .problem import Problem
from .stages import solve_stages

__all__ = ["solve_cascade"]


def solve_cascade(problem: Problem, backend, opts: Options,
                  rng: Optional[MatlabRandom] = None) -> SolveResult:
    """Run stage A, then FB on the seeds A could not certify."""
    t0 = time.time()
    if rng is None:
        rng = MatlabRandom(opts.ms_seed if opts.ms_seed >= 0 else 0)

    optsA = opts.copy(enforce_kkt_sign=True)
    resA = solve_stages(problem, backend, optsA, rng)

    runsA = resA.runs
    K = len(runsA)
    convA = np.array([r.converged for r in runsA], dtype=bool)
    kktA = np.array([r.kkt_certified for r in runsA], dtype=bool)

    needs_fb = ~(convA & kktA)
    n_fb = int(needs_fb.sum())

    runsFB: dict[int, RunResult] = {}
    t_fb = 0.0
    if n_fb > 0 and opts.fb_enable:
        # FB uses its own fresh scatter rather than warm-starting from A's
        # points, which are usually in the very basin FB needs to escape.
        x0 = np.asarray(problem.x0, float)
        x0 = x0[:, 0] if x0.ndim > 1 else x0
        optsFB = opts.copy(multi_start=True, ms_num_starts=n_fb,
                           ms_show_runs=False, verbose=False)
        t1 = time.time()
        try:
            rfb = solve_fb(problem, optsFB, x0, n_starts=n_fb, rng=rng)
            for slot, seed in enumerate(np.flatnonzero(needs_fb)):
                if slot < len(rfb.runs):
                    runsFB[int(seed)] = rfb.runs[slot]
        except Exception as exc:
            resA.info["fb_error"] = f"{type(exc).__name__}: {exc}"
        t_fb = time.time() - t1

    # ---- per-seed best-of ----
    best_runs: list[RunResult] = []
    picks: list[str] = []
    for c in range(K):
        a = runsA[c]
        b = runsFB.get(c)
        if b is None:
            best_runs.append(a)
            picks.append("A")
            continue
        score_a = 3 * a.kkt_certified + a.converged
        score_b = 3 * b.kkt_certified + b.converged
        if score_b > score_a or (score_b == score_a and b.fval < a.fval):
            best_runs.append(b)
            picks.append("FB")
        else:
            best_runs.append(a)
            picks.append("A")

    fvals = np.array([r.fval for r in best_runs])
    conv = np.array([r.converged for r in best_runs], dtype=bool)
    kkt = np.array([r.kkt_certified for r in best_runs], dtype=bool)

    finite = np.isfinite(fvals)
    scored = np.where(conv & kkt & finite, fvals, np.inf)
    if not np.isfinite(scored).any():
        scored = np.where(conv & finite, fvals, np.inf)
    if not np.isfinite(scored).any():
        scored = np.where(finite, fvals, np.inf)
    best = int(np.argmin(scored)) if scored.size else 0

    info = dict(resA.info)
    info.update({
        "cascade": True,
        "n_seeds_to_fb": n_fb,
        "n_picked_fb": picks.count("FB"),
        "pick": picks,
        "time_A": resA.elapsed,
        "time_FB": t_fb,
    })
    return SolveResult(
        theta=best_runs[best].theta if best_runs else np.zeros(problem.n),
        fval=float(fvals[best]) if fvals.size else np.inf,
        converged=bool(conv[best]) if conv.size else False,
        runs=best_runs, best_run=best, elapsed=time.time() - t0,
        solver_used=f"cascade[A={int(convA.sum())}/{K}, FB picked {picks.count('FB')}]",
        n_ineq_rows=resA.n_ineq_rows, info=info)
