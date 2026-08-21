"""Stages 0 to 2: warm-up, pre-feasibility and the main solve.

``adam_mode`` determines how the K multistart columns are processed:

``"A"``  Warm-up on every column, pre-feasibility on the first, then a single
         solve receiving the whole (n, K) matrix.
``"B"``  Warm-up on the first column only.
``"C"``  Stages 0 to 2 applied independently to each column. The default.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from .adam import AdamInfo, adam_warmup
from .core import SolveResult, _clip_matlab, _proj_bnd, scatter_starts, solve_multistart
from .matlab_rng import MatlabRandom
from .options import Options

__all__ = ["solve_stages"]


def _prefeas_options(opts: Options) -> Options:
    """Stage 1 runs the same iteration with f := 0 and a loose residual test."""
    return opts.copy(
        maxIter=max(500, int(np.ceil(opts.maxIter * 0.4))),
        tol_r=1.0,
        tol_h=opts.tol_h,
        output_file="",
        verbose=False,
        multi_start=False,
        force_single_start=True,
        ms_num_starts=1,
        check_sosc=False,
        enforce_kkt_sign=False,
    )


def solve_stages(
    problem,
    backend,
    opts: Options,
    rng: Optional[MatlabRandom] = None,
) -> SolveResult:
    """Run the full three-stage pipeline and return the aggregated result."""
    t_start = time.time()
    n = problem.n
    lb = np.asarray(problem.lb, float)
    ub = np.asarray(problem.ub, float)
    mode = str(opts.adam_mode).strip().upper()
    if mode not in ("A", "B", "C"):
        mode = "A"

    if opts.force_single_start:
        opts = opts.copy(multi_start=False, ms_num_starts=1, ms_show_runs=False)

    if rng is None:
        rng = MatlabRandom(opts.ms_seed if opts.ms_seed >= 0 else 0)

    X0 = np.asarray(problem.x0, float)
    X0 = X0.reshape(n, -1) if X0.ndim > 1 else X0.reshape(n, 1)
    if X0.shape[1] > 1 and not opts.multi_start:
        X0 = X0[:, :1]
    elif opts.multi_start and X0.shape[1] > opts.ms_num_starts:
        # An explicit start matrix wider than the requested run count is
        # truncated.
        X0 = X0[:, : opts.ms_num_starts]
    elif opts.multi_start and 1 < X0.shape[1] < opts.ms_num_starts:
        # One that is too narrow is topped up by scattering around its first
        # column, so ms_num_starts is always honoured.
        extra = scatter_starts(X0[:, 0], lb, ub,
                               opts.ms_num_starts - X0.shape[1] + 1,
                               opts.ms_scale, rng, opts.ms_x0)
        X0 = np.hstack([X0, extra[:, 1:]])
    X0 = _clip_matlab(X0, lb[:, None], ub[:, None])

    # Modes A and C scatter here so Adam can touch every column; mode B keeps
    # the legacy flow where the inner solver scatters.
    if mode != "B" and opts.multi_start and X0.shape[1] == 1 and opts.ms_num_starts > 1:
        X0 = scatter_starts(X0[:, 0], lb, ub, opts.ms_num_starts,
                            opts.ms_scale, rng, opts.ms_x0)

    info: dict = {"adam_mode": mode, "backend": backend.name}

    # ---------------- Stage 0: Adam ----------------
    X_adam = X0.copy()
    adam_info = AdamInfo(skipped=True)
    if opts.use_adam_warmup:
        cols = 1 if mode == "B" else X0.shape[1]
        try:
            for c in range(cols):
                X_adam[:, c], ic = adam_warmup(backend, lb, ub, X0[:, c], opts)
                if c == 0:
                    adam_info = ic
        except Exception as exc:                       # keep going from x0
            X_adam = X0.copy()
            adam_info = AdamInfo(skipped=True, error=str(exc))
    info["adam"] = adam_info

    # ---------------- Adam early exit (unconstrained only) ----------------
    if (opts.adam_early_exit and opts.use_adam_warmup and not adam_info.skipped
            and problem.m == 0):
        fv = np.array([backend.f(X_adam[:, c]) for c in range(X_adam.shape[1])])
        gn = np.array([np.linalg.norm(backend.grad_f(X_adam[:, c]))
                       for c in range(X_adam.shape[1])])
        accept = gn < opts.adam_exit_tol_g
        if accept.any():
            masked = np.where(accept, fv, np.inf)
            best = int(np.argmin(masked))
            from .core import RunResult
            runs = [RunResult(theta=X_adam[:, c].copy(), fval=float(fv[c]),
                              residual_converged=bool(accept[c]), iterations=0,
                              dual_feas=True, dual_feas_strict=True)
                    for c in range(X_adam.shape[1])]
            info["early_exit_adam"] = True
            return SolveResult(theta=X_adam[:, best].copy(), fval=float(fv[best]),
                               residual_converged=True, runs=runs, best_run=best,
                               elapsed=time.time() - t_start,
                               solver_used=f"adam-only [{int(accept.sum())}/{len(runs)} cols]",
                               n_ineq_rows=int(problem.ineq_row_mask.sum()), info=info)

    # ---------------- Stage 1: pre-feasibility ----------------
    X_feas = X_adam.copy()
    if opts.use_prefeasibility and problem.m > 0:
        pre = _prefeas_options(opts)
        if mode == "C":
            for c in range(X_adam.shape[1]):
                try:
                    r = solve_multistart(problem, backend, pre,
                                         x0=X_adam[:, c], objective_override=0.0, rng=rng)
                    if np.all(np.isfinite(r.theta)):
                        X_feas[:, c] = r.theta
                except Exception as exc:
                    info.setdefault("prefeas_errors", []).append(str(exc))
        else:
            try:
                r = solve_multistart(problem, backend, pre,
                                     x0=X_adam[:, 0], objective_override=0.0, rng=rng)
                if mode == "A":
                    X_feas[:, 0] = r.theta
                else:
                    X_feas = r.theta.reshape(n, 1)
            except Exception:
                pass
        info["prefeas_done"] = True

    # ---------------- Stage 2: main solve ----------------
    if mode == "C":
        one = opts.copy(multi_start=False, force_single_start=True,
                        ms_num_starts=1, verbose=False)
        runs = []
        for c in range(X_feas.shape[1]):
            oc = one
            if opts.sign_flip_multistart:
                oc = one.copy(lam0_sign=1 if c % 2 == 0 else -1)
            try:
                r = solve_multistart(problem, backend, oc, x0=X_feas[:, c], rng=rng)
                runs.extend(r.runs)
            except Exception as exc:
                from .core import RunResult
                runs.append(RunResult(theta=X_feas[:, c].copy(),
                                      error=f"{type(exc).__name__}: {exc}"))
            if opts.verbose and opts.ms_show_runs:
                last = runs[-1]
                tag = (f"Converged in {last.iterations:4d} steps" if last.residual_converged
                       else f"NOT converged ({last.iterations} steps)")
                print(f"  Run {c + 1:3d} | Obj: {last.fval:14.6e} | {tag}")

        # Only finite objectives from converged runs are eligible.
        fvals = np.array([r.fval for r in runs])
        conv = np.array([r.residual_converged for r in runs], dtype=bool)
        eligible = conv & np.isfinite(fvals)
        if eligible.any():
            search = np.where(eligible, fvals, np.inf)
            best = int(np.argmin(search))
            theta, fval, converged = runs[best].theta, float(fvals[best]), True
        elif np.isfinite(fvals).any():
            search = np.where(np.isfinite(fvals), fvals, np.inf)
            best = int(np.argmin(search))
            theta, fval, converged = runs[best].theta, float(fvals[best]), False
        else:
            best = 0 if len(runs) else None
            theta = runs[best].theta if best is not None else np.zeros(n)
            fval = float(fvals[best]) if best is not None else np.inf
            converged = False
        out = SolveResult(theta=theta, fval=fval, residual_converged=converged, runs=runs,
                          best_run=best, elapsed=time.time() - t_start,
                          solver_used=f"kronos[{backend.name}, mode=C, step={opts.step_method}]",
                          n_ineq_rows=int(problem.ineq_row_mask.sum()), info=info)
    else:
        out = solve_multistart(problem, backend, opts, x0=X_feas, rng=rng)
        out.elapsed = time.time() - t_start
        out.info.update(info)
        out.solver_used = f"kronos[{backend.name}, mode={mode}, step={opts.step_method}]"

    out.info["x0_initial"] = X0
    out.info["x0_after_adam"] = X_adam
    out.info["x0_after_prefeas"] = X_feas
    return out
