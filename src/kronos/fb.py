"""Fischer-Burmeister formulation.

The slack variables are eliminated and the complementarity conditions imposed
through the smoothed function

    phi(mu, b) = mu + b - sqrt(mu**2 + b**2 + eps**2),    b = -sigma * g(x)

whose zero is equivalent to ``mu >= 0``, ``b >= 0`` and ``mu * b = 0``.

The state is ``z = [xs, theta, lam, mu]`` and the residual

    R = [ dL/dxs ; dL/dtheta ; h_dummy ; h_eq ; phi ]
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import sympy as sp

from .backends import get_backend
from .core import RunResult, SolveResult, _clip_matlab, _mmax, _mmin, scatter_starts
from .linalg import min_norm_solve
from .matlab_rng import MatlabRandom
from .options import Options
from .problem import Problem

__all__ = ["FBSystem", "solve_fb"]

_XT = 1.0


@dataclass
class FBSystem:
    """The slack-eliminated problem the FB residual is built on."""

    reduced: Problem            # variables = free user vars; h = [h_eq..., g...]
    n_eq: int                   # number of pure equality rows in reduced.h
    signs: np.ndarray           # sigma_k for each inequality row
    kept: np.ndarray            # bool mask over the original variables
    pinned: np.ndarray          # bool mask over the original variables
    pinned_val: np.ndarray
    slack_of_row: dict          # original h-row -> original slack variable index
    g_rows: list                # original h-row index of each inequality
    lb_user: np.ndarray
    ub_user: np.ndarray


def build_fb_system(problem: Problem, opts: Options) -> FBSystem:
    """Eliminate squared slacks and split ``h`` into equalities and ``g``."""
    syms = problem.symbols
    n_full = problem.n
    mask = problem.ineq_row_mask
    sign = problem.ineq_row_sign
    slack_idx = problem.slack_var_index

    is_slack = np.zeros(n_full, dtype=bool)
    for sr in problem.slack_rows:
        is_slack[sr.slack_var] = True

    # g_k(x) = h_row with its slack set to zero
    g_rows = [int(i) for i in np.flatnonzero(mask)]
    g_exprs = [sp.expand(problem.h[i].subs({syms[int(slack_idx[i])]: 0})) for i in g_rows]
    signs = np.array([sign[i] for i in g_rows], dtype=float)
    h_eq = [problem.h[i] for i in range(problem.m) if not mask[i]]

    lb = np.asarray(problem.lb, float).copy()
    ub = np.asarray(problem.ub, float).copy()

    # Simple-bound rows tighten lb/ub; they are dropped only when the target
    # variable is pinned, or when native-bound mode is on.
    keep_row = np.ones(len(g_exprs), dtype=bool)
    for k, g in enumerate(g_exprs):
        involved = [j for j, s in enumerate(syms)
                    if not is_slack[j] and g.has(s)]
        if len(involved) != 1:
            continue
        j = involved[0]
        d = sp.diff(g, syms[j])
        if d.free_symbols:
            continue
        c1 = float(d)
        if abs(abs(c1) - 1.0) > 1e-12:
            continue
        c0 = -float(g.subs({syms[j]: 0}))
        if signs[k] > 0:
            if c1 > 0:
                ub[j] = min(ub[j], c0)
            else:
                lb[j] = max(lb[j], -c0)
        else:
            if c1 > 0:
                lb[j] = max(lb[j], c0)
            else:
                ub[j] = min(ub[j], -c0)
        if abs(ub[j] - lb[j]) < 1e-10:
            keep_row[k] = False

    # Keep g_rows aligned with the rows that actually survive into the reduced
    # problem -- dropped bound rows would otherwise shift every later index.
    g_rows = [r for r, k in zip(g_rows, keep_row) if k]
    g_exprs = [g for g, k in zip(g_exprs, keep_row) if k]
    signs = signs[keep_row]

    # Pinned free variables are substituted out entirely.
    pinned = (~is_slack) & (np.abs(ub - lb) < 1e-10)
    pinned_val = np.where(pinned, 0.5 * (lb + ub), 0.0)
    if pinned.any():
        sub = {syms[j]: sp.Float(float(pinned_val[j])) for j in np.flatnonzero(pinned)}
        f_red = problem.f.subs(sub)
        h_eq = [e.subs(sub) for e in h_eq]
        g_exprs = [e.subs(sub) for e in g_exprs]
    else:
        f_red = problem.f

    kept = (~is_slack) & (~pinned)
    keep_idx = np.flatnonzero(kept)
    names = [problem.var_names[j] for j in keep_idx]

    reduced = Problem(name=f"{problem.name}::fb", var_names=names,
                      f=f_red, h=list(h_eq) + list(g_exprs),
                      lb=lb[keep_idx], ub=ub[keep_idx], x0=np.zeros(len(names)))
    return FBSystem(reduced=reduced, n_eq=len(h_eq), signs=signs, kept=kept,
                    pinned=pinned, pinned_val=pinned_val,
                    slack_of_row={int(i): int(slack_idx[i]) for i in g_rows},
                    g_rows=g_rows, lb_user=lb[keep_idx], ub_user=ub[keep_idx])


def solve_fb(problem: Problem, opts: Options, x0: np.ndarray,
             n_starts: int = 1, rng: Optional[MatlabRandom] = None) -> SolveResult:
    """Multistart FB solve.  ``x0`` is a single column in the original space."""
    t0 = time.time()
    sysm = build_fb_system(problem, opts)
    red = sysm.reduced
    nu = red.n                      # free user variables
    n_eq = sysm.n_eq
    n_in = len(sysm.signs)
    backend = get_backend(red, opts.backend, switch_n=opts.backend_switch_n)

    if rng is None:
        rng = MatlabRandom(opts.ms_seed if opts.ms_seed >= 0 else 0)

    x0 = np.asarray(x0, float).ravel()[: problem.n]
    lb_f = np.asarray(problem.lb, float)
    ub_f = np.asarray(problem.ub, float)
    starts_full = (scatter_starts(x0, lb_f, ub_f, n_starts, opts.ms_scale, rng)
                   if n_starts > 1 else x0.reshape(-1, 1))
    starts = starts_full[sysm.kept, :]

    rank_rule = opts.rank_rule
    if rank_rule == "auto":
        rank_rule = "sparse" if nu >= opts.backend_switch_n else "dense"

    n_lam = 1 + n_eq                # dummy row + equality rows
    idx_th = slice(1, 1 + nu)
    idx_lam = slice(1 + nu, 1 + nu + n_lam)
    idx_mu = slice(1 + nu + n_lam, 1 + nu + n_lam + n_in)
    nZ = 1 + nu + n_lam + n_in
    eps2 = opts.fb_eps ** 2

    def residual_and_jac(z, want_jac=True):
        xs = z[0]
        th = z[idx_th]
        lam = z[idx_lam]
        mu = z[idx_mu]
        # combined multipliers for the reduced problem's h = [h_eq, g]
        lam_comb = np.concatenate([lam[1:], sysm.signs * mu])
        fval, gf, hall, Jall, Hlag = backend.kkt(th, lam_comb)
        h_eq = hall[:n_eq]
        g = hall[n_eq:]
        Jeq = Jall[:n_eq]
        Jg = Jall[n_eq:]

        b = -sysm.signs * g
        root = np.sqrt(mu ** 2 + b ** 2 + eps2)
        phi = mu + b - root

        R = np.empty(nZ)
        R[0] = 2.0 * (xs - _XT) + lam[0]
        R[idx_th] = gf + (Jeq.T @ lam[1:] if n_eq else 0.0) \
                       + (Jg.T @ (sysm.signs * mu) if n_in else 0.0)
        R[1 + nu] = xs - _XT
        R[2 + nu: 1 + nu + n_lam] = h_eq
        R[idx_mu] = phi
        if not want_jac:
            return R, None, fval

        J = np.zeros((nZ, nZ))
        J[0, 0] = 2.0
        J[0, 1 + nu] = 1.0
        J[1 + nu, 0] = 1.0
        J[idx_th, idx_th] = Hlag
        if n_eq:
            J[idx_th, 2 + nu: 1 + nu + n_lam] = Jeq.T
            J[2 + nu: 1 + nu + n_lam, idx_th] = Jeq
        if n_in:
            J[idx_th, idx_mu] = (sysm.signs * Jg.T)
            # dphi/dmu and dphi/dtheta
            J[idx_mu, idx_mu] = np.diag(1.0 - mu / root)
            J[idx_mu, idx_th] = ((1.0 - b / root) * (-sysm.signs))[:, None] * Jg
        return R, J, fval

    runs: list[RunResult] = []
    for s in range(starts.shape[1]):
        th = _clip_matlab(starts[:, s], sysm.lb_user, sysm.ub_user)
        z = np.concatenate([[_XT], th, np.zeros(n_lam), 0.01 * np.ones(n_in)])
        converged = False
        last_maxr = np.inf
        stag = 0
        maxr = np.inf
        k = 0
        for k in range(1, opts.maxIter + 1):
            R, J, _ = residual_and_jac(z)
            maxr = _mmax(np.abs(R))
            if maxr < opts.tol_r:
                converged = True
                break
            dz = min_norm_solve(J, R, method=opts.step_method,
                                tikhonov_mu=opts.tikhonov_mu, rule=rank_rule,
                                svd_tol_rule=opts.svd_tol_rule)
            alpha = 1.0
            Phi0 = 0.5 * float(R @ R)
            accepted = False
            for _ in range(opts.bt_max):
                z_try = z - alpha * dz
                z_try[idx_th] = _clip_matlab(z_try[idx_th], sysm.lb_user, sysm.ub_user)
                R_try, _, _ = residual_and_jac(z_try, want_jac=False)
                if 0.5 * float(R_try @ R_try) <= Phi0 * (1.0 - opts.bt_c1 * alpha):
                    z = z_try
                    accepted = True
                    break
                alpha *= opts.bt_rho
            if not accepted:
                z[idx_th] = _clip_matlab(z[idx_th] + 0.01 * rng.randn(nu),
                                         sysm.lb_user, sysm.ub_user)
            stag = stag + 1 if abs(maxr - last_maxr) < opts.stag_tol else 0
            last_maxr = maxr
            if stag > opts.stag_max:
                z[idx_th] = _clip_matlab(z[idx_th] + opts.kick_size * rng.randn(nu),
                                         sysm.lb_user, sysm.ub_user)
                z[idx_mu] = 0.5 * z[idx_mu]
                stag = 0

        th_final = z[idx_th]
        mu_final = z[idx_mu]

        # rebuild the full-length point, including reconstructed slacks
        theta_full = np.zeros(problem.n)
        theta_full[sysm.kept] = th_final
        theta_full[sysm.pinned] = sysm.pinned_val[sysm.pinned]
        _, _, hall, _, _ = backend.kkt(th_final, np.zeros(red.m))
        for k_row, row in enumerate(sysm.g_rows):
            j = sysm.slack_of_row[row]
            gv = float(hall[n_eq + k_row])
            # h_row = g + sigma * s**2 = 0  ->  s = sqrt(-sigma * g)
            val = -problem.ineq_row_sign[row] * gv
            theta_full[j] = float(np.sqrt(val)) if val > 0 else 0.0

        r = RunResult(theta=theta_full, fval=float(backend.f(th_final)),
                      converged=converged, iterations=k, max_r=maxr)
        # FB certification: mu >= 0, inequality feasible, complementarity small
        if converged and n_in:
            g_now = hall[n_eq:]
            b = -sysm.signs * g_now
            r.min_lam_strict = _mmin(mu_final)
            r.dual_feas_strict = bool(
                _mmin(mu_final) >= -opts.dual_feas_tol
                and np.all(b >= -opts.tol_h)
                and np.all(np.abs(mu_final * b) <= max(opts.dual_feas_tol, 1e-6)))
            r.dual_feas = r.dual_feas_strict
        elif converged:
            r.dual_feas = r.dual_feas_strict = True
            r.min_lam_strict = 0.0
        runs.append(r)

    fvals = np.array([r.fval for r in runs])
    conv = np.array([r.converged for r in runs])
    search = np.where(conv, fvals, np.inf)
    best = int(np.argmin(search)) if np.isfinite(search).any() else int(np.argmin(fvals))
    return SolveResult(theta=runs[best].theta, fval=float(fvals[best]),
                       converged=bool(conv[best]), runs=runs, best_run=best,
                       elapsed=time.time() - t0, solver_used="kronos-fb",
                       n_ineq_rows=n_in)
