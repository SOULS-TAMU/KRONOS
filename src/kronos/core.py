"""The KRONOS KKT iteration.

A line-by-line port of ``run_one_start`` / ``solve_nlp`` from the reference
MATLAB implementation.  Where the MATLAB code and the published algorithm box
disagree, **the code wins** -- the published results were produced by the code.
Those places are marked ``PARITY:`` below:

PARITY 1  An internal dummy variable ``xs`` is appended with ``(xs - 1)**2``
          added to the objective and the row ``xs - 1 = 0`` added to ``H``.
          It is zero at the solution but enlarges the KKT matrix by one row and
          column, which changes the minimum-norm step.  Controlled by
          ``Options.use_dummy_variable`` (default True).

PARITY 2  After feasibility restoration moves ``p``, the Newton step reuses the
          residual ``r_k`` and Jacobian ``J_k`` evaluated *before* restoration.
          The algorithm box says they are reconstructed; neither MATLAB backend
          does so.

PARITY 3  The Armijo test is ``Phi_try <= Phi0 * (1 - c1*alpha)``, i.e. a
          *relative* decrease, not the absolute ``Phi0 - c1*alpha*||r||^2`` of
          the algorithm box.

PARITY 4  The incumbent update at the foot of the loop pairs the objective
          value from the *start* of the iteration with the iterate from the
          *end* of it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .linalg import lsqminnorm, min_norm_solve, null_space
from .matlab_rng import MatlabRandom
from .options import Options

__all__ = ["RunResult", "SolveResult", "run_one_start", "solve_multistart"]

_XT = 1.0


# MATLAB's max/min omit NaN; NumPy's propagate it.  The distinction is not
# cosmetic here: at a point where some gradient entries are NaN (0/0 at a
# non-differentiable origin, say) MATLAB's max(abs(r)) reports the largest
# *finite* residual, so the run can converge.  NumPy would report NaN and the
# run could never terminate.  Reproduce MATLAB.
def _mmax(x) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 0.0
    with np.errstate(all="ignore"):
        finite = x[~np.isnan(x)]
    return float(np.max(finite)) if finite.size else float("nan")


def _mmin(x) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 0.0
    with np.errstate(all="ignore"):
        finite = x[~np.isnan(x)]
    return float(np.min(finite)) if finite.size else float("nan")


@dataclass
class RunResult:
    """Outcome of a single start."""

    theta: np.ndarray
    fval: float = np.inf
    converged: bool = False
    iterations: int = 0
    max_r: float = np.inf
    max_h: float = np.inf
    error: str = ""
    dual_feas: bool = False
    min_lam_ineq: float = np.nan
    dual_feas_strict: bool = False
    min_lam_strict: float = np.nan
    n_promoted: int = 0
    sosc_pass: bool = False
    sosc_measured: bool = False
    """Whether the second-order test actually ran.  The Fischer-Burmeister
    fallback does not form a reduced Hessian, so its runs leave this False --
    'not measured' is not the same as 'failed'."""
    lam_min_red: float = np.nan
    lam_user: np.ndarray = None          # type: ignore[assignment]
    elapsed: float = 0.0

    @property
    def kkt_certified(self) -> bool:
        return bool(self.converged and self.dual_feas_strict)


@dataclass
class SolveResult:
    """Outcome of a multistart solve."""

    theta: np.ndarray
    fval: float
    converged: bool
    runs: list[RunResult] = field(default_factory=list)
    best_run: Optional[int] = None
    elapsed: float = 0.0
    solver_used: str = ""
    n_ineq_rows: int = 0
    info: dict = field(default_factory=dict)

    # -- aggregate views used by the benchmark harness -------------------
    @property
    def all_fvals(self) -> np.ndarray:
        return np.array([r.fval for r in self.runs])

    @property
    def all_conv(self) -> np.ndarray:
        return np.array([r.converged for r in self.runs], dtype=bool)

    @property
    def all_kkt(self) -> np.ndarray:
        return np.array([r.kkt_certified for r in self.runs], dtype=bool)

    @property
    def all_sosc(self) -> np.ndarray:
        return np.array([r.sosc_pass for r in self.runs], dtype=bool)

    @property
    def n_conv(self) -> int:
        """Runs that converged **and** are KKT-certified.

        A run whose KKT residual reaches zero but whose inequality multipliers
        come out with the wrong sign is a stationary point of the reformulated
        problem, not a solution of yours.  Counting it as "converged" would
        overstate the result, so ``n_conv`` requires certification.  The looser
        count is ``n_residual_conv``.
        """
        return int(self.all_kkt.sum())

    @property
    def n_kkt(self) -> int:
        """Alias of :attr:`n_conv`."""
        return int(self.all_kkt.sum())

    @property
    def n_residual_conv(self) -> int:
        """Runs that met the residual test, certified or not."""
        return int(self.all_conv.sum())

    @property
    def all_sosc_measured(self) -> np.ndarray:
        return np.array([r.sosc_measured for r in self.runs], dtype=bool)

    @property
    def n_local(self) -> int:
        """Runs proven to be strict local minima (KKT-certified + SOSC)."""
        return int((self.all_kkt & self.all_sosc).sum())

    @property
    def n_stationary(self) -> int:
        """KKT-certified runs where the second-order test ran and did not pass."""
        return int((self.all_kkt & self.all_sosc_measured & ~self.all_sosc).sum())

    @property
    def n_sosc_unmeasured(self) -> int:
        """KKT-certified runs where the second-order test was not performed."""
        return int((self.all_kkt & ~self.all_sosc_measured).sum())

    def global_hits(self, fstar: Optional[float] = None) -> int:
        """Certified runs that reached the known optimum, if one is known."""
        if fstar is None:
            fstar = self.info.get("fstar")
        if fstar is None or not np.isfinite(fstar):
            return 0
        tol = max(1e-4, 1e-3 * max(1.0, abs(fstar)))
        return int((self.all_kkt & (np.abs(self.all_fvals - fstar) <= tol)).sum())

    def summary(self, fstar: Optional[float] = None,
                show_uncertified: bool = False) -> str:
        """A human-readable report of the solve.

        "Converged" means **KKT-certified**: the residual test passed *and* the
        multipliers have the right signs.  Pass ``show_uncertified=True`` to
        also see runs that merely met the residual test.

        If the problem carries a known optimum (or one is passed as ``fstar``),
        the report adds how many runs reached it, both as a fraction of all
        runs and as a fraction of the converged ones.

        Second-order information is not printed but remains available:
        ``n_local``, ``n_stationary``, ``n_sosc_unmeasured``, and per run
        ``sosc_pass`` / ``sosc_measured`` / ``lam_min_red``.
        """
        if fstar is None:
            fstar = self.info.get("fstar")
        K = len(self.runs)
        best = self.runs[self.best_run] if self.best_run is not None and self.runs else None
        L = []
        L.append("=" * 62)
        L.append(f"  KRONOS  |  {self.info.get('problem', 'problem')}"
                 f"   n={self.info.get('n', '?')}  m={self.info.get('m', '?')}")
        L.append("=" * 62)
        L.append(f"  solver              : {self.solver_used}")
        L.append(f"  objective           : {self.fval:.10g}")
        if fstar is not None and np.isfinite(fstar):
            L.append(f"  known optimum f*    : {fstar:.10g}   (gap {abs(self.fval - fstar):.3e})")
        L.append("  ---- multistart ----")
        L.append(f"  runs                : {K}")
        L.append(f"  converged           : {self.n_conv}/{K}  ({100*self.n_conv/max(K,1):.1f}%)"
                 f"   [KKT-certified]")
        if show_uncertified:
            nr = self.n_residual_conv
            L.append(f"  residual-converged  : {nr}/{K}  ({100*nr/max(K,1):.1f}%)"
                     f"   [not necessarily certified]")
        if fstar is not None and np.isfinite(fstar):
            g = self.global_hits(fstar)
            ratio = (100 * g / self.n_conv) if self.n_conv else float("nan")
            L.append(f"  reached f*          : {g}/{K}  ({100*g/max(K,1):.1f}%)")
            L.append(f"  f* / converged      : {ratio:.1f}%")
        if best is not None:
            L.append("  ---- best run ----")
            L.append(f"  certified           : {best.kkt_certified}"
                     f"   (min signed multiplier {best.min_lam_strict:.3e})")
            L.append(f"  iterations          : {best.iterations}")
            L.append(f"  final |KKT residual|: {best.max_r:.3e}")
            L.append(f"  final |constraints| : {best.max_h:.3e}")
        L.append("  ---- timing ----")
        L.append(f"  total               : {self.elapsed:.3f} s"
                 f"   ({self.elapsed/max(K,1):.3f} s/run)")
        L.append("=" * 62)
        return "\n".join(L)

    def __repr__(self) -> str:
        return (f"SolveResult(fval={self.fval:.8g}, converged={self.converged}, "
                f"n_conv={self.n_conv}/{len(self.runs)}, n_kkt={self.n_kkt})")


# ======================================================================
#  KKT assembly
# ======================================================================
class _Assembler:
    """Builds the KKT residual and Jacobian around a backend.

    The dummy variable's contribution is constant, so it is written in
    directly rather than differentiated (identical values, far cheaper).
    """

    def __init__(self, backend, n: int, m: int, use_dummy: bool, f_scale: float = 1.0):
        self.b = backend
        self.f_scale = f_scale
        self.n = n
        self.m = m
        self.nX = 1 if use_dummy else 0
        self.nP = self.nX + n
        self.nH = self.nX + m
        self.Ntot = self.nP + self.nH

    def __call__(self, p: np.ndarray, lam: np.ndarray):
        """Return ``(R, J, h, Jh_full, fval)``."""
        nX, n, m, nP, nH, Ntot = self.nX, self.n, self.m, self.nP, self.nH, self.Ntot
        th = p[nX:nX + n]
        lam_u = lam[nX:]

        with np.errstate(all="ignore"):
            fval, gf, hu, Ju, Htt = self.b.kkt(th, lam_u, self.f_scale)

        R = np.empty(Ntot)
        J = np.zeros((Ntot, Ntot))
        Jh_full = np.zeros((nH, nP))

        with np.errstate(all="ignore"):
            gt = gf + (Ju.T @ lam_u if m else 0.0)
        J[nX:nP, nX:nP] = Htt
        if m:
            J[nX:nP, nP + nX:] = Ju.T
            J[nP + nX:, nX:nP] = Ju
            Jh_full[nX:, nX:] = Ju

        if nX:
            xs = p[0]
            R[0] = 2.0 * (xs - _XT) + lam[0]
            R[1:nP] = gt
            R[nP] = xs - _XT
            R[nP + 1:] = hu
            J[0, 0] = 2.0
            J[0, nP] = 1.0
            J[nP, 0] = 1.0
            Jh_full[0, 0] = 1.0
            h = np.empty(nH)
            h[0] = xs - _XT
            h[1:] = hu
        else:
            R[:nP] = gt
            R[nP:] = hu
            h = hu

        return R, J, h, Jh_full, fval

    def h_and_Jh(self, p: np.ndarray):
        """Constraints and their Jacobian only -- used inside restoration,
        where the Hessian of the Lagrangian is not needed."""
        nX, n, m, nP, nH = self.nX, self.n, self.m, self.nP, self.nH
        th = p[nX:nX + n]
        hu, Ju = self.b.h_only(th) if hasattr(self.b, "h_only") else (self.b.h(th), self.b.Jh(th))
        Jh_full = np.zeros((nH, nP))
        if m:
            Jh_full[nX:, nX:] = Ju
        if nX:
            h = np.empty(nH)
            h[0] = p[0] - _XT
            h[1:] = hu
            Jh_full[0, 0] = 1.0
        else:
            h = hu
        return h, Jh_full


def _clip_matlab(x: np.ndarray, lo, hi) -> np.ndarray:
    """``min(max(x, lo), hi)`` with MATLAB's NaN semantics.

    MATLAB's two-argument ``max``/``min`` return the non-NaN operand, so
    ``min(max(NaN, lo), hi)`` is ``lo`` -- projection *sanitises* NaN to the
    lower bound and the run continues from a finite point.  ``np.clip`` would
    propagate the NaN and strand the run forever.  Problems whose starting
    point is itself NaN (hong_done's x0 is ``zeros ./ sum(zeros)``) depend on
    this.
    """
    x = np.where(np.isnan(x), lo, x)
    return np.clip(x, lo, hi)


def _proj_bnd(p: np.ndarray, nX: int, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """Clamp the dummy to [-100, 100] and theta to the numerical safety box."""
    if nX:
        p[:nX] = _clip_matlab(p[:nX], -100.0, 100.0)
    sl = slice(nX, nX + lb.size)
    p[sl] = _clip_matlab(p[sl], lb, ub)
    return p


# ======================================================================
#  Single start
# ======================================================================
def run_one_start(
    assemble: _Assembler,
    p0: np.ndarray,
    lam0: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    opts: Options,
    ineq_row_mask: np.ndarray,
    ineq_row_sign: np.ndarray,
    slack_var_idx: np.ndarray,
    rng: MatlabRandom,
    show_iters: bool = False,
) -> RunResult:
    """One Newton/KKT solve from a single starting point."""
    nX, nP, nH, Ntot = assemble.nX, assemble.nP, assemble.nH, assemble.Ntot
    n, m = assemble.n, assemble.m

    p = np.array(p0, dtype=float)
    lam = np.array(lam0, dtype=float)

    res = RunResult(theta=p[nX:nX + n].copy())
    res.lam_user = np.full(m, np.nan)

    sse_best = np.inf
    p_best = p.copy()
    last_maxr = np.inf
    stag_count = 0
    kkt_kicks_done = 0

    rank_rule = opts.rank_rule
    if rank_rule == "auto":
        rank_rule = "sparse" if n >= opts.backend_switch_n else "dense"

    def step(A, rhs):
        """The minimum-norm solve, used for both the Newton step and the
        feasibility-restoration step -- the two places the algorithm takes a
        pseudoinverse."""
        return min_norm_solve(A, rhs, method=opts.step_method,
                              tikhonov_mu=opts.tikhonov_mu, rule=rank_rule,
                              svd_tol_rule=opts.svd_tol_rule)

    has_ineq = m > 0 and bool(ineq_row_mask.any())
    sign_ineq = ineq_row_sign[ineq_row_mask] if has_ineq else np.zeros(0)

    promoted_rows = np.zeros(m, dtype=bool)
    promotion_done = False
    promotion_bailed = False
    promote_slack: list[int] = []
    promotion_iter = 0
    p_pre = lam_pre = None

    def signed_penalty(lam_now: np.ndarray) -> float:
        if opts.sign_bias_rho <= 0 or not has_ineq:
            return 0.0
        signed = sign_ineq * lam_now[nX:][ineq_row_mask]
        viol = np.maximum(0.0, -signed)
        return float(opts.sign_bias_rho * np.sum(viol ** 2))

    k = 0
    for k in range(1, opts.maxIter + 1):
        R, J, h, Jh_full, curr_sse = assemble(p, lam)
        max_r = _mmax(np.abs(R))
        max_h = _mmax(np.abs(h))

        if show_iters and (k == 1 or k % opts.print_freq == 0):
            print(f"  {k:4d}  | {curr_sse:12.4e} | {max_r:12.4e} | {max_h:12.4e}")

        # (G) promoted system failed to converge -> restore and accept sign-bad
        if promotion_done and (k - promotion_iter) > opts.max_iter_after_promotion:
            p, lam = p_pre.copy(), lam_pre.copy()
            promoted_rows[:] = False
            promote_slack = []
            promotion_done = False
            promotion_bailed = True
            kkt_kicks_done = opts.max_kkt_kicks
            continue

        # ---------------- convergence ----------------
        if max_h < opts.tol_h and max_r < opts.tol_r:
            active = ineq_row_mask & (~promoted_rows)
            res.lam_user = lam[nX:].copy()
            if active.any():
                signed_active = ineq_row_sign[active] * lam[nX:][active]
                min_lam = _mmin(signed_active)
                dual_ok = bool(np.all(signed_active >= -opts.dual_feas_tol))
            else:
                signed_active = np.zeros(0)
                min_lam, dual_ok = 0.0, True

            # (A) rejection kick on dual-infeasible multipliers
            if opts.enforce_kkt_sign and not dual_ok and kkt_kicks_done < opts.max_kkt_kicks:
                if show_iters:
                    print(f"  {k:4d}  | REJECTED (min sign*lam={min_lam:.2e}, "
                          f"kick {kkt_kicks_done + 1}/{opts.max_kkt_kicks})")
                bad = np.zeros(m, dtype=bool)
                bad[active] = signed_active < 0
                lam_u = lam[nX:]
                lam_u[bad] = opts.lam0_magnitude * ineq_row_sign[bad]
                lam[nX:] = lam_u
                p = p + opts.kick_size * rng.randn(p.size)
                stag_count = 0
                kkt_kicks_done += 1
                continue

            # (G) active-set promotion once the kick budget is spent
            if (opts.enforce_kkt_sign and not dual_ok and opts.promote_on_kick_exhaust
                    and not promotion_done and not promotion_bailed):
                p_pre, lam_pre = p.copy(), lam.copy()
                bad = np.zeros(m, dtype=bool)
                bad[active] = signed_active < 0
                for r_i in np.flatnonzero(bad):
                    j = int(slack_var_idx[r_i])
                    if j >= 0:
                        p[nX + j] = 0.0
                        promote_slack.append(j)
                        promoted_rows[r_i] = True
                promotion_done = True
                promotion_iter = k
                kkt_kicks_done = 0
                stag_count = 0
                continue

            sse_best = curr_sse
            p_best = p.copy()
            res.converged = True
            res.dual_feas = dual_ok
            res.min_lam_ineq = min_lam

            # strict check over every inequality row, promoted included
            if has_ineq:
                signed_all = ineq_row_sign[ineq_row_mask] * lam[nX:][ineq_row_mask]
                res.min_lam_strict = _mmin(signed_all)
                res.dual_feas_strict = bool(np.all(signed_all >= -opts.dual_feas_tol))
            else:
                res.dual_feas_strict = True
                res.min_lam_strict = 0.0
            res.n_promoted = int(promoted_rows.sum())

            # ---- SOSC on the reduced Hessian ----
            if opts.check_sosc:
                Hb = J[:nP, :nP]
                Hb = 0.5 * (Hb + Hb.T)
                Ac = J[nP:, :nP]
                # MATLAB's null/eig return NaN on a non-finite input rather
                # than raising; a NaN reduced Hessian simply fails SOSC.
                if not np.all(np.isfinite(Ac)):
                    res.sosc_pass = False
                    res.lam_min_red = np.nan
                else:
                    Z = null_space(Ac)
                    if Z.shape[1] == 0:
                        res.sosc_pass = True
                        res.sosc_measured = True
                        res.lam_min_red = np.inf
                    elif not np.all(np.isfinite(Hb)):
                        res.sosc_pass = False
                        res.lam_min_red = np.nan
                    else:
                        Hr = Z.T @ Hb @ Z
                        Hr = 0.5 * (Hr + Hr.T)
                        ev = np.real(np.linalg.eigvals(Hr))
                        res.lam_min_red = _mmin(ev)
                        res.sosc_pass = bool(res.lam_min_red > opts.sosc_tol)
                        res.sosc_measured = True
            break

        # ---------------- stagnation ----------------
        delta = abs(max_r - last_maxr) if np.isfinite(max_r) and np.isfinite(last_maxr) else np.inf
        stag_count = stag_count + 1 if delta < opts.stag_tol else 0
        last_maxr = max_r

        if stag_count > opts.stag_max:
            p = p + opts.kick_size * rng.randn(p.size)
            lam = 0.5 * lam
            if promote_slack:
                p[nX + np.asarray(promote_slack)] = 0.0
            stag_count = 0

        # ---------------- feasibility restoration ----------------
        # PARITY 2: R and J are deliberately NOT rebuilt afterwards.
        if max_h > opts.feas_tol and not opts.disable_restoration:
            h_i, Jh_i = h, Jh_full
            for _ in range(opts.feas_iters):
                p = p - opts.feas_step * step(Jh_i, h_i)
                p = _proj_bnd(p, nX, lb, ub)
                if promote_slack:
                    p[nX + np.asarray(promote_slack)] = 0.0
                h_i, Jh_i = assemble.h_and_Jh(p)
                if _mmax(np.abs(h_i)) < 1e-7:
                    break

        # ---------------- minimum-norm Newton step ----------------
        dz = step(J, R)
        if promote_slack:
            dz[nX + np.asarray(promote_slack)] = 0.0

        # ---------------- projected backtracking line search ----------------
        alpha = 1.0
        with np.errstate(over="ignore", invalid="ignore"):
            Phi0 = 0.5 * float(R @ R) + signed_penalty(lam)
        z = np.concatenate([p, lam])
        accepted = False
        for _ in range(opts.bt_max):
            z_try = z - alpha * dz
            p_try = _proj_bnd(z_try[:nP].copy(), nX, lb, ub)
            lam_try = z_try[nP:]
            R_try, _, _, _, _ = assemble(p_try, lam_try)
            with np.errstate(over="ignore", invalid="ignore"):
                Phi_try = 0.5 * float(R_try @ R_try) + signed_penalty(lam_try)
            if not np.isfinite(Phi_try):
                Phi_try = np.inf
            # PARITY 3: relative Armijo condition, as in the MATLAB code.
            if Phi_try <= Phi0 * (1.0 - opts.bt_c1 * alpha):
                p, lam = p_try, lam_try
                if opts.project_ineq_sign and has_ineq:
                    lam_u = lam[nX:]
                    signed = sign_ineq * lam_u[ineq_row_mask]
                    lam_u[ineq_row_mask] = sign_ineq * np.maximum(signed, 0.0)
                    lam[nX:] = lam_u
                accepted = True
                break
            alpha *= opts.bt_rho

        if not accepted:
            lam = 0.8 * lam

        # PARITY 4: old objective, new iterate.
        if max_h < opts.tol_h and curr_sse < sse_best:
            sse_best = curr_sse
            p_best = p.copy()

    if not np.isfinite(sse_best):
        p_best = p.copy()
        sse_best = assemble.b.f(p[nX:nX + n])

    res.theta = p_best[nX:nX + n].copy()
    res.fval = float(sse_best)
    res.iterations = k
    res.max_r = max_r
    res.max_h = max_h
    return res


# ======================================================================
#  Multistart driver  (port of the outer part of solve_nlp.m)
# ======================================================================
def scatter_starts(
    x0: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    n_starts: int,
    ms_scale: float,
    rng: MatlabRandom,
    ms_x0: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Generate the multistart scatter exactly as the reference does.

    Variables whose bound is a mere numerical clamp (``|bound| >= 1e5``) are
    scattered in a box of half-width ``5 * ms_scale`` around the centre; the
    rest are scattered across their real bounds.  The user's ``x0`` is kept as
    the first start.
    """
    n = x0.size
    center = np.asarray(ms_x0, float).ravel() if ms_x0 is not None else x0.copy()
    real_lb = lb > -1e5
    real_ub = ub < 1e5
    lb_use = np.where(real_lb, lb, center - 5.0 * ms_scale)
    ub_use = np.where(real_ub, ub, center + 5.0 * ms_scale)
    U = np.asarray(rng.rand(n, n_starts)).reshape(n, n_starts)
    starts = lb_use[:, None] + (ub_use - lb_use)[:, None] * U
    starts[:, 0] = center
    return starts


def solve_multistart(
    problem,
    backend,
    opts: Options,
    x0: Optional[np.ndarray] = None,
    objective_override=None,
    rng: Optional[MatlabRandom] = None,
) -> SolveResult:
    """Run the KKT iteration from one or many starts and pick the best.

    ``x0`` may be a single column or an ``(n, K)`` matrix of pre-generated
    starts; a matrix is used as-is, which is how stage 2 receives Adam-warmed
    columns.
    """
    import time
    t0 = time.time()

    n = problem.n
    m = problem.m
    lb = np.asarray(problem.lb, float)
    ub = np.asarray(problem.ub, float)
    if rng is None:
        rng = MatlabRandom(opts.ms_seed if opts.ms_seed >= 0 else 0)

    X0 = problem.x0 if x0 is None else np.asarray(x0, float)
    X0 = X0.reshape(n, -1) if X0.ndim > 1 else X0.reshape(n, 1)

    if X0.shape[1] > 1:
        starts = X0
    elif opts.multi_start and opts.ms_num_starts > 1:
        starts = scatter_starts(X0[:, 0], lb, ub, opts.ms_num_starts,
                                opts.ms_scale, rng, opts.ms_x0)
    else:
        starts = X0
    n_starts = starts.shape[1]

    assemble = _Assembler(backend, n, m, opts.use_dummy_variable,
                          1.0 if objective_override is None else float(objective_override))
    nX, nH = assemble.nX, assemble.nH

    mask = problem.ineq_row_mask
    sign = problem.ineq_row_sign
    slack_idx = problem.slack_var_index

    runs: list[RunResult] = []
    for s in range(n_starts):
        p0 = np.empty(assemble.nP)
        if nX:
            p0[0] = _XT
        p0[nX:] = starts[:, s]
        p0 = _proj_bnd(p0, nX, lb, ub)

        if opts.lam0_sign != 0:
            lam0 = opts.lam0_magnitude * np.sign(opts.lam0_sign) * np.ones(nH)
        elif opts.sign_flip_multistart and n_starts > 1:
            lam0 = opts.lam0_magnitude * ((s % 2 == 0) * 2 - 1) * np.ones(nH)
        else:
            lam0 = np.zeros(nH)

        import time as _t
        t1 = _t.time()
        r = run_one_start(assemble, p0, lam0, lb, ub, opts,
                          mask, sign, slack_idx, rng,
                          show_iters=opts.verbose and n_starts == 1)
        r.elapsed = _t.time() - t1
        runs.append(r)

        if opts.verbose and opts.ms_show_runs and n_starts > 1:
            tag = (f"Converged in {r.iterations:4d} steps" if r.converged
                   else f"Not converged ({r.iterations} steps)")
            print(f"  Run {s + 1:3d} | Obj: {r.fval:12.4e} | {tag}")

    # ---- global best: converged only, min fval, ties broken by iterations
    fvals = np.array([r.fval for r in runs])
    conv = np.array([r.converged for r in runs], dtype=bool)
    # Only finite objectives from converged runs are eligible, as in MATLAB.
    search = np.where(conv & np.isfinite(fvals), fvals, np.inf)
    best_f = float(np.min(search)) if search.size else np.inf

    if not np.isfinite(best_f):
        best_idx = int(np.argmin([r.max_h for r in runs])) if runs else None
        theta = runs[best_idx].theta if best_idx is not None else np.zeros(n)
        fval = backend.f(theta) if best_idx is not None else np.inf
        converged = False
    else:
        tied = np.flatnonzero(search == best_f)
        best_idx = int(tied[int(np.argmin([runs[i].iterations for i in tied]))])
        theta = runs[best_idx].theta
        fval = best_f
        converged = True

    return SolveResult(
        theta=np.asarray(theta, float),
        fval=float(fval),
        converged=converged,
        runs=runs,
        best_run=best_idx,
        elapsed=time.time() - t0,
        solver_used=f"kronos[{backend.name}, step={opts.step_method}]",
        n_ineq_rows=int(mask.sum()),
    )
