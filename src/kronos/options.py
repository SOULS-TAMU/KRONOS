"""Solver options.

``Options.describe()`` lists every setting with its default and purpose.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict, replace
from typing import Any, Literal, Optional

__all__ = ["Options"]


@dataclass
class Options:
    """KRONOS solver settings."""

    # -- Newton solver ---------------------------------------------------
    maxIter: int = 2000
    tol_r: float = 1e-5           # KKT residual tolerance, max|r| < tol_r
    tol_h: float = 1e-5           # feasibility tolerance, max|h| < tol_h
    stag_tol: float = 1e-4        # |max|r|_k - max|r|_{k-1}| below this = stagnant
    stag_max: int = 25            # stagnant steps before a random kick
    kick_size: float = 0.1        # magnitude of the random kick
    feas_tol: float = 1e-3        # max|h| above this triggers restoration
    feas_iters: int = 15          # inner restoration iterations
    feas_step: float = 0.5        # restoration step fraction
    bt_max: int = 30              # max backtracking steps
    bt_rho: float = 0.5           # backtracking contraction
    bt_c1: float = 1e-4           # Armijo constant

    # -- output ----------------------------------------------------------
    verbose: bool = False
    print_freq: int = 5
    output_file: str = ""

    # -- multistart ------------------------------------------------------
    multi_start: bool = False
    ms_num_starts: int = 25
    ms_seed: int = 42
    ms_x0: Optional[Any] = None
    ms_scale: float = 1.5
    ms_show_runs: bool = True
    force_single_start: bool = False

    # -- Adam warm-up (Stage 0) ------------------------------------------
    use_adam_warmup: bool = True
    adam_iters: int = 200
    adam_lr: float = 5e-2
    adam_rho: float = 10.0
    adam_b1: float = 0.9
    adam_b2: float = 0.999
    adam_eps: float = 1e-8
    adam_verbose: bool = False
    adam_mode: Literal["A", "B", "C"] = "C"
    adam_early_exit: bool = False
    adam_exit_tol_g: float = 1e-3

    # -- pre-feasibility (Stage 1) ---------------------------------------
    use_prefeasibility: bool = True

    # -- KKT certification (Stage 2/3) -----------------------------------
    check_kkt_sign: bool = True
    enforce_kkt_sign: bool = True
    dual_feas_tol: float = 1e-6
    max_kkt_kicks: int = 5
    check_sosc: bool = True
    sosc_tol: float = 1e-6

    # -- multiplier initialisation / sign handling -----------------------
    sign_flip_multistart: bool = False
    lam0_magnitude: float = 0.01
    lam0_sign: int = 0
    sign_bias_rho: float = 0.0
    project_ineq_sign: bool = False
    promote_on_kick_exhaust: bool = False
    max_iter_after_promotion: int = 300

    # -- Fischer-Burmeister fallback -------------------------------------
    fb_eps: float = 1e-6
    fb_enable: bool = True

    # -- known optimum (enables the global-hit metric) -------------------
    fstar: Optional[float] = None

    # -- Python-package additions ----------------------------------------
    backend: Literal["auto", "sympy", "casadi", "jax", "callable"] = "auto"
    backend_switch_n: int = 20
    """Problems with at least this many variables are routed to the
    algorithmic-differentiation backend."""

    step_method: Literal["pinv", "cod", "lstsq", "tikhonov", "backslash"] = "pinv"
    """Minimum-norm least-squares step, used for the Newton direction and for
    feasibility restoration. ``"pinv"`` uses the Moore-Penrose pseudoinverse via
    the SVD; ``"cod"`` uses a complete orthogonal decomposition."""

    svd_tol_rule: Literal["matlab", "numpy", "exact"] = "matlab"
    """Singular-value cutoff for ``step_method="pinv"``: below
    ``tol * sigma_max`` a direction is treated as null.  ``"exact"`` keeps
    every nonzero singular value."""

    rank_rule: Literal["auto", "dense", "sparse"] = "auto"
    """Rank tolerance for ``step_method="cod"``. ``"auto"`` uses ``"dense"``
    below ``backend_switch_n`` variables and ``"sparse"`` at or above it."""

    tikhonov_mu: float = 1e-8
    disable_restoration: bool = False
    use_dummy_variable: bool = True
    """Include the internal variable ``xs``, contributing ``(xs - 1)**2`` to
    the objective and ``xs - 1`` to ``h``. It vanishes at the solution but adds
    a row and column to the KKT matrix, which changes the minimum-norm step."""

    def copy(self, **changes: Any) -> "Options":
        return replace(self, **changes)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Options":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# Self-documentation.  ``Options.describe()`` prints every setting, grouped,
# with its default and what it does -- so users do not have to read source.
# ---------------------------------------------------------------------------
_GROUPS: list[tuple[str, list[tuple[str, str]]]] = [
    ("Convergence", [
        ("tol_r", "converged when max|KKT residual| < tol_r"),
        ("tol_h", "converged when max|constraint violation| < tol_h"),
        ("maxIter", "maximum Newton iterations per run"),
    ]),
    ("Multistart", [
        ("multi_start", "run from many starting points instead of one"),
        ("ms_num_starts", "how many starting points"),
        ("ms_seed", "random seed for the scatter (reproducible)"),
        ("ms_scale", "scatter half-width for variables with no finite bound"),
        ("ms_x0", "centre of the scatter (defaults to the problem's x0)"),
        ("ms_show_runs", "print each run as it finishes"),
        ("force_single_start", "override multi_start and run once"),
    ]),
    ("Line search and robustness", [
        ("bt_max", "maximum backtracking steps"),
        ("bt_rho", "step-size contraction factor per backtrack"),
        ("bt_c1", "Armijo sufficient-decrease constant"),
        ("stag_tol", "residual change below this counts as stagnation"),
        ("stag_max", "stagnant steps tolerated before a random kick"),
        ("kick_size", "magnitude of the random kick"),
        ("feas_tol", "constraint violation above this triggers restoration"),
        ("feas_iters", "inner iterations of feasibility restoration"),
        ("feas_step", "step fraction used during restoration"),
    ]),
    ("Warm-up (stage 0) and pre-feasibility (stage 1)", [
        ("use_adam_warmup", "first-order Adam pass before Newton"),
        ("adam_mode", "how the warm-up is spread over starts: 'A' all, 'B' first only, 'C' full pipeline each"),
        ("adam_iters", "Adam iterations"),
        ("adam_lr", "Adam learning rate"),
        ("adam_rho", "penalty weight on ||h||^2 during warm-up"),
        ("adam_b1", "Adam first-moment decay"),
        ("adam_b2", "Adam second-moment decay"),
        ("adam_eps", "Adam numerical epsilon"),
        ("adam_early_exit", "accept a warm-up point that is already stationary"),
        ("adam_exit_tol_g", "gradient norm below which early exit is allowed"),
        ("adam_verbose", "print Adam progress"),
        ("use_prefeasibility", "run a reduce-||h|| pass before optimising"),
    ]),
    ("Certification", [
        ("check_kkt_sign", "check inequality multiplier signs at convergence"),
        ("enforce_kkt_sign", "reject and retry runs with wrong-signed multipliers"),
        ("max_kkt_kicks", "retries allowed before giving up on the sign check"),
        ("dual_feas_tol", "tolerance on multiplier sign (dual feasibility)"),
        ("check_sosc", "test the second-order condition (verified minimum)"),
        ("sosc_tol", "reduced-Hessian eigenvalue above which SOSC passes"),
        ("fb_enable", "Fischer-Burmeister fallback for wrong-signed runs"),
        ("fb_eps", "smoothing parameter in the Fischer-Burmeister function"),
    ]),
    ("Backend", [
        ("backend", "'auto', 'sympy', 'casadi' or 'jax'"),
        ("backend_switch_n", "variable count at which 'auto' switches to casadi"),
    ]),
    ("Output", [
        ("verbose", "print the iteration log"),
        ("print_freq", "print every N iterations (single-start only)"),
        ("output_file", "CSV path for per-run results ('' = none)"),
        ("fstar", "known optimum, enables the 'reached f*' metric"),
    ]),
    ("Advanced multiplier handling", [
        ("sign_flip_multistart", "alternate the sign of the initial multipliers"),
        ("lam0_magnitude", "magnitude of nonzero initial multipliers"),
        ("lam0_sign", "force initial multiplier sign (+1 / -1, 0 = zero-init)"),
        ("sign_bias_rho", "penalty on wrong-signed multipliers in the merit function"),
        ("project_ineq_sign", "clip wrong-signed multipliers to zero every step"),
        ("promote_on_kick_exhaust", "pin stubborn inequalities as equalities"),
        ("max_iter_after_promotion", "iterations allowed after such a promotion"),
    ]),
]


# Numerical alternatives to the documented step. They remain selectable, but
# are omitted from ``describe()`` so the documented interface matches the
# published method.
_UNDOCUMENTED = {
    "step_method", "svd_tol_rule", "rank_rule", "tikhonov_mu",
    "use_dummy_variable", "disable_restoration",
}


def _describe(only: Optional[str] = None) -> str:
    """Render every option, grouped, with defaults and descriptions."""
    d = Options()
    width = max(len(f) for _, fields in _GROUPS for f, _ in fields)
    out: list[str] = []
    for title, fields in _GROUPS:
        if only and only.lower() not in title.lower():
            continue
        out.append("")
        out.append(title)
        out.append("-" * len(title))
        for name, doc in fields:
            val = getattr(d, name)
            val = repr(val) if not isinstance(val, str) else f'"{val}"'
            out.append(f"  {name:<{width}}  = {val:<10}  {doc}")
    return "\n".join(out).strip()


Options.describe = staticmethod(_describe)  # type: ignore[attr-defined]
