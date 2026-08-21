"""Benchmark harness.

Computes per-problem metrics (certified convergence, global hits, timings) and
compares them against a reference CSV.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from .api import solve
from .backends import get_backend
from .core import SolveResult
from .library import load_problem, problem_names
from .options import Options
from .problem import Problem

__all__ = ["Metrics", "options_from_problem", "compute_metrics",
           "run_problem", "run_suite", "load_reference", "compare"]


@dataclass
class Metrics:
    """Per-problem benchmark metrics, mirroring the MATLAB report block."""

    problem: str
    n: int
    m: int
    K: int
    n_conv: int
    n_reformulated_stationary: int
    n_kkt: int
    n_global: int
    n_kkt_global: int
    n_local: int
    n_stationary: int
    Lbar: float
    best_fval: float
    fstar: Optional[float]
    mean_time: float
    total_time: float
    backend: str
    error: str = ""

    def as_row(self) -> dict:
        return asdict(self)


def options_from_problem(problem: Problem, K: Optional[int] = None, **overrides) -> Options:
    """Rebuild the exact MATLAB options the reference run used.

    The benchmark scripts were run with ``bench_ms_override = K``, which forces
    ``ms_num_starts = K``, ``maxIter = 500`` and silences output.
    """
    mo = dict(problem.meta.get("matlab_opts", {}))
    field_map = {
        "tol_r": "tol_r", "tol_h": "tol_h", "maxIter": "maxIter",
        "ms_num_starts": "ms_num_starts", "ms_seed": "ms_seed", "ms_scale": "ms_scale",
        "adam_iters": "adam_iters", "adam_lr": "adam_lr", "adam_rho": "adam_rho",
        "adam_b1": "adam_b1", "adam_b2": "adam_b2", "adam_eps": "adam_eps",
        "adam_mode": "adam_mode", "adam_exit_tol_g": "adam_exit_tol_g",
        "max_kkt_kicks": "max_kkt_kicks", "dual_feas_tol": "dual_feas_tol",
        "sosc_tol": "sosc_tol", "stag_tol": "stag_tol", "stag_max": "stag_max",
        "kick_size": "kick_size", "feas_tol": "feas_tol", "feas_iters": "feas_iters",
        "feas_step": "feas_step", "bt_max": "bt_max", "bt_rho": "bt_rho", "bt_c1": "bt_c1",
        "lam0_magnitude": "lam0_magnitude", "lam0_sign": "lam0_sign",
        "sign_bias_rho": "sign_bias_rho", "max_iter_after_promotion": "max_iter_after_promotion",
    }
    kw: dict[str, Any] = {}
    for src, dst in field_map.items():
        if src in mo and mo[src] is not None:
            kw[dst] = mo[src]
    for flag in ("multi_start", "use_adam_warmup", "adam_early_exit", "enforce_kkt_sign",
                 "check_kkt_sign", "force_single_start", "sign_flip_multistart",
                 "project_ineq_sign", "promote_on_kick_exhaust", "use_prefeasibility"):
        if flag in mo and mo[flag] is not None:
            kw[flag] = bool(mo[flag])
    for k in ("maxIter", "ms_num_starts", "ms_seed", "stag_max", "feas_iters",
              "bt_max", "max_kkt_kicks", "max_iter_after_promotion"):
        if k in kw:
            kw[k] = int(kw[k])

    kw.setdefault("adam_mode", "C")
    kw.setdefault("verbose", False)
    kw.setdefault("ms_show_runs", False)
    if K is not None:
        kw["ms_num_starts"] = int(K)
        kw["multi_start"] = K > 1
        kw["force_single_start"] = K == 1
    kw["fstar"] = problem.fstar
    kw.update(overrides)
    return Options(**{k: v for k, v in kw.items() if k in Options.__dataclass_fields__})


def compute_metrics(problem: Problem, result: SolveResult, K: int,
                    elapsed: float, backend_name: str) -> Metrics:
    """Reduce a solve to the reported benchmark metrics."""
    fv = result.all_fvals
    cv = result.all_conv
    kkt = result.all_kkt
    sosc = result.all_sosc
    measured = result.all_sosc_measured

    n_conv, n_kkt = int(kkt.sum()), int(kkt.sum())   # "converged" == certified
    n_reformulated_stationary = int(cv.sum())
    n_local = int((kkt & sosc).sum())
    n_sosc_unknown = int((kkt & ~measured).sum())
    n_stationary = n_kkt - n_local - n_sosc_unknown

    fstar = problem.fstar
    n_global = n_kkt_global = 0
    if fstar is not None and np.isfinite(fstar):
        tol = max(1e-4, 1e-3 * max(1.0, abs(fstar)))
        is_glob = kkt & (np.abs(fv - fstar) <= tol)
        n_global = int(is_glob.sum())
        n_kkt_global = n_global

    if int(cv.sum()) > 0:
        fc = fv[cv]
        best = float(np.min(fc))
        gap = np.maximum(np.abs(fc - best) / max(1.0, abs(best)), 1e-12)
        Lbar = float(np.log10(np.median(gap)))
    else:
        Lbar = float("nan")

    return Metrics(
        problem=problem.name, n=problem.n, m=problem.m, K=K,
        n_conv=n_conv, n_reformulated_stationary=n_reformulated_stationary, n_kkt=n_kkt, n_global=n_global, n_kkt_global=n_kkt_global,
        n_local=n_local, n_stationary=n_stationary, Lbar=Lbar,
        best_fval=float(result.fval), fstar=fstar,
        mean_time=elapsed / max(K, 1), total_time=elapsed, backend=backend_name,
    )


def run_problem(name: str | Problem, K: int = 25, backend: Optional[str] = None,
                **overrides) -> Metrics:
    """Solve one benchmark problem and return its metrics."""
    problem = load_problem(name) if isinstance(name, str) else name
    opts = options_from_problem(problem, K=K, **overrides)
    if backend:
        opts = opts.copy(backend=backend)
    t0 = time.time()
    try:
        bk = get_backend(problem, opts.backend, switch_n=opts.backend_switch_n)
        result = solve(problem, opts, backend=bk)
        return compute_metrics(problem, result, K, time.time() - t0, bk.name)
    except Exception as exc:                       # keep the sweep going
        return Metrics(problem=problem.name, n=problem.n, m=problem.m, K=K,
                       n_conv=0, n_reformulated_stationary=0, n_kkt=0, n_global=0, n_kkt_global=0, n_local=0,
                       n_stationary=0, Lbar=float("nan"), best_fval=float("nan"),
                       fstar=problem.fstar, mean_time=float("nan"),
                       total_time=time.time() - t0, backend=backend or "auto",
                       error=f"{type(exc).__name__}: {exc}")


def run_suite(names: Optional[Iterable[str]] = None, K: int = 25,
              out_csv: Optional[str | Path] = None, progress: bool = True,
              **overrides) -> list[Metrics]:
    """Run a set of problems and optionally write a CSV of the metrics."""
    names = list(names) if names is not None else list(problem_names())
    rows: list[Metrics] = []
    for i, nm in enumerate(names, 1):
        mt = run_problem(nm, K=K, **overrides)
        rows.append(mt)
        if progress:
            tag = mt.error or (f"conv {mt.n_conv}/{K}  glob {mt.n_global}  "
                               f"f={mt.best_fval:.6g}  {mt.total_time:.1f}s")
            print(f"[{i:3d}/{len(names)}] {nm:<28} {tag}", flush=True)
    if out_csv:
        with open(out_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].as_row()))
            w.writeheader()
            for r in rows:
                w.writerow(r.as_row())
    return rows


def load_reference(csv_path: str | Path, config: str = "v3") -> dict[str, dict]:
    """Load the published per-problem reference rows."""
    ref: dict[str, dict] = {}
    with open(csv_path) as fh:
        for row in csv.DictReader(fh):
            if row.get("config") == config:
                ref[row["problem"]] = row
    return ref


def compare(rows: Iterable[Metrics], csv_path: str | Path, config: str = "v3"):
    """Diff Python metrics against the reference CSV.

    Returns a list of dicts with the reference and Python values side by side.
    """
    ref = load_reference(csv_path, config)
    out = []
    for m in rows:
        r = ref.get(m.problem)
        if r is None:
            continue
        def num(key):
            try:
                return float(r[key])
            except (TypeError, ValueError, KeyError):
                return float("nan")
        out.append({
            "problem": m.problem, "n": m.n, "K": m.K,
            "conv_ref": num("n_conv"), "conv_py": m.n_conv,
            "glob_ref": num("n_global"), "glob_py": m.n_global,
            "glob_py_kkt": m.n_kkt_global,
            "L_ref": num("Ln_abs"), "L_py": m.Lbar,
            "t_ref": num("mean_time"), "t_py": m.mean_time,
            "fstar": m.fstar, "f_py": m.best_fval, "error": m.error,
        })
    return out
