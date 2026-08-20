"""Result visualisation.

All functions return a matplotlib ``Figure`` and never call ``show()``, so they
compose into larger reports.  ``matplotlib`` is an optional dependency
(``pip install kronos[plot]``).
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np

__all__ = ["plot_runs", "plot_convergence_profile", "plot_comparison", "summary_table"]


def _mpl():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError as exc:                    # pragma: no cover
        raise ImportError(
            "plotting needs matplotlib -- install with:  pip install kronos[plot]"
        ) from exc


def plot_runs(result, fstar: Optional[float] = None, title: str = ""):
    """Objective value per multistart run, coloured by certification status.

    Green = KKT-certified, amber = converged but not certified, grey = failed.
    """
    plt = _mpl()
    fv = result.all_fvals
    cv = result.all_conv
    kkt = result.all_kkt
    idx = np.arange(1, len(fv) + 1)

    colours = np.where(kkt, "#2a9d3f", np.where(cv, "#e8a33d", "#b0b0b0"))
    fig, ax = plt.subplots(figsize=(9, 4.2))
    finite = np.isfinite(fv)
    ax.scatter(idx[finite], fv[finite], c=colours[finite], s=48,
               edgecolor="black", linewidth=0.5, zorder=3)
    if (~finite).any():
        ax.scatter(idx[~finite], np.full((~finite).sum(), np.nanmax(fv[finite], initial=0)),
                   marker="x", c="#b0b0b0", s=40, label="non-finite")

    if fstar is not None and np.isfinite(fstar):
        ax.axhline(fstar, ls="--", lw=1.2, c="#3f6fb0", zorder=2,
                   label=f"$f^*$ = {fstar:.6g}")
        ax.legend(frameon=False)

    ax.set_xlabel("multistart run")
    ax.set_ylabel("objective")
    ax.set_title(title or f"{result.info.get('problem', 'problem')}  "
                         f"({result.n_kkt}/{len(fv)} KKT-certified)")
    ax.grid(alpha=0.25, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def plot_convergence_profile(rows, key: str = "n_conv", K: int = 25):
    """Cumulative distribution of a per-problem metric across the suite."""
    plt = _mpl()
    vals = np.array([getattr(r, key) for r in rows], dtype=float)
    vals = vals[np.isfinite(vals)]
    order = np.sort(vals)
    frac = np.arange(1, order.size + 1) / order.size

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.step(order, frac, where="post", lw=2, c="#2a6fb0")
    ax.set_xlabel(f"{key} (out of {K})")
    ax.set_ylabel("fraction of problems <= x")
    ax.set_title(f"Profile of {key} over {order.size} problems")
    ax.grid(alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def plot_comparison(comparison: Sequence[dict], metric: str = "conv",
                    logtime: bool = True):
    """Scatter reference vs Python for a metric; points on the diagonal agree.

    ``metric`` is one of ``"conv"``, ``"glob"`` or ``"time"``.
    """
    plt = _mpl()
    ref = np.array([c[f"{metric}_ref"] for c in comparison], dtype=float)
    py = np.array([c[f"{metric}_py"] for c in comparison], dtype=float)
    ok = np.isfinite(ref) & np.isfinite(py)
    ref, py = ref[ok], py[ok]

    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    ax.scatter(ref, py, s=42, alpha=0.75, c="#2a6fb0", edgecolor="black", linewidth=0.4)
    lo = min(ref.min(), py.min()) if ref.size else 0
    hi = max(ref.max(), py.max()) if ref.size else 1
    if metric == "time" and logtime:
        ax.set_xscale("log"); ax.set_yscale("log")
        lo = max(lo, 1e-4)
    ax.plot([lo, hi], [lo, hi], ls="--", c="#888", lw=1)
    ax.set_xlabel(f"MATLAB reference ({metric})")
    ax.set_ylabel(f"kronos / Python ({metric})")
    same = int(np.sum(ref == py))
    ax.set_title(f"{metric}: {same}/{ref.size} identical")
    ax.grid(alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def summary_table(rows: Iterable, K: int = 25) -> str:
    """Plain-text summary of a suite run."""
    rows = list(rows)
    n_err = sum(1 for r in rows if r.error)
    conv = np.array([r.n_conv for r in rows], dtype=float)
    glob = np.array([r.n_global for r in rows], dtype=float)
    t = np.array([r.mean_time for r in rows], dtype=float)
    t = t[np.isfinite(t)]
    lines = [
        f"problems           : {len(rows)}  ({n_err} errored)",
        f"mean converged     : {np.mean(conv):.2f} / {K}",
        f"problems all-conv  : {int(np.sum(conv == K))}",
        f"mean global hits   : {np.mean(glob):.2f} / {K}",
        f"problems >=1 global: {int(np.sum(glob > 0))}",
        f"median time / run  : {np.median(t):.3f} s" if t.size else "median time: n/a",
        f"total wall time    : {np.sum([r.total_time for r in rows]) / 60:.1f} min",
    ]
    return "\n".join(lines)
