"""Stage 0: Adam warm-up.

First-order descent on the merit function ``f(theta) + rho * ||h(theta)||^2``,
run before the Newton KKT solver commits to a basin.  Momentum plus a
per-coordinate adaptive step traverses saddles and shallow valleys that Newton
would otherwise be pulled into.  A direct port of ``adam_warmup.m``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .options import Options

__all__ = ["AdamInfo", "adam_warmup"]


@dataclass
class AdamInfo:
    iters: int = 0
    merit_initial: float = np.nan
    merit_final: float = np.nan
    final_h_norm: float = np.nan
    elapsed: float = 0.0
    skipped: bool = True
    error: Optional[str] = None


def adam_warmup(backend, lb, ub, theta0, opts: Options) -> tuple[np.ndarray, AdamInfo]:
    """Refine a single starting point with Adam on the penalised merit."""
    import time
    t0 = time.time()

    from .core import _clip_matlab
    theta = _clip_matlab(np.asarray(theta0, float).ravel(), lb, ub)
    have_h = backend.m > 0
    info = AdamInfo(skipped=False)

    def merit(x: np.ndarray) -> float:
        val = float(backend.f(x))
        if have_h:
            hv = backend.h(x)
            val += opts.adam_rho * float(hv @ hv)
        return val

    info.merit_initial = merit(theta)

    mom = np.zeros(theta.size)
    vel = np.zeros(theta.size)
    b1, b2 = opts.adam_b1, opts.adam_b2

    t = 0
    for t in range(1, opts.adam_iters + 1):
        with np.errstate(all="ignore"):
            g = np.asarray(backend.grad_f(theta), float).ravel()
            if have_h:
                g = g + 2.0 * opts.adam_rho * (backend.Jh(theta).T @ backend.h(theta))
        if not np.all(np.isfinite(g)):
            break                       # overflow: keep the last good iterate

        with np.errstate(all="ignore"):
            mom = b1 * mom + (1.0 - b1) * g
            vel = b2 * vel + (1.0 - b2) * (g * g)
            m_hat = mom / (1.0 - b1 ** t)
            v_hat = vel / (1.0 - b2 ** t)
            theta = theta - opts.adam_lr * m_hat / (np.sqrt(v_hat) + opts.adam_eps)
            theta = _clip_matlab(theta, lb, ub)

        if opts.adam_verbose and (t == 1 or t % max(1, opts.adam_iters // 10) == 0):
            print(f"  adam {t:4d}/{opts.adam_iters}  merit={merit(theta):.6e}")

    info.iters = t
    info.merit_final = merit(theta)
    info.final_h_norm = (float(np.max(np.abs(backend.h(theta)))) if have_h else np.nan)
    info.elapsed = time.time() - t0
    return theta, info
