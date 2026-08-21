"""JAX backend: forward and reverse-mode AD with JIT compilation.

Requires 64-bit mode, which is enabled on import.
"""

from __future__ import annotations

import sys

import numpy as np
import sympy as sp

from .base import BaseBackend, KKTParts

__all__ = ["JaxBackend"]


_INT64_MAX = 2 ** 63 - 1


def _floatify(expr):
    """Convert integer coefficients too large for int64 into floats.

    ``lambdify`` emits integer literals as Python ``int``. JAX rejects an
    ``int`` outside the int64 range as an argument to a jitted computation, so
    such coefficients are converted to floats. Exponents are left alone, since
    an integer power is both faster and more accurate than a float one.
    """
    big = {n for n in expr.atoms(sp.Integer) if abs(int(n)) > _INT64_MAX}
    if not big:
        return expr
    exponents = {e.exp for e in expr.atoms(sp.Pow) if e.exp.is_Integer}
    subs = {n: sp.Float(int(n)) for n in big if n not in exponents}
    return expr.xreplace(subs) if subs else expr


def _enable_x64():
    import jax
    jax.config.update("jax_enable_x64", True)
    return jax


class JaxBackend(BaseBackend):
    """All KKT derivatives from one jitted JAX function."""

    name = "jax"

    def __init__(self, problem, jit: bool = True):
        jax = _enable_x64()
        import jax.numpy as jnp

        self._jnp = jnp
        syms = problem.symbols
        self.n = len(syms)
        self.m = len(problem.h)

        old = sys.getrecursionlimit()
        sys.setrecursionlimit(max(old, 100000))
        try:
            f_fn = sp.lambdify(syms, _floatify(sp.sympify(problem.f)), modules="jax")
            h_fns = [sp.lambdify(syms, _floatify(sp.sympify(e)), modules="jax")
                     for e in problem.h]
        finally:
            sys.setrecursionlimit(old)

        def f_of(theta):
            return f_fn(*[theta[i] for i in range(self.n)])

        def h_of(theta):
            if not self.m:
                return jnp.zeros(0)
            args = [theta[i] for i in range(self.n)]
            return jnp.stack([jnp.asarray(fn(*args), dtype=jnp.float64).reshape(())
                              for fn in h_fns])

        def lag(theta, lam, f_scale):
            val = f_scale * f_of(theta)
            if self.m:
                val = val + jnp.dot(lam[: self.m], h_of(theta))
            return val

        grad_f = jax.grad(lambda t, s: s * f_of(t), argnums=0)
        jac_h = jax.jacfwd(h_of)
        hess_lag = jax.jacfwd(jax.jacrev(lag, argnums=0), argnums=0)

        def fused(theta, lam, f_scale):
            return (f_scale * f_of(theta), grad_f(theta, f_scale), h_of(theta),
                    jac_h(theta) if self.m else jnp.zeros((0, self.n)),
                    hess_lag(theta, lam, f_scale))

        self._fused = jax.jit(fused) if jit else fused

    # ------------------------------------------------------------------
    def kkt(self, theta, lam, f_scale: float = 1.0) -> KKTParts:
        jnp = self._jnp
        th = jnp.asarray(np.asarray(theta, float).ravel())
        lm = jnp.asarray(np.asarray(lam, float).ravel()
                         if np.size(lam) else np.zeros(max(self.m, 1)))
        if lm.size < max(self.m, 1):
            lm = jnp.concatenate([lm, jnp.zeros(max(self.m, 1) - lm.size)])
        f, g, h, J, H = self._fused(th, lm, float(f_scale))
        return (float(f),
                np.asarray(g, float).reshape(self.n),
                np.asarray(h, float).reshape(self.m),
                np.asarray(J, float).reshape(self.m, self.n),
                np.asarray(H, float).reshape(self.n, self.n))

    def f(self, theta) -> float:
        return self.kkt(theta, np.zeros(max(self.m, 1)))[0]

    def grad_f(self, theta) -> np.ndarray:
        return self.kkt(theta, np.zeros(max(self.m, 1)))[1]

    def h(self, theta) -> np.ndarray:
        return self.kkt(theta, np.zeros(max(self.m, 1)))[2]

    def Jh(self, theta) -> np.ndarray:
        return self.kkt(theta, np.zeros(max(self.m, 1)))[3]

    def h_only(self, theta):
        r = self.kkt(theta, np.zeros(max(self.m, 1)))
        return r[2], r[3]

    def hess_lag(self, theta, lam, f_scale: float = 1.0) -> np.ndarray:
        return self.kkt(theta, lam, f_scale)[4]
