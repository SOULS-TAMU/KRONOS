"""SymPy backend: symbolic differentiation compiled to NumPy.

This mirrors the reference implementation's ``matlabFunction`` path and is the
default for small problems (``n < 20``), where forming the Hessian of the
Lagrangian symbolically is cheap and gives the tightest agreement with MATLAB.
"""

from __future__ import annotations

import sys
import warnings
from contextlib import contextmanager
from typing import Sequence

import numpy as np
import sympy as sp

from .base import BaseBackend, KKTParts


@contextmanager
def _quiet():
    """Silence expected float noise.

    Evaluating a compiled expression at a singular point legitimately produces
    0/0 and overflow.  The solver handles the resulting NaN (see
    ``kronos.core``), so the warnings are noise, not information.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with np.errstate(all="ignore"):
            yield

__all__ = ["SympyBackend"]


def _dirac(x):
    """MATLAB's ``dirac``: Inf at zero, 0 elsewhere.

    The second derivative of ``|x|`` is ``2*delta(x)``, which SymPy emits as
    ``DiracDelta``.  ``lambdify``'s numpy namespace has no such name, so a
    Hessian containing it raises ``NameError`` -- silently failing any problem
    built from ``abs()``.  Match MATLAB rather than dropping the term.
    """
    x = np.asarray(x, dtype=float)
    out = np.where(x == 0.0, np.inf, 0.0)
    return out if out.ndim else float(out)


def _heaviside(x, half=0.5):
    x = np.asarray(x, dtype=float)
    out = np.where(x < 0, 0.0, np.where(x > 0, 1.0, half))
    return out if out.ndim else float(out)


_EXTRA_NS = {"DiracDelta": _dirac, "Heaviside": _heaviside,
             "dirac": _dirac, "heaviside": _heaviside}


def _lambdify(args, expr, cse: bool = True):
    mods = [_EXTRA_NS, "numpy"]
    try:
        return sp.lambdify(args, expr, modules=mods, cse=cse)
    except TypeError:                       # older sympy without cse kwarg
        return sp.lambdify(args, expr, modules=mods)


class SympyBackend(BaseBackend):
    """Symbolic derivatives, compiled once at construction."""

    name = "sympy"

    def __init__(self, problem):
        old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(max(old_limit, 100000))
        try:
            syms = problem.symbols
            self.n = len(syms)
            self.m = len(problem.h)
            self._syms = syms

            f = sp.sympify(problem.f)
            h = sp.Matrix([sp.sympify(e) for e in problem.h]) if self.m else sp.Matrix(0, 1, [])
            lam = sp.symbols(f"kronos_lam0:{max(self.m, 1)}", real=True)
            lam = list(lam)[: self.m]
            fs = sp.Symbol("kronos_fscale", real=True)

            X = sp.Matrix(syms)
            grad_f = sp.Matrix([sp.diff(f, s) for s in syms])
            Jh = h.jacobian(X) if self.m else sp.Matrix(0, self.n, [])
            lag = fs * f + sum((lam[i] * h[i] for i in range(self.m)), sp.Integer(0))
            grad_lag = sp.Matrix([sp.diff(lag, s) for s in syms])
            hess_lag = grad_lag.jacobian(X)

            self._f = _lambdify((syms,), f)
            self._grad_f = _lambdify((syms,), grad_f)
            self._h = _lambdify((syms,), h) if self.m else None
            self._Jh = _lambdify((syms,), Jh) if self.m else None
            self._hess = (_lambdify((syms, lam, fs), hess_lag) if self.m
                          else _lambdify((syms, fs), hess_lag))
        finally:
            sys.setrecursionlimit(old_limit)

    # ------------------------------------------------------------------
    def f(self, theta: np.ndarray) -> float:
        with _quiet():
            return float(self._f(theta))

    def grad_f(self, theta: np.ndarray) -> np.ndarray:
        with _quiet():
            return np.asarray(self._grad_f(theta), dtype=float).reshape(self.n)

    def h(self, theta: np.ndarray) -> np.ndarray:
        if self.m == 0:
            return np.zeros(0)
        with _quiet():
            return np.asarray(self._h(theta), dtype=float).reshape(self.m)

    def Jh(self, theta: np.ndarray) -> np.ndarray:
        if self.m == 0:
            return np.zeros((0, self.n))
        with _quiet():
            return np.asarray(self._Jh(theta), dtype=float).reshape(self.m, self.n)

    def h_only(self, theta: np.ndarray):
        """Constraints and their Jacobian, skipping the Hessian."""
        return self.h(theta), self.Jh(theta)

    def hess_lag(self, theta: np.ndarray, lam: np.ndarray,
                 f_scale: float = 1.0) -> np.ndarray:
        with _quiet():
            if self.m == 0:
                H = self._hess(theta, f_scale)
            else:
                H = self._hess(theta, np.asarray(lam, float).ravel(), f_scale)
            return np.asarray(H, dtype=float).reshape(self.n, self.n)

    def kkt(self, theta, lam, f_scale: float = 1.0):
        return (f_scale * self.f(theta), f_scale * self.grad_f(theta),
                self.h(theta), self.Jh(theta), self.hess_lag(theta, lam, f_scale))
