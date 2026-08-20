"""CasADi backend: sparse algorithmic differentiation.

Used for larger problems (``n >= 20`` by default), matching the reference
implementation's routing.  The SymPy expression graph is translated to CasADi
``SX`` once via ``lambdify`` with a CasADi namespace -- operator overloading
does the rest -- and all five KKT quantities are compiled into a *single*
``casadi.Function``.  One call per Newton iteration then returns everything,
which is where most of the speed comes from: CasADi evaluates the shared
subgraph once instead of five times.
"""

from __future__ import annotations

import sys

import numpy as np
import sympy as sp

from .base import BaseBackend, KKTParts

__all__ = ["CasadiBackend", "sympy_to_casadi"]


def _casadi_namespace(ca):
    return {
        "sqrt": ca.sqrt, "exp": ca.exp, "log": ca.log, "log10": ca.log10,
        "sin": ca.sin, "cos": ca.cos, "tan": ca.tan,
        "asin": ca.asin, "acos": ca.acos, "atan": ca.atan, "atan2": ca.atan2,
        "sinh": ca.sinh, "cosh": ca.cosh, "tanh": ca.tanh,
        "asinh": ca.asinh, "acosh": ca.acosh, "atanh": ca.atanh,
        "Abs": ca.fabs, "fabs": ca.fabs, "sign": ca.sign,
        "Max": ca.fmax, "Min": ca.fmin, "fmax": ca.fmax, "fmin": ca.fmin,
        "floor": ca.floor, "ceiling": ca.ceil,
        "pi": np.pi, "E": np.e, "erf": ca.erf,
    }


def sympy_to_casadi(expr, syms, sx_vars, ca):
    """Translate a SymPy expression (or Matrix) into CasADi ``SX``."""
    old = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old, 100000))
    try:
        ns = _casadi_namespace(ca)
        try:
            fn = sp.lambdify(syms, expr, modules=[ns], cse=True)
        except TypeError:
            fn = sp.lambdify(syms, expr, modules=[ns])
        out = fn(*sx_vars)
    finally:
        sys.setrecursionlimit(old)

    if isinstance(out, ca.SX):
        return out
    arr = np.asarray(out, dtype=object)
    flat = [x if isinstance(x, ca.SX) else ca.SX(float(x)) for x in arr.ravel()]
    if arr.ndim <= 1:
        return ca.vertcat(*flat) if flat else ca.SX.zeros(0, 1)
    return ca.reshape(ca.vertcat(*flat), arr.shape[1], arr.shape[0]).T


class CasadiBackend(BaseBackend):
    """All KKT derivatives in one compiled CasADi ``Function``."""

    name = "casadi"

    def __init__(self, problem, jit: bool = False):
        import casadi as ca

        self._ca = ca
        syms = problem.symbols
        self.n = len(syms)
        self.m = len(problem.h)

        th = ca.SX.sym("theta", self.n)
        lam = ca.SX.sym("lam", max(self.m, 1))
        fs = ca.SX.sym("f_scale")
        sx_vars = [th[i] for i in range(self.n)]

        f = sympy_to_casadi(sp.sympify(problem.f), syms, sx_vars, ca)
        if self.m:
            h = ca.vertcat(*[sympy_to_casadi(sp.sympify(e), syms, sx_vars, ca)
                             for e in problem.h])
            lam_u = lam[: self.m]
            lag = fs * f + ca.dot(lam_u, h)
            Jh = ca.jacobian(h, th)
        else:
            h = ca.SX.zeros(0, 1)
            lag = fs * f
            Jh = ca.SX.zeros(0, self.n)

        grad_f = ca.gradient(f, th)
        hess_lag = ca.hessian(lag, th)[0]

        opts = {}
        if jit:
            opts = {"jit": True, "compiler": "shell",
                    "jit_options": {"flags": ["-O2"], "verbose": False}}
        self._hfun = ca.Function("hJ", [th], [h, Jh], ["theta"], ["h", "Jh"], opts)
        self._fun = ca.Function("kkt", [th, lam, fs],
                                [fs * f, fs * grad_f, h, Jh, hess_lag],
                                ["theta", "lam", "f_scale"],
                                ["f", "grad_f", "h", "Jh", "hess_lag"], opts)

    # ------------------------------------------------------------------
    def _call(self, theta, lam, f_scale=1.0):
        lam = np.zeros(1) if self.m == 0 else np.asarray(lam, float).ravel()
        if lam.size < 1:
            lam = np.zeros(1)
        return self._fun(np.asarray(theta, float).ravel(), lam, f_scale)

    def kkt(self, theta, lam, f_scale: float = 1.0) -> KKTParts:
        f, g, h, Jh, H = self._call(theta, lam, f_scale)
        return (float(f),
                np.asarray(g).reshape(self.n),
                np.asarray(h).reshape(self.m),
                np.asarray(Jh).reshape(self.m, self.n),
                np.asarray(H).reshape(self.n, self.n))

    def f(self, theta) -> float:
        return self.kkt(theta, np.zeros(max(self.m, 1)))[0]

    def grad_f(self, theta) -> np.ndarray:
        return self.kkt(theta, np.zeros(max(self.m, 1)))[1]

    def h(self, theta) -> np.ndarray:
        return self.kkt(theta, np.zeros(max(self.m, 1)))[2]

    def Jh(self, theta) -> np.ndarray:
        return self.kkt(theta, np.zeros(max(self.m, 1)))[3]

    def h_only(self, theta):
        """Constraints and their Jacobian, skipping the Hessian."""
        h, Jh = self._hfun(np.asarray(theta, float).ravel())
        return (np.asarray(h).reshape(self.m),
                np.asarray(Jh).reshape(self.m, self.n))

    def hess_lag(self, theta, lam, f_scale: float = 1.0) -> np.ndarray:
        return self.kkt(theta, lam, f_scale)[4]
