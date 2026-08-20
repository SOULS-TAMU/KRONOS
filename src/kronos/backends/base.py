"""Backend interface.

A backend supplies the derivative information required by the KKT iteration.
The internal variable ``xs`` is not a backend concern: its contribution to the
KKT system is constant and is assembled analytically in :mod:`kronos.core`, so
a backend sees only the problem variables.

Required at each evaluation point ``(theta, lam)``:

    f(theta)                     scalar objective
    grad_f(theta)                (n,)
    h(theta)                     (m,)
    Jh(theta)                    (m, n)
    hess_lag(theta, lam)         (n, n)   d^2/dtheta^2 [ f + lam' h ]
"""

from __future__ import annotations

from typing import Protocol, Tuple

import numpy as np

__all__ = ["Backend", "KKTParts"]

KKTParts = Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


class Backend(Protocol):
    """Derivative provider for a :class:`~kronos.problem.Problem`."""

    n: int
    m: int
    name: str

    def f(self, theta: np.ndarray) -> float: ...
    def grad_f(self, theta: np.ndarray) -> np.ndarray: ...
    def h(self, theta: np.ndarray) -> np.ndarray: ...
    def Jh(self, theta: np.ndarray) -> np.ndarray: ...
    def hess_lag(self, theta: np.ndarray, lam: np.ndarray) -> np.ndarray: ...

    def kkt(self, theta: np.ndarray, lam: np.ndarray) -> KKTParts:
        """Fused evaluation: ``(f, grad_f, h, Jh, hess_lag)``.

        Backends that can compute these together (CasADi, JAX) should override
        this; the default just calls the pieces.
        """
        ...


class BaseBackend:
    """Default fused evaluation for backends that do not specialise it."""

    n: int
    m: int
    name: str = "base"

    def kkt(self, theta: np.ndarray, lam: np.ndarray) -> KKTParts:
        return (self.f(theta), self.grad_f(theta), self.h(theta),
                self.Jh(theta), self.hess_lag(theta, lam))

    def __repr__(self) -> str:
        return f"<{type(self).__name__} n={self.n} m={self.m}>"
