"""Derivative backends.
"""

from __future__ import annotations

from .base import Backend, BaseBackend
from .sympy_backend import SympyBackend

__all__ = ["Backend", "BaseBackend", "SympyBackend", "get_backend", "available_backends"]


def available_backends() -> list[str]:
    names = ["sympy"]
    try:
        import casadi  # noqa: F401
        names.append("casadi")
    except ImportError:
        pass
    try:
        import jax  # noqa: F401
        names.append("jax")
    except ImportError:
        pass
    return names


def get_backend(problem, name: str = "auto", switch_n: int = 20, **kwargs):
    """Construct a backend for ``problem``.

    ``"auto"`` reproduces the reference routing: symbolic below ``switch_n``
    variables, CasADi at or above it (falling back to SymPy if CasADi is not
    installed).
    """
    if name == "auto":
        name = "casadi" if problem.n >= switch_n else "sympy"
        if name == "casadi" and "casadi" not in available_backends():
            # Reported rather than applied silently, since SymPy is markedly
            # slower at this size.
            import warnings
            warnings.warn(
                f"{problem.name!r} has {problem.n} variables, which routes to the "
                f"CasADi backend, but CasADi is not installed - falling back to "
                f"SymPy, which is far slower at this size. "
                f"Install it with:  pip install casadi",
                RuntimeWarning, stacklevel=2)
            name = "sympy"

    if name == "sympy":
        return SympyBackend(problem)
    if name == "casadi":
        from .casadi_backend import CasadiBackend
        return CasadiBackend(problem, **kwargs)
    if name == "jax":
        from .jax_backend import JaxBackend
        return JaxBackend(problem, **kwargs)
    raise ValueError(f"unknown backend {name!r}; available: {available_backends()}")
