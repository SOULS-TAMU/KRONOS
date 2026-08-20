"""KRONOS: KKT-certified nonlinear programming.

The solver forms the full KKT system and drives its residual to zero with
Newton's method. The Newton step is the minimum-norm least-squares solution of
the KKT linear system, which keeps the iteration well defined when that system
is rank deficient. Inequalities and variable bounds are converted to equalities
using squared slack variables.

Example
-------
>>> from kronos import Problem, solve
>>> p = Problem.build("rosen", ["x1", "x2"],
...                   objective="100*(x2 - x1**2)**2 + (1 - x1)**2",
...                   x0=[-1.2, 1.0])
>>> result = solve(p)
>>> bool(result.converged)
True
"""

from .options import Options
from .problem import Problem, SlackRow
from .core import RunResult, SolveResult
from .api import solve
from .library import load_problem, problem_names, iter_problems, find
from .linalg import lsqminnorm, min_norm_solve, null_space
from .backends import get_backend, available_backends

__version__ = "0.3.1"

__all__ = [
    "Options", "Problem", "SlackRow", "RunResult", "SolveResult", "solve",
    "load_problem", "problem_names", "iter_problems", "find",
    "lsqminnorm", "min_norm_solve", "null_space",
    "get_backend", "available_backends", "__version__",
]
