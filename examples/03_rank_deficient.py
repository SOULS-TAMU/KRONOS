"""A rank-deficient constraint set.

    min  (x-1)^2 + (y-2)^2
    s.t. x + y = 3
         2x + 2y = 6          (twice the first, carrying no new information)

The constraint Jacobian has rank 1 rather than 2, which makes the KKT matrix
singular. A Newton step formed by inverting it fails; the minimum-norm
least-squares step is defined regardless.
"""
import numpy as np

from kronos import Options, Problem, get_backend, lsqminnorm, solve
from kronos.core import _Assembler

p = Problem.build("redundant", ["x", "y"],
                  objective  = "(x - 1)**2 + (y - 2)**2",
                  equalities = ["x + y - 3", "2*x + 2*y - 6"],
                  x0 = [0.0, 0.0])

b = get_backend(p, "sympy")
Jh = b.Jh(np.array([0.0, 0.0]))
print("constraint Jacobian:\n", Jh)
print("rank =", np.linalg.matrix_rank(Jh), "of", min(Jh.shape), "\n")

A = _Assembler(b, p.n, p.m, Options().use_dummy_variable)
R, J, _, _, _ = A(np.array([1.0, 1.0, 2.0]), np.zeros(1 + p.m))
print("KKT matrix   :", J.shape, " rank", np.linalg.matrix_rank(J), "of", J.shape[0])
print("condition no.: %.3e" % np.linalg.cond(J))
try:
    np.linalg.solve(J, R)
except np.linalg.LinAlgError as e:
    print("np.linalg.solve ->", type(e).__name__ + ":", e)
print("minimum-norm    ->", np.round(lsqminnorm(J, R), 12), "\n")

r = solve(p, multi_start=True, ms_num_starts=10, ms_seed=42)
print(r.summary())
