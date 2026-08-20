"""State a problem, solve it, and read the certification."""
import numpy as np

from kronos import Problem, solve

# Give the objective, the constraints and the bounds.  Inequalities and bounds
# are handled internally -- you do not add anything yourself.
p = Problem.build(
    "my_problem", ["x1", "x2", "x3", "x4"],
    objective    = "x1*x4*(x1 + x2 + x3) + x3",
    equalities   = ["x1**2 + x2**2 + x3**2 + x4**2 - 40"],   # each means == 0
    inequalities = ["25 - x1*x2*x3*x4"],                     # each means <= 0
    lb = [1, 1, 1, 1],
    ub = [5, 5, 5, 5],
    x0 = [1, 5, 5, 1],
    fstar = 17.0140173,                                      # optional
)

r = solve(p, multi_start=True, ms_num_starts=25, ms_seed=42)

print(r.summary())
print("x* =", np.round(r.theta[:4], 8))
