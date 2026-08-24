# KRONOS

An algorithm for solving ill-conditioned nonlinear programs.

KRONOS solves

$$
\begin{aligned}
\min_{x} \quad & f(x) \\
\textrm{s.t.} \quad & h(x) = 0 \\
& g(x) \le 0 \\
& x^{L} \le x \le x^{U}
\end{aligned}
$$

by forming the full KKT system and driving its residual to zero iteratively, taking the minimum-norm least-squares solution of the KKT linear system
as the step. Inequalities and variable bounds are converted to equalities using
squared slack variables.

Each returned point is classified as first-order KKT certified, certified with
the second-order sufficient conditions verified, or uncertified.

The method is described in KRONOS: An algorithm for solving ill-conditioned nonlinear programs; see [Citation](#citation).

## Install

```bash
pip install kronos-v1
```

The package imports as `kronos`. It requires Python 3.9 or later and installs
`numpy`, `scipy`, `sympy` and `casadi`.

Optional extras add matplotlib for figures, or JAX as a third backend:

```bash
pip install "kronos-v1[plot]"
pip install "kronos-v1[all]"
```

To install the current state of the main branch instead of the last release:

```bash
pip install git+https://github.com/SOULS-TAMU/KRONOS.git
```

For development:

```bash
git clone https://github.com/SOULS-TAMU/KRONOS.git
cd KRONOS
pip install -e ".[dev]"
pytest -q
```

Verify the installation:

```python
import kronos
print(kronos.__version__, len(kronos.problem_names()))     # 0.4.8 244
```

## Quick start

Solve one of the built-in problems:

```python
from kronos import load_problem, solve

p = load_problem("hs053")
r = solve(p, multi_start=True, ms_num_starts=25)
print(r.summary())
```

Define a problem by supplying the objective, the constraints and the bounds.

```python
from kronos import Problem, solve

p = Problem.build(
    "my_problem", ["x1", "x2", "x3", "x4"],
    objective    = "x1*x4*(x1 + x2 + x3) + x3",
    equalities   = ["x1**2 + x2**2 + x3**2 + x4**2 - 40"],   # == 0
    inequalities = ["25 - x1*x2*x3*x4"],                     # <= 0
    lb = [1, 1, 1, 1],
    ub = [5, 5, 5, 5],
    x0 = [1, 5, 5, 1],
)

r = solve(p, multi_start=True, ms_num_starts=25)
print(r.fval)      # 17.014017
print(r.theta[:4]) # [1.0, 4.743, 3.8211503, 1.3794082]
```

Set `inequality_sense=">="` for constraints of the form `g(x) >= 0`, or pass a
list with one entry per row. Expressions may be given as strings or as SymPy
objects. Problems can be written to and read from JSON with `p.save(path)` and
`Problem.load(path)`.

## Reading the result

`r.summary()` reports the outcome:

```
==============================================================
  KRONOS  |  hs053   n=15  m=13
==============================================================
  converged   : 25/25  (100.0%)
  reached f*  : 25/25  (100.0%)
  time        : 0.659 s   (0.026 s/run)
  ---- best run ----
  status      : converged (KKT certified)
  objective   : 4.093023256      f* = 4.093
  x*          : theta1     = -0.7674418605
                theta2     = 0.2558139535
                theta3     = 0.6279069767
                theta4     = -0.1162790698
                theta5     = 0.2558139535
  iterations  : 3
==============================================================
```

The first three lines summarise all the multistart runs; everything below
`best run` describes the single run that produced the reported solution. Its
`status` states whether that run is KKT certified, merely stationary for the
reformulated system, or did not converge at all, so a reported objective is
never mistaken for a solution.

`x*` lists the problem's own variables; the slack variables introduced for the
inequalities and bounds are internal and not shown.

**Converged means KKT certified**: the residual test is satisfied *and* the
inequality multipliers have the correct signs. Because inequalities are carried
as squared slacks, a run can drive the residual to zero and still end with a
wrong-signed multiplier. Such a point is stationary for the reformulated system
but is not a KKT point of the original problem, and is not counted as
converged. That looser count is available separately:

```python
r.n_conv                        # converged, i.e. KKT certified
r.n_reformulated_stationary     # stationary for the reformulated system
```

The report is plain ASCII with no colour codes or terminal control characters,
so it appears identically in a terminal, in a Jupyter or VS Code notebook, and
in a redirected file. Lines are at most 62 characters.

### Supplying a known optimum

If the optimal objective value is known in advance, supply it as `fstar`. The
report then adds how many runs reached it, and `r.global_hits()` becomes
available. Without it those lines are omitted, and everything else is unchanged:
`fstar` is used only for reporting and never influences the solve.

Three equivalent ways:

```python
p = Problem.build(..., fstar=17.0140173)    # stored with the problem
r = solve(p, fstar=17.0140173)              # for one solve
print(r.summary(fstar=17.0140173))          # for one report
```

A run counts as having reached `f*` when it is certified and

```
|f - f*| <= max(1e-4, 1e-3 * max(1, |f*|))
```

so the test is absolute for small optima and relative for large ones. Query it
directly with:

```python
r.global_hits(17.0140173)      # number of certified runs that reached it
```

The built-in problems already carry their known optima, so `load_problem`
supplies `fstar` automatically.

The same quantities are available programmatically:

| attribute | meaning |
|---|---|
| `r.fval`, `r.theta` | best objective and the corresponding point |
| `r.n_conv` | converged and certified runs (`r.n_kkt` is an alias) |
| `r.n_residual_conv` | runs satisfying the residual test, certified or not |
| `r.global_hits(fstar)` | certified runs that reached a known optimum |
| `r.runs` | one `RunResult` per multistart run |
| `r.all_fvals`, `r.all_conv`, `r.all_kkt` | per-run arrays |

Each `RunResult` provides `converged`, `reformulated_stationary`,
`min_lam_strict`, `iterations`, `max_r` and `max_h`.

### Strict local minima

A certified KKT point is not necessarily a minimum; it can be a maximum or a
saddle. The second-order sufficient condition separates them, and is checked
automatically. To count how many runs are proven minima:

```python
r = solve(p, multi_start=True, ms_num_starts=25)
print(r.n_local)          # runs proven to be strict local minima
```

The gap between `n_conv` and `n_local` splits two ways:

```python
print(f"converged {r.n_conv}, minima {r.n_local}, "
      f"failed SOSC {r.n_stationary}, untested {r.n_sosc_unmeasured}")
```

| attribute | meaning |
|---|---|
| `r.n_local` | certified **and** second-order test passed |
| `r.n_stationary` | certified, test ran, point is not a minimum |
| `r.n_sosc_unmeasured` | certified, test not performed |

Those three always sum to `r.n_conv`. Runs solved by the Fischer-Burmeister
formulation fall into the third category, because that path does not form a
reduced Hessian; not measured is not the same as failed.

Per run, the same information is available as `sosc_pass`, `sosc_measured` and
`lam_min_red`, the last being the smallest eigenvalue of the Hessian of the
Lagrangian restricted to the null space of the active constraint Jacobian. A
positive value means a strict local minimum:

```python
b = r.runs[r.best_run]
b.sosc_pass, b.sosc_measured, b.lam_min_red
```

`check_sosc=False` skips the test; `sosc_tol` sets the eigenvalue threshold.

## Built-in problems

244 problems are included, drawn from the Hock-Schittkowski collection,
CUTEst, and standard global-optimisation test functions. They range from 2 to
1000 variables and each carries its known optimum.

```python
from kronos import problem_names, find, load_problem

problem_names()                        # all 244
find("hs")                             # search by name
find(max_n=10, constrained=True)       # filter by size and constraints
find(constrained=False)                # unconstrained only
p = load_problem("hs110")
```

```bash
kronos list                            # full list with n, m and f*
kronos list --max-n 10
```

Sizes: 2-4: 97, 5-9: 72, 10-19: 41, 20-49: 18, 50+: 16. Of these, 96 are unconstrained.

```
a01_beale              a02_bohachevsky1       a03_bohachevsky2       a04_bohachevsky3       a05_branin_rcos
a06_colville           a07_dixon_price        a08_hump               a09_matyas             a10_perm
a11_powell_singular    a13_sphere             a14_sum_squares        a15_trid               a16_zakharov
a17_branin_rcos2       a18_ackley1            a19_ackley2            a20_camel3             a21_booth
a22_brown              a24_exponential        a25_freudenstein_roth  a26_miele_cantrell     a27_quadratic
a28_rotated_ellipse    a29_rump               aircrfta               aircrftb               aircrtfb
allinitu               alsotame               argauss                arglina                arglinb
arglinc                avgasa                 b03                    b06                    b08
b09                    b11                    b12                    b13                    b14
b15                    b16                    b17                    b18                    b22
b24                    b30                    bard                   bdvalue                beale
biggs3                 biggs5                 biggs6                 booth                  box2
box3                   bqp1var                brkmcc                 brownal                brownden
bt1                    bt10                   bt11                   bt12                   bt13
bt2                    bt3                    bt5                    bt6                    bt7
bt8                    byrdsphr               camel6                 catena                 chnrosnb
cliff                  cluster                coolhans               cube                   denschna
denschnb               denschnc               denschnd               denschne               denschnf
dixchlng               dixon3dq               eg1                    eigencco               eigminc
engval2                errinros               expfit                 extrasim               fccu
fletcbv2               fletchcr               genhs28                genhumps               gottfr
gridneth               growth                 growthls               hairy                  hatfldc
hatfldd                hatflde                hatfldf                hatfldg                heart6
heart8                 helix                  hilberta               hilbertb               himmelba
himmelbb               himmelbc               himmelbe               himmelbf               himmelbg
himmelbh               himmelp1               hong                   hs001                  hs003
hs004                  hs005                  hs006                  hs007                  hs008
hs009                  hs010                  hs012                  hs014                  hs015
hs017                  hs018                  hs019                  hs020                  hs023
hs026                  hs027                  hs028                  hs029                  hs030
hs031                  hs032                  hs033                  hs034                  hs038
hs040                  hs042                  hs043                  hs046                  hs047
hs048                  hs049                  hs050                  hs051                  hs052
hs053                  hs054                  hs055                  hs056                  hs060
hs061                  hs063                  hs065                  hs066                  hs071
hs077                  hs078                  hs079                  hs080                  hs081
hs100nlp               hs107                  hs108                  hs110                  hs113
hs3mod                 humps                  hypcir                 jensmp                 kowosb
lch                    lewispol               loghairy               lotschd                lsnnodoc
madsen                 maratos                mconcon                mexhat                 mwright
nasty                  orthregb               osbornea               oslbqp                 palmer1c
palmer1d               palmer2c               palmer3c               palmer3e               palmer4c
palmer5b               palmer5c               palmer5d               palmer6c               palmer6e
palmer7c               palmer7e               palmer8c               palmer8e               penalty2
pfit4ls                pk3                    powellsq               power                  pspdoc
qrtquad                rk23                   rosenbr                sim2bqp                simbqp
sineval                sisser                 ssnlbeam               supersim               tame
tointqor               try_b                  twobars                vardim                 yfit
yfitu                  zangwil2               zangwil3               zecevic3
```

## Plots

```python
from kronos.plotting import plot_runs
fig = plot_runs(r, fstar=p.fstar)      # per-run objective, coloured by status
fig.savefig("runs.png", dpi=150)
```

Green marks converged (KKT-certified) runs, amber runs that met the residual
test without certification, and grey failed runs.

## Command line

```bash
kronos list                            # built-in problems
kronos solve hs110 -K 25 --plot runs.png
kronos bench hs001 hs053 -K 25 --out bench.csv
kronos options                         # every setting, grouped
```

## Options

Options may be passed as keyword arguments to `solve` or collected in an
`Options` object. The convergence tolerances are the most commonly adjusted:

```python
r = solve(p, tol_r=1e-8, tol_h=1e-8, maxIter=5000)   # default is 1e-5
```

```python
from kronos import Options
opts = Options(tol_r=1e-8, multi_start=True, ms_num_starts=50)
r = solve(p, opts)
```

| option | default | meaning |
|---|---|---|
| `tol_r`, `tol_h` | `1e-5` | tolerances on the KKT residual and the constraint violation |
| `maxIter` | `2000` | Newton iterations per run |
| `multi_start` | `False` | solve from several starting points |
| `ms_num_starts`, `ms_seed` | `25`, `42` | number of starting points and the random seed |
| `use_adam_warmup` | `True` | first-order warm-up before Newton |
| `adam_mode` | `"C"` | how the warm-up is applied across starting points, see below |
| `backend` | `"auto"` | SymPy below 20 variables, CasADi at or above |
| `fstar` | `None` | known optimum, enabling the `reached f*` lines |
| `max_kkt_kicks` | `5` | retries when a run converges without certification |

When many runs converge without being certified, increasing `max_kkt_kicks`
often helps.

### `adam_mode`

The solve has three stages: a first-order warm-up, a pre-feasibility pass, and
the main Newton solve. With `K` starting points, `adam_mode` decides how the
first two are distributed across them.

| mode | warm-up | pre-feasibility | main solve |
|---|---|---|---|
| `"A"` | every starting point | first point only | one call receiving all `K` points |
| `"B"` | first point only | first point only | one call, which scatters `K` fresh points around it |
| `"C"` | every starting point | every starting point | one call per starting point |

`"C"` is the default and the most thorough: each starting point goes through
the whole pipeline independently. `"B"` is the cheapest, but note that it
scatters its `K-1` remaining points *after* the warm-up, so those never benefit
from it.

The choice matters on harder problems. On `dixchlng`, for example, `"A"` and
`"C"` reach the known optimum from all 25 starts while `"B"` reaches it from
one.

All 58 options are documented in place:

```python
from kronos import Options
print(Options.describe())              # all options
print(Options.describe("Convergence")) # a single group
```

```bash
kronos options
kronos options certification
```

The groups are Convergence, Multistart, Line search and robustness, Warm-up and
pre-feasibility, Certification, Numerics, Backend, Output, and Advanced
multiplier handling.

## Backends

With `backend="auto"` the solver uses SymPy below 20 variables and CasADi at or
above that threshold. JAX is available as a third option. The backends produce
mathematically equivalent derivatives; floating-point evaluation differs
between them, so a run may follow a slightly different trajectory.

```python
from kronos import get_backend
b = get_backend(p, "casadi")     # compile once
r = solve(p, backend=b)          # reuse across repeated solves
```

Compiling the derivatives is often the dominant one-off cost, so reusing a
backend is worthwhile when solving the same problem many times.

When running several solves in parallel, limit BLAS threads to avoid
oversubscription:

```bash
OMP_NUM_THREADS=1 python my_script.py
```

## Citation

Ahmed, M G T. and Hasan, M. M. F. (2026) 'KRONOS: An algorithm for solving
ill-conditioned nonlinear programs', *Computers & Chemical Engineering*, 215,
109839. doi: 10.1016/j.compchemeng.2026.109839.

```
M G Toufik Ahmed, M.M. Faruque Hasan,
KRONOS: An algorithm for solving ill-conditioned nonlinear programs,
Computers & Chemical Engineering,
Volume 215,
2026,
109839,
ISSN 0098-1354,
https://doi.org/10.1016/j.compchemeng.2026.109839.
```

```bibtex
@article{ahmed2026kronos,
  title   = {KRONOS: An algorithm for solving ill-conditioned nonlinear programs},
  author  = {Ahmed, M G Toufik and Hasan, M. M. Faruque},
  journal = {Computers \& Chemical Engineering},
  volume  = {215},
  pages   = {109839},
  year    = {2026},
  issn    = {0098-1354},
  doi     = {10.1016/j.compchemeng.2026.109839},
}
```

## Tutorial

A worked introduction is provided in [`docs/tutorial.ipynb`](docs/tutorial.ipynb),
covering problem definition, the certification output, backends, plotting and
the built-in problem set.

## Tests

```bash
pytest -q
```

## License

MIT. See [`LICENSE`](LICENSE).

Copyright (c) 2026 M G Toufik Ahmed and M. M. Faruque Hasan.
