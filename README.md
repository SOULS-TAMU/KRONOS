# KRONOS

KKT-certified nonlinear programming.

KRONOS solves

$$
\begin{aligned}
\min_{x} \quad & f(x) \\
\textrm{s.t.} \quad & h(x) = 0 \\
& g(x) \le 0 \\
& x^{L} \le x \le x^{U}
\end{aligned}
$$

by forming the full KKT system and driving its residual to zero with Newton's
method. The Newton step is the minimum-norm least-squares solution of the KKT
linear system, which keeps the iteration well defined when that system is rank
deficient, as it commonly is at a solution. Inequalities and variable bounds
are converted to equalities using squared slack variables, so the method uses
no active-set strategy and no barrier parameter.

Each returned point is classified as first-order KKT certified, certified with
the second-order sufficient conditions verified, or uncertified.

## Install

```bash
pip install https://github.com/toufik3078/test_kronos/archive/refs/heads/main.zip
```

This requires no `git` and no manual download. It installs `numpy`, `scipy`,
`sympy` and `casadi`. CasADi is a required dependency because the solver routes
to it above roughly 20 variables, where it builds derivatives one to two orders
of magnitude faster than SymPy (for n = 1000, 0.2 s against 35 s).

If `git` is available:

```bash
pip install git+https://github.com/toufik3078/test_kronos.git
```

Optional extras add matplotlib for figures, or JAX as a third backend:

```bash
pip install "kronos[plot] @ https://github.com/toufik3078/test_kronos/archive/refs/heads/main.zip"
pip install "kronos[all]  @ https://github.com/toufik3078/test_kronos/archive/refs/heads/main.zip"
```

For development:

```bash
git clone https://github.com/toufik3078/test_kronos.git
cd test_kronos
pip install -e ".[dev]"
pytest -q
```

Verify the installation:

```python
import kronos
print(kronos.__version__, len(kronos.problem_names()))     # 0.3.1 244
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
Slack variables are introduced internally and need not be provided:

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

`r.summary()` reports the objective, the multistart statistics and the timing:

```
  objective           : 4.093023256
  known optimum f*    : 4.093   (gap 2.326e-05)
  ---- multistart ----
  runs                : 25
  converged           : 25/25  (100.0%)   [KKT-certified]
  reached f*          : 25/25  (100.0%)
  f* / converged      : 100.0%
  ---- best run ----
  certified           : True   (min signed multiplier -6.0e-20)
  iterations          : 11
  final |KKT residual|: 2.276e-08
```

A run counts as converged only if it is KKT certified, meaning the residual
test is satisfied and the inequality multipliers have the correct signs. A run
that reduces the residual but ends with a wrong-signed multiplier is a
stationary point of the reformulated problem rather than a KKT point of the
original one, and is excluded. The looser count is available with

```python
print(r.summary(show_uncertified=True))     # adds a "residual-converged" line
```

The two `f*` lines are shown only when a known optimum is available, either
from a built-in problem or because one was supplied:

```python
r = solve(p, fstar=17.0140173)              # or Problem.build(..., fstar=...)
```

The same quantities are available programmatically:

| attribute | meaning |
|---|---|
| `r.fval`, `r.theta` | best objective and the corresponding point |
| `r.n_conv` | converged and certified runs (`r.n_kkt` is an alias) |
| `r.n_residual_conv` | runs satisfying the residual test, certified or not |
| `r.global_hits(fstar)` | certified runs that reached a known optimum |
| `r.runs` | one `RunResult` per multistart run |
| `r.all_fvals`, `r.all_conv`, `r.all_kkt` | per-run arrays |

Each `RunResult` provides `converged`, `kkt_certified`, `min_lam_strict`,
`iterations`, `max_r` and `max_h`.

Second-order information is computed but not printed. Use `r.n_local` for the
number of runs proven to be strict local minima, and `sosc_pass`,
`sosc_measured` and `lam_min_red` on individual runs.

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

Green marks KKT-certified runs, amber converged but uncertified runs, and grey
failed runs.

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

The defaults match the configuration used to validate the built-in problems:
Adam warm-up enabled, `adam_mode="C"`, and the multiplier-sign check enforced.
This configuration is both more reliable and generally faster than a plain
Newton solve, since a warm start requires far fewer iterations.

| option | default | meaning |
|---|---|---|
| `tol_r`, `tol_h` | `1e-5` | tolerances on the KKT residual and the constraint violation |
| `maxIter` | `2000` | Newton iterations per run |
| `multi_start` | `False` | solve from several starting points |
| `ms_num_starts`, `ms_seed` | `25`, `42` | number of starting points and the random seed |
| `use_adam_warmup` | `True` | first-order warm-up before Newton |
| `adam_mode` | `"C"` | full pipeline per starting point |
| `backend` | `"auto"` | SymPy below 20 variables, CasADi at or above |
| `fstar` | `None` | known optimum, enabling the `reached f*` lines |
| `max_kkt_kicks` | `5` | retries when a run converges without certification |

When many runs converge without being certified, increasing `max_kkt_kicks` is
usually the most effective adjustment. Measured across 28 problems it raised
the certified count by about 3% at a 35% increase in run time.

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

Derivatives are supplied by an interchangeable backend; the algorithm itself is
fixed. With `backend="auto"` the solver uses SymPy below 20 variables and
CasADi at or above that threshold. JAX is available as a third option. All
three produce identical derivatives, so the choice affects speed only.

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

## Method

The solve proceeds in stages:

1. **Adam warm-up.** First-order descent on `f + rho*||h||^2` to move the
   starting point into a better basin.
2. **Pre-feasibility.** The same KKT iteration with `f := 0`, reducing `||h||`.
3. **Main solve.** Minimum-norm Newton steps on the full KKT system, with a
   projected backtracking line search, feasibility restoration, and random
   perturbations on stagnation.
4. **Fischer-Burmeister fallback.** Starting points that converge to a
   dual-infeasible multiplier are retried with the slacks eliminated and the
   complementarity conditions imposed through a smoothed Fischer-Burmeister
   function, which makes `mu >= 0` structural.
5. **Classification.** The reduced Hessian on the null space of the active
   constraint Jacobian distinguishes strict local minima from other stationary
   points.

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
