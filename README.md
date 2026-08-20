# KRONOS

**KKT-certified nonlinear programming.**

KRONOS solves

```
min  f(x)     s.t.   h(x) = 0,   g(x) <= 0,   xL <= x <= xU
```

by driving the full KKT system to zero with a Newton iteration whose step is
the **minimum-norm least-squares solution** of the KKT linear system. That step
is what keeps the iteration well defined when the system is rank deficient,
which at a solution it usually is. There is no active-set logic and no barrier
parameter.

Every result is certified: first-order KKT verified, second-order conditions
verified (a genuine local minimum), or explicitly uncertified.

---

## Install

```bash
pip install git+https://github.com/toufik3078/test_kronos.git
```

That is the whole installation — one command, nothing to clone or build. It
pulls `numpy`, `scipy`, `sympy` and `casadi`. CasADi is included because the
solver routes to it above ~20 variables, where it builds derivatives one to two
orders of magnitude faster than SymPy (n=1000: 0.2 s versus 35 s).

With figures, or with JAX as a third backend:

```bash
pip install "kronos[plot] @ git+https://github.com/toufik3078/test_kronos.git"
pip install "kronos[all]  @ git+https://github.com/toufik3078/test_kronos.git"
```

To work on the code instead of just using it:

```bash
git clone https://github.com/toufik3078/test_kronos.git
cd test_kronos
pip install -e ".[dev]"
pytest -q
```

Check it worked:

```python
import kronos
print(kronos.__version__, len(kronos.problem_names()))     # 0.3.1 244
```

---

## Quick start

Solve one of the built-in problems:

```python
from kronos import load_problem, solve

p = load_problem("hs053")
r = solve(p, multi_start=True, ms_num_starts=25)
print(r.summary())
```

Or state your own. Give it the objective, the constraints and the bounds —
inequalities and bounds are handled internally, you do not add anything
yourself:

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

Use `inequality_sense=">="` for `>=` constraints (or a list, one per row).
Expressions can be strings or SymPy objects. `p.save("mine.json")` and
`Problem.load(...)` round-trip.

---

## Reading the result

`r.summary()` prints the full picture:

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

**"Converged" means KKT-certified**: the residual test passed *and* the
multipliers have the right signs. A run that drives the residual to zero but
ends with a wrong-signed multiplier is a stationary point of the reformulated
problem, not a solution of yours, so it is not counted. To see the looser
number too:

```python
print(r.summary(show_uncertified=True))     # adds a "residual-converged" line
```

The `reached f*` lines appear only when a known optimum is available — either
from a bundled problem, or because you supplied one:

```python
r = solve(p, fstar=17.0140173)              # or Problem.build(..., fstar=...)
print(r.summary())
```

Programmatically:

| | |
|---|---|
| `r.fval`, `r.theta` | best objective and the point that achieves it |
| `r.n_conv` | runs that converged **and** are certified (`r.n_kkt` is an alias) |
| `r.n_residual_conv` | runs that met the residual test, certified or not |
| `r.global_hits(fstar)` | certified runs that reached a known optimum |
| `r.runs` | one `RunResult` per multistart run |
| `r.all_fvals`, `r.all_conv`, `r.all_kkt` | per-run arrays |

Each `RunResult` carries `converged`, `kkt_certified`, `min_lam_strict`,
`iterations`, `max_r`, `max_h`.

**Converged is not the same as certified.** A run can drive the KKT residual to
zero yet end with an inequality multiplier of the wrong sign — a stationary
point of the reformulated problem that is not a KKT point of yours.
`kkt_certified` is the one to check.

A second-order test is also available if you want proof of a strict local
minimum: `r.n_local`, and per run `sosc_pass` / `sosc_measured` /
`lam_min_red`.

---

## Built-in problems

244 problems ship with the package — Hock-Schittkowski, CUTEst, and standard
global-optimisation test functions, from 2 to 1000 variables, each with its
known optimum.

```python
from kronos import problem_names, find, load_problem

problem_names()                        # all 244
find("hs")                             # by name
find(max_n=10, constrained=True)       # by size / constrainedness
find(constrained=False)                # unconstrained only
p = load_problem("hs110")
```

```bash
kronos list                            # the whole list, with n, m and f*
kronos list --max-n 10                 # just the small ones
```

All 244 names (n=2-4: 97  n=5-9: 72  n=10-19: 41  n=20-49: 18  n=50+: 16; 96 unconstrained):

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

---

## Plots

```python
from kronos.plotting import plot_runs
fig = plot_runs(r, fstar=p.fstar)      # per-run objective, coloured by status
fig.savefig("runs.png", dpi=150)
```

Green = KKT-certified, amber = converged but uncertified, grey = failed.

---

## Command line

```bash
kronos list                            # the built-in problems
kronos solve hs110 -K 25 --plot runs.png
kronos bench hs001 hs053 -K 25 --out bench.csv
```

---

## Options

Pass any option as a keyword to `solve`, or build an `Options` object and reuse
it. The most common one is the convergence tolerance:

```python
r = solve(p, tol_r=1e-8, tol_h=1e-8, maxIter=5000)   # tighter than the default 1e-5
```

```python
from kronos import Options
opts = Options(tol_r=1e-8, multi_start=True, ms_num_starts=50)
r = solve(p, opts)
```

The defaults are the configuration the bundled problems were validated with —
Adam warm-up on, `adam_mode="C"`, multiplier-sign check enforced. These are not
only more reliable than a bare Newton solve, they are usually *faster*, because
a warm start converges in far fewer iterations than one that thrashes.

The ones you are most likely to touch:

| option | default | |
|---|---|---|
| `tol_r`, `tol_h` | `1e-5` | converged when the KKT residual and the constraint violation fall below these |
| `maxIter` | `2000` | Newton iterations per run |
| `multi_start` | `False` | run from many starting points |
| `ms_num_starts`, `ms_seed` | `25`, `42` | how many, and the seed |
| `use_adam_warmup` | `True` | first-order warm-up before Newton |
| `adam_mode` | `"C"` | full pipeline per starting point; the most thorough |
| `backend` | `"auto"` | `sympy` below 20 variables, `casadi` above |
| `fstar` | `None` | known optimum; enables the `reached f*` lines |
| `max_kkt_kicks` | `5` | retries when a run converges but fails certification — raise it if `converged` is much lower than `residual-converged` |

If `summary(show_uncertified=True)` shows many runs converging without being
certified, raising `max_kkt_kicks` is usually the most effective single change
(measured across 28 problems: +3% certified runs for +35% time).

**All 58 options, grouped and explained**, without leaving Python:

```python
from kronos import Options
print(Options.describe())              # everything
print(Options.describe("Convergence")) # just one group
```

or from the shell:

```bash
kronos options
kronos options certification
```

Groups: Convergence, Multistart, Line search and robustness, Warm-up and
pre-feasibility, Certification, Numerics, Backend, Output, Advanced multiplier
handling.

---

## Backends

Derivatives are pluggable; the algorithm is not. `"auto"` uses SymPy below 20
variables and CasADi at or above it. JAX is also supported. All three produce
identical derivatives; pick CasADi or JAX for large problems.

```python
from kronos import get_backend
b = get_backend(p, "casadi")     # compile once
r = solve(p, backend=b)          # reuse across many solves
```

For large problems, avoid oversubscribing BLAS threads when running solves in
parallel — set `OMP_NUM_THREADS=1` per process.

---

## Tutorial

A full walkthrough is in [`docs/tutorial.ipynb`](docs/tutorial.ipynb)
(rendered: `docs/tutorial.html`).

---

## License

MIT — see [`LICENSE`](LICENSE).

Copyright (c) 2026 M G Toufik Ahmed & M. M. Faruque Hasan.
