"""Run several built-in problems and plot one of them."""
from kronos import find, load_problem, solve
from kronos.plotting import plot_runs

print(f"{'problem':<14}{'n':>4}{'m':>4}{'conv':>8}{'certified':>11}{'reached f*':>12}")
for name in find(max_n=12, constrained=True)[:8]:
    p = load_problem(name)
    r = solve(p, multi_start=True, ms_num_starts=25, ms_seed=42)
    print(f"{name:<14}{p.n:>4}{p.m:>4}{r.n_conv:>6}/25{r.n_kkt:>9}/25"
          f"{r.global_hits(p.fstar):>10}/25")

p = load_problem("hs030")
r = solve(p, multi_start=True, ms_num_starts=25, ms_seed=42)
fig = plot_runs(r, fstar=p.fstar)
fig.savefig("hs030_runs.png", dpi=150)
print("\nwrote hs030_runs.png")
