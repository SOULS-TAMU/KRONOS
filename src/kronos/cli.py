"""Command line interface: ``kronos <command> ...``
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def _cmd_list(args):
    from .library import load_problem, problem_names
    names = problem_names()
    print(f"{len(names)} bundled problems\n")
    print(f"{'name':<30}{'n':>6}{'m':>6}{'f*':>16}")
    for nm in names:
        p = load_problem(nm)
        if args.max_n is not None and p.n > args.max_n:
            continue
        fs = "-" if p.fstar is None else f"{p.fstar:.6g}"
        print(f"{nm:<30}{p.n:>6}{p.m:>6}{fs:>16}")


def _cmd_solve(args):
    from .api import solve
    from .benchmark import options_from_problem
    from .library import load_problem
    p = load_problem(args.problem)
    opts = options_from_problem(p, K=args.K, verbose=args.verbose)
    if args.backend:
        opts = opts.copy(backend=args.backend)
    if args.step:
        opts = opts.copy(step_method=args.step)
    r = solve(p, opts)
    K = len(r.runs)
    print(f"\nproblem   : {p.name}  (n={p.n}, m={p.m})")
    print(f"objective : {r.fval:.10g}" + (f"   f* = {p.fstar:.10g}" if p.fstar is not None else ""))
    print(f"converged : {r.n_conv}/{K}")
    if p.fstar is not None:
        g = r.global_hits(p.fstar)
        ratio = f"   ({100 * g / r.n_conv:.1f}% of converged)" if r.n_conv else ""
        print(f"reached f*: {g}/{K}{ratio}")
    if args.verbose:
        print(f"residual  : {r.n_residual_conv}/{K} met the residual test "
              f"({r.n_residual_conv - r.n_conv} of them uncertified)")
        print(f"solver    : {r.solver_used}")
    print(f"time      : {r.elapsed:.2f} s total, {r.elapsed / max(K, 1):.3f} s/run")
    print("x* =", np.array2string(r.theta[:min(10, r.theta.size)], precision=8))
    if args.plot:
        from .plotting import plot_runs
        fig = plot_runs(r, fstar=p.fstar)
        fig.savefig(args.plot, dpi=140)
        print(f"plot      : {args.plot}")


def _cmd_options(args):
    from .options import Options
    print(Options.describe(args.group))


def _cmd_bench(args):
    from .benchmark import run_suite, compare
    from .library import load_problem, problem_names, resolve_name
    if args.problems:
        names = [resolve_name(nm) for nm in args.problems]
    else:
        names = [nm for nm in problem_names()
                 if args.min_n <= load_problem(nm).n
                 <= (args.max_n if args.max_n is not None else 10 ** 9)]
    rows = run_suite(names, K=args.K, out_csv=args.out)
    from .plotting import summary_table
    print("\n" + summary_table(rows, K=args.K))
    if args.reference:
        cmp = compare(rows, args.reference)
        same_c = sum(1 for c in cmp if c["conv_ref"] == c["conv_py"])
        same_g = sum(1 for c in cmp if c["glob_ref"] == c["glob_py"])
        print(f"\nvs reference: n_conv identical on {same_c}/{len(cmp)}, "
              f"n_global identical on {same_g}/{len(cmp)}")
        if args.K != 25:
            print(f"  note: the reference was produced at K=25; you ran K={args.K}, "
                  f"so the counts are not directly comparable.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="kronos", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list", help="list bundled benchmark problems")
    p.add_argument("--max-n", type=int, default=None)
    p.set_defaults(func=_cmd_list)

    p = sub.add_parser("solve", help="solve one bundled problem")
    p.add_argument("problem")
    p.add_argument("-K", type=int, default=25, help="multistart runs")
    p.add_argument("--backend", choices=["auto", "sympy", "casadi", "jax"])
    p.add_argument("--step", choices=["cod", "pinv", "lstsq", "tikhonov", "backslash"])
    p.add_argument("--plot", help="write a PNG of the per-run objectives")
    p.add_argument("--verbose", action="store_true",
                   help="also show the solver used and the uncertified count")
    p.set_defaults(func=_cmd_solve)

    p = sub.add_parser("options", help="list every solver option with its default")
    p.add_argument("group", nargs="?", default=None,
                   help="only show groups matching this text, e.g. 'converg'")
    p.set_defaults(func=_cmd_options)

    p = sub.add_parser("bench", help="run the benchmark suite")
    p.add_argument("problems", nargs="*", help="problem names (default: all matching --min-n/--max-n)")
    p.add_argument("--min-n", type=int, default=0)
    p.add_argument("--max-n", type=int, default=None)
    p.add_argument("-K", type=int, default=25)
    p.add_argument("--out", default="kronos_bench.csv")
    p.add_argument("--reference", help="reference CSV to diff against")
    p.set_defaults(func=_cmd_bench)

    args = ap.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
