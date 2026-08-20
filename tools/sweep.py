"""Run a slice of the benchmark library and write metrics to CSV.

BLAS threading must be pinned *before* numpy is imported.  Several sweeps
running in parallel each spawning a full thread pool oversubscribes the CPU
badly -- measured here at ~200x slower dense linear algebra -- so default to
one thread per process and let the caller run several sweeps side by side.
"""
from __future__ import annotations
import argparse, csv, json, os, signal, sys, time
from pathlib import Path

_THREADS = os.environ.get("KRONOS_BLAS_THREADS", "1")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, _THREADS)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.setrecursionlimit(200000)

from kronos.benchmark import Metrics, run_problem
from kronos.library import load_problem, problem_names


class Timeout(Exception):
    pass


def _alarm(sig, frame):
    raise Timeout()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-n", type=int, default=0)
    ap.add_argument("--max-n", type=int, default=10**9)
    ap.add_argument("--K", type=int, default=25)
    ap.add_argument("--timeout", type=int, default=900, help="seconds per problem")
    ap.add_argument("--backend", default=None)
    ap.add_argument("--step", default=None,
                    help="step_method override: cod | svd | pinv | lstsq | tikhonov | backslash")
    args = ap.parse_args()

    names = []
    for nm in problem_names():
        p = load_problem(nm)
        if args.min_n <= p.n <= args.max_n:
            names.append((nm, p.n, p.m))
    names.sort(key=lambda t: (t[1], t[2]))
    print(f"{len(names)} problems with {args.min_n} <= n <= {args.max_n}", flush=True)

    signal.signal(signal.SIGALRM, _alarm)
    done = {}
    out = Path(args.out)
    if out.is_file():
        for row in csv.DictReader(open(out)):
            done[row["problem"]] = row

    fh = open(out, "w", newline="")
    writer = None
    t_all = time.time()
    for i, (nm, n, m) in enumerate(names, 1):
        t0 = time.time()
        signal.alarm(args.timeout)
        try:
            kw = {"step_method": args.step} if args.step else {}
            mt = run_problem(nm, K=args.K, backend=args.backend, **kw)
        except Timeout:
            p = load_problem(nm)
            mt = Metrics(problem=nm, n=n, m=m, K=args.K, n_conv=0, n_kkt=0, n_global=0,
                         n_kkt_global=0, n_local=0, n_stationary=0, Lbar=float("nan"),
                         best_fval=float("nan"), fstar=p.fstar, mean_time=float("nan"),
                         total_time=time.time() - t0, backend="-",
                         error=f"TIMEOUT after {args.timeout}s")
        except Exception as exc:
            p = load_problem(nm)
            mt = Metrics(problem=nm, n=n, m=m, K=args.K, n_conv=0, n_kkt=0, n_global=0,
                         n_kkt_global=0, n_local=0, n_stationary=0, Lbar=float("nan"),
                         best_fval=float("nan"), fstar=p.fstar, mean_time=float("nan"),
                         total_time=time.time() - t0, backend="-",
                         error=f"{type(exc).__name__}: {exc}")
        finally:
            signal.alarm(0)

        if writer is None:
            writer = csv.DictWriter(fh, fieldnames=list(mt.as_row()))
            writer.writeheader()
        writer.writerow(mt.as_row())
        fh.flush()
        tag = mt.error or f"conv {mt.n_conv}/{args.K} glob {mt.n_global} f={mt.best_fval:.6g}"
        print(f"[{i:3d}/{len(names)}] {nm:<28} n={n:<4} {time.time()-t0:7.1f}s  {tag}", flush=True)
    fh.close()
    print(f"SWEEP DONE in {(time.time()-t_all)/60:.1f} min -> {out}", flush=True)


main()
