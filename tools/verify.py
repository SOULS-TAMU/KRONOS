"""Verify the shipped library against the MATLAB reference table."""
from __future__ import annotations
import argparse, csv, os, sys, time
from pathlib import Path
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.setrecursionlimit(200000)

from kronos.benchmark import run_problem
from kronos.library import load_problem, problem_names

ap = argparse.ArgumentParser()
ap.add_argument("--reference", required=True)
ap.add_argument("-K", type=int, default=25)
ap.add_argument("--out", default="results/verify.csv")
args = ap.parse_args()

ref = {r["problem"]: r for r in csv.DictReader(open(args.reference)) if r.get("config") == "v3"}
names = list(problem_names())
rows, t0 = [], time.time()
print(f"{'problem':<22}{'n':>4}{'conv':>9}{'global':>10}{'ref c/g':>11}{'s':>7}")
ce = cb = cw = ge = gb = gw = 0
for i, nm in enumerate(names, 1):
    p = load_problem(nm)
    src = p.meta.get("source_name", nm)
    m = run_problem(nm, K=args.K)
    R = ref.get(src, {})
    rc = float(R.get("n_conv", "nan")); rg = float(R.get("n_global", "nan"))
    ce += m.n_conv == rc; cb += m.n_conv > rc; cw += m.n_conv < rc
    ge += m.n_global == rg; gb += m.n_global > rg; gw += m.n_global < rg
    flag = "" if (m.n_conv >= rc and m.n_global >= rg) else "  <-- below"
    print(f"{nm:<22}{p.n:>4}{m.n_conv:>6}/{args.K:<3}{m.n_global:>7}/{args.K:<3}"
          f"{int(rc):>6}/{int(rg):<4}{m.total_time:>7.1f}{flag}", flush=True)
    rows.append(m)
n = len(names)
print("\n" + "=" * 62)
print(f"  {n} problems, K={args.K}, {(time.time()-t0)/60:.1f} min total")
print(f"  n_conv    identical {ce:3d}  better {cb:3d}  worse {cw:3d}  -> {100*(ce+cb)/n:5.1f}% match-or-beat")
print(f"  n_global  identical {ge:3d}  better {gb:3d}  worse {gw:3d}  -> {100*(ge+gb)/n:5.1f}% match-or-beat")
print("=" * 62)
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
with open(args.out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].as_row())); w.writeheader()
    for r in rows: w.writerow(r.as_row())
