"""Compare sweep results against the MATLAB reference and emit a report."""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("--reference", required=True)
    ap.add_argument("--plots", default=None, help="directory to write PNGs into")
    args = ap.parse_args()

    ref = {r["problem"]: r for r in csv.DictReader(open(args.reference))
           if r.get("config") == "v3"}
    rows = []
    for path in args.csvs:
        rows.extend(list(csv.DictReader(open(path))))

    stats = dict(n=0, ce=0, cb=0, cw=0, ge=0, gb=0, gw=0)
    errs, worse, tr, tp = [], [], [], []
    for r in rows:
        R = ref.get(r["problem"])
        if R is None:
            continue
        if r.get("error"):
            errs.append((r["problem"], r["error"][:70]))
            continue
        stats["n"] += 1
        cr, cp = num(R["n_conv"]), num(r["n_conv"])
        gr, gp = num(R["n_global"]), num(r["n_global"])
        stats["ce"] += cp == cr; stats["cb"] += cp > cr; stats["cw"] += cp < cr
        stats["ge"] += gp == gr; stats["gb"] += gp > gr; stats["gw"] += gp < gr
        if cp < cr or gp < gr:
            worse.append((r["problem"], int(cr), int(cp), int(gr), int(gp),
                          num(r["best_fval"]), num(R["fstar"]), num(r["n"])))
        tr.append(num(R["mean_time"])); tp.append(num(r["mean_time"]))

    n = max(stats["n"], 1)
    print("=" * 74)
    print(f"  KRONOS python vs MATLAB reference   ({stats['n']} problems compared, "
          f"{len(errs)} errored)")
    print("=" * 74)
    print(f"  n_conv    identical {stats['ce']:3d}   better {stats['cb']:3d}   "
          f"worse {stats['cw']:3d}    -> {100*(stats['ce']+stats['cb'])/n:5.1f}% match-or-beat")
    print(f"  n_global  identical {stats['ge']:3d}   better {stats['gb']:3d}   "
          f"worse {stats['gw']:3d}    -> {100*(stats['ge']+stats['gb'])/n:5.1f}% match-or-beat")

    tr, tp = np.array(tr), np.array(tp)
    ok = np.isfinite(tr) & np.isfinite(tp) & (tp > 0)
    if ok.any():
        sp = tr[ok] / tp[ok]
        print(f"\n  speed     median {np.median(sp):8.0f}x   mean {np.mean(sp):8.0f}x   "
              f"min {np.min(sp):6.1f}x   max {np.max(sp):8.0f}x")
        print(f"            MATLAB {tr[ok].sum()*25/3600:7.2f} h   ->   "
              f"python {tp[ok].sum()*25/60:7.1f} min")

    if errs:
        print(f"\n  ERRORS ({len(errs)}):")
        for p, e in errs[:15]:
            print(f"    {p:<28} {e}")
    if worse:
        print(f"\n  BELOW REFERENCE ({len(worse)}):")
        print(f"    {'problem':<26}{'n':>4}{'conv r/p':>11}{'glob r/p':>11}"
              f"{'f_py':>16}{'fstar':>16}")
        for p, cr, cp, gr, gp, fp, fs, nn in sorted(worse, key=lambda w: (w[3]-w[4])+(w[1]-w[2])):
            print(f"    {p:<26}{int(nn):>4}{cr:>6}/{cp:<4}{gr:>6}/{gp:<4}{fp:>16.8g}{fs:>16.8g}")

    if args.plots:
        outdir = Path(args.plots); outdir.mkdir(parents=True, exist_ok=True)
        from kronos.plotting import plot_comparison
        cmp = [{"conv_ref": num(ref[r["problem"]]["n_conv"]), "conv_py": num(r["n_conv"]),
                "glob_ref": num(ref[r["problem"]]["n_global"]), "glob_py": num(r["n_global"]),
                "time_ref": num(ref[r["problem"]]["mean_time"]), "time_py": num(r["mean_time"])}
               for r in rows if r["problem"] in ref and not r.get("error")]
        for metric in ("conv", "glob", "time"):
            fig = plot_comparison(cmp, metric=metric)
            fig.savefig(outdir / f"compare_{metric}.png", dpi=140)
        print(f"\n  plots -> {outdir}/compare_{{conv,glob,time}}.png")


main()
