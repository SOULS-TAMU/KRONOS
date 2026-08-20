"""Head-to-head: the COD step vs the SVD/Moore-Penrose
step , both against the MATLAB reference."""
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


def load(paths):
    out = {}
    for p in paths:
        if Path(p).is_file():
            for r in csv.DictReader(open(p)):
                out[r["problem"]] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cod", nargs="+", required=True)
    ap.add_argument("--svd", nargs="+", required=True)
    ap.add_argument("--reference", required=True)
    args = ap.parse_args()

    cod, svd = load(args.cod), load(args.svd)
    ref = {r["problem"]: r for r in csv.DictReader(open(args.reference))
           if r.get("config") == "v3"}
    common = [p for p in cod if p in svd and p in ref]

    print("=" * 78)
    print(f"  COD (lsqminnorm, the code)  vs  SVD (J^dagger, the algorithm box)")
    print(f"  {len(common)} problems, K=25")
    print("=" * 78)

    agg = {}
    for tag, src in (("COD", cod), ("SVD", svd)):
        ce = cb = cw = ge = gb = gw = 0
        tot_c = tot_g = 0
        times = []
        for p in common:
            r, x = ref[p], src[p]
            if x.get("error"):
                continue
            cr, cp = num(r["n_conv"]), num(x["n_conv"])
            gr, gp = num(r["n_global"]), num(x["n_global"])
            ce += cp == cr; cb += cp > cr; cw += cp < cr
            ge += gp == gr; gb += gp > gr; gw += gp < gr
            tot_c += cp; tot_g += gp
            times.append(num(x["mean_time"]))
        n = max(ce + cb + cw, 1)
        agg[tag] = dict(ce=ce, cb=cb, cw=cw, ge=ge, gb=gb, gw=gw,
                        conv=tot_c, glob=tot_g,
                        match_c=100 * (ce + cb) / n, match_g=100 * (ge + gb) / n,
                        t=np.nansum(times))
        print(f"\n  {tag} vs MATLAB reference")
        print(f"    n_conv   identical {ce:3d}  better {cb:3d}  worse {cw:3d}"
              f"   -> {agg[tag]['match_c']:5.1f}% match-or-beat")
        print(f"    n_global identical {ge:3d}  better {gb:3d}  worse {gw:3d}"
              f"   -> {agg[tag]['match_g']:5.1f}% match-or-beat")

    a, b = agg["COD"], agg["SVD"]
    print("\n" + "-" * 78)
    print(f"  {'metric':<34}{'COD':>12}{'SVD':>12}{'difference':>14}")
    print(f"  {'total converged runs':<34}{a['conv']:>12.0f}{b['conv']:>12.0f}{b['conv']-a['conv']:>+14.0f}")
    print(f"  {'total global hits':<34}{a['glob']:>12.0f}{b['glob']:>12.0f}{b['glob']-a['glob']:>+14.0f}")
    print(f"  {'n_conv match-or-beat (%)':<34}{a['match_c']:>12.1f}{b['match_c']:>12.1f}{b['match_c']-a['match_c']:>+14.1f}")
    print(f"  {'n_global match-or-beat (%)':<34}{a['match_g']:>12.1f}{b['match_g']:>12.1f}{b['match_g']-a['match_g']:>+14.1f}")
    print(f"  {'total wall time (min, K=25)':<34}{a['t']*25/60:>12.1f}{b['t']*25/60:>12.1f}{(b['t']-a['t'])*25/60:>+14.1f}")

    # per-problem disagreements
    diffs = []
    for p in common:
        c, s = cod[p], svd[p]
        if c.get("error") or s.get("error"):
            continue
        dc = num(s["n_conv"]) - num(c["n_conv"])
        dg = num(s["n_global"]) - num(c["n_global"])
        if dc or dg:
            diffs.append((p, num(c["n"]), num(c["n_conv"]), num(s["n_conv"]),
                          num(c["n_global"]), num(s["n_global"]),
                          num(ref[p]["n_conv"]), num(ref[p]["n_global"]), dc))
    print(f"\n  problems where the two steps disagree: {len(diffs)} of {len(common)}")
    if diffs:
        print(f"    {'problem':<28}{'n':>4}{'conv COD/SVD':>15}{'glob COD/SVD':>15}{'MATLAB c/g':>13}")
        for p, nn, cc, sc, cg, sg, rc, rg, _dc in sorted(diffs, key=lambda d: d[-1])[:40]:
            print(f"    {p:<28}{int(nn):>4}{int(cc):>8}/{int(sc):<6}{int(cg):>8}/{int(sg):<6}"
                  f"{int(rc):>8}/{int(rg):<5}")


main()
