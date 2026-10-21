"""v1 diagnostics, part 1: slice the committed eval.json files (no torch needed)."""
import json, glob, os
import numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
pk = np.load("data/packed_v1.npz")
blk, hgt = pk["blockage"], pk["height"]
runs = sorted(glob.glob("runs/grid/n*_s?"))
print("== quick-val tau over training (16 val solves) ==")
for r in runs:
    q = [json.loads(l) for l in open(r + "/metrics.jsonl")]
    q = [(d["step"], d["quick_val_tau_rel_l2"]) for d in q if "quick_val_tau_rel_l2" in d]
    print(f"{os.path.basename(r):18s}", " ".join(f"{v:.3f}" for _, v in q))
print("\n== test_B / val tau err by flow k (median) ==")
for split in ("val", "test_B"):
    for r in runs:
        rows = json.load(open(r + "/eval.json"))["splits"][split]["per_solve"]
        by = {k: np.median([x["tau_rel_l2"] for x in rows if x["k"] == k]) for k in range(4)}
        re = {k: np.median([x["Re"] for x in rows if x["k"] == k]) for k in range(4)}
        print(f"{split:6s} {os.path.basename(r):18s}", " ".join(f"k{k}(Re~{re[k]:.1f}) {by[k]:.3f}" for k in range(4)))
print("\n== test_B: tau err vs blockage / height (Spearman, n160 runs) ==")
def spear(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b)); return np.corrcoef(ra, rb)[0, 1]
for r in [x for x in runs if "n160" in x]:
    rows = json.load(open(r + "/eval.json"))["splits"]["test_B"]["per_solve"]
    e = np.array([x["tau_rel_l2"] for x in rows]); s = np.array([x["solve"] for x in rows])
    print(os.path.basename(r), f"rho(err,blockage)={spear(e, blk[s]):+.2f} rho(err,height)={spear(e, hgt[s]):+.2f} rho(err,Re)={spear(e,[x['Re'] for x in rows]):+.2f}")
print("\nblockage ranges: ", {sp: (round(float(blk[pk['split']==c].min()),2), round(float(np.median(blk[pk['split']==c])),2), round(float(blk[pk['split']==c].max()),2)) for c, sp in enumerate(pk["split_names"])})
print("\n== train vs val vs test_B gap (n160, median tau err) ==")
for r in [x for x in runs if "n160" in x]:
    ev = json.load(open(r + "/eval.json"))["splits"]
    print(os.path.basename(r), {k: round(float(np.median([x["tau_rel_l2"] for x in v["per_solve"]])), 3) for k, v in ev.items()},
          {k: round(float(np.median([x["u_rel_l2"] for x in v["per_solve"]])), 3) for k, v in ev.items()})
