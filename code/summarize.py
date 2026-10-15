"""Build summary.json (consumed by report.html) from results_b/c and trace_examples."""

import json
import os
import random
import re
import statistics as st
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
rng = random.Random(0)
DENIAL = re.compile(r"can't|cannot|don't have|unable|not able|no way to", re.I)


def load(name):
    p = os.path.join(HERE, name)
    return [json.loads(l) for l in open(p, encoding="utf-8")] if os.path.exists(p) else []


def auroc(pos, neg):
    if not pos or not neg:
        return None
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def boot_auroc(pos, neg, n=1000):
    vals = sorted(
        auroc([rng.choice(pos) for _ in pos], [rng.choice(neg) for _ in neg]) for _ in range(n)
    )
    return [vals[int(0.025 * n)], vals[int(0.975 * n)]]


def boot_mean(xs, n=1000):
    vals = sorted(st.mean(rng.choice(xs) for _ in xs) for _ in range(n))
    return [vals[int(0.025 * n)], vals[int(0.975 * n)]]


cells = defaultdict(list)
for r in load("results_b.jsonl"):
    cells[(r["inj"], r["budget"], r["ablation"])].append(r)

b_rows = []
for (inj, budget, abl), rows in sorted(cells.items()):
    det = [r["det"] for r in rows]
    ctrl = [r["det"] for r in cells.get(("none", budget, abl), [])]
    row = {
        "inj": inj, "budget": budget, "ablation": abl, "n": len(rows),
        "mrr": st.mean(r["rr"] for r in rows),
        "top10": sum(r["rank"] < 10 for r in rows),
        "det": st.mean(det), "det_ci": boot_mean(det),
        "a_rank": st.median(r["a_best_rank"] for r in rows),
        "q_rank": st.median(r["q_best_rank"] for r in rows),
        "denial": sum(bool(DENIAL.search(r["think"])) for r in rows) if budget else None,
    }
    if inj != "none" and ctrl:
        row["auroc"] = auroc(det, ctrl)
        row["auroc_ci"] = boot_auroc(det, ctrl)
    b_rows.append(row)

c_by = defaultdict(list)
for r in load("results_c.jsonl"):
    c_by[r["inj"]].append(r)
c_rows = []
for inj, rows in c_by.items():
    n = [r["n_think"] for r in rows]
    c_rows.append({
        "inj": inj, "n": len(rows), "mean": st.mean(n), "ci": boot_mean(n),
        "median": st.median(n), "hit_budget": sum(r["hit_budget"] for r in rows),
        "det": st.mean(r["det"] for r in rows), "lengths": n,
    })

traces = {}
p = os.path.join(HERE, "trace_examples.json")
if os.path.exists(p):
    traces = json.load(open(p, encoding="utf-8"))

def load_json(name):
    p = os.path.join(HERE, name)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


d = {}
for key, name in (("injected", "results_d.json"), ("aware", "results_d_aware.json")):
    r = load_json(name)
    if r:
        d[key] = {
            "grid": [{k: g[k] for k in ("cond", "edit", "det", "named_top1", "w_top1")} for g in r["grid"]],
            "validation": r["validation"], "H_added": r["H_added"], "samples": r["samples"],
        }
ft = load_json("results_d_freetext.json")
if ft:
    d["free"] = {k: {"n": len(v), "says_word": sum(x["says_word"] for x in v),
                     "claims": sum(x["claims_a_word"] for x in v), "denies": sum(x["denies"] for x in v),
                     "leaks": [x["text"][:200] for x in v if x["says_word"]][:4]}
                 for k, v in ft.items()}
ha = load_json("results_d_hadded.json")
if ha:
    d["hadded"] = ha

json.dump({"b": b_rows, "c": c_rows, "traces": traces, "d": d},
          open(os.path.join(HERE, "summary.json"), "w", encoding="utf-8"))
for r in b_rows:
    ci = r.get("auroc_ci")
    print(f"{r['inj']:8s} {r['budget']:3d} {r['ablation']:7s} det={r['det']:+6.2f} "
          f"auroc={r.get('auroc') or float('nan'):.2f} {ci and [round(x, 2) for x in ci]} "
          f"mrr={r['mrr']:.3f} denial={r['denial']}")
for r in c_rows:
    print(f"C {r['inj']:8s} mean={r['mean']:.0f} ci={[round(x) for x in r['ci']]} "
          f"median={r['median']} hit={r['hit_budget']}/{r['n']} det={r['det']:+.2f}")
