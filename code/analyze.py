"""Summarise results_b.jsonl into one row per (injection, budget, ablation) cell."""

import json
import os
import statistics as st
import sys
from collections import defaultdict

PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "results_b.jsonl"
)

cells = defaultdict(list)
for line in open(PATH, encoding="utf-8"):
    r = json.loads(line)
    cells[(r["inj"], r["budget"], r["ablation"])].append(r)


def auroc(pos, neg):
    """P(pos > neg) + 0.5 P(pos == neg)."""
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


hdr = (f"{'injection':9s} {'budget':>6s} {'ablation':8s} | {'MRR':>5s} {'top10':>5s} "
       f"{'det':>6s} {'AUROC':>5s} | {'said':>4s} {'q_rank':>6s} {'a_rank':>6s} {'a<10':>4s} | "
       f"{'think':>5s}")
print(hdr)
print("-" * len(hdr))
order = ["none", "jl_band", "rnd_band", "jl_L11", "rnd_L11"]
for abl in ("none", "jspace", "random", "matched"):
    for budget in (0, 64, 256):
        ctrl = cells.get(("none", budget, abl), [])
        for inj in order:
            rows = cells.get((inj, budget, abl))
            if not rows:
                continue
            det = [r["det"] for r in rows]
            auc = auroc(det, [r["det"] for r in ctrl]) if inj != "none" else float("nan")
            print(
                f"{inj:9s} {budget:6d} {abl:8s} | "
                f"{st.mean(r['rr'] for r in rows):5.3f} {sum(r['rank'] < 10 for r in rows):5d} "
                f"{st.mean(det):+6.2f} {auc:5.2f} | "
                f"{sum(r['says_in_think'] for r in rows):4d} "
                f"{st.median(r['q_best_rank'] for r in rows):6.0f} "
                f"{st.median(r['a_best_rank'] for r in rows):6.0f} "
                f"{sum(r['a_best_rank'] < 10 for r in rows):4d} | "
                f"{st.mean(r['n_think'] for r in rows):5.0f}"
            )
        print()
print("MRR/top10: naming the injected word at the report (n=40 concepts).")
print("det: log P(Yes) - log P(No) for 'do you detect an injected thought'. AUROC: injected vs no-injection.")
print("said: thinking text contains the word. q_rank / a_rank: best J-lens rank of the word over the band,")
print("at the question tokens (where injected) / the assistant tokens (thinking + answer). a<10: trials in lens top-10.")
