"""Experiment B grid: injection type x thinking budget x J-space ablation.

Writes one JSON row per (cell, concept) to results_b.jsonl; finished cells are
skipped on rerun.
"""

import json
import os
import time

from core import Rig
from trial import run_cell

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_b.jsonl")
BAND = list(range(10, 25))

INJECTIONS = {
    "none": dict(strength=0.0, kind="jlens", inj_layers=None),
    "jl_band": dict(strength=0.1, kind="jlens", inj_layers=None),
    "rnd_band": dict(strength=0.1, kind="random", inj_layers=None),
    "jl_L11": dict(strength=1.0, kind="jlens", inj_layers=[11]),
    "rnd_L11": dict(strength=1.0, kind="random", inj_layers=[11]),
}
CELLS = [(inj, b, "none") for b in (0, 64, 256) for inj in INJECTIONS]
CELLS += [(inj, b, abl) for abl in ("jspace", "random") for b in (0, 256)
          for inj in ("none", "jl_band", "jl_L11")]

done = set()
if os.path.exists(OUT):
    for line in open(OUT, encoding="utf-8"):
        r = json.loads(line)
        done.add((r["inj"], r["budget"], r["ablation"]))

rig = Rig(band=BAND)
concepts = rig.concepts[:40]
print(f"{len(CELLS)} cells, {len(done)} already done", flush=True)
for i, (inj, budget, abl) in enumerate(CELLS):
    if (inj, budget, abl) in done:
        continue
    t0 = time.time()
    rows = run_cell(
        rig, concepts, budget=budget,
        ablation=None if abl == "none" else abl, abl_layers=BAND, k=25,
        trace=True, seed=1, **INJECTIONS[inj],
    )
    with open(OUT, "a", encoding="utf-8") as f:
        for r in rows:
            r["inj"] = inj
            f.write(json.dumps(r) + "\n")
    mrr = sum(r["rr"] for r in rows) / len(rows)
    det = sum(r["det"] for r in rows) / len(rows)
    print(f"[{i + 1}/{len(CELLS)}] inj={inj:8s} budget={budget:3d} abl={abl:6s} "
          f"MRR={mrr:.3f} det={det:+.2f} ({time.time() - t0:.0f}s)", flush=True)
print("DONE", flush=True)
