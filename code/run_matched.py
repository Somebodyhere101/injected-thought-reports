"""Norm-matched control for the J-space ablation cells of run_b.py: perturb the
assistant turn by a random vector of the same size as the J-space projection.
Appends to results_b.jsonl with ablation="matched"."""

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
    "jl_L11": dict(strength=1.0, kind="jlens", inj_layers=[11]),
}

rig = Rig(band=BAND)
concepts = rig.concepts[:40]
for budget in (0, 256):
    for inj, kw in INJECTIONS.items():
        t0 = time.time()
        rows = run_cell(rig, concepts, budget=budget, ablation="matched", abl_layers=BAND,
                        k=25, trace=True, seed=1, **kw)
        with open(OUT, "a", encoding="utf-8") as f:
            for r in rows:
                r["inj"] = inj
                f.write(json.dumps(r) + "\n")
        det = sum(r["det"] for r in rows) / len(rows)
        print(f"inj={inj:8s} budget={budget:3d} abl=matched det={det:+.2f} ({time.time() - t0:.0f}s)",
              flush=True)
print("DONE", flush=True)
