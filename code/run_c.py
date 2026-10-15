"""Experiment C: does an injected signal change how long the model chooses to think?
Thinking budget is generous (1024); the model stops when it closes </think>."""

import json
import os
import statistics as st
import time

from core import Rig
from trial import run_cell

# Same conditions as run_b.py (not imported: run_b runs its grid at import).
BAND = list(range(10, 25))
INJECTIONS = {
    "none": dict(strength=0.0, kind="jlens", inj_layers=None),
    "jl_band": dict(strength=0.1, kind="jlens", inj_layers=None),
    "rnd_band": dict(strength=0.1, kind="random", inj_layers=None),
    "jl_L11": dict(strength=1.0, kind="jlens", inj_layers=[11]),
    "rnd_L11": dict(strength=1.0, kind="random", inj_layers=[11]),
}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_c.jsonl")
BUDGET = 1024
SEEDS = (1,)

done = set()
if os.path.exists(OUT):
    for line in open(OUT, encoding="utf-8"):
        r = json.loads(line)
        done.add((r["inj"], r["seed"]))

rig = Rig(band=BAND)
concepts = rig.concepts[:40]
for inj in INJECTIONS:
    for seed in SEEDS:
        if (inj, seed) in done:
            continue
        t0 = time.time()
        # Batches of 20 with a preallocated cache: a growing cache fragments 8 GB.
        rows = []
        for half in (concepts[i : i + 20] for i in range(0, len(concepts), 20)):
            rows += run_cell(rig, half, budget=BUDGET, seed=seed, static_cache=True,
                             **INJECTIONS[inj])
        with open(OUT, "a", encoding="utf-8") as f:
            for r in rows:
                r["inj"], r["seed"] = inj, seed
                f.write(json.dumps(r) + "\n")
        n = [r["n_think"] for r in rows]
        print(f"inj={inj:8s} seed={seed} mean_think={st.mean(n):6.0f} median={st.median(n):5.0f} "
              f"hit_budget={sum(r['hit_budget'] for r in rows)}/{len(rows)} ({time.time() - t0:.0f}s)",
              flush=True)
print("DONE", flush=True)
