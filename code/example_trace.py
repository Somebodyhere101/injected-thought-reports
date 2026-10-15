"""Save full layer x token J-lens rank maps of injected concepts for the report figure."""

import json
import os

from core import Rig
from trial import run_cell

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trace_examples.json")

rig = Rig(band=list(range(10, 25)))
pick = [c for c in rig.concepts if c[0] in ("truck", "diamond", "pool", "photo")]
assert len(pick) == 4, pick
out = {}
for name, kw in {
    "jl_band": dict(strength=0.1, kind="jlens", inj_layers=None),
    "jl_L11": dict(strength=1.0, kind="jlens", inj_layers=[11]),
}.items():
    rows = run_cell(rig, pick, budget=96, trace=True, keep_trace=True, seed=3, **kw)
    out[name] = [
        {k: r[k] for k in ("concept", "rank", "det", "think", "trace")} for r in rows
    ]
    print(name, [(r["concept"], r["q_best_rank"], r["a_best_rank"]) for r in rows], flush=True)
json.dump(out, open(OUT, "w", encoding="utf-8"))
print("DONE")
