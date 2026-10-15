"""Experiment B (effort x workspace) on a larger model, reduced grid.

    python run_b_scale.py 4b|8b

Cells: injection {none, word across the band} x thinking {off, 256 tokens} x
assistant turn {untouched, J-space removed, same-size random push}. 40 words.
"""

import json
import os
import statistics as st
import sys
import time

from core import LENS_FILE, Rig, model_spec
from dlib import auroc
from trial import run_cell

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = {
    "1.7b": dict(model="Qwen/Qwen3-1.7B", lens_file=LENS_FILE, gpu_layers=None),
    "4b": dict(model="Qwen/Qwen3-4B", lens_file="qwen3-4b/jlens/Salesforce-wikitext/Qwen3-4B_jacobian_lens.pt",
               gpu_layers=12),
    "8b": dict(model="Qwen/Qwen3-8B", lens_file="qwen3-8b/jlens/Salesforce-wikitext/Qwen3-8B_jacobian_lens.pt",
               gpu_layers=3),
}
KEY = sys.argv[1]
OUT = os.path.join(HERE, f"results_b_{KEY}.jsonl")
T0 = time.time()

rig = Rig(**model_spec(MODELS, KEY))
concepts = rig.concepts[:40]
done = set()
if os.path.exists(OUT):
    for line in open(OUT, encoding="utf-8"):
        r = json.loads(line)
        done.add((r["inj"], r["budget"], r["ablation"]))

INJ = {"none": 0.0, "word": 0.1}
cells = [(i, b, a) for a in ("none", "jspace", "matched") for b in (0, 256) for i in INJ]
for inj, budget, abl in cells:
    if (inj, budget, abl) in done:
        continue
    t0 = time.time()
    rows = []
    for half in (concepts[:20], concepts[20:]):
        rows += run_cell(rig, half, INJ[inj], budget=budget, ablation=None if abl == "none" else abl,
                         abl_layers=rig.band, k=25, seed=1, static_cache=budget > 0)
    with open(OUT, "a", encoding="utf-8") as f:
        for r in rows:
            r["inj"] = inj
            f.write(json.dumps(r) + "\n")
    print(f"[{time.time() - T0:6.0f}s] inj={inj:5s} budget={budget:3d} abl={abl:8s} "
          f"det={st.mean(r['det'] for r in rows):+.2f} MRR={st.mean(r['rr'] for r in rows):.3f} "
          f"top10={sum(r['rank'] < 10 for r in rows)}/40 ({time.time() - t0:.0f}s)", flush=True)

# summary: detection AUROC (word vs none) per budget x ablation
rows = [json.loads(l) for l in open(OUT, encoding="utf-8")]
for abl in ("none", "jspace", "matched"):
    for b in (0, 256):
        w = [r["det"] for r in rows if r["inj"] == "word" and r["budget"] == b and r["ablation"] == abl]
        n = [r["det"] for r in rows if r["inj"] == "none" and r["budget"] == b and r["ablation"] == abl]
        if w and n:
            print(f"AUROC word vs none | budget {b:3d} | {abl:8s}: {auroc(w, n):.2f}  "
                  f"(none det {st.mean(n):+.2f}, word det {st.mean(w):+.2f})", flush=True)
print("DONE", flush=True)
