"""Does a word in the workspace drive the report because it is a word, or because it is
any disturbance there? Same size, same layers, same positions: word direction vs random
direction on the assistant turn, each with H untouched and held at clean.

    python w_control.py 1.7b|4b|8b
"""

import json
import os
import statistics as st
import sys
import time

import torch

from core import LENS_FILE, PREFILL, Rig, model_spec
from dlib import Ctx, auroc, rnd_inj, unit, word_inj

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = {
    "1.7b": dict(model="Qwen/Qwen3-1.7B", lens_file=LENS_FILE, gpu_layers=None),
    "4b": dict(model="Qwen/Qwen3-4B", lens_file="qwen3-4b/jlens/Salesforce-wikitext/Qwen3-4B_jacobian_lens.pt",
               gpu_layers=12),
    "8b": dict(model="Qwen/Qwen3-8B", lens_file="qwen3-8b/jlens/Salesforce-wikitext/Qwen3-8B_jacobian_lens.pt",
               gpu_layers=3),
}
KEY = sys.argv[1]
T0 = time.time()
rig = Rig(**model_spec(MODELS, KEY))
L, band = rig.n_layers, rig.band
EDIT = list(range(band[0] + 1, L))
concepts = rig.concepts[:40]
TIDS = [c[1] for c in concepts]
N = len(TIDS)
ctx = Ctx(rig, list(rig.messages), PREFILL)
dirs_path = os.path.join(HERE, f"dirs_{KEY}.pt")
if os.path.exists(dirs_path):
    H_raw = torch.load(dirs_path)["H_raw"].to(rig.device)
else:  # older battery runs did not save H; re-estimate it the same way (deterministic)
    H_raw = torch.stack(ctx.estimate_H(seeds=(100, 200, 300), n=60, single_layer=round(0.39 * L))[1]).mean(0)
Hu = unit(H_raw)
held = {l: ctx.clamp(Hu[l], l) for l in EDIT}

out = {"model": KEY}
for s in (0.02, 0.05):
    base = st.mean(ctx.run(N)["det"])
    base_h = st.mean(ctx.run(N, edits=held)["det"])
    row = {"clean": base}
    for name, inj in (("word", word_inj(rig, TIDS, band, s, ctx.ASPAN)),
                      ("random", rnd_inj(rig, N, 777, band, s, ctx.ASPAN))):
        r = ctx.run(N, [inj], tids=TIDS)
        rh = ctx.run(N, [inj], edits=held, tids=TIDS)
        row[name] = {"det_change": st.mean(r["det"]) - base, "frac_above": auroc(r["det"], [base]),
                     "named_top1": sum(x == 0 for x in r["rank"]),
                     "det_change_H_held": st.mean(rh["det"]) - base_h,
                     "named_top1_H_held": sum(x == 0 for x in rh["rank"])}
        print(f"[{time.time() - T0:5.0f}s] {KEY} s={s} {name:6s} change={row[name]['det_change']:+.2f} "
              f"above={row[name]['frac_above']:.2f} named={row[name]['named_top1']}/{N} | H held: "
              f"change={row[name]['det_change_H_held']:+.2f} named={row[name]['named_top1_H_held']}/{N}", flush=True)
    out[str(s)] = row
json.dump(out, open(os.path.join(HERE, f"w_control_{KEY}.json"), "w", encoding="utf-8"), indent=1)
print("DONE", flush=True)
