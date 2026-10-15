"""Sanity checks: lens readout by layer, prompt spans, and an injection-strength sweep
with thinking off and no ablation."""

import statistics as st
import time

import torch

from core import Rig
from trial import run_cell

BAND = list(range(10, 25))

t0 = time.time()
rig = Rig(band=BAND)
print(f"loaded in {time.time() - t0:.0f}s; layers={rig.n_layers} d={rig.d} concepts={len(rig.concepts)}")
print("mean norms:", {l: round(v) for l, v in rig.mean_norm.items() if l % 3 == 0})

for think in (False, True):
    p = rig.prompt_text(think)
    print(f"--- prompt (think={think}) spans={rig.spans(p)} ---")
    print(repr(p[-260:]))

# Where in depth does the J-lens read out meaningful content?
with torch.no_grad():
    for text, pos in [
        ("Fact: The currency used in the country shaped like a boot is", -2),
        ("Q: What color is the sky on a clear day? A: The sky is", -1),
    ]:
        ids = rig.tok(text, return_tensors="pt").input_ids.cuda()
        hs = rig.hf(ids, output_hidden_states=True).hidden_states
        print(f"\n{text!r} @ {pos}")
        for l in range(0, rig.n_layers, 2):
            top = rig.lens_logits(hs[l + 1][0, pos][None], l)[0].topk(5).indices
            print(f"  L{l:2d}", [rig.tok.decode([t]) for t in top])

concepts = rig.concepts[:40]
print(f"\nstrength sweep, thinking off, {len(concepts)} concepts")
for s in [0, 0.02, 0.05, 0.1, 0.2, 0.4]:
    t0 = time.time()
    rows = run_cell(rig, concepts, s, budget=0)
    rr = [r["rr"] for r in rows]
    print(
        f"  s={s:<5} MRR={st.mean(rr):.3f} top1={sum(r['rank'] == 0 for r in rows)}/{len(rows)} "
        f"top10={sum(r['rank'] < 10 for r in rows)} median_rank={st.median(r['rank'] for r in rows):.0f} "
        f"det(yes-no)={st.mean(r['det'] for r in rows):+.2f}  ({time.time() - t0:.1f}s)"
    )
