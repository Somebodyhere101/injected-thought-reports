"""Calibration: residual composition, free-text reports, and which injection layers
let the concept reach the report position."""

import statistics as st

import torch

from core import Rig
from trial import run_cell

rig = Rig(band=list(range(10, 25)))
concepts = rig.concepts[:40]


def summary(rows):
    return (
        f"MRR={st.mean(r['rr'] for r in rows):.3f} top10={sum(r['rank'] < 10 for r in rows):2d} "
        f"med_rank={st.median(r['rank'] for r in rows):6.0f} det={st.mean(r['det'] for r in rows):+.2f}"
    )


# 1. How much of the residual norm sits in a few outlier dimensions?
with torch.no_grad():
    ids = rig.tok(rig.prompt_text(False), return_tensors="pt", add_special_tokens=False).input_ids.cuda()
    hs = rig.hf(ids, output_hidden_states=True).hidden_states
    for l in (10, 14, 18, 22, 24):
        h = hs[l + 1][0, 1:].float()
        top = h.abs().mean(0).topk(8).indices
        rest = h.clone()
        rest[:, top] = 0
        print(f"L{l}: median norm {h.norm(dim=-1).median():7.1f}  without top-8 dims {rest.norm(dim=-1).median():6.1f}")

# 2. Single-layer injection sweep (question turn only), thinking off.
print("\nsingle-layer injection, thinking off")
for l in (8, 11, 14, 17, 20, 23):
    for s in (0.5, 1, 2, 4):
        rows = run_cell(rig, concepts, s, budget=0, inj_layers=[l])
        print(f"  L{l:2d} s={s:<4} {summary(rows)}")

# 3. Ceiling check: band injection on the question turn AND the assistant turn.
print("\nband injection on question + assistant turn, thinking off")
for s in (0.02, 0.05, 0.1):
    rows = run_cell(rig, concepts, s, budget=0, inj_span="all")
    print(f"  s={s:<4} {summary(rows)}")
