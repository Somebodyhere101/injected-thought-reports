"""Follow-ups to gib.py, same model, prompts and G.

    python gib_ext.py 1.7b|4b|8b

1. Sufficiency: add G at larger strengths (1.5, 2, 3) while the model writes coherent text.
2. Generality of the "ordered letters" effect: remove G (0.25, 0.5, 0.75) while the model writes
   gibberish for the second plain prompt, and for the lexical prompt (G was estimated from the
   first plain prompt only).
Each against one random direction of the same per-layer size at strengths 1-12, with the output
change (KL on the unsteered text) recorded so G and random can be compared at matched KL.
Outputs: gib_ext_<model>.json (numbers), gib_ext_texts_<model>.json (texts, kept local).
Text measures (real-letter share, order, compressibility) are computed by gib_summary.py.
"""

import json
import os
import sys

import torch

import gib  # loads the model named in sys.argv[1]
from gib import (COHERENT, GIB_LEXICAL, GIB_PLAIN, HERE, KEY, L, N, analyze, dev, external_strings, generate,
                 log, prompt, rig, summarize, tok, unit)

# ---------- G, recomputed exactly as in gib.main() ----------
ext_ids = [tok.encode(s, add_special_tokens=False) for s in external_strings(N, 99)]
ids_g, a0_g = prompt(GIB_PLAIN[0])
ids_c, a0_c = prompt(COHERENT[0])
own = generate(GIB_PLAIN[0], N, seed=11)[2]
diffs = []
for conts in (ext_ids, own):
    rg = analyze(ids_g, a0_g, conts, want_mean=True)
    rc = analyze(ids_c, a0_c, conts, want_mean=True)
    diffs.append(rg["mean_resid"] - rc["mean_resid"])
G = torch.cat(diffs).mean(0)
torch.save({"G": G.cpu()}, os.path.join(HERE, f"gib_G_{KEY}.pt"))
log("G recomputed")

step = lambda V: {l: V[l] - (V[l - 1] if l > 0 else 0) for l in range(L)}
dG = step(G)
r = unit(torch.randn(rig.d, generator=torch.Generator().manual_seed(501)).to(dev))
dR = step(torch.stack([r * G[l].norm() for l in range(L)]))


def steer(d, alpha):
    return lambda a0: [{"vecs": {l: (alpha * d[l])[None].to(torch.bfloat16) for l in range(L)}, "span": (a0, 10**9)}]


TESTS = [
    ("add G to coherent writing (strong)", COHERENT[0], +1.0, (1.5, 2.0, 3.0), (2.0, 4.0, 8.0, 12.0)),
    ("remove G, second plain gibberish prompt", GIB_PLAIN[1], -1.0, (0.25, 0.5, 0.75), (1.0, 2.0, 4.0, 8.0)),
    ("remove G, lexical gibberish prompt", GIB_LEXICAL, -1.0, (0.25, 0.5, 0.75), (1.0, 2.0, 4.0, 8.0)),
]
OUT = {"model": KEY, "n": N, "tests": {}}
TEXTS = {}
for task, user, sign, g_alphas, r_alphas in TESTS:
    ids0, a00, base_gen = generate(user, N, seed=30)
    base = analyze(ids0, a00, base_gen, ref_lp="return", light=True)
    ref = base.pop("lp")
    OUT["tests"][task] = {"baseline": {"kl": 0.0}}
    TEXTS[f"{task} | baseline"] = [tok.decode(g, skip_special_tokens=True) for g in base_gen]
    for name, d, alphas in (("G", dG, g_alphas), ("random 1", dR, r_alphas)):
        for a in alphas:
            inj = steer(d, sign * a)
            g = generate(user, N, injects=inj, seed=30)[2]
            kl = summarize(analyze(ids0, a00, base_gen, injects=inj, ref_lp=ref, light=True))["kl"]
            OUT["tests"][task][f"{name} x{a}"] = {"kl": kl}
            TEXTS[f"{task} | {name} x{a}"] = [tok.decode(x, skip_special_tokens=True) for x in g]
            log(f"{task:42s} {name:8s} x{a}: KL {kl[0]:.2f}")
            json.dump(OUT, open(os.path.join(HERE, f"gib_ext_{KEY}.json"), "w", encoding="utf-8"), indent=1)
            json.dump(TEXTS, open(os.path.join(HERE, f"gib_ext_texts_{KEY}.json"), "w", encoding="utf-8"), indent=1)
    del ref
log("DONE")
