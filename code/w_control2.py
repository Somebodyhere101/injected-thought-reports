"""Uncertainty and matched magnitudes for "a word in the workspace moves the report".

    python w_control2.py 1.7b|4b|8b

A word's J-lens direction, or a random direction, is added on the assistant turn across
the band at several strengths. Per trial we record the yes-score change, the KL divergence
of the answer distribution from the clean one, and the size of the change in the final-layer
residual at the answer position. Random directions: 3 sets of 40.

Compared two ways, each with a 95% bootstrap interval:
  - matched input size: same strength (same norm added at the same layers and positions);
  - matched downstream effect: the random dose-response curve interpolated to the word's
    KL divergence (and, separately, to its final-layer residual change).
"""

import json
import os
import random
import statistics as st
import sys
import time

import torch

from core import LENS_FILE, PREFILL, Rig, model_spec
from dlib import Ctx, rnd_inj, word_inj

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = {
    "1.7b": dict(model="Qwen/Qwen3-1.7B", lens_file=LENS_FILE, gpu_layers=None),
    "4b": dict(model="Qwen/Qwen3-4B", lens_file="qwen3-4b/jlens/Salesforce-wikitext/Qwen3-4B_jacobian_lens.pt",
               gpu_layers=12),
    "8b": dict(model="Qwen/Qwen3-8B", lens_file="qwen3-8b/jlens/Salesforce-wikitext/Qwen3-8B_jacobian_lens.pt",
               gpu_layers=3),
}
KEY = sys.argv[1]
OUT = os.path.join(HERE, f"w_control2_{KEY}.json")
T0 = time.time()
rng = random.Random(0)
WORD_S = (0.01, 0.02, 0.03, 0.05, 0.08)
RAND_S = (0.02, 0.05, 0.08, 0.12, 0.2, 0.3)
RAND_SEEDS = (777, 778, 779)


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


rig = Rig(**model_spec(MODELS, KEY))
L, band = rig.n_layers, rig.band
concepts = rig.concepts[:40]
TIDS, NAMES = [c[1] for c in concepts], [c[0] for c in concepts]
N = len(TIDS)
ctx = Ctx(rig, list(rig.messages), PREFILL)

c = ctx.run(N, kl_ref="return", hidden=True)
clean = st.mean(c["det"])
ref_lp = c["_lp"][0]
ref_h = c["hid"][0, L - 1]


def trial_stats(injects):
    r = ctx.run(N, injects, kl_ref=ref_lp, hidden=True)
    dh = (r["hid"][:, L - 1] - ref_h).norm(dim=-1) / ref_h.norm()
    return [{"det_change": d - clean, "kl": k, "dh": x} for d, k, x in zip(r["det"], r["kl"], dh.tolist())]


cells = {"word": {}, "random": {}}
for s in WORD_S:
    cells["word"][str(s)] = trial_stats([word_inj(rig, TIDS, band, s, ctx.ASPAN)])
    m = cells["word"][str(s)]
    log(f"word   s={s:<5} change {st.mean(t['det_change'] for t in m):+6.2f}  KL {st.mean(t['kl'] for t in m):.3f}  "
        f"dh {st.mean(t['dh'] for t in m):.3f}")
for s in RAND_S:
    trials = []
    for sd in RAND_SEEDS:
        trials += trial_stats([rnd_inj(rig, N, sd, band, s, ctx.ASPAN)])
    cells["random"][str(s)] = trials
    log(f"random s={s:<5} change {st.mean(t['det_change'] for t in trials):+6.2f}  KL {st.mean(t['kl'] for t in trials):.3f}  "
        f"dh {st.mean(t['dh'] for t in trials):.3f}")


def mean(cell, k):
    return st.mean(t[k] for t in cell)


def interp(xs, ys, x):
    """Piecewise-linear interpolation of y at x; None outside the measured range."""
    pts = sorted(zip(xs, ys))
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0 + 1e-12)
    return None


def resample(cell):
    return rng.choices(cell, k=len(cell))


summary = {"clean_det": clean, "matched_input": {}, "matched_kl": {}, "matched_dh": {}}
B = 2000
for s in WORD_S:
    w = cells["word"][str(s)]
    # matched input size
    if str(s) in cells["random"]:
        r = cells["random"][str(s)]
        diffs = sorted(mean(resample(w), "det_change") - mean(resample(r), "det_change") for _ in range(B))
        summary["matched_input"][str(s)] = {
            "word": mean(w, "det_change"), "random": mean(r, "det_change"),
            "diff": mean(w, "det_change") - mean(r, "det_change"), "ci": [diffs[int(.025 * B)], diffs[int(.975 * B)]],
            "word_kl": mean(w, "kl"), "random_kl": mean(r, "kl")}
    # matched downstream effect
    for key, out in (("kl", "matched_kl"), ("dh", "matched_dh")):
        def one(wc, rc):
            target = mean(wc, key)
            xs = [mean(rc[k], key) for k in rc]
            ys = [mean(rc[k], "det_change") for k in rc]
            y = interp(xs, ys, target)
            return None if y is None else mean(wc, "det_change") - y
        point = one(w, cells["random"])
        if point is None:
            summary[out][str(s)] = {"diff": None, "note": "word effect outside the random range"}
            continue
        boots = [one(resample(w), {k: resample(v) for k, v in cells["random"].items()}) for _ in range(B // 4)]
        boots = sorted(b for b in boots if b is not None)
        summary[out][str(s)] = {"diff": point, "ci": [boots[int(.025 * len(boots))], boots[int(.975 * len(boots))]],
                                "word_" + key: mean(w, key)}
    mi = summary["matched_input"].get(str(s))
    mk = summary["matched_kl"].get(str(s))
    log(f"s={s}: matched size diff " + (f"{mi['diff']:+.2f} [{mi['ci'][0]:+.2f}, {mi['ci'][1]:+.2f}]" if mi else "n/a")
        + "; matched KL diff " + (f"{mk['diff']:+.2f} [{mk['ci'][0]:+.2f}, {mk['ci'][1]:+.2f}]" if mk and mk.get("diff") is not None else "n/a"))

json.dump({"model": KEY, "words": NAMES, "word_strengths": WORD_S, "random_strengths": RAND_S,
           "random_seeds": RAND_SEEDS, "summary": summary, "trials": cells},
          open(OUT, "w", encoding="utf-8"), indent=1)
log("DONE")
