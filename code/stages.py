"""Which stage does each intervention identify? A reporting bottleneck vs an
introspection mechanism.

    python stages.py 1.7b|4b|8b

Stage 1, the question tokens: the injected word sits in the residual where it was added
    (measured earlier: J-lens rank 0 there).
Stage 2, transfer: when does the injection's influence leave the question tokens? Inject a
    word at one layer S on the question tokens, then overwrite those tokens' residual with
    its clean value at layer l. Later positions can only pick the injection up through
    attention at layers S+1..l. The yes-score as a function of l is the transfer window.
Stage 3, what arrives at the answer position: is the word's identity linearly present
    there, layer by layer, with H untouched and with H held at its clean value?
    Nearest-centroid decoding over 40 words, leave-one-strength-out (chance 2.5%).
Stage 4, the report: hold H in one window of layers at a time, or only at the answer
    position, and see where H is needed for the yes-score to register the injection.

Every per-trial number is saved to stages_<model>.json.
"""

import json
import os
import random
import statistics as st
import sys
import time

import torch

from core import LENS_FILE, PREFILL, Rig, model_spec
from dlib import Ctx, rnd_inj, unit, word_inj

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = {
    "1.7b": dict(model="Qwen/Qwen3-1.7B", lens_file=LENS_FILE, gpu_layers=None),
    "4b": dict(model="Qwen/Qwen3-4B", lens_file="qwen3-4b/jlens/Salesforce-wikitext/Qwen3-4B_jacobian_lens.pt",
               gpu_layers=12),
    "8b": dict(model="Qwen/Qwen3-8B", lens_file="qwen3-8b/jlens/Salesforce-wikitext/Qwen3-8B_jacobian_lens.pt",
               gpu_layers=3),
}
KEY = sys.argv[1]
OUT = os.path.join(HERE, f"stages_{KEY}.json")
T0 = time.time()
rng = random.Random(0)


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def ci(xs, n=2000):
    """Mean and 95% bootstrap interval over trials."""
    m = [st.mean(rng.choices(xs, k=len(xs))) for _ in range(n)]
    m.sort()
    return [st.mean(xs), m[int(0.025 * n)], m[int(0.975 * n)]]


rig = Rig(**model_spec(MODELS, KEY))
L, band = rig.n_layers, rig.band
single = round(0.39 * L)
EDIT = list(range(band[0] + 1, L))
concepts = rig.concepts[:40]
TIDS, NAMES = [c[1] for c in concepts], [c[0] for c in concepts]
N = len(TIDS)
ctx = Ctx(rig, list(rig.messages), PREFILL)
DET = ctx.DET

dirs_path = os.path.join(HERE, f"dirs_{KEY}.pt")
if os.path.exists(dirs_path):
    H_raw = torch.load(dirs_path)["H_raw"].to(rig.device)
else:
    H_raw = torch.stack(ctx.estimate_H(seeds=(100, 200, 300), n=60, single_layer=single)[1]).mean(0)
    torch.save({"H_raw": H_raw.cpu()}, dirs_path)
Hu = unit(H_raw)
res = {"model": KEY, "layers": L, "single_layer": single, "edit_layers": [EDIT[0], EDIT[-1]],
       "words": NAMES, "det_position": DET, "question_span": [ctx.q_lo, ctx.q_hi], "assistant_start": ctx.a_start}


def save():
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)


clean_trials = ctx.run(N)["det"]
clean = st.mean(clean_trials)
res["clean_det"] = clean


@torch.no_grad()
def capture(pos_lo, pos_hi, injects=()):
    """Residuals at positions [pos_lo, pos_hi) for every layer, batch of one."""
    caps = {}
    hds = [rig.layers[l].register_forward_hook(
        lambda m, a, o, l=l: caps.__setitem__(l, (o if torch.is_tensor(o) else o[0])[0, pos_lo:pos_hi].float().clone()))
        for l in range(L)]
    ctl = rig.ctl
    ctl.reset()
    ctl.injects = list(injects)
    try:
        rig.hf.model(input_ids=ctx.X1)
    finally:
        ctl.reset()
        for hd in hds:
            hd.remove()
    return caps


# ---------- Stage 2: transfer window ----------
q_clean = capture(ctx.q_lo, ctx.q_hi)
inj_w = word_inj(rig, TIDS, [single], 1.0, ctx.QSPAN)
inj_r = rnd_inj(rig, N, 301, [single], 1.0, ctx.QSPAN)
full_w = ctx.run(N, [inj_w])["det"]
full_r = ctx.run(N, [inj_r])["det"]
stage2 = {"full_word": ci([d - clean for d in full_w]), "full_random": ci([d - clean for d in full_r]), "cut": {}}
step = 1 if L <= 28 else 2
for l in range(single, L, step):
    patch = [{"layer": l, "lo": ctx.q_lo, "values": q_clean[l]}]
    w = [d - clean for d in ctx.run(N, [inj_w], patches=patch)["det"]]
    r = [d - clean for d in ctx.run(N, [inj_r], patches=patch)["det"]]
    stage2["cut"][str(l)] = {"word": ci(w), "random": ci(r), "word_trials": w, "random_trials": r}
    log(f"stage 2: question tokens restored at L{l:2d}: word {ci(w)[0]:+.2f}  random {ci(r)[0]:+.2f} "
        f"(full: {stage2['full_word'][0]:+.2f}, {stage2['full_random'][0]:+.2f})")
res["stage2_transfer"] = stage2
save()

# ---------- Stage 4: where is H needed? ----------


def clamp_at(U, l, only_pos=None):
    base = ctx.clamp(U, l)
    if only_pos is None:
        return base

    def fn(x, first):
        y = base(x, first)
        pos = torch.arange(first, first + x.shape[1], device=x.device)
        keep = (pos == only_pos)[None, :, None]
        return torch.where(keep, y, x)

    return fn


chunk = (len(EDIT) + 3) // 4
windows = [EDIT[i : i + chunk] for i in range(0, len(EDIT), chunk)]
inj_main = word_inj(rig, TIDS, band, 0.1, ctx.QSPAN)  # the original protocol
R1 = torch.nn.functional.normalize(torch.randn(L, rig.d, generator=torch.Generator().manual_seed(999)), dim=-1).to(rig.device)
holds = {
    "untouched": None,
    "H held, all layers, whole assistant turn": {l: clamp_at(Hu[l], l) for l in EDIT},
    "H held, all layers, answer position only": {l: clamp_at(Hu[l], l, DET) for l in EDIT},
    "random direction held, all layers": {l: clamp_at(R1[l], l) for l in EDIT},
}
for w in windows:
    holds[f"H held, layers {w[0]}-{w[-1]} only"] = {l: clamp_at(Hu[l], l) for l in w}
stage4 = {}
for name, ed in holds.items():
    base = st.mean(ctx.run(N, edits=ed)["det"]) if ed else clean
    t = ctx.run(N, [inj_main], edits=ed)["det"]
    ch = [d - base for d in t]
    stage4[name] = {"change": ci(ch), "frac_above": sum(d > base for d in t) / N, "baseline": base, "trials": ch}
    log(f"stage 4: {name:42s} change {ci(ch)[0]:+6.2f} [{ci(ch)[1]:+.2f}, {ci(ch)[2]:+.2f}] above {stage4[name]['frac_above']:.2f}")
res["stage4_report"] = stage4
save()

# ---------- Stage 3: is the word's identity present at the answer position? ----------


@torch.no_grad()
def hid_at(pos, injects, edits=None):
    ctl = rig.ctl
    ctl.reset()
    ctl.injects = list(injects)
    if edits:
        ctl.edits = {"a_start": ctx.a_start, "fns": edits}
    ctl.record = {"pos": pos, "out": {}}
    try:
        rig.hf.model(input_ids=ctx.X1.expand(N, -1))
        return torch.stack([ctl.record["out"][l] for l in range(L)], 1)  # [N, L, d]
    finally:
        ctl.reset()


def decode(X):
    """X: {strength: [N, d]} shift vectors. Leave-one-strength-out nearest centroid (cosine)."""
    ks = list(X)
    Xn = {k: torch.nn.functional.normalize(v, dim=-1) for k, v in X.items()}
    accs = []
    for k in ks:
        cent = torch.nn.functional.normalize(torch.stack([Xn[j] for j in ks if j != k]).mean(0), dim=-1)
        pred = (Xn[k] @ cent.T).argmax(-1)
        accs.append((pred == torch.arange(N, device=pred.device)).float().mean().item())
    return st.mean(accs)


STRENGTHS = (0.6, 0.8, 1.0, 1.2, 1.4)
held_all = holds["H held, all layers, whole assistant turn"]
stage3 = {"chance": 1 / N, "by_layer": {}}
for where, pos in (("answer position", DET), ("last question token", ctx.q_hi - 1)):
    cl = hid_at(pos, [])[0]
    for cond, ed in (("untouched", None), ("H held", held_all)):
        if where == "last question token" and cond == "H held":
            continue  # holding acts on the assistant turn only
        H = {s: hid_at(pos, [word_inj(rig, TIDS, [single], s, ctx.QSPAN)], ed) for s in STRENGTHS}
        for l in range(single, L):
            X = {s: H[s][:, l] - cl[l] for s in STRENGTHS}
            Xp = {s: x - (x @ Hu[l])[:, None] * Hu[l][None] for s, x in X.items()}
            key = f"{where} | {cond}"
            stage3["by_layer"].setdefault(key, {})[str(l)] = {"acc": decode(X), "acc_without_H": decode(Xp)}
        best = max(stage3["by_layer"][f"{where} | {cond}"].items(), key=lambda kv: kv[1]["acc"])
        log(f"stage 3: {where:20s} {cond:9s} best identity decoding L{best[0]} {best[1]['acc']:.2f} "
            f"(without H {best[1]['acc_without_H']:.2f}; chance {1 / N:.3f})")
        del H
res["stage3_identity"] = stage3
save()
log("DONE")
