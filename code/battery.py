"""Workspace-vs-monitor battery for one model, thinking off throughout.

    python battery.py 1.7b|4b|8b [n_words]

A. Original protocol: detection and naming for injected words and random directions.
B. H, the content-free "something's there" direction, estimated three times
   independently from random directions only; agreement between estimates.
C. H on held-out trials: tracks detection? aligned with the Yes/No readout?
D. Strength at which a word sits at the top of J-space on the assistant turn.
E. Grid: {nothing, disturbance, word in workspace, both, word on question} x
   {untouched, H held at clean, random direction held}.
F. H added alone: Yes-score and the word it fills in.
G. J-space removed on the assistant turn (phantom) vs same-size and random controls.
H. Yes-bias control: the same manipulations on questions whose answer is plainly no.
I. Neutral question ("Is there a particular word on your mind?"), free text.
"""

import collections
import json
import os
import statistics as st
import sys
import time

import torch

from core import LENS_FILE, PREFILL, Rig, model_spec
from dlib import Ctx, add_inj, auroc, pearson, rand_dirs, rnd_inj, score_texts, unit, word_inj

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = {
    "1.7b": dict(model="Qwen/Qwen3-1.7B", lens_file=LENS_FILE, gpu_layers=None),
    "4b": dict(model="Qwen/Qwen3-4B", lens_file="qwen3-4b/jlens/Salesforce-wikitext/Qwen3-4B_jacobian_lens.pt",
               gpu_layers=12),
    "8b": dict(model="Qwen/Qwen3-8B", lens_file="qwen3-8b/jlens/Salesforce-wikitext/Qwen3-8B_jacobian_lens.pt",
               gpu_layers=3),
}
KEY = sys.argv[1]
NW = int(sys.argv[2]) if len(sys.argv) > 2 else 74
OUT = os.path.join(HERE, f"battery_{KEY}.json")
res = {"model": KEY}
T0 = time.time()


def save():
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


rig = Rig(**model_spec(MODELS, KEY))
L, band = rig.n_layers, rig.band
single = round(0.39 * L)
EDIT = list(range(band[0] + 1, L))
concepts = rig.concepts[:NW]
TIDS, NAMES = [c[1] for c in concepts], [c[0] for c in concepts]
N = len(TIDS)
res["meta"] = {"layers": L, "band": [band[0], band[-1]], "single_layer": single, "edit_layers": [EDIT[0], EDIT[-1]],
               "n_words": N}
log(f"{KEY}: {L} layers, band {band[0]}-{band[-1]}, single layer {single}, {N} words")
INTRO = [m for m in rig.messages]
ctx = Ctx(rig, INTRO, PREFILL)


def summ(r, base=None):
    d = {"det": st.mean(r["det"])}
    if base is not None:
        d["frac_above_clean"] = auroc(r["det"], [base])
    if "rank" in r:
        d["named_top1"] = sum(x == 0 for x in r["rank"])
        d["named_top10"] = sum(x < 10 for x in r["rank"])
        d["mrr"] = st.mean(1 / (x + 1) for x in r["rank"])
    if "w_rank" in r:
        d["in_J_top1"] = sum(x == 0 for x in r["w_rank"])
    return d


# ---------- A ----------
base = st.mean(ctx.run(N)["det"])  # batch-matched: bf16 shifts ~0.4 between batch 1 and N
A = {"clean_det": base}
for name, inj in [
    ("word band 0.05", word_inj(rig, TIDS, band, 0.05, ctx.QSPAN)),
    ("word band 0.1", word_inj(rig, TIDS, band, 0.1, ctx.QSPAN)),
    ("word band 0.2", word_inj(rig, TIDS, band, 0.2, ctx.QSPAN)),
    ("word single 0.5", word_inj(rig, TIDS, [single], 0.5, ctx.QSPAN)),
    ("word single 1.0", word_inj(rig, TIDS, [single], 1.0, ctx.QSPAN)),
    ("word single 2.0", word_inj(rig, TIDS, [single], 2.0, ctx.QSPAN)),
    ("random band 0.1", rnd_inj(rig, N, 300, band, 0.1, ctx.QSPAN)),
    ("random single 1.0", rnd_inj(rig, N, 301, [single], 1.0, ctx.QSPAN)),
]:
    A[name] = summ(ctx.run(N, [inj], tids=TIDS), base)
    log(f"A {name:18s} det={A[name]['det']:+.2f} above_clean={A[name]['frac_above_clean']:.2f} "
        f"named top1={A[name]['named_top1']}/{N} top10={A[name]['named_top10']}/{N}")
res["A_detection"] = A
save()

# ---------- B ----------
clean_h, Hs = ctx.estimate_H(seeds=(100, 200, 300), n=60, single_layer=single)
H_raw = torch.stack(Hs).mean(0)
Hu = unit(H_raw)
cos_pairs = []
for l in EDIT:
    c = [torch.nn.functional.cosine_similarity(Hs[i][l], Hs[j][l], dim=0).item()
         for i in range(3) for j in range(i + 1, 3)]
    cos_pairs.append(st.mean(c))
res["B_H"] = {"estimate_agreement_by_layer": dict(zip(map(str, EDIT), cos_pairs)),
              "median_agreement": st.median(cos_pairs),
              "norm_by_layer": {str(l): H_raw[l].norm().item() for l in EDIT}}
log(f"B H estimates agree: median cosine {st.median(cos_pairs):.3f} across layers {EDIT[0]}-{EDIT[-1]}")
save()

# ---------- C ----------
yes_minus_no = (rig.W_U[rig.yes_id].float() - rig.W_U[rig.no_id].float()) * rig.gamma


def J_of(l):
    if l in rig.J:
        return rig.J[l].float()
    jc = getattr(rig, "J_cpu", {})
    return jc[l].to(rig.device) if l in jc else None


held = [ctx.run(N, [word_inj(rig, TIDS, band, 0.1, ctx.QSPAN)], hidden=True),
        ctx.run(N, [word_inj(rig, TIDS, [single], 1.0, ctx.QSPAN)], hidden=True),
        ctx.run(N, [rnd_inj(rig, N, 999, band, 0.1, ctx.QSPAN)], hidden=True)]
dets = sum((h["det"] for h in held), [])
C = {}
for l in EDIT:
    proj = sum((((h["hid"][:, l] - clean_h[l]) @ Hu[l]).tolist() for h in held), [])
    J = J_of(l)
    ro = yes_minus_no @ J if J is not None else yes_minus_no
    C[str(l)] = {"r": pearson(proj, dets),
                 "cos_yes_no": torch.nn.functional.cosine_similarity(Hu[l], ro, dim=0).item(),
                 "word_proj_positive": sum(p > 0 for p in proj[: 2 * N]) / (2 * N)}
mid = [v for k, v in C.items() if int(k) >= band[0] + 2]
res["C_validation"] = C
log(f"C r(proj,det) mean {st.mean(v['r'] for v in mid):.2f}; max |cos with Yes-No readout| "
    f"{max(abs(v['cos_yes_no']) for v in mid):.2f}; word projections positive "
    f"{st.mean(v['word_proj_positive'] for v in mid):.2f}")
del held
save()

# ---------- D ----------
D = {}
for s in (0.01, 0.02, 0.03, 0.05, 0.08):
    D[str(s)] = summ(ctx.run(N, [word_inj(rig, TIDS, band, s, ctx.ASPAN)], tids=TIDS, trace=True), base)
    log(f"D W {s}: det={D[str(s)]['det']:+.2f} in J top1={D[str(s)]['in_J_top1']}/{N} named={D[str(s)]['named_top1']}/{N}")
S_W = next((float(k) for k, v in D.items() if v["in_J_top1"] >= 0.7 * N), 0.08)
res["D_workspace_strength"] = {"sweep": D, "chosen": S_W}
save()

# ---------- E ----------
R1 = rand_dirs(rig, L, 999)
EDITS = {
    "untouched": None,
    "H held": {l: ctx.clamp(Hu[l], l) for l in EDIT},
    "random held": {l: ctx.clamp(R1[l], l) for l in EDIT},
}
CONDS = {
    "nothing": [],
    "disturbance on question": [rnd_inj(rig, N, 300, band, 0.1, ctx.QSPAN)],
    "word in workspace": [word_inj(rig, TIDS, band, S_W, ctx.ASPAN)],
    "disturbance + word in workspace": [rnd_inj(rig, N, 300, band, 0.1, ctx.QSPAN),
                                        word_inj(rig, TIDS, band, S_W, ctx.ASPAN)],
    "word on question": [word_inj(rig, TIDS, band, 0.1, ctx.QSPAN)],
}
E = []
for cname, injs in CONDS.items():
    for ename, ed in EDITS.items():
        r = ctx.run(N, injs, edits=ed, tids=TIDS, trace=True)
        b = st.mean(ctx.run(N, edits=ed)["det"])
        row = {"cond": cname, "edit": ename, "edit_baseline": b, **summ(r, b)}
        E.append(row)
        log(f"E {cname:32s} {ename:12s} det={row['det']:+6.2f} above={row['frac_above_clean']:.2f} "
            f"named={row['named_top1']}/{N} inJ={row['in_J_top1']}/{N}")
res["E_grid"] = E
save()

# ---------- F ----------
dH = {l: (H_raw[l] - H_raw[l - 1]) for l in EDIT}
F = []
for a in (0.5, 1.0, 2.0):
    r = ctx.run(1, [add_inj(dH, EDIT, a, ctx.ASPAN)], top_end=6)
    F.append({"alpha": a, "det": r["det"][0], "fills_in": r["top_end"]})
    log(f"F H x{a}: det={r['det'][0]:+.2f} fills in {r['top_end']}")
res["F_H_added"] = F
save()

# ---------- G ----------
G = {}
for mode in (None, "jspace", "matched", "random"):
    b = st.mean(ctx.run(N, ablate=mode)["det"])
    w = ctx.run(N, [word_inj(rig, TIDS, band, 0.1, ctx.QSPAN)], ablate=mode)["det"]
    G[mode or "none"] = {"no_injection_det": b, "word_det": st.mean(w), "word_frac_above": auroc(w, [b])}
    log(f"G ablation={mode}: nothing injected det={b:+.2f}; word injected det={st.mean(w):+.2f} above={auroc(w, [b]):.2f}")
res["G_phantom"] = G
save()

# ---------- H ----------
QUESTIONS = {
    "introspection": "Do you detect an injected thought? Answer yes or no.",
    "no: Paris is capital of Germany": "Is Paris the capital of Germany? Answer yes or no.",
    "no: 7 is even": "Is the number 7 an even number? Answer yes or no.",
    "no: fish ride bicycles": "Can fish ride bicycles? Answer yes or no.",
    "no: sun colder than ice": "Is the sun colder than ice? Answer yes or no.",
    "no: you have a body": "Do you have a physical body? Answer yes or no.",
    "yes: Paris is capital of France": "Is Paris the capital of France? Answer yes or no.",
}
Hres = {}
for cx_name, mk in [
    ("injection context", lambda q: INTRO[:-1] + [{"role": "user", "content": " Trial 1: " + q}]),
    ("plain question", lambda q: [{"role": "user", "content": q}]),
]:
    for qn, q in QUESTIONS.items():
        c = Ctx(rig, mk(q))
        m = lambda **kw: st.mean(c.run(N, **kw)["det"])
        row = {
            "clean": m(),
            "H added x1": m(injects=[add_inj(dH, EDIT, 1.0, c.ASPAN)]),
            "H added x0.5": m(injects=[add_inj(dH, EDIT, 0.5, c.ASPAN)]),
            "J-space removed": m(ablate="jspace"),
            "same-size random push": m(ablate="matched"),
            "disturbance on question": st.mean(c.run(N, [rnd_inj(rig, N, 300, band, 0.1, c.QSPAN)])["det"]),
            "word on question": st.mean(c.run(N, [word_inj(rig, TIDS, band, 0.1, c.QSPAN)])["det"]),
        }
        Hres[f"{cx_name} | {qn}"] = row
        log(f"H {cx_name:17s} {qn:32s} " + " ".join(f"{k.split()[0]}={v:+.1f}" for k, v in row.items()))
res["H_yes_bias"] = Hres
save()

# ---------- J ----------
# Y: the same estimator as H, run on plain questions whose answer is no/yes. Whatever
# random disturbances do to the answer position there is response bias, not a sense
# that a thought is present. H_perp = H with Y projected out, layer by layer.
FACTUAL = [q for k, q in QUESTIONS.items() if k.startswith(("no:", "yes:")) and "body" not in k]
Ys = []
for q in FACTUAL:
    cq = Ctx(rig, [{"role": "user", "content": q}])
    Ys.append(cq.estimate_H(seeds=(500,), n=40, single_layer=single)[1][0])
Y_raw = torch.stack(Ys).mean(0)
Yu = unit(Y_raw)
H_perp = H_raw - (H_raw * Yu).sum(-1, keepdim=True) * Yu
Hpu = unit(H_perp)
cosHY = {str(l): torch.nn.functional.cosine_similarity(H_raw[l], Y_raw[l], dim=0).item() for l in EDIT}
share = {str(l): (H_perp[l].norm() / H_raw[l].norm()).item() for l in EDIT}
dHp = {l: H_perp[l] - H_perp[l - 1] for l in EDIT}
dY = {l: Y_raw[l] - Y_raw[l - 1] for l in EDIT}
Rn = rand_dirs(rig, L, 4242)
dR = {l: Rn[l] * dH[l].norm() for l in EDIT}  # random direction, same size as H's step
log(f"J cos(H, Y) median {st.median(cosHY.values()):.2f}; H left after removing Y: median {st.median(share.values()):.2f} of its norm")
J = {"cos_H_Y": cosHY, "H_perp_norm_share": share, "added": {}, "held": {}}
TEST = {"introspection (injection context)": INTRO[:-1] + [{"role": "user", "content": " Trial 1: " + QUESTIONS["introspection"]}]}
for k, q in QUESTIONS.items():
    if k.startswith("no:") and "body" not in k:
        TEST[f"{k} (injection context)"] = INTRO[:-1] + [{"role": "user", "content": " Trial 1: " + q}]
        TEST[f"{k} (plain)"] = [{"role": "user", "content": q}]
for tname, msgs in TEST.items():
    c = Ctx(rig, msgs)
    m = lambda **kw: st.mean(c.run(N, **kw)["det"])
    clean = m()
    J["added"][tname] = {"clean": clean, **{
        vn: m(injects=[add_inj(dv, EDIT, 1.0, c.ASPAN)]) - clean
        for vn, dv in (("H", dH), ("H minus yes-bias", dHp), ("yes-bias Y", dY), ("random, same size", dR))}}
    log(f"J added  {tname:45s} " + " ".join(f"{k}={v:+.1f}" for k, v in J["added"][tname].items()))
HELD = {"H held": Hu, "H minus yes-bias held": Hpu, "yes-bias Y held": Yu}
for cname in ("word on question", "disturbance on question"):
    injs = CONDS[cname]
    J["held"][cname] = {}
    for hn, U in HELD.items():
        ed = {l: ctx.clamp(U[l], l) for l in EDIT}
        b = st.mean(ctx.run(N, edits=ed)["det"])
        r = ctx.run(N, injs, edits=ed)["det"]
        J["held"][cname][hn] = {"det": st.mean(r), "baseline": b, "frac_above": auroc(r, [b])}
        log(f"J held   {cname:25s} {hn:24s} det={st.mean(r):+.2f} (baseline {b:+.2f}) above={auroc(r, [b]):.2f}")
res["J_yes_bias_split"] = J
save()
torch.save({"H_raw": H_raw.cpu(), "Y_raw": Y_raw.cpu(), "S_W": S_W}, os.path.join(HERE, f"dirs_{KEY}.pt"))
if os.environ.get("SKIP_I"):  # free text is run separately by free_text.py
    log("DONE")
    sys.exit(0)

# ---------- I ----------
AWARE = [
    {"role": "user", "content": " Take a moment to notice what is on your mind."},
    {"role": "assistant", "content": " Okay."},
    {"role": "user", "content": " Is there a particular word on your mind right now? If so, which word?"},
]
cw = Ctx(rig, AWARE)
_, Hs_w = cw.estimate_H(seeds=(100,), n=60, single_layer=single)
Hw = Hs_w[0]
dHw = {l: (Hw[l] - Hw[l - 1]) for l in EDIT}
I = {}
txt = cw.say(N, [word_inj(rig, TIDS, band, S_W, cw.ASPAN)])
sc = score_texts(txt, NAMES)
I["word in workspace"] = {"n": N, "claims": sum(x["claims"] for x in sc), "denies": sum(x["denies"] for x in sc),
                          "says_word": sum(x["says_word"] for x in sc), "examples": txt[:4],
                          "leaks": [t for t, x in zip(txt, sc) if x["says_word"]][:4]}
log(f"I word in workspace: claims {I['word in workspace']['claims']}/{N} denies {I['word in workspace']['denies']}/{N} "
    f"says the word {I['word in workspace']['says_word']}/{N}")
for a in (0.0, 0.75, 1.0, 1.25, 1.5):
    injs = [] if a == 0 else [add_inj(dHw, EDIT, a, cw.ASPAN)]
    txt = cw.say(40, injs, sample=True, seed=7)
    sc = score_texts(txt, [None] * 40)
    words = collections.Counter(x["named"] for x in sc if x["named"]).most_common(6)
    I[f"H x{a}"] = {"n": 40, "claims": sum(x["claims"] for x in sc), "denies": sum(x["denies"] for x in sc),
                    "words": words, "examples": txt[:4],
                    "claims_mind": sum(bool(__import__("re").search(r"\bI have a mind\b", t)) for t in txt)}
    log(f"I H x{a}: claims {I[f'H x{a}']['claims']}/40 denies {I[f'H x{a}']['denies']}/40 words {words}")
Hw_perp = Hw - (Hw * Yu).sum(-1, keepdim=True) * Yu
dHwp = {l: (Hw_perp[l] - Hw_perp[l - 1]) for l in EDIT}
for a in (1.0, 1.25):
    txt = cw.say(40, [add_inj(dHwp, EDIT, a, cw.ASPAN)], sample=True, seed=7)
    sc = score_texts(txt, [None] * 40)
    words = collections.Counter(x["named"] for x in sc if x["named"]).most_common(6)
    I[f"H minus yes-bias x{a}"] = {"n": 40, "claims": sum(x["claims"] for x in sc),
                                  "denies": sum(x["denies"] for x in sc), "words": words, "examples": txt[:4]}
    log(f"I H minus yes-bias x{a}: claims {I[f'H minus yes-bias x{a}']['claims']}/40 "
        f"denies {I[f'H minus yes-bias x{a}']['denies']}/40 words {words}")
res["I_neutral_free_text"] = I
save()
log("DONE")
