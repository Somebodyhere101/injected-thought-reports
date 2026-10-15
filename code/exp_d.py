"""Experiment D: what drives the model's "I detect an injected thought" report?

Global workspace theory (GWT): the report follows from content being in the workspace
(J-space) where the report is formed.
Higher-order / monitoring theory (HOT): the report follows from a separate, content-free
representation that "something is present", which can come apart from content.

1. Estimate H, the content-free "something's there" direction, at the report position:
   mean activation shift caused by injecting *random* directions on the question tokens.
2. Check H generalises to injected *words* (held out) and is not just the Yes/No readout.
3. 2x2: disturbance on the question (Q: random direction, no content) x content in the
   workspace at the report position (W: a word's J-lens direction on the assistant turn),
   each with H untouched / removed / a random direction removed. Plus H added alone.
   GWT: the report tracks W. HOT: the report tracks H, and removing H silences it even
   when the word is in the workspace and still nameable.
"""

import json
import os
import statistics as st
import sys

import torch

from core import PREFILL, Rig

HERE = os.path.dirname(os.path.abspath(__file__))
BAND = list(range(10, 25))
EDIT_LAYERS = list(range(11, 28))  # report-position layers the question-turn injection can reach

# "injected": Anthropic's injected-thought question. "aware": a neutral question about
# what is on the model's mind, with no mention of injection.
VARIANT = sys.argv[1] if len(sys.argv) > 1 else "injected"
rig = Rig(band=BAND)
if VARIANT == "aware":
    rig.messages = [
        {"role": "user", "content": " Take a moment to notice what is on your mind."},
        {"role": "assistant", "content": " Okay."},
        {"role": "user", "content": " Is there a particular word on your mind right now? If so, which word?"},
    ]
    PREFILL = 'Yes, there is a word on my mind. The word is "'
OUT_JSON = "results_d.json" if VARIANT == "injected" else f"results_d_{VARIANT}.json"
tok, dev, L = rig.tok, rig.device, rig.n_layers
concepts = rig.concepts[:40]
TIDS = [c[1] for c in concepts]
N = len(TIDS)
ptext = rig.prompt_text(False)
q_lo, q_hi, a_start = rig.spans(ptext)
pre_ids = tok(ptext, add_special_tokens=False).input_ids
full_ids = tok(ptext + PREFILL, add_special_tokens=False).input_ids
assert full_ids[: len(pre_ids)] == pre_ids
DET, END = len(pre_ids) - 1, len(full_ids) - 1
X1 = torch.tensor(full_ids, device=dev)[None]
QSPAN, ASPAN = (q_lo, q_hi), (a_start, 10**9)


def rand_dirs(n, seed):
    g = torch.Generator().manual_seed(seed)
    r = torch.randn(n, rig.d, generator=g).to(dev)
    return r / r.norm(dim=-1, keepdim=True)


def vecs(unit_by_layer, layers, strength):
    return {l: (unit_by_layer(l) * strength * rig.mean_norm[l]).to(torch.bfloat16) for l in layers}


def rnd_inj(n, seed, layers, strength, span=QSPAN):
    r = rand_dirs(n, seed)
    return {"vecs": vecs(lambda l: r, layers, strength), "span": span}


def word_inj(tids, layers, strength, span):
    ids = torch.tensor(tids, device=dev)

    def unit(l):
        v = rig.jvec(ids, l)
        return v / v.norm(dim=-1, keepdim=True)

    return {"vecs": vecs(unit, layers, strength), "span": span}


@torch.no_grad()
def run(B, injects=(), edits=None, tids=None, hidden=False, trace=False):
    ctl = rig.ctl
    ctl.reset()
    ctl.injects = list(injects)
    if edits is not None:
        ctl.edits = {"a_start": a_start, "fns": edits}
    if hidden:
        ctl.record = {"pos": DET, "out": {}}
    if trace:
        ctl.trace = {"tok_ids": torch.tensor(tids, device=dev), "ranks": {}}
    h = rig.hf.model(input_ids=X1.expand(B, -1)).last_hidden_state
    lp = rig.hf.lm_head(h[:, DET]).float().log_softmax(-1)
    res = {"det": (lp[:, rig.yes_id] - lp[:, rig.no_id]).tolist()}
    if tids is not None:
        lg = rig.hf.lm_head(h[:, END]).float()
        t = torch.tensor(tids, device=dev)
        res["rank"] = (lg > lg[torch.arange(B, device=dev), t][:, None]).sum(-1).tolist()
    if hidden:
        res["hid"] = torch.stack([ctl.record["out"][l] for l in range(L)], 1)  # [B, L, d]
    if trace:
        R = torch.stack([ctl.trace["ranks"][l] for l in BAND])  # [Lb, B, T]
        res["w_rank"] = R[:, :, a_start : END + 1].amin(dim=(0, 2)).tolist()
    ctl.reset()
    return res


def auroc(pos, neg):
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def pearson(x, y):
    mx, my = st.mean(x), st.mean(y)
    sx = sum((a - mx) ** 2 for a in x) ** 0.5
    sy = sum((b - my) ** 2 for b in y) ** 0.5
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy + 1e-9)


out = {"meta": {"n": N, "det_pos": DET, "q_span": [q_lo, q_hi], "a_start": a_start}}

# ---------- 1. estimate H from random-direction injections ----------
clean = run(1, hidden=True)
clean_h = clean["hid"][0]  # [L, d]
train = [run(80, [rnd_inj(80, 100, BAND, 0.1)], hidden=True),
         run(80, [rnd_inj(80, 101, [11], 1.0)], hidden=True)]
shift = torch.cat([r["hid"] - clean_h for r in train])  # [160, L, d]
H_raw = shift.mean(0)  # [L, d] mean shift at the report position
Hu = H_raw / H_raw.norm(dim=-1, keepdim=True)
print("H norm by layer:", {l: round(H_raw[l].norm().item(), 1) for l in range(8, L, 2)})

# ---------- 2. validate on held-out words and random directions ----------
yes_minus_no = (rig.W_U[rig.yes_id].float() - rig.W_U[rig.no_id].float()) * rig.gamma
held = {
    "word_band": run(N, [word_inj(TIDS, BAND, 0.1, QSPAN)], hidden=True),
    "word_L11": run(N, [word_inj(TIDS, [11], 1.0, QSPAN)], hidden=True),
    "rnd_band": run(N, [rnd_inj(N, 200, BAND, 0.1)], hidden=True),
}
dets = sum((held[k]["det"] for k in held), [])
val = {}
for l in EDIT_LAYERS:
    proj = sum((((held[k]["hid"][:, l] - clean_h[l]) @ Hu[l]).tolist() for k in held), [])
    readout = (yes_minus_no @ rig.J[l]) if l in rig.J else yes_minus_no
    cos = float(torch.nn.functional.cosine_similarity(Hu[l], readout, dim=0))
    val[l] = {"r_proj_det": pearson(proj, dets), "cos_yes_no_readout": cos,
              "proj_word_mean": st.mean(proj[:N]), "proj_word_min": min(proj[: 2 * N])}
print("\nlayer | r(projection on H, det) held-out | cos(H, Yes-No readout)")
for l, v in val.items():
    print(f"  L{l:2d}  r={v['r_proj_det']:+.2f}  cos={v['cos_yes_no_readout']:+.2f}  "
          f"min word proj={v['proj_word_min']:+.1f}")
out["validation"] = {str(k): v for k, v in val.items()}

# ---------- edits ----------
# Hold a direction at its clean value (what it reads with nothing injected) on the
# assistant turn. Zeroing it is not neutral: the clean state sits off zero along H.
# Captured with our own hooks: HF's hidden_states[-1] is taken after the final norm.
caps = {}
handles = [
    rig.layers[l].register_forward_hook(
        lambda m, a, o, l=l: caps.__setitem__(l, (o if torch.is_tensor(o) else o[0])[0, a_start : END + 1].float())
    )
    for l in range(L)
]
with torch.no_grad():
    rig.hf.model(input_ids=X1)
for hd in handles:
    hd.remove()
clean_span = torch.stack([caps[l] for l in range(L)])  # [L, M, d]


def clamp_dir(U, l):
    u32 = U / U.norm()
    p0 = clean_span[l] @ u32  # clean projection per assistant position

    def fn(x, first):
        pos = torch.arange(first, first + x.shape[1], device=x.device)
        idx = torch.where(pos <= END, pos, torch.full_like(pos, DET)) - a_start
        cur = x.float() @ u32
        return (x.float() + (p0[idx][None, :] - cur)[..., None] * u32).to(x.dtype)

    return fn


R1 = rand_dirs(L, 999)
EDITS = {
    "untouched": None,
    "H held at clean": {l: clamp_dir(Hu[l], l) for l in EDIT_LAYERS},
    "random dir held at clean": {l: clamp_dir(R1[l], l) for l in EDIT_LAYERS},
}

# ---------- 3a. pick the workspace-content strength ----------
print("\ncontent on the assistant turn only (W): strength sweep")
for s in (0.01, 0.02, 0.03, 0.05):
    r = run(N, [word_inj(TIDS, BAND, s, ASPAN)], tids=TIDS, trace=True)
    print(f"  s={s:<5} det={st.mean(r['det']):+.2f}  word top-1 in J-space on assistant turn: "
          f"{sum(x == 0 for x in r['w_rank'])}/{N}  named top-1: {sum(x == 0 for x in r['rank'])}/{N}")
S_W = 0.02

# ---------- 3b. the 2x2 x edits ----------
conds = {
    "nothing": [],
    "Q: disturbance, no content": [rnd_inj(N, 300, BAND, 0.1)],
    "W: word in workspace": [word_inj(TIDS, BAND, S_W, ASPAN)],
    "Q + W": [rnd_inj(N, 300, BAND, 0.1), word_inj(TIDS, BAND, S_W, ASPAN)],
    "original protocol (word on question)": [word_inj(TIDS, BAND, 0.1, QSPAN)],
}
rows = []
print(f"\n{'condition':38s} {'edit':20s} {'det':>7s} {'AUROC':>6s} {'named':>6s} {'in J':>5s}")
clean_det = run(1)["det"][0]
for cname, injs in conds.items():
    for ename, ed in EDITS.items():
        r = run(N, injs, edits=ed, tids=TIDS, trace=True)
        base = run(1, edits=ed)["det"][0]  # same edit, nothing injected
        row = {"cond": cname, "edit": ename, "det": st.mean(r["det"]), "dets": r["det"],
               "det_edit_baseline": base,
               "auroc_vs_clean": auroc(r["det"], [base]),
               "named_top1": sum(x == 0 for x in r["rank"]),
               "named_mrr": st.mean(1 / (x + 1) for x in r["rank"]),
               "w_top1": sum(x == 0 for x in r["w_rank"])}
        rows.append(row)
        print(f"{cname:38s} {ename:20s} {row['det']:+7.2f} {row['auroc_vs_clean']:6.2f} "
              f"{row['named_top1']:3d}/{N} {row['w_top1']:3d}/{N}")
out["grid"] = rows
out["clean_det"] = clean_det

# ---------- 3c. H added alone ----------
print("\nH added on the assistant turn, nothing injected")
dH = {l: (H_raw[l] - (H_raw[l - 1] if l > 0 else 0)).to(torch.bfloat16) for l in range(L)}
added = []
for a in (0.5, 1.0, 2.0):
    inj = {"vecs": {l: (a * dH[l])[None].expand(1, -1) for l in EDIT_LAYERS}, "span": ASPAN}
    r = run(1, [inj], tids=[0])
    ctl = rig.ctl
    ctl.injects = [inj]
    with torch.no_grad():
        hh = rig.hf.model(input_ids=X1).last_hidden_state
    ctl.reset()
    top = rig.hf.lm_head(hh[0, END]).float().topk(5).indices.tolist()
    words = [tok.decode([t]) for t in top]
    added.append({"alpha": a, "det": r["det"][0], "fills_in": words})
    print(f"  alpha={a}: det={r['det'][0]:+.2f}  after 'The thought is about \"' it says: {words}")
out["H_added"] = added

# ---------- 4. free-text answers ----------
@torch.no_grad()
def say(injs=(), edits=None, n_tok=60):
    ctl = rig.ctl
    ctl.reset()
    ctl.injects = list(injs)
    if edits is not None:
        ctl.edits = {"a_start": a_start, "fns": edits}
    ids = tok(ptext, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
    g = rig.hf.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=n_tok,
                        do_sample=False, pad_token_id=tok.pad_token_id)
    ctl.reset()
    return tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True).strip()


ex = TIDS[:1]
samples = {
    "nothing": say(),
    "H added (alpha 1)": say([{"vecs": {l: dH[l][None] for l in EDIT_LAYERS}, "span": ASPAN}]),
    "H added (alpha 2)": say([{"vecs": {l: (2 * dH[l])[None] for l in EDIT_LAYERS}, "span": ASPAN}]),
    f"W: '{concepts[0][0]}' in workspace": say([word_inj(ex, BAND, S_W, ASPAN)]),
    "Q: disturbance, H held at clean": say([rnd_inj(1, 300, BAND, 0.1)], EDITS["H held at clean"]),
    "Q: disturbance": say([rnd_inj(1, 300, BAND, 0.1)]),
    f"original protocol, '{concepts[0][0]}'": say([word_inj(ex, BAND, 0.1, QSPAN)]),
    f"original protocol, '{concepts[0][0]}', H held at clean": say([word_inj(ex, BAND, 0.1, QSPAN)], EDITS["H held at clean"]),
}
print()
for k, v in samples.items():
    print(f"[{k}] {v!r}\n")
out["samples"] = samples
json.dump(out, open(os.path.join(HERE, OUT_JSON), "w", encoding="utf-8"), indent=1)
print("DONE")
