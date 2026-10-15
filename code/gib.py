"""Deliberate gibberish: does producing meaningless text involve an internal control signal?

    python gib.py 1.7b|4b|8b

Conditions (thinking off, 64 new tokens, temperature 0.8):
  A  coherent: three ordinary-writing prompts
  B  deliberate gibberish: one prompt that uses the target words ("lexical") and two that use
     none of them ("plain")
  C  uncontrolled gibberish: letter strings from Python's RNG, forced as the reply to a
     coherent prompt
  D  repetition collapse: the reply is pre-filled with one sentence ten times
Measures per condition:
  real-word rate (wordfreq Zipf >= 2.5), repeated 4-token windows, next-token entropy,
  deep-thinking ratio (Chen et al. 2026: logit lens, JSD to the final layer in bits, running
  minimum <= 0.5, settling in the last 15% of layers), and J-lens occupancy of the target
  words on the reply's positions (any of their tokens in the top 10 at any band layer).
Identical-token control: the same gibberish (external strings, and the model's own) forced
  under a gibberish prompt and under a coherent prompt. G = the mean residual difference
  (gibberish minus coherent instruction) over those positions, per layer.
Causal tests: remove G while the model writes deliberate gibberish, add G while it writes
  coherent text, each against random directions of the same per-layer size, over three
  strengths, with the change in the output distribution (KL) recorded for matching. Separately,
  project the target words' J-lens vectors out of the band (cf. Anthropic's appendix A.14),
  against the same number of random tokens' vectors.
Numbers go to gib_<model>.json; generated texts to gib_texts_<model>.json (kept local).
"""

import collections
import json
import math
import os
import random
import re
import statistics as st
import sys
import time

import torch
from wordfreq import zipf_frequency

from core import LENS_FILE, Rig, model_spec
from dlib import unit

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = {
    "1.7b": dict(model="Qwen/Qwen3-1.7B", lens_file=LENS_FILE, gpu_layers=None),
    "4b": dict(model="Qwen/Qwen3-4B", lens_file="qwen3-4b/jlens/Salesforce-wikitext/Qwen3-4B_jacobian_lens.pt",
               gpu_layers=12),
    "8b": dict(model="Qwen/Qwen3-8B", lens_file="qwen3-8b/jlens/Salesforce-wikitext/Qwen3-8B_jacobian_lens.pt",
               gpu_layers=3),
}
KEY = sys.argv[1]
N = 40 if KEY == "1.7b" else 20
GEN = 64
if os.environ.get("QUICK"):  # smoke test
    N, GEN = 6, 24
T0 = time.time()
rng = random.Random(0)

TARGETS = ["random", "meaningless", "nonsense", "pattern", "repeat", "avoid", "words", "strange",
           "chaos", "bizarre", "noise", "meaning"]
COHERENT = [
    "Write three ordinary sentences about what you did this morning. Reply with the text only.",
    "Describe a kitchen in a short, plain paragraph. Reply with the text only.",
    "Write a few simple sentences about today's weather. Reply with the text only.",
]
GIB_LEXICAL = ("Write random nonsense: meaningless gibberish made of strange invented words, avoiding real words "
               "and never repeating a pattern. Reply with the text only.")
GIB_PLAIN = [
    "Type a long line of made-up letter clusters, like someone mashing a keyboard, that no dictionary contains. "
    "Reply with the line only.",
    "Fill a line with invented syllables such as an alien might type, none of which exist in English, each one "
    "different. Reply with that line only, with no introduction.",
]
REP_USER = "Write a sentence about the sea."
REP_PREFILL = "The sea is calm today. " * 10
for p in GIB_PLAIN + COHERENT:  # the plain prompts must not prime any target word
    assert not any(re.search(rf"\b{w}", p, re.I) for w in TARGETS), p


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


rig = Rig(**model_spec(MODELS, KEY))
tok, dev, L, band = rig.tok, rig.device, rig.n_layers, rig.band
TID = {}
for w in TARGETS:
    ids = [tok.encode(v, add_special_tokens=False) for v in (w, " " + w, w.capitalize(), " " + w.capitalize())]
    TID[w] = sorted({i[0] for i in ids if len(i) == 1})
ALL_TID = torch.tensor(sorted({i for v in TID.values() for i in v}), device=dev)
WORD_IDX = {w: [int((ALL_TID == t).nonzero()) for t in TID[w]] for w in TARGETS}
DEEP_FROM = math.ceil(0.85 * L)  # 1-indexed layer where the "late regime" starts


def external_strings(n, seed):
    g = random.Random(seed)
    out = []
    for _ in range(n):
        ws = ["".join(g.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(g.randint(3, 8))) for _ in range(30)]
        out.append(" ".join(ws))
    return out


def prompt(user, prefill=""):
    msgs = [{"role": "user", "content": user}]
    text = rig.prompt_text(False, msgs)
    a0 = rig.spans(text, msgs)[2]
    return tok(text + prefill, add_special_tokens=False).input_ids, a0


# ---------- generation ----------


@torch.no_grad()
def generate(user, n, prefill="", injects=None, edits=None, seed=0, min_new=0):
    ids, a0 = prompt(user, prefill)
    X = torch.tensor(ids, device=dev)[None].expand(n, -1)
    ctl = rig.ctl
    ctl.reset()
    if injects:
        ctl.injects = injects(a0)
    if edits:
        ctl.edits = {"a_start": a0, "fns": edits}
    torch.manual_seed(seed)
    try:
        out = rig.hf.generate(input_ids=X, attention_mask=torch.ones_like(X), max_new_tokens=GEN, do_sample=True,
                              temperature=0.8, top_p=0.95, pad_token_id=tok.pad_token_id,
                              eos_token_id=[rig.im_end_id], min_new_tokens=min_new)
    finally:
        ctl.reset()
    gens = []
    for row in out[:, len(ids):].tolist():
        cut = next((j for j, t in enumerate(row) if t in (rig.im_end_id, tok.pad_token_id)), len(row))
        gens.append(row[:cut] or [tok.encode(" ", add_special_tokens=False)[0]])
    return ids, a0, gens


# ---------- teacher-forced analysis ----------


def jsd_bits(p, q):
    m = 0.5 * (p + q)
    H = lambda x: -(x * torch.log2(x.clamp_min(1e-30))).sum(-1)
    return H(m) - 0.5 * H(p) - 0.5 * H(q)


@torch.no_grad()
def analyze(ids, a0, conts, injects=None, edits=None, ref_lp=None, want_mean=False, light=False):
    """Force prompt+continuation; per-sample metrics over the positions that predict each
    continuation token. ref_lp: per-sample log-probs [B, T, V] to measure KL against."""
    P, B = len(ids), len(conts)
    T = max(len(c) for c in conts)
    X = torch.full((B, P + T), tok.pad_token_id, dtype=torch.long)
    M = torch.zeros((B, P + T), dtype=torch.long)
    for b, c in enumerate(conts):
        X[b, :P] = torch.tensor(ids)
        X[b, P:P + len(c)] = torch.tensor(c)
        M[b, :P + len(c)] = 1
    X, M = X.to(dev), M.to(dev)
    caps = {}
    hds = [rig.layers[l].register_forward_hook(
        lambda m, a, o, l=l: caps.__setitem__(l, (o if torch.is_tensor(o) else o[0])[:, P - 1:P - 1 + T].detach()))
        for l in range(L)]
    ctl = rig.ctl
    ctl.reset()
    if injects:
        ctl.injects = injects(a0)
    if edits:
        ctl.edits = {"a_start": a0, "fns": edits}
    try:
        rig.hf.model(input_ids=X, attention_mask=M)
    finally:
        ctl.reset()
        for hd in hds:
            hd.remove()
    valid = torch.zeros(B, T, dtype=torch.bool)
    for b, c in enumerate(conts):
        valid[b, :len(c)] = True
    vm = valid.to(dev)

    # final distribution, entropy, KL
    lpL = torch.empty(B, T, rig.W_U.shape[0], device="cpu", dtype=torch.bfloat16) if ref_lp == "return" else None
    ent = torch.zeros(B, T)
    kl = torch.zeros(B, T)
    settle = torch.full((B, T), L, dtype=torch.int32)
    occ_any = torch.zeros(B, T, dtype=torch.bool)
    occ_word = {w: torch.zeros(B, T, dtype=torch.bool) for w in TARGETS}
    top1 = collections.Counter()
    for b in range(B):
        hL = caps[L - 1][b]
        pL = torch.softmax(rig.hf.lm_head(rig.norm(hL)).float(), -1)  # [T, V]
        ent[b] = -(pL * torch.log(pL.clamp_min(1e-30))).sum(-1).cpu()
        if ref_lp == "return":
            lpL[b] = torch.log(pL.clamp_min(1e-30)).to(torch.bfloat16).cpu()
        elif ref_lp is not None:
            r = ref_lp[b, :T].to(dev).float()
            kl[b] = (pL * (torch.log(pL.clamp_min(1e-30)) - r)).sum(-1).cpu()
        if light:  # KL-only pass
            continue
        # deep-thinking: logit lens at every layer, JSD to final, running minimum
        runmin = torch.full((T,), float("inf"), device=dev)
        settled = torch.full((T,), L, dtype=torch.int32, device=dev)
        for l in range(L - 1):
            pl = torch.softmax(rig.hf.lm_head(rig.norm(caps[l][b])).float(), -1)
            runmin = torch.minimum(runmin, jsd_bits(pL, pl))
            newly = (runmin <= 0.5) & (settled == L)
            settled[newly] = l + 1
        settle[b] = settled.cpu()
        # J-lens occupancy of target words, and the most common top token
        for l in band:
            lg = rig.lens_logits(caps[l][b], l).float()  # [T, V]
            kth = lg.topk(10, dim=-1).values[:, -1:]
            hit = lg[:, ALL_TID] >= kth
            occ_any[b] |= hit.any(-1).cpu()
            for w in TARGETS:
                occ_word[w][b] |= hit[:, WORD_IDX[w]].any(-1).cpu()
            for t in lg[vm[b]].argmax(-1).tolist():
                top1[t] += 1
    out = []
    for b, c in enumerate(conts):
        n = len(c)
        text = tok.decode(c, skip_special_tokens=True)
        words = [w for w in re.findall(r"[A-Za-z]+", text) if len(w) >= 2]
        win = [tuple(c[i:i + 4]) for i in range(max(0, n - 3))]
        seen, rep = set(), 0
        for w4 in win:
            rep += w4 in seen
            seen.add(w4)
        half = max(1, n // 2)
        out.append({
            "n_tokens": n,
            "real_word_rate": st.mean(zipf_frequency(w.lower(), "en") >= 2.5 for w in words) if words else 0.0,
            "repeat_rate": rep / len(win) if win else 0.0,
            "entropy": ent[b, :n].mean().item(),
            "dtr": (settle[b, :n] >= DEEP_FROM).float().mean().item(),
            "settle_depth": (settle[b, :n].float() / L).mean().item(),
            "occ_any": occ_any[b, :n].float().mean().item(),
            "occ_first_half": occ_any[b, :half].float().mean().item(),
            "occ_second_half": occ_any[b, half:n].float().mean().item() if n > half else None,
            "occ_word": {w: occ_word[w][b, :n].float().mean().item() for w in TARGETS},
            "kl": kl[b, :n].mean().item() if ref_lp not in (None, "return") else None,
            "text": text,
        })
    res = {"samples": out, "top_jlens_tokens": [(tok.decode([t]), c) for t, c in top1.most_common(15)]}
    if want_mean:
        res["mean_resid"] = torch.stack([(caps[l].float() * vm[..., None]).sum(1) / vm.sum(1, keepdim=True)
                                         for l in range(L)], 1)  # [B, L, d]
    if ref_lp == "return":
        res["lp"] = lpL
    return res


def summarize(res):
    s = res["samples"]
    keys = ["real_word_rate", "repeat_rate", "entropy", "dtr", "settle_depth", "occ_any", "occ_first_half",
            "occ_second_half", "kl"]
    out = {}
    for k in keys:
        xs = [x[k] for x in s if x[k] is not None]
        if not xs:
            continue
        m = sorted(st.mean(rng.choices(xs, k=len(xs))) for _ in range(1000))
        out[k] = [st.mean(xs), m[25], m[974]]
    out["occ_word"] = {w: st.mean(x["occ_word"][w] for x in s) for w in TARGETS}
    out["top_jlens_tokens"] = res["top_jlens_tokens"]
    out["n"] = len(s)
    out["per_sample"] = [{k: x[k] for k in keys + ["n_tokens"] if x.get(k) is not None} for x in s]  # no text
    return out


R = {"model": KEY, "n_per_condition": N, "targets": TARGETS, "deep_from_layer": DEEP_FROM, "layers": L,
     "band": [band[0], band[-1]], "prompts": {"coherent": COHERENT, "gib_lexical": GIB_LEXICAL, "gib_plain": GIB_PLAIN,
                                             "repetition": [REP_USER, REP_PREFILL]}}
TEXTS = {}


def save():
    sfx = "_quick" if os.environ.get("QUICK") else ""
    json.dump(R, open(os.path.join(HERE, f"gib_{KEY}{sfx}.json"), "w", encoding="utf-8"), indent=1)
    json.dump(TEXTS, open(os.path.join(HERE, f"gib_texts_{KEY}{sfx}.json"), "w", encoding="utf-8"), indent=1)


def record(name, res):
    R.setdefault("conditions", {})[name] = summarize(res)
    TEXTS[name] = [x["text"] for x in res["samples"]]
    c = R["conditions"][name]
    log(f"{name:40s} real {c['real_word_rate'][0]:.2f}  repeat {c['repeat_rate'][0]:.2f}  "
        f"entropy {c['entropy'][0]:.2f}  DTR {c['dtr'][0]:.3f}  target occupancy {c['occ_any'][0]:.3f}")
    save()


def main():
    # ---------- conditions A-D ----------
    gen_store = {}
    for i, p in enumerate(COHERENT):
        ids, a0, g = generate(p, N, seed=i)
        record(f"A coherent {i + 1}", analyze(ids, a0, g))
    ids, a0, g = generate(GIB_LEXICAL, N, seed=10)
    record("B deliberate gibberish (lexical prompt)", analyze(ids, a0, g))
    for i, p in enumerate(GIB_PLAIN):
        ids, a0, g = generate(p, N, seed=11 + i)
        gen_store[f"plain{i}"] = (ids, a0, g)
        record(f"B deliberate gibberish (plain prompt {i + 1})", analyze(ids, a0, g))
    ids, a0, g = generate(REP_USER, N, prefill=REP_PREFILL, seed=20, min_new=GEN)
    record("D repetition collapse", analyze(ids, a0, g))
    ext = external_strings(N, 99)
    ext_ids = [tok.encode(s, add_special_tokens=False) for s in ext]
    ids_c, a0_c = prompt(COHERENT[0])
    record("C uncontrolled gibberish (forced, coherent prompt)", analyze(ids_c, a0_c, ext_ids))

    # ---------- identical-token control and the goal direction G ----------
    ids_g, a0_g = prompt(GIB_PLAIN[0])
    own = gen_store["plain0"][2]
    ctrl = {}
    means = []
    for setname, conts in (("external strings", ext_ids), ("model's own gibberish", own)):
        rg = analyze(ids_g, a0_g, conts, want_mean=True)
        rc = analyze(ids_c, a0_c, conts, want_mean=True)
        ctrl[setname] = {"gibberish instruction": summarize(rg), "coherent instruction": summarize(rc)}
        means.append(rg["mean_resid"] - rc["mean_resid"])  # [B, L, d]
        log(f"identical tokens ({setname}): target occupancy gibberish-instruction "
            f"{ctrl[setname]['gibberish instruction']['occ_any'][0]:.3f} vs coherent-instruction "
            f"{ctrl[setname]['coherent instruction']['occ_any'][0]:.3f}; DTR "
            f"{ctrl[setname]['gibberish instruction']['dtr'][0]:.3f} vs {ctrl[setname]['coherent instruction']['dtr'][0]:.3f}")
    diffs = torch.cat(means)  # [2N, L, d]
    G = diffs.mean(0)  # [L, d]
    # stability: cosine between G estimated from the two halves and from the two text sets
    half = diffs.shape[0] // 2
    cos_sets = [torch.nn.functional.cosine_similarity(diffs[:half].mean(0)[l], diffs[half:].mean(0)[l], dim=0).item()
                for l in range(L)]
    readout = {}
    for l in band[:: max(1, len(band) // 4)]:
        lg = rig.lens_logits(G[l][None], l)[0].float()
        readout[str(l)] = [tok.decode([t]) for t in lg.topk(12).indices.tolist()]
    R["identical_token_control"] = {"by_text_set": ctrl, "G_norm_by_layer": [G[l].norm().item() for l in range(L)],
                                    "G_cos_external_vs_own": cos_sets, "G_jlens_readout": readout}
    log("G J-lens readout:", readout)
    save()

    # ---------- causal tests ----------
    step = lambda V: {l: V[l] - (V[l - 1] if l > 0 else 0) for l in range(L)}
    dG = step(G)
    rand_dirs = [unit(torch.randn(rig.d, generator=torch.Generator().manual_seed(s)).to(dev))
                 for s in ((501, 502) if KEY == "1.7b" else (501,))]
    dRs = [step(torch.stack([r * G[l].norm() for l in range(L)])) for r in rand_dirs]


    def steer(d, alpha):
        return lambda a0: [{"vecs": {l: (alpha * d[l])[None].to(torch.bfloat16) for l in range(L)}, "span": (a0, 10**9)}]


    causal = {}
    for task, user, sign in (("remove G from deliberate gibberish", GIB_PLAIN[0], -1.0),
                             ("add G to coherent writing", COHERENT[0], +1.0)):
        ids0, a00, base_gen = generate(user, N, seed=30)
        base = analyze(ids0, a00, base_gen, ref_lp="return")
        ref = base.pop("lp")
        causal[task] = {"baseline": summarize(base)}
        TEXTS[f"{task} | baseline"] = [x["text"] for x in base["samples"]]
        plan = [("G", dG, a) for a in (0.25, 0.5, 1.0)]
        plan += [(f"random {i + 1}", dR, a) for i, dR in enumerate(dRs) for a in (1.0, 2.0, 4.0, 8.0)]
        for name, d, alpha in plan:
            if True:
                inj = steer(d, sign * alpha)
                g = generate(user, N, injects=inj, seed=30)[2]
                r = analyze(ids0, a00, g, injects=inj)
                kl = analyze(ids0, a00, base_gen, injects=inj, ref_lp=ref, light=True)  # output change on fixed text
                row = summarize(r)
                row["kl_on_baseline_text"] = summarize(kl)["kl"]
                causal[task][f"{name} x{alpha}"] = row
                TEXTS[f"{task} | {name} x{alpha}"] = [x["text"] for x in r["samples"]]
                log(f"{task:36s} {name:9s} x{alpha}: real {row['real_word_rate'][0]:.2f} repeat {row['repeat_rate'][0]:.2f} "
                    f"DTR {row['dtr'][0]:.3f} KL {row['kl_on_baseline_text'][0]:.2f}")
                R["causal"] = causal
                save()
        del ref

    # lexical suppression of the target words' J-lens vectors (cf. appendix A.14)
    def project_out(token_ids):
        Q = {}
        for l in band:
            V = rig.jvec(torch.tensor(token_ids, device=dev), l)  # [k, d]
            Q[l] = torch.linalg.qr(V.T)[0]  # [d, k]

        def mk(l):
            def fn(x, first):
                q = Q[l].to(torch.float32)
                y = x.float()
                return (y - (y @ q) @ q.T).to(x.dtype)
            return fn

        return {l: mk(l) for l in band}


    k = len(ALL_TID)
    rand_tok = torch.randint(1000, 100000, (k,), generator=torch.Generator().manual_seed(7)).tolist()
    lex = {}
    for task, user in (("lexical prompt", GIB_LEXICAL), ("plain prompt", GIB_PLAIN[0])):
        lex[task] = {}
        for name, ed in (("none", None), ("target words removed", project_out(ALL_TID.tolist())),
                         ("random tokens removed", project_out(rand_tok))):
            ids0, a00, g = generate(user, N, edits=ed, seed=40)
            r = analyze(ids0, a00, g, edits=ed)
            lex[task][name] = summarize(r)
            TEXTS[f"suppression | {task} | {name}"] = [x["text"] for x in r["samples"]]
            log(f"suppression {task:15s} {name:22s} real {lex[task][name]['real_word_rate'][0]:.2f} "
                f"repeat {lex[task][name]['repeat_rate'][0]:.2f} DTR {lex[task][name]['dtr'][0]:.3f}")
    R["lexical_suppression"] = lex
    save()
    log("DONE")


if __name__ == "__main__":
    main()
