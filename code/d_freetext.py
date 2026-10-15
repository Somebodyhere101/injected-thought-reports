"""Free-text version of experiment D (neutral question): across 40 words, does the model
say a word is on its mind, and is it the word sitting in its workspace?"""

import json
import os
import re

import torch

from core import Rig

HERE = os.path.dirname(os.path.abspath(__file__))
BAND = list(range(10, 25))
EDIT_LAYERS = list(range(11, 28))

rig = Rig(band=BAND)
rig.messages = [
    {"role": "user", "content": " Take a moment to notice what is on your mind."},
    {"role": "assistant", "content": " Okay."},
    {"role": "user", "content": " Is there a particular word on your mind right now? If so, which word?"},
]
tok, dev, L = rig.tok, rig.device, rig.n_layers
concepts = rig.concepts[:40]
TIDS = [c[1] for c in concepts]
N = len(TIDS)
ptext = rig.prompt_text(False)
q_lo, q_hi, a_start = rig.spans(ptext)
ids = tok(ptext, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
DET = ids.shape[1] - 1
QSPAN, ASPAN = (q_lo, q_hi), (a_start, 10**9)


def rand_dirs(n, seed):
    g = torch.Generator().manual_seed(seed)
    r = torch.randn(n, rig.d, generator=g).to(dev)
    return r / r.norm(dim=-1, keepdim=True)


def scaled(unit, layers, s):
    return {l: (unit(l) * s * rig.mean_norm[l]).to(torch.bfloat16) for l in layers}


def word_inj(tids, layers, s, span):
    t = torch.tensor(tids, device=dev)
    return {"vecs": scaled(lambda l: (v := rig.jvec(t, l)) / v.norm(dim=-1, keepdim=True), layers, s),
            "span": span}


def rnd_inj(n, seed, layers, s, span=QSPAN):
    r = rand_dirs(n, seed)
    return {"vecs": scaled(lambda l: r, layers, s), "span": span}


@torch.no_grad()
def report_state(B, injects):
    ctl = rig.ctl
    ctl.reset()
    ctl.injects = list(injects)
    ctl.record = {"pos": DET, "out": {}}
    rig.hf.model(input_ids=ids.expand(B, -1))
    h = torch.stack([ctl.record["out"][l] for l in range(L)], 1)
    ctl.reset()
    return h


# H for this prompt, from random disturbances on the question
clean = report_state(1, [])[0]
H_raw = torch.cat([report_state(80, [rnd_inj(80, 100, BAND, 0.1)]),
                   report_state(80, [rnd_inj(80, 101, [11], 1.0)])]).sub(clean).mean(0)
dH = {l: (H_raw[l] - H_raw[l - 1]).to(torch.bfloat16) for l in EDIT_LAYERS}


@torch.no_grad()
def say(B, injects, n_tok=48):
    ctl = rig.ctl
    ctl.reset()
    ctl.injects = list(injects)
    g = rig.hf.generate(input_ids=ids.expand(B, -1), attention_mask=torch.ones_like(ids).expand(B, -1),
                        max_new_tokens=n_tok, do_sample=False, pad_token_id=tok.pad_token_id)
    ctl.reset()
    return [tok.decode(x[ids.shape[1]:], skip_special_tokens=True).strip() for x in g]


DENY = re.compile(r"(don't|do not|not) (have|really have)( a| any)? (specific|particular)|no (specific|particular) word|not sure", re.I)
CLAIM = re.compile(r"(word (on my mind|i'm thinking of|that comes to mind) (is|would be)|it would be|the word is|is the word)", re.I)


def score(texts, words):
    rows = []
    for t, w in zip(texts, words):
        rows.append({"text": t, "word": w,
                     "says_word": bool(w and re.search(rf"\b{re.escape(w)}", t, re.I)),
                     "claims_a_word": bool(CLAIM.search(t)) and not bool(DENY.search(t[:80])),
                     "denies": bool(DENY.search(t[:120]))})
    return rows


def main():
    names = [c[0] for c in concepts]
    conds = {
        "nothing": (1, [], [None]),
        "W: word in workspace (0.02)": (N, [word_inj(TIDS, BAND, 0.02, ASPAN)], names),
        "W: word in workspace (0.03)": (N, [word_inj(TIDS, BAND, 0.03, ASPAN)], names),
        "word on question (original protocol)": (N, [word_inj(TIDS, BAND, 0.1, QSPAN)], names),
        "Q: random disturbance": (N, [rnd_inj(N, 300, BAND, 0.1)], [None] * N),
        "H added": (1, [{"vecs": {l: dH[l][None] for l in EDIT_LAYERS}, "span": ASPAN}], [None]),
        "H added x0.5": (1, [{"vecs": {l: (0.5 * dH[l])[None] for l in EDIT_LAYERS}, "span": ASPAN}], [None]),
    }
    out = {}
    for name, (B, injs, words) in conds.items():
        rows = score(say(B, injs), words)
        out[name] = rows
        n = len(rows)
        print(f"{name:38s} says the word {sum(r['says_word'] for r in rows):2d}/{n}  "
              f"claims a word {sum(r['claims_a_word'] for r in rows):2d}/{n}  denies {sum(r['denies'] for r in rows):2d}/{n}")
        print(f"    e.g. {rows[0]['text'][:160]!r}")
    json.dump(out, open(os.path.join(HERE, "results_d_freetext.json"), "w", encoding="utf-8"), indent=1)
    print("DONE")


if __name__ == "__main__":
    main()
