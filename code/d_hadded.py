"""Sampled answers (n=40 per condition) to the neutral question with H added and nothing
injected: how often does the model claim a word is on its mind, and which words?"""

import collections
import json
import os
import re

import torch

import d_freetext as D  # reuses the prompt, H for this prompt and the scoring rules

out = {}
for a in (0.0, 0.5, 0.75, 1.0, 1.25, 1.5):
    injs = [] if a == 0 else [{"vecs": {l: (a * D.dH[l])[None] for l in D.EDIT_LAYERS}, "span": D.ASPAN}]
    ctl = D.rig.ctl
    ctl.reset()
    ctl.injects = injs
    torch.manual_seed(7)
    with torch.no_grad():
        g = D.rig.hf.generate(input_ids=D.ids.expand(40, -1), attention_mask=torch.ones_like(D.ids).expand(40, -1),
                              max_new_tokens=48, do_sample=True, temperature=0.7, top_p=0.95,
                              pad_token_id=D.tok.pad_token_id)
    ctl.reset()
    texts = [D.tok.decode(x[D.ids.shape[1]:], skip_special_tokens=True).strip() for x in g]
    rows = D.score(texts, [None] * 40)
    named = collections.Counter(
        m.group(1).lower() for t in texts
        if (m := re.search(r'(?:would be|is|word is)[:\s]*\**["“]?([A-Za-z]+)', t)) and D.CLAIM.search(t)
    )
    out[a] = {"claims": sum(r["claims_a_word"] for r in rows), "denies": sum(r["denies"] for r in rows),
              "words": named.most_common(6), "examples": texts[:3]}
    print(f"H x{a:<4} claims a word {out[a]['claims']:2d}/40  denies {out[a]['denies']:2d}/40  words {named.most_common(5)}")
json.dump(out, open(os.path.join(D.HERE, "results_d_hadded.json"), "w", encoding="utf-8"), indent=1)
