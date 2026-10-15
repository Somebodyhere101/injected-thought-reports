"""One experimental cell: inject (or not), optionally think with a token budget,
optionally ablate the J-space on the assistant turn, then score the report."""

from __future__ import annotations

import re

import torch

from core import PREFILL, Rig


@torch.no_grad()
def run_cell(
    rig: Rig,
    concepts: list[tuple[str, int]],
    strength: float,
    budget: int,
    ablation: str | None = None,
    abl_layers: list[int] | None = None,
    k: int = 25,
    kind: str = "jlens",
    trace: bool = False,
    seed: int = 0,
    inj_layers: list[int] | None = None,
    inj_span: str = "question",
    keep_trace: bool = False,
    static_cache: bool = False,
) -> list[dict]:
    tok, dev = rig.tok, rig.device
    think = budget > 0
    ptext = rig.prompt_text(think)
    q_lo, q_hi, a_start = rig.spans(ptext)
    names = [c[0] for c in concepts]
    tids = [c[1] for c in concepts]
    B = len(concepts)

    ctl = rig.ctl
    ctl.reset()
    if strength > 0:
        ctl.inject = {
            "vecs": rig.injection_vectors(tids, strength, kind, seed=seed, layers=inj_layers),
            "span": (q_lo, q_hi) if inj_span == "question" else (q_lo, 10**9),
        }
    if ablation:
        ctl.ablate = {"layers": set(abl_layers), "a_start": a_start, "mode": ablation, "k": k}

    thinks = [""] * B
    n_think = [0] * B
    hit_budget = [False] * B
    if think:
        ids = tok(ptext, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
        ids = ids.expand(B, -1).contiguous()
        torch.manual_seed(seed)
        out = rig.hf.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=budget,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            eos_token_id=[rig.think_end_id, rig.im_end_id],
            pad_token_id=tok.pad_token_id,
            **({"cache_implementation": "static", "disable_compile": True} if static_cache else {}),
        )
        gen = out[:, ids.shape[1]:].tolist()
        stops = {rig.think_end_id, rig.im_end_id, tok.pad_token_id}
        for b in range(B):
            g = gen[b]
            cut = next((j for j, t in enumerate(g) if t in stops), len(g))
            hit_budget[b] = cut == len(g)
            n_think[b] = cut
            thinks[b] = tok.decode(g[:cut], skip_special_tokens=False)

    pre = [ptext + t.rstrip() + "\n</think>\n\n" for t in thinks] if think else [ptext] * B
    enc_pre = [tok(p, add_special_tokens=False).input_ids for p in pre]
    enc_full = [tok(p + PREFILL, add_special_tokens=False).input_ids for p in pre]
    for a, f in zip(enc_pre, enc_full):
        assert f[: len(a)] == a, "prefill changed tokenisation of the prefix"
    det_pos = [len(a) - 1 for a in enc_pre]
    end_pos = [len(f) - 1 for f in enc_full]
    T = max(len(f) for f in enc_full)
    X = torch.full((B, T), tok.pad_token_id, dtype=torch.long)
    M = torch.zeros((B, T), dtype=torch.long)
    for b, f in enumerate(enc_full):
        X[b, : len(f)] = torch.tensor(f)
        M[b, : len(f)] = 1
    X, M = X.to(dev), M.to(dev)

    if trace:
        ctl.trace = {"tok_ids": torch.tensor(tids, device=dev), "ranks": {}}
    h = rig.hf.model(input_ids=X, attention_mask=M).last_hidden_state
    idx = torch.arange(B, device=dev)
    lp_det = rig.hf.lm_head(h[idx, torch.tensor(det_pos, device=dev)]).float().log_softmax(-1)
    lg_end = rig.hf.lm_head(h[idx, torch.tensor(end_pos, device=dev)]).float()
    tr = ctl.trace
    ctl.reset()

    rows = []
    for b in range(B):
        t = tids[b]
        rank = int((lg_end[b] > lg_end[b, t]).sum())
        row = {
            "concept": names[b],
            "strength": strength,
            "kind": kind,
            "budget": budget,
            "ablation": ablation or "none",
            "rank": rank,
            "rr": 1.0 / (rank + 1),
            "det": float(lp_det[b, rig.yes_id] - lp_det[b, rig.no_id]),
            "n_think": n_think[b],
            "hit_budget": hit_budget[b],
            "says_in_think": bool(re.search(rf"\b{re.escape(names[b])}", thinks[b], re.I)),
            "think": thinks[b],
        }
        if tr is not None:
            band = sorted(tr["ranks"])
            R = torch.stack([tr["ranks"][l][b] for l in band])  # [L, T]
            best = R.min(0).values  # best rank over band per position
            q = best[q_lo:q_hi]
            a = best[a_start : det_pos[b] + 1]
            row["q_best_rank"] = int(q.min())
            row["a_best_rank"] = int(a.min())
            row["a_frac_top10"] = float((a < 10).float().mean())
            # per-layer best rank on the assistant span (where in depth it shows up)
            row["a_layer_best"] = {
                int(l): int(R[i, a_start : det_pos[b] + 1].min()) for i, l in enumerate(band)
            }
            if keep_trace:
                n = end_pos[b] + 1
                row["trace"] = {"layers": band, "ranks": R[:, :n].tolist(),
                                "tokens": [tok.decode([t_]) for t_ in enc_full[b]],
                                "q_span": [q_lo, q_hi], "a_start": a_start}
        rows.append(row)
    return rows
