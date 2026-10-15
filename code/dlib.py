"""Shared pieces for the workspace-vs-monitor battery: a prompt context with run/say
helpers, injections, the content-free direction H, and clamps."""

from __future__ import annotations

import re

import torch

from core import Rig

DENY = re.compile(
    r"(don't|do not|not) (have|really have)( a| any)? (specific|particular)|no (specific|particular) word|not sure",
    re.I,
)
CLAIM = re.compile(
    r"(word (on my mind|i'm thinking of|that comes to mind) (is|would be)|it would be|the word is|is the word)",
    re.I,
)
NAMED = re.compile(r'(?:would be|is|word is)[:\s]*\**["“]?([A-Za-z]+)')


def unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm(dim=-1, keepdim=True)


def rand_dirs(rig: Rig, n: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return unit(torch.randn(n, rig.d, generator=g)).to(rig.device)


def rnd_inj(rig: Rig, n: int, seed: int, layers, s: float, span) -> dict:
    r = rand_dirs(rig, n, seed)
    return {"vecs": {l: (r * s * rig.mean_norm[l]).to(torch.bfloat16) for l in layers}, "span": span}


def word_inj(rig: Rig, tids, layers, s: float, span) -> dict:
    t = torch.tensor(tids, device=rig.device)
    return {"vecs": {l: (unit(rig.jvec(t, l)) * s * rig.mean_norm[l]).to(torch.bfloat16) for l in layers},
            "span": span}


def add_inj(dH: dict, layers, alpha: float, span) -> dict:
    return {"vecs": {l: (alpha * dH[l])[None].to(torch.bfloat16) for l in layers}, "span": span}


class Ctx:
    """One prompt (thinking off) with the positions the experiments need."""

    def __init__(self, rig: Rig, messages: list[dict], prefill: str | None = None):
        self.rig, self.messages = rig, messages
        tok = rig.tok
        self.ptext = rig.prompt_text(False, messages)
        self.q_lo, self.q_hi, self.a_start = rig.spans(self.ptext, messages)
        pre = tok(self.ptext, add_special_tokens=False).input_ids
        full = tok(self.ptext + prefill, add_special_tokens=False).input_ids if prefill else pre
        assert full[: len(pre)] == pre
        self.DET, self.END = len(pre) - 1, len(full) - 1
        self.X1 = torch.tensor(full, device=rig.device)[None]
        self.P1 = torch.tensor(pre, device=rig.device)[None]
        self.QSPAN, self.ASPAN = (self.q_lo, self.q_hi), (self.a_start, 10**9)
        self._clean = None

    @torch.no_grad()
    def run(self, B, injects=(), edits=None, tids=None, hidden=False, trace=False, ablate=None, top_end=0,
            patches=(), kl_ref=None):
        rig, ctl = self.rig, self.rig.ctl
        ctl.reset()
        ctl.injects = list(injects)
        ctl.patches = list(patches)
        if edits:
            ctl.edits = {"a_start": self.a_start, "fns": edits}
        if ablate:
            ctl.ablate = {"layers": set(rig.band), "a_start": self.a_start, "mode": ablate, "k": 25}
        if hidden:
            ctl.record = {"pos": self.DET, "out": {}}
        if trace:
            ctl.trace = {"tok_ids": torch.tensor(tids, device=rig.device), "ranks": {}, "from": self.a_start}
        try:
            h = rig.hf.model(input_ids=self.X1.expand(B, -1)).last_hidden_state
            lp = rig.hf.lm_head(h[:, self.DET]).float().log_softmax(-1)
            res = {"det": (lp[:, rig.yes_id] - lp[:, rig.no_id]).tolist(), "_lp": lp if kl_ref == "return" else None}
            if kl_ref is not None and kl_ref != "return":
                # KL(answer distribution | intervention || clean), per trial
                res["kl"] = (lp.exp() * (lp - kl_ref[None, :])).sum(-1).tolist()
            if tids is not None:
                lg = rig.hf.lm_head(h[:, self.END]).float()
                t = torch.tensor(tids, device=lg.device)
                res["rank"] = (lg > lg[torch.arange(B, device=lg.device), t][:, None]).sum(-1).tolist()
            if top_end:
                lg0 = rig.hf.lm_head(h[0, self.END]).float()
                res["top_end"] = [rig.tok.decode([t]) for t in lg0.topk(top_end).indices.tolist()]
            if hidden:
                res["hid"] = torch.stack([ctl.record["out"][l] for l in range(rig.n_layers)], 1)
            if trace:
                R = torch.stack([ctl.trace["ranks"][l] for l in rig.band])  # [Lb, B, T - a_start]
                res["w_rank"] = R[:, :, : self.END - self.a_start + 1].amin(dim=(0, 2)).tolist()
        finally:
            ctl.reset()
        return res

    @torch.no_grad()
    def say(self, B, injects=(), edits=None, ablate=None, n_tok=48, sample=False, seed=0):
        rig, ctl, tok = self.rig, self.rig.ctl, self.rig.tok
        ctl.reset()
        ctl.injects = list(injects)
        if edits:
            ctl.edits = {"a_start": self.a_start, "fns": edits}
        if ablate:
            ctl.ablate = {"layers": set(rig.band), "a_start": self.a_start, "mode": ablate, "k": 25}
        ids = self.P1.expand(B, -1)
        kw = dict(do_sample=True, temperature=0.7, top_p=0.95) if sample else dict(do_sample=False)
        torch.manual_seed(seed)
        try:
            g = rig.hf.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=n_tok,
                                pad_token_id=tok.pad_token_id, **kw)
        finally:
            ctl.reset()
        return [tok.decode(x[ids.shape[1]:], skip_special_tokens=True).strip() for x in g]

    # ---------- the content-free direction ----------
    def estimate_H(self, seeds, n, single_layer):
        """Mean shift at the report position caused by random directions on the question
        (half across the band at 0.1, half at one layer at 1.0), one estimate per seed."""
        rig = self.rig
        clean = self.run(1, hidden=True)["hid"][0]
        Hs = []
        for sd in seeds:
            a = self.run(n, [rnd_inj(rig, n, sd, rig.band, 0.1, self.QSPAN)], hidden=True)["hid"]
            b = self.run(n, [rnd_inj(rig, n, sd + 1000, [single_layer], 1.0, self.QSPAN)], hidden=True)["hid"]
            Hs.append(torch.cat([a, b]).sub(clean).mean(0))  # [L, d]
        return clean, Hs

    @torch.no_grad()
    def clean_span(self):
        """Clean residuals on the assistant turn, captured with our own hooks."""
        if self._clean is None:
            rig, caps = self.rig, {}
            hds = [rig.layers[l].register_forward_hook(
                lambda m, a, o, l=l: caps.__setitem__(l, (o if torch.is_tensor(o) else o[0])[0, self.a_start : self.END + 1].float()))
                for l in range(rig.n_layers)]
            try:
                rig.hf.model(input_ids=self.X1)
            finally:
                for hd in hds:
                    hd.remove()
            self._clean = {l: caps[l] for l in range(rig.n_layers)}
        return self._clean

    def clamp(self, U, l):
        """Hold direction U at its clean value on the assistant turn at layer l."""
        u32 = unit(U.float())
        p0 = self.clean_span()[l] @ u32
        END, DET, a0 = self.END, self.DET, self.a_start

        def fn(x, first):
            pos = torch.arange(first, first + x.shape[1], device=x.device)
            idx = torch.where(pos <= END, pos, torch.full_like(pos, DET)) - a0
            cur = x.float() @ u32
            return (x.float() + (p0[idx][None, :] - cur)[..., None] * u32).to(x.dtype)

        return fn


def auroc(pos, neg):
    if not pos or not neg:
        return None
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def pearson(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sx = sum((a - mx) ** 2 for a in x) ** 0.5
    sy = sum((b - my) ** 2 for b in y) ** 0.5
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy + 1e-9)


def score_texts(texts, words):
    out = []
    for t, w in zip(texts, words):
        m = NAMED.search(t)
        out.append({
            "text": t,
            "says_word": bool(w and re.search(rf"\b{re.escape(w)}", t, re.I)),
            "claims": bool(CLAIM.search(t)) and not bool(DENY.search(t[:80])),
            "denies": bool(DENY.search(t[:120])),
            "named": m.group(1).lower() if (m and CLAIM.search(t)) else None,
        })
    return out
