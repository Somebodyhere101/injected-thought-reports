"""Shared rig for the effort x J-space experiments (Qwen3-1.7B by default).

Injection follows Anthropic's verbal-introspection protocol (jacobian-lens repo,
data/experiments/README.md): a concept's J-lens direction, unit-normalised and
scaled by the layer's mean residual norm times a strength, is added at every
band layer on every token of the user's question turn.

J-space ablation (approximation of the paper's sparse decomposition): at each
ablated (layer, position), take the top-k tokens of the J-lens readout and
project the residual off the span of their J-lens vectors. The control removes
the span of k fixed random tokens' J-lens vectors instead.
"""

from __future__ import annotations

import json
import os

import jlens
import torch
import transformers

HERE = os.path.dirname(os.path.abspath(__file__))
INTRO_JSON = os.path.join(
    HERE, "..", "third_party", "jacobian-lens", "data", "experiments",
    "verbal-introspection.json",
)
MODEL = "Qwen/Qwen3-1.7B"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_FILE = "qwen3-1.7b/jlens/Salesforce-wikitext/Qwen3-1.7B_jacobian_lens.pt"
PREFILL = 'Yes, I detect an injected thought. The thought is about "'


PIN_BUDGET = 6 * 2**30
_PINNED = 0


def _stream_from_cpu(layer: torch.nn.Module, device: str) -> None:
    """Hold a block's weights in pinned CPU memory and copy them to the GPU only while
    the block runs. Registered before the Controller's hooks, so it runs first."""
    def pin(t):
        # Windows caps page-locked memory and a failed pin poisons the CUDA context, so
        # pin only within a fixed budget and leave the rest in ordinary memory.
        global _PINNED
        nbytes = t.numel() * t.element_size()
        if _PINNED + nbytes > PIN_BUDGET:
            return t
        _PINNED += nbytes
        return t.pin_memory()

    cpu = {name: pin(p.data) for name, p in layer.named_parameters()}
    params = dict(layer.named_parameters())
    for name, p in params.items():
        p.data = cpu[name]

    def pre(module, args):
        for name, p in params.items():
            p.data = cpu[name].to(device, non_blocking=True)

    def post(module, args, output):
        for name, p in params.items():
            p.data = cpu[name]

    layer.register_forward_pre_hook(pre)
    layer.register_forward_hook(post)


def model_spec(models: dict, key: str) -> dict:
    """A script's model table plus Qwen3-14B. GPU_LAYERS overrides how many blocks stay on the
    GPU ("all" keeps the whole model there), for machines with more memory than the laptop."""
    extra = {"14b": dict(model="Qwen/Qwen3-14B",
                         lens_file="qwen3-14b/jlens/Salesforce-wikitext/Qwen3-14B_jacobian_lens.pt",
                         gpu_layers=34)}
    spec = dict({**extra, **models}[key])
    g = os.environ.get("GPU_LAYERS")
    if g:
        spec["gpu_layers"] = None if g == "all" else int(g)
    return spec


def default_band(n_layers: int) -> list[int]:
    """The workspace band as a fraction of depth (layers 10-24 of Qwen3-1.7B's 28)."""
    return list(range(round(0.36 * n_layers), round(0.89 * n_layers)))


class Rig:
    def __init__(self, band: list[int] | None = None, device: str = "cuda", seed: int = 0,
                 model: str = MODEL, lens_file: str = LENS_FILE, gpu_layers: int | None = None):
        """gpu_layers: keep only the first N blocks resident on the GPU and stream the rest
        from CPU memory each forward pass (for models larger than the GPU)."""
        self.device = device
        self.model_name = model
        self.tok = transformers.AutoTokenizer.from_pretrained(model)
        if gpu_layers is None:
            self.hf = transformers.AutoModelForCausalLM.from_pretrained(
                model, dtype=torch.bfloat16
            ).to(device).eval()
        else:
            # Keep embeddings, final norm, LM head and the first `gpu_layers` blocks on the
            # GPU; stream the remaining blocks' weights from pinned CPU memory per forward.
            self.hf = transformers.AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16).eval()
            m = self.hf.model
            for mod in (m.embed_tokens, m.norm, m.rotary_emb, self.hf.lm_head):
                mod.to(device)
            for i, layer in enumerate(m.layers):
                if i < gpu_layers:
                    layer.to(device)
                else:
                    _stream_from_cpu(layer, device)
        for p in self.hf.parameters():
            p.requires_grad_(False)
        self.layers = self.hf.model.layers
        self.n_layers = len(self.layers)
        self.d = self.hf.config.hidden_size
        self.norm = self.hf.model.norm
        self.W_U = self.hf.lm_head.weight  # [V, d]
        self.gamma = self.norm.weight.float()
        self.band = band if band is not None else default_band(self.n_layers)
        lens = jlens.JacobianLens.from_pretrained(LENS_REPO, filename=lens_file)
        if gpu_layers is None:
            self.J = {l: J.to(device) for l, J in lens.jacobians.items()}
        else:
            # Large lenses: keep band layers on the GPU in bf16, the rest on the CPU.
            self.J = {l: J.to(device, torch.bfloat16) for l, J in lens.jacobians.items() if l in self.band}
            self.J_cpu = {l: J.float() for l, J in lens.jacobians.items()}
        del lens

        g = torch.Generator().manual_seed(seed)
        V = self.W_U.shape[0]
        self.rand_idx = {
            l: torch.randint(0, V, (64,), generator=g).to(device) for l in range(self.n_layers)
        }
        self._rand_Q: dict[tuple[int, int], torch.Tensor] = {}

        intro = json.load(open(INTRO_JSON, encoding="utf-8"))
        self.messages = [m for m in intro["intro_prompt"] if m["content"] != ""]
        self.concepts = []
        for c in intro["concepts"]:
            ids = self.tok.encode(c["surface"], add_special_tokens=False)
            if len(ids) == 1:
                self.concepts.append((c["surface"], ids[0]))
        self.yes_id = self.tok.encode("Yes", add_special_tokens=False)[0]
        self.no_id = self.tok.encode("No", add_special_tokens=False)[0]
        self.think_end_id = self.tok.convert_tokens_to_ids("</think>")
        self.im_end_id = self.tok.convert_tokens_to_ids("<|im_end|>")

        self.ctl = Controller(self)
        self.mean_norm = self._mean_norms()

    # ---------- lens helpers ----------
    def lens_logits(self, h: torch.Tensor, layer: int) -> torch.Tensor:
        J = self.J[layer]
        z = h.to(J.dtype) @ J.T
        return self.hf.lm_head(self.norm(z.to(torch.bfloat16)))

    def jvec(self, token_ids: torch.Tensor, layer: int) -> torch.Tensor:
        """J-lens vectors J_l^T (gamma * W_U[t]) for token ids [..., ] -> [..., d]."""
        u = self.W_U[token_ids].float() * self.gamma
        return u @ self.J[layer].float()

    def ablate_rows(self, x: torch.Tensor, layer: int, mode: str, k: int) -> torch.Tensor:
        shape, dtype = x.shape, x.dtype
        flat = x.reshape(-1, shape[-1])
        out = torch.empty_like(flat)
        for s in range(0, flat.shape[0], 256):
            c = flat[s : s + 256].float()
            if mode in ("jspace", "matched"):
                idx = self.lens_logits(c, layer).topk(k, dim=-1).indices  # [n, k]
                Vv = self.jvec(idx, layer)  # [n, k, d]
                Q, _ = torch.linalg.qr(Vv.transpose(1, 2))  # [n, d, k]
                proj = (Q @ (Q.transpose(1, 2) @ c[..., None])).squeeze(-1)
                if mode == "matched":
                    # Same-size perturbation as the J-space removal, random direction.
                    r = torch.randn_like(c)
                    proj = r * (proj.norm(dim=-1, keepdim=True) / r.norm(dim=-1, keepdim=True))
            elif mode == "random":
                key = (layer, k)
                if key not in self._rand_Q:
                    Vv = self.jvec(self.rand_idx[layer][:k], layer)  # [k, d]
                    self._rand_Q[key] = torch.linalg.qr(Vv.T)[0]  # [d, k]
                Q = self._rand_Q[key]
                proj = (c @ Q) @ Q.T
            else:
                raise ValueError(mode)
            out[s : s + 256] = (c - proj).to(dtype)
        return out.reshape(shape)

    def concept_ranks(self, h: torch.Tensor, layer: int, tok_ids: torch.Tensor) -> torch.Tensor:
        """Lens rank (0 = top) of tok_ids[b] at every row of h[b] -> [B, T]."""
        B, T, _ = h.shape
        out = torch.empty(B, T, dtype=torch.int32)
        for b in range(B):
            for s in range(0, T, 256):
                lg = self.lens_logits(h[b, s : s + 256], layer).float()
                tgt = lg[:, tok_ids[b]]
                out[b, s : s + 256] = (lg > tgt[:, None]).sum(-1).int().cpu()
        return out

    # ---------- prompt helpers ----------
    def prompt_text(self, think: bool, messages: list[dict] | None = None) -> str:
        return self.tok.apply_chat_template(
            messages or self.messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=think,
        )

    def spans(self, text: str, messages: list[dict] | None = None) -> tuple[int, int, int]:
        """(q_lo, q_hi, a_start) token positions: the user's question turn content and
        the start of the assistant turn."""
        enc = self.tok(text, return_offsets_mapping=True, add_special_tokens=False)
        q = (messages or self.messages)[-1]["content"]
        c_lo = text.rfind(q)
        c_hi = c_lo + len(q)
        c_a = text.rfind("<|im_start|>assistant")
        offs = enc["offset_mapping"]
        qpos = [i for i, (s, e) in enumerate(offs) if s < c_hi and e > c_lo]
        apos = [i for i, (s, e) in enumerate(offs) if s >= c_a]
        return qpos[0], qpos[-1] + 1, apos[0]

    @torch.no_grad()
    def _mean_norms(self) -> dict[int, float]:
        text = self.prompt_text(False)
        ids = self.tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        out = self.hf(ids, output_hidden_states=True)
        # hidden_states[l + 1] is the output of block l. Median rather than mean: a few
        # massive-activation positions dominate the mean at mid/late layers.
        return {
            l: out.hidden_states[l + 1][0, 1:].float().norm(dim=-1).median().item()
            for l in range(self.n_layers)
        }

    def injection_vectors(self, tok_ids: list[int], strength: float, kind: str = "jlens",
                          seed: int = 0, layers: list[int] | None = None) -> dict[int, torch.Tensor]:
        """{layer: [B, d]} vectors to add on the question turn."""
        ids = torch.tensor(tok_ids, device=self.device)
        g = torch.Generator(device="cpu").manual_seed(seed)
        rnd = torch.randn(len(tok_ids), self.d, generator=g).to(self.device)
        vecs = {}
        for l in layers or self.band:
            v = self.jvec(ids, l) if kind == "jlens" else rnd
            v = v / v.norm(dim=-1, keepdim=True)
            vecs[l] = (v * strength * self.mean_norm[l]).to(torch.bfloat16)
        return vecs


class Controller:
    """Forward hooks that inject on the question turn, ablate on the assistant turn,
    and optionally record the concept's lens rank (pre-ablation) at band layers."""

    def __init__(self, rig: Rig):
        self.rig = rig
        self.start = self.end = 0
        self.inject: dict | None = None  # {"vecs": {l: [B,d]}, "span": (lo, hi)}
        self.injects: list[dict] = []  # further injections, same shape as `inject`
        self.ablate: dict | None = None  # {"layers": set, "a_start": int, "mode": str, "k": int}
        self.edits: dict | None = None  # {"a_start": int, "fns": {l: fn(x [B,M,d], first_pos) -> x}}
        self.trace: dict | None = None  # {"tok_ids": tensor[B], "ranks": {l: [B,T]}}
        self.record: dict | None = None  # {"pos": int, "out": {l: [B,d]}} after all edits
        self.patches: list[dict] = []  # {"layer": l, "lo": pos, "values": [M,d]}: overwrite positions
        rig.hf.model.embed_tokens.register_forward_pre_hook(self._pre_embed)
        for l in range(rig.n_layers):
            rig.layers[l].register_forward_hook(self._make_hook(l))

    def reset(self):
        self.inject = self.ablate = self.trace = self.edits = self.record = None
        self.injects = []
        self.patches = []

    def _pre_embed(self, module, args):
        T = args[0].shape[1]
        self.start = 0 if T > 1 else self.end
        self.end = self.start + T

    def _make_hook(self, l: int):
        def hook(module, args, output):
            h = output if torch.is_tensor(output) else output[0]
            # Positions are tracked on the CPU so the hook never forces a GPU sync.
            new = None
            for inj in ([self.inject] if self.inject is not None else []) + self.injects:
                if l not in inj["vecs"]:
                    continue
                lo, hi = inj["span"]
                s, e = max(lo, self.start), min(hi, self.end)
                if s < e:
                    if new is None:
                        new = h.clone()
                    new[:, s - self.start : e - self.start] += inj["vecs"][l][:, None, :]
            cur = h if new is None else new
            if self.trace is not None and l in self.rig.band and cur.shape[1] > 1:
                frm = max(self.trace.get("from", 0) - self.start, 0)  # optional: only from a position on
                self.trace["ranks"][l] = self.rig.concept_ranks(cur[:, frm:], l, self.trace["tok_ids"])
            if self.ablate is not None and l in self.ablate["layers"]:
                s = max(self.ablate["a_start"], self.start)
                if s < self.end:
                    if new is None:
                        new = h.clone()
                    i = s - self.start
                    new[:, i:] = self.rig.ablate_rows(
                        new[:, i:], l, self.ablate["mode"], self.ablate["k"]
                    )
            if self.edits is not None and l in self.edits["fns"]:
                s = max(self.edits["a_start"], self.start)
                if s < self.end:
                    if new is None:
                        new = h.clone()
                    i = s - self.start
                    new[:, i:] = self.edits["fns"][l](new[:, i:], s)
            for p in self.patches:
                if p["layer"] != l:
                    continue
                lo, hi = p["lo"], p["lo"] + p["values"].shape[0]
                s, e = max(lo, self.start), min(hi, self.end)
                if s < e:
                    if new is None:
                        new = h.clone()
                    new[:, s - self.start : e - self.start] = p["values"][s - lo : e - lo].to(new.dtype)
            if self.record is not None and self.start <= self.record["pos"] < self.end:
                self.record["out"][l] = (h if new is None else new)[:, self.record["pos"] - self.start].float()
            if new is None:
                return None
            return new if torch.is_tensor(output) else (new, *output[1:])

        return hook
