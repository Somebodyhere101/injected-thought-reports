"""Summarise gib_<model>.json for all models -> gib_summary.json, and print the key comparisons.

    python gib_summary.py              # 1.7b, 4b, 8b -> gib_summary.json
    python gib_summary.py 14b          # named models only -> gib_summary_14b.json

Matched-output-change comparison: for each G strength, each random direction's dose-response
(metric against log KL) is interpolated at G's KL; the difference G - random is reported with an
approximate 95% interval combining the two conditions' bootstrap intervals (half-widths added in
quadrature). Outside a random direction's KL range no comparison is made.
"""

import json
import math
import os
import sys

HERE = os.environ.get("GIB_DIR") or os.path.dirname(os.path.abspath(__file__))
KEYS = sys.argv[1:] or ["1.7b", "4b", "8b"]
OUT_NAME = "gib_summary.json" if not sys.argv[1:] else f"gib_summary_{'_'.join(KEYS)}.json"
METRICS = ["real_word_rate", "repeat_rate", "dtr", "occ_any", "entropy", "compress", "alpha_run", "real_chars", "chars"]


def interp(points, x):
    """points: [(x, (mean, lo, hi))] -> (mean, half) at x by linear interpolation on log x."""
    pts = sorted((math.log(max(px, 1e-6)), v) for px, v in points)
    lx = math.log(max(x, 1e-6))
    for (x0, v0), (x1, v1) in zip(pts, pts[1:]):
        if x0 <= lx <= x1:
            t = (lx - x0) / (x1 - x0 + 1e-12)
            m = v0[0] + t * (v1[0] - v0[0])
            h = (1 - t) * (v0[2] - v0[1]) / 2 + t * (v1[2] - v1[1]) / 2
            return m, h
    return None


import random
import re
import statistics as st
import zlib

from wordfreq import zipf_frequency

rng = random.Random(0)


def text_metrics(t):
    """Measures computed from the text itself (the texts are kept local; only these numbers are published).
    compress: zlib-compressed length / raw length (lower = more regular).
    alpha_run: share of adjacent letter pairs (ignoring separators) that step forward by one in the
      alphabet, e.g. "a b c" (higher = more ordered).
    real_chars: share of letters that belong to common English words of 3+ letters (Zipf >= 2.5)."""
    raw = t.encode("utf-8")
    # zlib's fixed overhead dominates very short texts, so compressibility is only defined from 40 bytes
    comp = len(zlib.compress(raw, 9)) / len(raw) if len(raw) >= 40 else None
    letters = [c.lower() for c in t if c.isalpha() and c.isascii()]
    pairs = list(zip(letters, letters[1:]))
    alpha = sum(ord(b) - ord(a) == 1 for a, b in pairs) / len(pairs) if pairs else 0.0
    words = re.findall(r"[A-Za-z]+", t)
    tot = sum(len(w) for w in words)
    real = sum(len(w) for w in words if len(w) >= 3 and zipf_frequency(w.lower(), "en") >= 2.5)
    return {"compress": comp, "alpha_run": alpha, "real_chars": real / tot if tot else 0.0, "chars": len(t)}


def boot(xs, n=1000):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    m = sorted(st.mean(rng.choices(xs, k=len(xs))) for _ in range(n))
    return [st.mean(xs), m[int(0.025 * n)], m[int(0.975 * n)]]


TEXT_METRICS = ["compress", "alpha_run", "real_chars", "chars"]

out = {}
for k in KEYS:
    p = os.path.join(HERE, f"gib_{k}.json")
    if not os.path.exists(p):
        continue
    R = json.load(open(p, encoding="utf-8"))
    if "lexical_suppression" not in R:
        continue
    # add text-derived measures to every condition from the local texts
    TX = json.load(open(os.path.join(HERE, f"gib_texts_{k}.json"), encoding="utf-8"))

    def add_text(cond, key):
        if key in TX:
            ms = [text_metrics(t) for t in TX[key]]
            for m in TEXT_METRICS:
                b = boot([x[m] for x in ms])
                if b is not None:
                    cond[m] = b

    for n, v in R["conditions"].items():
        add_text(v, n)
    for task, rows in R["causal"].items():
        for n, v in rows.items():
            add_text(v, f"{task} | {n}")
    for task, d in R["lexical_suppression"].items():
        for n, v in d.items():
            add_text(v, f"suppression | {task} | {n}")
    C = R["conditions"]
    pick = lambda c: {m: c[m] for m in METRICS + ["occ_first_half", "occ_second_half"] if m in c}
    groups = {
        "A coherent": [v for n, v in C.items() if n.startswith("A ")],
        "B deliberate, lexical prompt": [v for n, v in C.items() if "lexical prompt" in n],
        "B deliberate, plain prompts": [v for n, v in C.items() if "plain prompt" in n],
        "C uncontrolled (forced)": [v for n, v in C.items() if n.startswith("C ")],
        "D repetition collapse": [v for n, v in C.items() if n.startswith("D ")],
    }
    cond = {}
    for g, vs in groups.items():
        cond[g] = {m: [sum(v[m][i] for v in vs) / len(vs) for i in range(3)] for m in METRICS if m in vs[0]}
        cond[g]["occ_word"] = {w: sum(v["occ_word"][w] for v in vs) / len(vs) for w in R["targets"]}
        cond[g]["top_jlens_tokens"] = vs[0]["top_jlens_tokens"][:10]
    ctrl = R["identical_token_control"]
    causal = {}
    for task, rows in R["causal"].items():
        base = rows["baseline"]
        G = {kk: v for kk, v in rows.items() if kk.startswith("G ")}
        rnd = {}
        for kk, v in rows.items():
            if kk.startswith("random"):
                name, a = kk.rsplit(" x", 1)
                rnd.setdefault(name, []).append((v["kl_on_baseline_text"][0], v))
        comp = {}
        for gk, gv in G.items():
            kl = gv["kl_on_baseline_text"][0]
            comp[gk] = {"kl": kl, "G": {m: gv[m] for m in METRICS if m in gv}, "vs_random": {}}
            for m in ("real_word_rate", "repeat_rate", "dtr", "occ_any", "compress", "alpha_run", "real_chars", "chars"):
                if m not in gv:
                    continue
                diffs = []
                for name, pts in rnd.items():
                    r = interp([(x, v[m]) for x, v in pts if m in v], kl)
                    if r is not None:
                        gm, gh = gv[m][0], (gv[m][2] - gv[m][1]) / 2
                        diffs.append((gm - r[0], math.sqrt(gh ** 2 + r[1] ** 2)))
                if diffs:
                    d = sum(x for x, _ in diffs) / len(diffs)
                    h = max(x for _, x in diffs)
                    comp[gk]["vs_random"][m] = [d, d - h, d + h]
        causal[task] = {"baseline": {m: base[m] for m in METRICS if m in base},
                        "G": comp,
                        "random_curves": {name: [{"kl": x, **{m: v[m] for m in ("real_word_rate", "repeat_rate", "dtr", "compress", "alpha_run", "real_chars", "chars") if m in v}}
                                                 for x, v in sorted(pts, key=lambda t: t[0])]
                                          for name, pts in rnd.items()}}
    out[k] = {"n": R["n_per_condition"], "layers": R["layers"], "deep_from": R["deep_from_layer"],
              "conditions": cond, "control": {
                  s: {inst: {m: v[inst][m] for m in ("occ_any", "dtr", "entropy")} for inst in v}
                  for s, v in ctrl["by_text_set"].items()},
              "G_cos_external_vs_own": ctrl["G_cos_external_vs_own"], "G_readout": ctrl["G_jlens_readout"],
              "causal": causal,
              "suppression": {t: {n: {m: v[m] for m in METRICS if m in v} for n, v in d.items()}
                              for t, d in R["lexical_suppression"].items()}}

    # follow-up tests (gib_ext.py): text measures and matched-KL comparisons
    ep = os.path.join(HERE, f"gib_ext_{k}.json")
    if os.path.exists(ep):
        E = json.load(open(ep, encoding="utf-8"))
        ET = json.load(open(os.path.join(HERE, f"gib_ext_texts_{k}.json"), encoding="utf-8"))
        ext = {}
        for task, rows in E["tests"].items():
            pts = {}
            for n, v in rows.items():
                key = f"{task} | {n}"
                if key not in ET:
                    continue
                ms = [text_metrics(t) for t in ET[key]]
                pts[n] = {"kl": v["kl"][0] if isinstance(v["kl"], list) else v["kl"],
                          **{m: b for m in TEXT_METRICS if (b := boot([x[m] for x in ms])) is not None}}
            rnd = [(v["kl"], v) for n, v in pts.items() if n.startswith("random")]
            comp = {}
            for n, v in pts.items():
                if not n.startswith("G "):
                    continue
                comp[n] = {"kl": v["kl"], "G": {m: v[m] for m in TEXT_METRICS}, "vs_random": {}}
                for m in TEXT_METRICS:
                    if m not in v:
                        continue
                    r = interp([(x, rv[m]) for x, rv in rnd if m in rv], v["kl"])
                    if r is not None:
                        h = math.sqrt(((v[m][2] - v[m][1]) / 2) ** 2 + r[1] ** 2)
                        comp[n]["vs_random"][m] = [v[m][0] - r[0], v[m][0] - r[0] - h, v[m][0] - r[0] + h]
            ext[task] = {"baseline": pts.get("baseline"), "G": comp,
                         "random_curve": sorted([{"kl": x, **{m: rv[m] for m in TEXT_METRICS}} for x, rv in rnd],
                                                key=lambda t: t["kl"])}
        out[k]["followup"] = ext

json.dump(out, open(os.path.join(HERE, OUT_NAME), "w", encoding="utf-8"), indent=1)
for k, o in out.items():
    print(f"\n===== Qwen3-{k.upper()} (n={o['n']} per condition)")
    cols = ("real_chars", "chars", "repeat_rate", "compress", "alpha_run", "dtr", "occ_any")
    print(f"{'condition':30s} " + " ".join(f"{c:>17s}" for c in cols))
    for g, c in o["conditions"].items():
        f = lambda m: f"{c[m][0]:.2f}[{c[m][1]:.2f},{c[m][2]:.2f}]" if m in c else "-"
        print(f"{g:30s} " + " ".join(f"{f(m):>17s}" for m in cols))
    print("identical tokens, gibberish vs coherent instruction:")
    for s, v in o["control"].items():
        gi, ci = v["gibberish instruction"], v["coherent instruction"]
        print(f"   {s:22s} occupancy {gi['occ_any'][0]:.3f} vs {ci['occ_any'][0]:.3f}; DTR {gi['dtr'][0]:.3f} vs {ci['dtr'][0]:.3f}")
    cs = o["G_cos_external_vs_own"]
    print(f"G from external vs own text: median cosine {sorted(cs)[len(cs) // 2]:.2f}")
    for task, c in o["causal"].items():
        print(f"{task}: baseline real {c['baseline']['real_word_rate'][0]:.2f} repeat {c['baseline']['repeat_rate'][0]:.2f}")
        for gk, g in c["G"].items():
            vr = g["vs_random"]
            s = "; ".join(f"{m} {g['G'][m][0]:.2f} (vs random at same KL {vr[m][0]:+.2f} [{vr[m][1]:+.2f},{vr[m][2]:+.2f}])"
                          for m in ("real_chars", "chars", "repeat_rate", "compress", "alpha_run") if m in vr)
            print(f"   {gk:6s} KL {g['kl']:6.2f}: {s or 'outside random range'}")
    for task, c in o.get("followup", {}).items():
        b = c["baseline"]
        print(f"follow-up {task}: baseline real chars {b['real_chars'][0]:.2f} order {b['alpha_run'][0]:.2f} compress {b['compress'][0]:.2f}")
        for gk, g in c["G"].items():
            vr = g["vs_random"]
            s = "; ".join(f"{m} {g['G'][m][0]:.2f} (vs random {vr[m][0]:+.2f} [{vr[m][1]:+.2f},{vr[m][2]:+.2f}])"
                          for m in ("real_chars", "alpha_run", "compress") if m in vr)
            print(f"   {gk:7s} KL {g['kl']:6.2f}: {s or 'outside random range'}")
    for t, d in o["suppression"].items():
        print(f"suppression {t}: " + "; ".join(f"{n} real chars {v['real_chars'][0]:.2f} [{v['real_chars'][1]:.2f},{v['real_chars'][2]:.2f}] repeat {v['repeat_rate'][0]:.2f}"
                                             for n, v in d.items()))
