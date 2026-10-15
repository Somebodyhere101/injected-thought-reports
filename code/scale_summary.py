"""Cross-model summary of battery_*.json and free_text_*.json -> scale.json + a printed table."""

import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
KEYS = ["1.7b", "4b", "8b", "14b"]


def load(name):
    p = os.path.join(HERE, name)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def auroc(pos, neg):
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def effort(k):
    """Detection AUROC (word vs no injection) and clean yes-score, thinking off vs 256 tokens."""
    path = os.path.join(HERE, "results_b.jsonl" if k == "1.7b" else f"results_b_{k}.jsonl")
    if not os.path.exists(path):
        return None
    word = "jl_band" if k == "1.7b" else "word"
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    out = {}
    for abl in ("none", "jspace", "matched"):
        for b in (0, 256):
            w = [r["det"] for r in rows if r["inj"] == word and r["budget"] == b and r["ablation"] == abl]
            n = [r["det"] for r in rows if r["inj"] == "none" and r["budget"] == b and r["ablation"] == abl]
            if w and n:
                out[f"eff_auroc_{abl}_{b}"] = auroc(w, n)
                if abl == "none":
                    out[f"eff_clean_{b}"] = st.mean(n)
                    out[f"eff_top10_{b}"] = sum(r["rank"] < 10 for r in rows
                                                if r["inj"] == word and r["budget"] == b and r["ablation"] == abl)
    return out


out = {}
for k in KEYS:
    b, f = load(f"battery_{k}.json"), load(f"free_text_{k}.json")
    if not b or "J_yes_bias_split" not in b:
        continue
    A, E, G, H, J = b["A_detection"], b["E_grid"], b["G_phantom"], b["H_yes_bias"], b["J_yes_bias_split"]
    e = {(r["cond"], r["edit"]): r for r in E}
    clean = A["clean_det"]
    row = {
        "n_words": b["meta"]["n_words"],
        # detection and naming, original protocol
        "detect_word": A["word band 0.1"]["det"] - clean,
        "detect_word_frac": A["word band 0.1"]["frac_above_clean"],
        "detect_random": A["random band 0.1"]["det"] - clean,
        "named_top10_best": max(v["named_top10"] for kk, v in A.items() if kk.startswith("word")),
        # the direction
        "H_agreement": b["B_H"]["median_agreement"],
        "H_r_det": st.mean(v["r"] for v in b["C_validation"].values()),
        "H_cos_yesno_max": max(abs(v["cos_yes_no"]) for v in b["C_validation"].values()),
        # content in workspace
        "W_strength": b["D_workspace_strength"]["chosen"],
        "W_inJ": e[("word in workspace", "untouched")]["in_J_top1"],
        "W_named": e[("word in workspace", "untouched")]["named_top1"],
        "W_det_change": e[("word in workspace", "untouched")]["det"] - e[("nothing", "untouched")]["det"],
        # necessity
        "word_frac_untouched": e[("word on question", "untouched")]["frac_above_clean"],
        "word_frac_H_held": e[("word on question", "H held")]["frac_above_clean"],
        "word_frac_random_held": e[("word on question", "random held")]["frac_above_clean"],
        "word_frac_Hperp_held": J["held"]["word on question"]["H minus yes-bias held"]["frac_above"],
        "word_frac_Y_held": J["held"]["word on question"]["yes-bias Y held"]["frac_above"],
        # sufficiency and specificity
        "H_added_det": [x for x in b["F_H_added"] if x["alpha"] == 1.0][0]["det"] - clean,
        "cos_H_Y": st.median(J["cos_H_Y"].values()),
    }
    for ctxn in ("injection context", "plain question"):
        intro = H[f"{ctxn} | introspection"]
        falses = [v for kk, v in H.items() if kk.startswith(ctxn) and "| no:" in kk and "body" not in kk]
        tag = "inj" if ctxn.startswith("injection") else "plain"
        row[f"H_delta_intro_{tag}"] = intro["H added x1"] - intro["clean"]
        row[f"H_delta_false_{tag}"] = st.mean(v["H added x1"] - v["clean"] for v in falses)
    adds = J["added"]
    row["Hperp_delta_intro"] = adds["introspection (injection context)"]["H minus yes-bias"]
    row["Hperp_delta_false"] = st.mean(v["H minus yes-bias"] for kk, v in adds.items() if kk.startswith("no:"))
    row["phantom_jspace"] = G["jspace"]["no_injection_det"] - G["none"]["no_injection_det"]
    row["phantom_matched"] = G["matched"]["no_injection_det"] - G["none"]["no_injection_det"]
    row["detect_under_jspace_removal"] = G["jspace"]["word_frac_above"]
    if f:
        T1, T2 = f["T1_injection_open_question"], f["T2_neutral_question"]
        T3 = f["T3 Is the sun colder than ice"]
        row.update({
            "ft_word_affirms": T1["word on question"]["summary"]["affirms"],
            "ft_word_names": T1["word on question"]["summary"]["names_word"],
            "ft_clean_affirms": T1["clean"]["summary"]["affirms"],
            "ft_H_affirms": T1["H added"]["summary"]["affirms"],
            "ft_neutral_clean_claims": T2["clean"]["summary"]["claims"],
            "ft_neutral_W_claims": T2["word in workspace"]["summary"]["claims"],
            "ft_neutral_W_says_word": T2["word in workspace"]["summary"]["says_word"],
            "ft_neutral_H_claims": T2["H (this prompt) added"]["summary"]["claims"],
            "ft_sun_clean_yes": T3["clean"]["summary"]["yes"],
            "ft_sun_H_yes": T3["H added"]["summary"]["yes"],
        })
    wc = load(f"w_control_{k}.json")
    if wc:
        for sk in ("0.02", "0.05"):
            tag = sk.replace("0.", "")
            row[f"ws_word_{tag}"] = wc[sk]["word"]["det_change"]
            row[f"ws_random_{tag}"] = wc[sk]["random"]["det_change"]
            row[f"ws_word_Hheld_{tag}"] = wc[sk]["word"]["det_change_H_held"]
    eff = effort(k)
    if eff:
        row.update(eff)
    out[k] = row

json.dump(out, open(os.path.join(HERE, "scale.json"), "w", encoding="utf-8"), indent=1)
keys = list(next(iter(out.values())).keys())
print(f"{'metric':28s}" + "".join(f"{k:>10s}" for k in out))
for m in keys:
    vals = []
    for k in out:
        v = out[k].get(m)
        vals.append(f"{v:10.2f}" if isinstance(v, float) else f"{str(v):>10s}")
    print(f"{m:28s}" + "".join(vals))
