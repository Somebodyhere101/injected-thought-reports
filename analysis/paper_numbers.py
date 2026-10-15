"""Every number in the paper, with its interval, recomputed from the result files.

    python analysis/paper_numbers.py [folder ...]

Reads results/ (small summaries, in this repository) and data/ (per-trial files, from the
Zenodo archive) and writes results/paper_numbers.json; prints a readable report. Folders given
on the command line are searched first, e.g. code/ after rerunning the experiments there.
Standard library only.

Intervals:
  - means over trials (words or random directions): 95% bootstrap percentile, 2,000 resamples;
  - AUROC: both groups resampled independently, 2,000 resamples;
  - counts k of n (shares of trials, free-text counts, decoding accuracy): Wilson score interval;
  - intervals stored in the summaries by the experiment scripts are passed through unchanged
    (stages.py and w_control2.py: 2,000 bootstrap resamples; gib_summary.py: 1,000).
Numbers from single deterministic prompts (the yes-bias battery, H added alone) have no interval;
the report gives the spread across questions instead. Battery means whose per-word values were not
saved are marked "point".
"""

import json
import math
import os
import random
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DIRS = sys.argv[1:] + [os.path.join(HERE, "..", "results"), os.path.join(HERE, "..", "data")]
OUT_DIR = os.path.join(HERE, "..", "results")
MODELS = ["1.7b", "4b", "8b", "14b"]
B = 2000
rng = random.Random(20261015)


def load(name, jsonl=False):
    for d in DIRS:
        p = os.path.join(d, name)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                return [json.loads(l) for l in f] if jsonl else json.load(f)
    return None


def wilson(k, n, z=1.959964):
    if n == 0:
        return None
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [p, max(0.0, c - h), min(1.0, c + h)]


def boot_mean(xs):
    m = sorted(st.mean(rng.choices(xs, k=len(xs))) for _ in range(B))
    return [st.mean(xs), m[int(0.025 * B)], m[int(0.975 * B)]]


def boot_diff(xs, ys):
    m = sorted(st.mean(rng.choices(xs, k=len(xs))) - st.mean(rng.choices(ys, k=len(ys))) for _ in range(B))
    return [st.mean(xs) - st.mean(ys), m[int(0.025 * B)], m[int(0.975 * B)]]


def auroc(pos, neg):
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def boot_auroc(pos, neg):
    m = sorted(auroc(rng.choices(pos, k=len(pos)), rng.choices(neg, k=len(neg))) for _ in range(B))
    return [auroc(pos, neg), m[int(0.025 * B)], m[int(0.975 * B)]]


def unique_trials(rows, key=lambda r: (r["det"], r.get("n_think"))):
    """Sampled cells were run in batches of 20 with one seed per batch. Where a cell has no
    injection (or a fixed set of random directions), later batches repeat the first exactly
    (same yes-score and thinking length); keep one period so that intervals are not computed on
    copies. Deterministic cells (thinking off, nothing injected) reduce to their distinct values."""
    ks = [key(r) for r in rows]
    for p in range(1, len(ks)):
        if len(ks) % p == 0 and all(ks[i] == ks[i % p] for i in range(len(ks))):
            return rows[:p]
    return rows


def frac_above(trials):
    """Share of trials whose change from the batch-matched baseline is positive."""
    return wilson(sum(t > 0 for t in trials), len(trials))


OUT = {"models": MODELS}

# ---------------------------------------------------------------- detection, naming, H
det = {}
for m in MODELS:
    S, Bt = load(f"stages_{m}.json"), load(f"battery_{m}.json")
    A, s4, s2 = Bt["A_detection"], S["stage4_report"], S["stage2_transfer"]
    clean = A["clean_det"]
    row = {
        "layers": S["layers"], "band": Bt["meta"]["band"], "single_layer": S["single_layer"],
        "edit_layers": S["edit_layers"], "n_words_battery": Bt["meta"]["n_words"], "n_words": len(S["words"]),
        "clean_yes_score": S["clean_det"],
        # original protocol (word across the band, strength 0.1, question tokens), 40 words
        "word_change": s4["untouched"]["change"],
        "word_above_clean": frac_above(s4["untouched"]["trials"]),
        # battery means (point): band injection, word vs random direction of the same size
        "point_word_band_change": A["word band 0.1"]["det"] - clean,
        "point_random_band_change": A["random band 0.1"]["det"] - clean,
        "random_band_above_clean": A["random band 0.1"]["frac_above_clean"],
        # single layer, strength 1.0, 40 words vs 40 random directions, with intervals
        "word_single_change": s2["full_word"],
        "random_single_change": s2["full_random"],
        "named_top10_best": max(v["named_top10"] for k, v in A.items() if k.startswith("word")),
        "named_top1_best": max(v["named_top1"] for k, v in A.items() if k.startswith("word")),
        # H
        "H_agreement_median_cos": Bt["B_H"]["median_agreement"],
        "H_r_det_mean_over_layers": st.mean(v["r"] for v in Bt["C_validation"].values()),
        "H_cos_yesno_max_abs": max(abs(v["cos_yes_no"]) for v in Bt["C_validation"].values()),
        "cos_H_Y_median": st.median(Bt["J_yes_bias_split"]["cos_H_Y"].values()),
        "H_perp_norm_share_median": st.median(Bt["J_yes_bias_split"]["H_perp_norm_share"].values()),
    }
    # holds, original protocol, with intervals (stages.py)
    holds = {}
    for k, v in s4.items():
        holds[k] = {"change": v["change"], "above_clean": frac_above(v["trials"])}
    row["holds"] = holds
    # holds of Y and H-minus-Y (battery, shares of n_words, Wilson)
    n = Bt["meta"]["n_words"]
    J = Bt["J_yes_bias_split"]["held"]["word on question"]
    for hn, key in (("H held", "battery_H_held"), ("H minus yes-bias held", "battery_Hperp_held"), ("yes-bias Y held", "battery_Y_held")):
        row[key] = wilson(round(J[hn]["frac_above"] * n), n)
    E = {(r["cond"], r["edit"]): r for r in Bt["E_grid"]}
    row["battery_untouched_above"] = wilson(round(E[("word on question", "untouched")]["frac_above_clean"] * n), n)
    row["battery_random_held_above"] = wilson(round(E[("word on question", "random held")]["frac_above_clean"] * n), n)
    det[m] = row
OUT["detection"] = det

# ---------------------------------------------------------------- yes-bias battery (deterministic prompts)
yb = {}
for m in MODELS:
    Bt = load(f"battery_{m}.json")
    H, J = Bt["H_yes_bias"], Bt["J_yes_bias_split"]["added"]
    r = {}
    for ctxn, tag in (("plain question", "plain"), ("injection context", "inj")):
        intro = H[f"{ctxn} | introspection"]
        falses = {kk.split("| no: ")[1]: v["H added x1"] - v["clean"] for kk, v in H.items()
                  if kk.startswith(ctxn) and "| no:" in kk and "body" not in kk}
        r[tag] = {"intro_clean": intro["clean"], "intro_H": intro["H added x1"] - intro["clean"],
                  "intro_H_half": intro["H added x0.5"] - intro["clean"],
                  "false_H": falses, "false_H_mean": st.mean(falses.values()),
                  "false_H_min": min(falses.values()), "false_H_max": max(falses.values()),
                  "true_H": H[f"{ctxn} | yes: Paris is capital of France"]["H added x1"] - H[f"{ctxn} | yes: Paris is capital of France"]["clean"],
                  "body_H": H[f"{ctxn} | no: you have a body"]["H added x1"] - H[f"{ctxn} | no: you have a body"]["clean"],
                  "intro_word_on_q": intro["word on question"] - intro["clean"],
                  "false_word_on_q_mean": st.mean(v["word on question"] - v["clean"] for kk, v in H.items()
                                                  if kk.startswith(ctxn) and "| no:" in kk and "body" not in kk),
                  "intro_random_on_q": intro["disturbance on question"] - intro["clean"],
                  "false_random_on_q_mean": st.mean(v["disturbance on question"] - v["clean"] for kk, v in H.items()
                                                    if kk.startswith(ctxn) and "| no:" in kk and "body" not in kk)}
    # J: H, H minus Y, Y and a same-size random direction added (change from clean)
    r["J_added"] = {
        "intro_inj": {k: v for k, v in J["introspection (injection context)"].items()},
        "false_plain_mean": {vn: st.mean(v[vn] for k, v in J.items() if k.startswith("no:") and k.endswith("(plain)"))
                             for vn in ("H", "H minus yes-bias", "yes-bias Y", "random, same size")},
        "false_inj_mean": {vn: st.mean(v[vn] for k, v in J.items() if k.startswith("no:") and k.endswith("(injection context)"))
                           for vn in ("H", "H minus yes-bias", "yes-bias Y", "random, same size")},
    }
    r["H_added_alone_change"] = [x["det"] - Bt["A_detection"]["clean_det"] for x in Bt["F_H_added"] if x["alpha"] == 1.0][0]
    yb[m] = r
OUT["yes_bias"] = yb

# ---------------------------------------------------------------- free text (counts, Wilson)
ft = {}
checks = load("free_text_checks.json")
for m in MODELS:
    F = load(f"free_text_{m}.json")
    T1, T2 = F["T1_injection_open_question"], F["T2_neutral_question"]
    T3s, T3f = F["T3 Is the sun colder than ice"], F["T3 Can fish ride bicycles"]
    c = lambda blk, cond, k: wilson(blk[cond]["summary"][k], blk[cond]["summary"]["n"])
    ft[m] = {
        "T1_clean_affirms": c(T1, "clean", "affirms"), "T1_word_affirms": c(T1, "word on question", "affirms"),
        "T1_word_denies": c(T1, "word on question", "denies"), "T1_word_names": c(T1, "word on question", "names_word"),
        "T1_random_affirms": c(T1, "random on question", "affirms"),
        "T1_ws_affirms": c(T1, "word in workspace", "affirms"), "T1_ws_denies": c(T1, "word in workspace", "denies"),
        "T1_ws_names": c(T1, "word in workspace", "names_word"),
        "T1_H_affirms": c(T1, "H added", "affirms"), "T1_H_denies": c(T1, "H added", "denies"),
        "T1_Y_affirms": c(T1, "Y added", "affirms"),
        "T2_clean_claims": c(T2, "clean", "claims"), "T2_ws_claims": c(T2, "word in workspace", "claims"),
        "T2_ws_says_word": c(T2, "word in workspace", "says_word"), "T2_ws_denies": c(T2, "word in workspace", "denies"),
        "T2_H_claims": c(T2, "H (this prompt) added", "claims"),
        "T3_sun_clean_yes_first": c(T3s, "clean", "yes"), "T3_sun_H_yes_first": c(T3s, "H added", "yes"),
        "T3_fish_clean_yes_first": c(T3f, "clean", "yes"), "T3_fish_H_yes_first": c(T3f, "H added", "yes"),
    }
    if checks and m in checks["models"]:
        for k, v in checks["models"][m].items():
            ft[m][k] = wilson(v["k"], v["n"])
OUT["free_text"] = ft

# ---------------------------------------------------------------- stages 2 and 3
stg = {}
for m in MODELS:
    S = load(f"stages_{m}.json")
    s2, s3 = S["stage2_transfer"], S["stage3_identity"]["by_layer"]
    full = s2["full_word"][0]
    ls = sorted(map(int, s2["cut"]))
    esc = next((l for l in ls if s2["cut"][str(l)]["word"][0] >= 0.9 * full), None)
    dec = {}
    for cond, d in s3.items():
        ks = sorted(map(int, d))
        first = next((l for l in ks if d[str(l)]["acc"] >= 0.9), None)
        last = ks[-1]
        n = 5 * len(S["words"])  # five held-out strengths x 40 words
        dec[cond] = {"first_layer_ge_0.9": first, "last_layer": last,
                     "acc_last": wilson(round(d[str(last)]["acc"] * n), n),
                     "acc_without_H_last": wilson(round(d[str(last)]["acc_without_H"] * n), n),
                     "acc_at_injection_layer": d[str(ks[0])]["acc"],
                     "min_acc_after_first": min(d[str(l)]["acc"] for l in ks if first is not None and l >= first) if first is not None else None}
    stg[m] = {"escape_layer_90": esc, "L": S["layers"], "single": S["single_layer"], "full_word": s2["full_word"],
              "full_random": s2["full_random"],
              "cut_word": {l: s2["cut"][l]["word"] for l in s2["cut"]},
              "cut_random": {l: s2["cut"][l]["random"] for l in s2["cut"]},
              "decode": dec,
              "decode_curve": {cond: {l: d[l]["acc"] for l in sorted(d, key=int)} for cond, d in s3.items()}}
OUT["stages"] = stg

# ---------------------------------------------------------------- word in the workspace, matched controls
wc = {}
for m in MODELS:
    W, Bt = load(f"w_control2_{m}.json"), load(f"battery_{m}.json")
    Sm, cells = W["summary"], W["trials"]
    r = {"strengths": {}}
    for s in map(str, W["word_strengths"]):
        w = cells["word"][s]
        e = {"word_change": boot_mean([t["det_change"] for t in w]),
             "word_kl_mean": st.mean(t["kl"] for t in w), "word_dh_mean": st.mean(t["dh"] for t in w)}
        if s in cells["random"]:
            rr = cells["random"][s]
            e.update({"random_change": boot_mean([t["det_change"] for t in rr]),
                      "random_kl_mean": st.mean(t["kl"] for t in rr), "random_dh_mean": st.mean(t["dh"] for t in rr)})
        for crit in ("matched_input", "matched_kl", "matched_dh"):
            v = Sm[crit].get(s)
            if v and v.get("diff") is not None:
                e[crit] = [v["diff"], v["ci"][0], v["ci"][1]]
            else:
                e[crit] = None
        r["strengths"][s] = e
    r["random_curve"] = {s: {"kl_mean": st.mean(t["kl"] for t in v), "change": boot_mean([t["det_change"] for t in v]),
                             "dh_mean": st.mean(t["dh"] for t in v)} for s, v in cells["random"].items()}
    E = {(x["cond"], x["edit"]): x for x in Bt["E_grid"]}
    n = Bt["meta"]["n_words"]
    ws, nb = E[("word in workspace", "untouched")], E[("nothing", "untouched")]
    wsH = E[("word in workspace", "H held")]
    r["battery_strength"] = Bt["D_workspace_strength"]["chosen"]
    r["in_J_top1"] = wilson(ws["in_J_top1"], n)
    r["named_top1_forced"] = wilson(ws["named_top1"], n)
    r["point_change_untouched"] = ws["det"] - nb["det"]
    r["point_change_H_held"] = wsH["det"] - wsH["edit_baseline"]
    wcold = load(f"w_control_{m}.json")
    if wcold:
        r["w_control_H_held"] = {s: wcold[s]["word"]["det_change_H_held"] for s in ("0.02", "0.05") if s in wcold}
    wc[m] = r
OUT["workspace"] = wc

# ---------------------------------------------------------------- thinking and J-space removal
th = {}
den = load("thinking_denials.json")
for m in MODELS:
    rows = load("results_b.jsonl" if m == "1.7b" else f"results_b_{m}.jsonl", jsonl=True)
    word = "jl_band" if m == "1.7b" else "word"
    cell = lambda inj, b, a: unique_trials([r for r in rows if r["inj"] == inj and r["budget"] == b and r["ablation"] == a])
    r = {}
    for abl in ("none", "jspace", "matched"):
        for b in (0, 256):
            w, n = cell(word, b, abl), cell("none", b, abl)
            if not w or not n:
                continue
            wd, nd = [x["det"] for x in w], [x["det"] for x in n]
            r[f"{abl}_{b}"] = {"auroc": boot_auroc(wd, nd), "clean": boot_mean(nd), "word": boot_mean(wd),
                               "named_top10": wilson(sum(x["rank"] < 10 for x in w), len(w)),
                               "hit_budget": sum(x["hit_budget"] for x in w + n) if b else None}
            if den:
                d = den["models"][m]
                for inj, tag in ((word, "word"), ("none", "none")):
                    key = f"{inj} | {b} | {abl}"
                    if key in d:
                        r[f"{abl}_{b}"][f"denial_{tag}"] = wilson(d[key]["denial"], d[key]["n"])
    # phantom: nothing injected, thinking off, change from untouched
    n0 = [x["det"] for x in cell("none", 0, "none")]
    for abl in ("jspace", "matched"):
        r[f"phantom_{abl}"] = boot_diff([x["det"] for x in cell("none", 0, abl)], n0)
    if m == "1.7b":
        for inj in ("jl_L11", "rnd_band", "rnd_L11"):
            for b in (0, 64, 256):
                w, n = cell(inj, b, "none"), cell("none", b, "none")
                if w and n:
                    r[f"{inj}_none_{b}"] = {"auroc": boot_auroc([x["det"] for x in w], [x["det"] for x in n]),
                                            "named_top10": wilson(sum(x["rank"] < 10 for x in w), len(w))}
        w, n = cell(word, 64, "none"), cell("none", 64, "none")
        r["none_64"] = {"auroc": boot_auroc([x["det"] for x in w], [x["det"] for x in n]),
                        "clean": boot_mean([x["det"] for x in n]),
                        "named_top10": wilson(sum(x["rank"] < 10 for x in w), len(w))}
        for abl in ("random",):
            for b in (0, 256):
                w, n = cell(word, b, abl), cell("none", b, abl)
                r[f"{abl}_{b}"] = {"auroc": boot_auroc([x["det"] for x in w], [x["det"] for x in n]),
                                   "clean": boot_mean([x["det"] for x in n])}
        r["phantom_random"] = boot_diff([x["det"] for x in cell("none", 0, "random")], n0)
        # where the word is: median best J-lens rank on the question vs the assistant turn
        w256 = cell(word, 256, "none")
        r["q_best_rank_median"] = st.median(x["q_best_rank"] for x in w256)
        r["a_best_rank_median"] = st.median(x["a_best_rank"] for x in w256)
        r["named_top10_max_any_cell"] = max(sum(x["rank"] < 10 for x in cell(i, b, a))
                                            for i, b, a in {(x["inj"], x["budget"], x["ablation"]) for x in rows})
        # chosen thinking length (cap 1,024)
        rc = load("results_c.jsonl", jsonl=True)
        rcu = {inj: unique_trials([x for x in rc if x["inj"] == inj]) for inj in ("none", "jl_band", "rnd_band", "jl_L11", "rnd_L11")}
        r["chosen_length"] = {inj: {"mean": boot_mean([x["n_think"] for x in v]), "hit_cap": sum(x["hit_budget"] for x in v),
                                    "n": len(v)} for inj, v in rcu.items()}
    th[m] = r
OUT["thinking"] = th

# ---------------------------------------------------------------- experiment D (1.7B), H added on the neutral question
D = load("results_d.json")
OUT["exp_d"] = {"clean": D["clean_det"],
                "grid": {f"{g['cond']} | {g['edit']}": {"det": g["det"], "named_top1": g["named_top1"], "w_top1": g["w_top1"],
                                                       "auroc_vs_clean": g["auroc_vs_clean"]} for g in D["grid"]},
                "validation_r_mean_l12plus": st.mean(v["r_proj_det"] for l, v in D["validation"].items() if int(l) >= 12),
                "validation_cos_max_l12plus": max(abs(v["cos_yes_no_readout"]) for l, v in D["validation"].items() if int(l) >= 12)}
Hd = load("results_d_hadded.json")
OUT["exp_d"]["H_added_neutral"] = {a: {"claims": wilson(v["claims"], 40), "denies": wilson(v["denies"], 40), "words": v["words"]}
                                   for a, v in Hd.items()}

# ---------------------------------------------------------------- deliberate gibberish
G = dict(load("gib_summary.json"))
G.update(load("gib_summary_14b.json"))
gib = {}
for m in MODELS:
    g = G[m]
    C = g["conditions"]
    pick = ("real_chars", "chars", "repeat_rate", "compress", "alpha_run", "dtr", "occ_any", "entropy")
    cg = g["causal"]
    rem, add = cg["remove G from deliberate gibberish"], cg["add G to coherent writing"]
    fu = g.get("followup", {})
    gib[m] = {
        "n": g["n"], "layers": g["layers"], "deep_from": g["deep_from"],
        "conditions": {k: {mm: v[mm] for mm in pick if mm in v} for k, v in C.items()},
        "occ_word_plain": C["B deliberate, plain prompts"]["occ_word"],
        "control": g["control"],
        "G_cos_ext_own_median": sorted(g["G_cos_external_vs_own"])[len(g["G_cos_external_vs_own"]) // 2],
        "remove": {"baseline": rem["baseline"], **{k: {"kl": v["kl"], "G": {mm: v["G"][mm] for mm in pick if mm in v["G"]},
                                                       "vs_random": v["vs_random"]} for k, v in rem["G"].items()}},
        "add": {"baseline": add["baseline"], **{k: {"kl": v["kl"], "G": {mm: v["G"][mm] for mm in pick if mm in v["G"]},
                                                    "vs_random": v["vs_random"]} for k, v in add["G"].items()}},
        "followup": fu,
        "suppression": g["suppression"],
    }
OUT["gibberish"] = gib

json.dump(OUT, open(os.path.join(OUT_DIR, "paper_numbers.json"), "w", encoding="utf-8"), indent=1)


# ---------------------------------------------------------------- readable report
def f(v, d=2, sign=False):
    if v is None:
        return "-"
    if isinstance(v, list):
        fs = f"{{:+.{d}f}}" if sign else f"{{:.{d}f}}"
        return f"{fs.format(v[0])} [{fs.format(v[1])}, {fs.format(v[2])}]"
    return (f"{{:+.{d}f}}" if sign else f"{{:.{d}f}}").format(v)


for m in MODELS:
    x, s, w, t, y, fx = det[m], stg[m], wc[m], th[m], yb[m], ft[m]
    print(f"\n==================== Qwen3-{m.upper()}  (L={x['layers']}, band {x['band']}, single {x['single_layer']}, edit {x['edit_layers']})")
    print(f"clean yes-score {x['clean_yes_score']:+.2f}")
    print(f"word on question (band, 0.1): change {f(x['word_change'], 2, True)}; above clean {f(x['word_above_clean'])}")
    print(f"  battery points: word {x['point_word_band_change']:+.2f} vs random {x['point_random_band_change']:+.2f} (random above {x['random_band_above_clean']:.2f})")
    print(f"  single layer 1.0: word {f(x['word_single_change'], 2, True)} vs random {f(x['random_single_change'], 2, True)}")
    print(f"  named top10 best {x['named_top10_best']}/{x['n_words_battery']}, top1 best {x['named_top1_best']}")
    print(f"H: agreement {x['H_agreement_median_cos']:.3f}; r {x['H_r_det_mean_over_layers']:.2f}; max|cos yes-no| {x['H_cos_yesno_max_abs']:.2f}; cos(H,Y) {x['cos_H_Y_median']:.2f}; H-perp share {x['H_perp_norm_share_median']:.2f}")
    for k, v in x["holds"].items():
        print(f"  hold {k:45s} change {f(v['change'], 2, True)} above {f(v['above_clean'])}")
    print(f"  battery: untouched {f(x['battery_untouched_above'])} H {f(x['battery_H_held'])} Hperp {f(x['battery_Hperp_held'])} Y {f(x['battery_Y_held'])} random {f(x['battery_random_held_above'])}")
    print(f"yes-bias plain: intro {y['plain']['intro_H']:+.2f} (half {y['plain']['intro_H_half']:+.2f}); false mean {y['plain']['false_H_mean']:+.2f} [{y['plain']['false_H_min']:+.2f}..{y['plain']['false_H_max']:+.2f}]; true {y['plain']['true_H']:+.2f}; body {y['plain']['body_H']:+.2f}")
    print(f"yes-bias inj:   intro {y['inj']['intro_H']:+.2f}; false mean {y['inj']['false_H_mean']:+.2f} [{y['inj']['false_H_min']:+.2f}..{y['inj']['false_H_max']:+.2f}]; word-on-q intro {y['inj']['intro_word_on_q']:+.2f} false {y['inj']['false_word_on_q_mean']:+.2f}; random-on-q intro {y['inj']['intro_random_on_q']:+.2f} false {y['inj']['false_random_on_q_mean']:+.2f}")
    print(f"  plain word-on-q intro {y['plain']['intro_word_on_q']:+.2f} false {y['plain']['false_word_on_q_mean']:+.2f}; random-on-q intro {y['plain']['intro_random_on_q']:+.2f} false {y['plain']['false_random_on_q_mean']:+.2f}")
    print(f"  J added false plain mean: " + ", ".join(f"{k} {v:+.2f}" for k, v in y['J_added']['false_plain_mean'].items()))
    print(f"  J added false inj mean:   " + ", ".join(f"{k} {v:+.2f}" for k, v in y['J_added']['false_inj_mean'].items()))
    print(f"  J added intro inj:        " + ", ".join(f"{k} {v:+.2f}" for k, v in y['J_added']['intro_inj'].items()))
    print("free text: " + "; ".join(f"{k} {f(v)}" for k, v in fx.items()))
    print(f"stage 2: escape (90%) by layer {s['escape_layer_90']} of {s['L']} (single {s['single']})")
    for c, v in s["decode"].items():
        print(f"stage 3 {c:32s}: first>=0.9 L{v['first_layer_ge_0.9']}, last L{v['last_layer']} acc {f(v['acc_last'])}, without H {f(v['acc_without_H_last'])}, min after first {v['min_acc_after_first']}")
    print(f"workspace: battery strength {w['battery_strength']}, in J top1 {f(w['in_J_top1'])}, named {f(w['named_top1_forced'])}, change {w['point_change_untouched']:+.2f}, H held {w['point_change_H_held']:+.2f}")
    for sk, e in w["strengths"].items():
        print(f"  s={sk}: word {f(e['word_change'], 2, True)} KL {e['word_kl_mean']:.3f} dh {e['word_dh_mean']:.2f} | random {f(e.get('random_change'), 2, True)} KL {e.get('random_kl_mean', float('nan')):.3f} dh {e.get('random_dh_mean', float('nan')):.2f} | input {f(e['matched_input'], 2, True)} KL {f(e['matched_kl'], 2, True)} dh {f(e['matched_dh'], 2, True)}")
    for k, v in t.items():
        if isinstance(v, dict) and "auroc" in v:
            print(f"  thinking {k:22s} AUROC {f(v['auroc'])} clean {f(v.get('clean'), 2, True)} word {f(v.get('word'), 2, True)} top10 {f(v.get('named_top10'))} denial w {f(v.get('denial_word'))} n {f(v.get('denial_none'))}")
    print(f"  phantom jspace {f(t['phantom_jspace'], 2, True)} matched {f(t['phantom_matched'], 2, True)}" + (f" random {f(t['phantom_random'], 2, True)}" if 'phantom_random' in t else ""))
    if "chosen_length" in t:
        for k, v in t["chosen_length"].items():
            print(f"  chosen length {k:8s} {f(v['mean'], 0)} hit cap {v['hit_cap']}/{v['n']}")
        print(f"  ranks: question {t['q_best_rank_median']} answer {t['a_best_rank_median']}; max top10 any cell {t['named_top10_max_any_cell']}")
print("\nexp D:", json.dumps(OUT["exp_d"], indent=0)[:3000])
