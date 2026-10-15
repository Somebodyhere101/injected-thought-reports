"""Collect the model-written texts behind paper 3, screen them, and write model_texts/.

    python build_texts.py <folder with the full experiment outputs> <folder with the 14B gibberish texts> [output folder]

Outputs (output folder, default model_texts/ in the current directory):
  free_text_<size>.json        free-text answers (free_text.py), by question block and condition
  thinking_<size>.jsonl        thinking traces of the thinking grids, one row per trial
  thinking_chosen_length_1.7b.jsonl   thinking traces when Qwen3-1.7B chose its own length (run_c.py)
  gibberish_<size>.json        replies in the deliberate-gibberish study (gib.py, gib_ext.py), by condition
  experiment_d_1.7b.json       free-text answers from Experiment D and the battery's neutral question (1.7B)
  SCREENING.txt                what the automated screen matched, with context
Also writes _samples.txt next to the output folder (not published): a few texts per condition, for reading.
Standard library only.
"""

import collections
import json
import os
import random
import re
import sys

SRC, SRC14 = sys.argv[1], sys.argv[2]
OUT = os.path.abspath(sys.argv[3] if len(sys.argv) > 3 else "model_texts")
os.makedirs(OUT, exist_ok=True)
MODELS = ["1.7b", "4b", "8b", "14b"]
rng = random.Random(7)

TERMS = (r"fuck\w*|shit\w*|bitch\w*|cunt\w*|whore\w*|slut\w*|bastard\w*|asshole\w*|dick|cock|pussy|"
         r"nigg\w*|fag|faggot\w*|retard\w*|kike\w*|chink\w*|spic|tranny|wetback\w*|"
         r"porn\w*|rape\w*|raping|sexual\w*|sex|nude|naked|orgasm\w*|erotic\w*|"
         r"kill\w*|murder\w*|suicid\w*|self-harm|bomb\w*|weapon\w*|gun\w*|shoot\w*|terror\w*|explosiv\w*|"
         r"overdose\w*|cocaine|heroin|meth|nazi\w*|hitler|jihad\w*|isis|kkk|genocide")
SENSITIVE = re.compile(r"\b(" + TERMS + r")\b", re.I)
PII = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+|https?://\S+|www\.\S+|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b")

samples, screen = [], []
stats = collections.Counter()


def take(source, cond, texts):
    """Screen every text and keep a reading sample of up to three unique ones per condition."""
    stats["texts"] += len(texts)
    uniq = list(dict.fromkeys(t for t in texts if t))
    stats["unique"] += len(uniq)
    for t in uniq:
        for rx, kind in ((SENSITIVE, "term"), (PII, "pii")):
            for m in rx.finditer(t):
                screen.append((source, cond, kind, m.group(0), t[max(0, m.start() - 90):m.end() + 90].replace("\n", " ")))
    for t in rng.sample(uniq, min(3, len(uniq))):
        samples.append((source, cond, t[:500].replace("\n", " / ")))


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f] if path.endswith(".jsonl") else json.load(f)


# free text
for m in MODELS:
    F = load(os.path.join(SRC, f"free_text_{m}.json"))
    out = {"model": f"Qwen/Qwen3-{m.upper()}", "note": "Per-word conditions list the answers in the order of the first 40 words (METHODS.md).",
           "blocks": {}}
    for blk, conds in F.items():
        if not isinstance(conds, dict):
            continue
        out["blocks"][blk] = {}
        for c, v in conds.items():
            if isinstance(v, dict) and "texts" in v:
                out["blocks"][blk][c] = {"summary": v.get("summary"), "texts": v["texts"]}
                take(f"free_text_{m}", f"{blk} | {c}", v["texts"])
    json.dump(out, open(os.path.join(OUT, f"free_text_{m}.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)

# thinking traces
for m, name in (("1.7b", "results_b.jsonl"), ("4b", "results_b_4b.jsonl"), ("8b", "results_b_8b.jsonl"), ("14b", "results_b_14b.jsonl")):
    rows = load(os.path.join(SRC, name))
    keep = ("concept", "inj", "budget", "ablation", "det", "rank", "n_think", "hit_budget", "think")
    cells = collections.defaultdict(list)
    with open(os.path.join(OUT, f"thinking_{m}.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            if r.get("budget"):
                f.write(json.dumps({k: r.get(k) for k in keep}, ensure_ascii=False) + "\n")
                cells[f"{r['inj']} | {r['budget']} | {r['ablation']}"].append(r.get("think", ""))
    for c, ts in cells.items():
        take(f"thinking_{m}", c, ts)
rows = load(os.path.join(SRC, "results_c.jsonl"))
cells = collections.defaultdict(list)
with open(os.path.join(OUT, "thinking_chosen_length_1.7b.jsonl"), "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps({k: r.get(k) for k in ("concept", "inj", "seed", "det", "n_think", "hit_budget", "think")}, ensure_ascii=False) + "\n")
        cells[r["inj"]].append(r.get("think", ""))
for c, ts in cells.items():
    take("thinking_chosen_length_1.7b", c, ts)

# deliberate gibberish
for m in MODELS:
    src = SRC14 if m == "14b" else SRC
    out = {"model": f"Qwen/Qwen3-{m.upper()}", "main": load(os.path.join(src, f"gib_texts_{m}.json")),
           "follow_up": load(os.path.join(src, f"gib_ext_texts_{m}.json"))}
    for part in ("main", "follow_up"):
        for c, ts in out[part].items():
            take(f"gibberish_{m}", f"{part} | {c}", ts)
    json.dump(out, open(os.path.join(OUT, f"gibberish_{m}.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)

# experiment D and the battery's neutral question (1.7B)
D = load(os.path.join(SRC, "results_d_freetext.json"))
Hd = load(os.path.join(SRC, "results_d_hadded.json"))
B = load(os.path.join(SRC, "battery_1.7b.json")).get("I_neutral_free_text", {})
out = {"experiment_d_free_text": {c: [r.get("text") for r in v] for c, v in D.items()},
       "experiment_d_H_added_examples": {a: v.get("examples", []) for a, v in Hd.items()},
       "battery_neutral_question_examples": {c: {"examples": v.get("examples", []), "leaks": v.get("leaks", [])} for c, v in B.items()}}
for c, ts in out["experiment_d_free_text"].items():
    take("experiment_d_1.7b", c, ts)
for a, ts in out["experiment_d_H_added_examples"].items():
    take("experiment_d_1.7b", f"H added x{a}", ts)
for c, v in out["battery_neutral_question_examples"].items():
    take("experiment_d_1.7b", f"battery | {c}", v["examples"] + v["leaks"])
json.dump(out, open(os.path.join(OUT, "experiment_d_1.7b.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)

with open(os.path.join(OUT, "SCREENING.txt"), "w", encoding="utf-8") as f:
    f.write(f"Automated screen of {stats['texts']} texts ({stats['unique']} unique).\n")
    f.write("Patterns: offensive, sexual, violent, self-harm, drug and extremist terms; e-mail addresses, URLs, phone numbers.\n")
    f.write(f"Matches: {len(screen)}\n\n")
    by = collections.Counter((s[2], s[3].lower()) for s in screen)
    for (kind, w), n in by.most_common():
        f.write(f"{kind:5s} {w:20s} {n}\n")
    f.write("\nEvery match with context:\n")
    for s in screen:
        f.write(f"[{s[0]} | {s[1]}] {s[2]} '{s[3]}': ...{s[4]}...\n")
with open(os.path.join(os.path.dirname(OUT), "_samples.txt"), "w", encoding="utf-8") as f:
    for s in samples:
        f.write(f"[{s[0]} | {s[1]}] {s[2]}\n")
print(stats, "matches:", len(screen), "samples:", len(samples))
print(collections.Counter((s[2], s[3].lower()) for s in screen).most_common(40))
