"""Checks on the free-text answers to "Is the sun colder than ice?" that need the texts.

    python analysis/free_text_checks.py <folder with the full free_text_<model>.json>

free_text.py scores only the first word of these answers. Reading them showed that, with H
added, Qwen3-1.7B starts with "Yes" but usually goes on to say that the sun is hotter. This
script counts answers that assert the false claim and answers that state the correct fact, and
writes only the counts to results/free_text_checks.json.
"""

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FALSE = re.compile(r"\bsun is colder than ice", re.I)
NEGATED = re.compile(r"\bsun is not colder than ice", re.I)
CORRECT = re.compile(r"hotter than ice|not colder|much hotter|far hotter|is hotter", re.I)

src = sys.argv[1]
out = {"patterns": {"asserts": FALSE.pattern, "negated": NEGATED.pattern, "correct": CORRECT.pattern}, "models": {}}
for m in ("1.7b", "4b", "8b", "14b"):
    F = json.load(open(os.path.join(src, f"free_text_{m}.json"), encoding="utf-8"))["T3 Is the sun colder than ice"]
    row = {}
    for cond, tag in (("clean", "clean"), ("H added", "H")):
        t = F[cond]["texts"]
        row[f"T3_sun_{tag}_asserts_false"] = {"k": sum(bool(FALSE.search(x)) and not NEGATED.search(x) for x in t), "n": len(t)}
        row[f"T3_sun_{tag}_states_correct"] = {"k": sum(bool(CORRECT.search(x)) for x in t), "n": len(t)}
    out["models"][m] = row
    print(m, {k: f"{v['k']}/{v['n']}" for k, v in row.items()})
json.dump(out, open(os.path.join(HERE, "..", "results", "free_text_checks.json"), "w", encoding="utf-8"), indent=1)
