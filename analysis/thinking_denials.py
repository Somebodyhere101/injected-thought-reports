"""Count thinking traces that contain a capability disclaimer, per cell of the thinking grids.

    python analysis/thinking_denials.py <folder with the full results_b*.jsonl>

The published per-trial files have the thinking text removed, so this script needs the
unpublished originals (the folder the experiments wrote to). It writes only counts, to
results/thinking_denials.json. The pattern is the one summarize.py uses for Qwen3-1.7B.
"""

import json
import os
import re
import sys
from collections import defaultdict

DENIAL = re.compile(r"can't|cannot|don't have|unable|not able|no way to", re.I)
FILES = {"1.7b": "results_b.jsonl", "4b": "results_b_4b.jsonl", "8b": "results_b_8b.jsonl", "14b": "results_b_14b.jsonl"}
HERE = os.path.dirname(os.path.abspath(__file__))


def unique_trials(rows):
    """Cells without an injection were sampled in batches of 20 with one seed per batch, so later
    batches repeat the first; keep one period (the same rule as paper_numbers.py)."""
    ks = [(r["det"], r["n_think"]) for r in rows]
    for p in range(1, len(ks)):
        if len(ks) % p == 0 and all(ks[i] == ks[i % p] for i in range(len(ks))):
            return rows[:p]
    return rows


src = sys.argv[1]
out = {"pattern": DENIAL.pattern, "models": {}}
for key, name in FILES.items():
    cells = defaultdict(list)
    for line in open(os.path.join(src, name), encoding="utf-8"):
        r = json.loads(line)
        if r["budget"]:
            cells[f"{r['inj']} | {r['budget']} | {r['ablation']}"].append(r)
    res = {}
    for k, rows in sorted(cells.items()):
        u = unique_trials(rows)
        res[k] = {"denial": sum(bool(DENIAL.search(r["think"])) for r in u), "n": len(u)}
    out["models"][key] = res
    print(key, {k: f"{v['denial']}/{v['n']}" for k, v in res.items()})
json.dump(out, open(os.path.join(HERE, "..", "results", "thinking_denials.json"), "w", encoding="utf-8"), indent=1)
