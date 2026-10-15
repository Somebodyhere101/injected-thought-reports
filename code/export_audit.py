"""Build the audit package.

audit/code/   every script, the page template and METHODS.md
audit/data/   per-trial numeric outputs with all model-generated text removed
audit/MANIFEST.json   size and SHA-256 of every file in audit/
audit_full.zip        code, logs and every raw output including model text (kept local)

Every string left in audit/data is listed at the end so it can be checked by eye.
"""

import glob
import hashlib
import json
import os
import shutil
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "audit")
TEXT_KEYS = {"think", "text", "texts", "examples", "leaks", "samples", "tokens"}


def strip(x):
    """Drop model-text fields recursively."""
    if isinstance(x, dict):
        return {k: strip(v) for k, v in x.items() if k not in TEXT_KEYS}
    if isinstance(x, list):
        return [strip(v) for v in x]
    return x


def load(path):
    if path.endswith(".jsonl"):
        return [json.loads(l) for l in open(path, encoding="utf-8")]
    return json.load(open(path, encoding="utf-8"))


def dump(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            for r in obj:
                f.write(json.dumps(r) + "\n")
        else:
            json.dump(obj, f, indent=1)


shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(os.path.join(OUT, "code"))
os.makedirs(os.path.join(OUT, "data"))

for p in sorted(glob.glob(os.path.join(HERE, "*.py"))) + [os.path.join(HERE, "METHODS.md")]:
    shutil.copy(p, os.path.join(OUT, "code", os.path.basename(p)))
# the page template, as text (served as HTML it would render as a broken page)
shutil.copy(os.path.join(HERE, "report_template.html"), os.path.join(OUT, "code", "report_template.html.txt"))

DATA = ["results_b.jsonl", "results_c.jsonl", "results_b_4b.jsonl", "results_b_8b.jsonl", "results_b_14b.jsonl", "trace_examples.json",
        "results_d.json", "results_d_aware.json", "results_d_freetext.json", "results_d_hadded.json",
        "battery_1.7b.json", "battery_4b.json", "battery_8b.json", "battery_14b.json",
        "free_text_1.7b.json", "free_text_4b.json", "free_text_8b.json", "free_text_14b.json",
        "stages_1.7b.json", "stages_4b.json", "stages_8b.json", "stages_14b.json",
        "w_control2_1.7b.json", "w_control2_4b.json", "w_control2_8b.json", "w_control2_14b.json",
        "w_control_1.7b.json", "w_control_4b.json", "w_control_8b.json", "scale.json"]
written = []
for name in DATA:
    src = os.path.join(HERE, name)
    if not os.path.exists(src):
        print(f"missing, skipped: {name}")
        continue
    obj = load(src)
    if name == "trace_examples.json":  # keep the prompt tokens, drop everything the model generated
        for exs in obj.values():
            for ex in exs:
                tr = ex["trace"]
                ex["prompt_tokens"] = tr["tokens"][: tr["a_start"]]
    obj = strip(obj)
    dump(obj, os.path.join(OUT, "data", name))
    written.append(name)

# every string left in the published data, for a by-eye check
strings = set()


def walk(x):
    if isinstance(x, dict):
        for k, v in x.items():
            strings.add(str(k))
            walk(v)
    elif isinstance(x, list):
        for v in x:
            walk(v)
    elif isinstance(x, str):
        strings.add(x)


for name in written:
    walk(load(os.path.join(OUT, "data", name)))
long = sorted(s for s in strings if len(s) > 30)
print(f"{len(written)} data files, {len(strings)} distinct strings; longest ones:")
for s in long:
    print("  ", repr(s))
assert max(len(s) for s in strings) < 120, "a long string survived stripping"

manifest = {}
for root, _, files in os.walk(OUT):
    for f in sorted(files):
        p = os.path.join(root, f)
        rel = os.path.relpath(p, OUT).replace("\\", "/")
        manifest[rel] = {"bytes": os.path.getsize(p), "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest()}
json.dump(manifest, open(os.path.join(OUT, "MANIFEST.json"), "w", encoding="utf-8"), indent=1)

with zipfile.ZipFile(os.path.join(HERE, "audit_full.zip"), "w", zipfile.ZIP_DEFLATED) as z:
    for pat in ("*.py", "*.md", "*.html", "*.json", "*.jsonl", "*.log"):
        for p in glob.glob(os.path.join(HERE, pat)):
            z.write(p, os.path.join("effort_jspace", os.path.basename(p)))
print(f"audit/: {len(manifest)} files; audit_full.zip: {os.path.getsize(os.path.join(HERE, 'audit_full.zip')) / 1e6:.1f} MB")
