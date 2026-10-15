"""Figures for paper 3, drawn with pgfplots from results/paper_numbers.json (standard library only).

    python figures.py            # writes figures/*.tex and compiles each to figures/*.pdf

Needs pdflatex with pgfplots and standalone. The numbers file is produced by
analysis/paper_numbers.py in the release repository.
"""

import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
CANDIDATES = [os.path.join(HERE, "repo", "results", "paper_numbers.json"),
              os.path.join(HERE, "..", "results", "paper_numbers.json")]
P = json.load(open(next(p for p in CANDIDATES if os.path.exists(p)), encoding="utf-8"))
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)
MODELS = P["models"]
LABEL = {"1.7b": "1.7B", "4b": "4B", "8b": "8B", "14b": "14B"}
MCOL = {"1.7b": "cA", "4b": "cB", "8b": "cC", "14b": "cD"}
MMARK = {"1.7b": "*", "4b": "square*", "8b": "triangle*", "14b": "diamond*"}

PREAMBLE = r"""\documentclass[border=2pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{times}
\usepackage{pgfplots}
\usepgfplotslibrary{groupplots}
\pgfplotsset{compat=1.18}
\definecolor{cA}{HTML}{2A78D6}
\definecolor{cB}{HTML}{EB6834}
\definecolor{cC}{HTML}{1BAF7A}
\definecolor{cD}{HTML}{EDA100}
\definecolor{cGrey}{HTML}{6F7988}
\definecolor{cLight}{HTML}{AEB4BD}
\definecolor{cRule}{HTML}{D9DEE5}
\pgfplotsset{every axis/.append style={font=\footnotesize, axis line style={cGrey}, tick style={cGrey},
  major grid style={cRule, line width=0.3pt}, ymajorgrids, tick pos=left, label style={font=\footnotesize},
  title style={font=\footnotesize\bfseries, yshift=-2pt}, legend style={font=\scriptsize, draw=none, fill=none}},
  err/.style={error bars/.cd, y dir=both, y explicit, error mark options={rotate=90, mark size=1.5pt}, /pgfplots/.cd}}
\begin{document}
"""
END = "\\end{document}\n"


def ci_coord(x, v):
    """pgfplots coordinate with an asymmetric y error bar from [est, lo, hi]."""
    e, lo, hi = v
    return f"({x:.4f},{e:.5f}) += (0,{hi - e:.5f}) -= (0,{e - lo:.5f})"


def write(name, body):
    path = os.path.join(OUT, name + ".tex")
    with open(path, "w", encoding="utf-8") as f:
        f.write(PREAMBLE + body + END)
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", name + ".tex"], cwd=OUT,
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        raise SystemExit(f"pdflatex failed on {name}")
    for ext in (".aux", ".log"):
        p = os.path.join(OUT, name + ext)
        if os.path.exists(p):
            os.remove(p)
    print("wrote", name + ".pdf")


XT = "xtick={1,2,3,4}, xticklabels={Qwen3-1.7B,4B,8B,14B}, xmin=0.5, xmax=4.5"

# ------------------------------------------------------------------ Figure: H is necessary and is a yes direction
D, Y = P["detection"], P["yes_bias"]
series = [("untouched", lambda m: D[m]["holds"]["untouched"]["above_clean"], "cGrey", "square*", -0.24),
          ("random direction held", lambda m: D[m]["holds"]["random direction held, all layers"]["above_clean"], "cLight", "triangle*", -0.08),
          ("$Y$ held", lambda m: D[m]["battery_Y_held"], "cB", "diamond*", 0.08),
          ("$H$ held", lambda m: D[m]["holds"]["H held, all layers, whole assistant turn"]["above_clean"], "cA", "*", 0.24)]
a = []
for name, get, col, mk, dx in series:
    pts = " ".join(ci_coord(i + 1 + dx, get(m)) for i, m in enumerate(MODELS))
    a.append(f"\\addplot[only marks, mark={mk}, mark size=2.2pt, color={col}, err] coordinates {{{pts}}};\n\\addlegendentry{{{name}}}")
b = []
for name, get, col, mk, dx in [
        ("introspection question", lambda m: Y[m]["plain"]["intro_H"], "cA", "*", -0.18),
        ("false questions (mean, range)", None, "cB", "diamond*", 0.0),
        ("same-size random direction", lambda m: Y[m]["J_added"]["false_plain_mean"]["random, same size"], "cGrey", "triangle*", 0.18)]:
    if get is None:
        pts = " ".join(ci_coord(i + 1 + dx, [Y[m]["plain"]["false_H_mean"], Y[m]["plain"]["false_H_min"], Y[m]["plain"]["false_H_max"]])
                       for i, m in enumerate(MODELS))
        b.append(f"\\addplot[only marks, mark={mk}, mark size=2.4pt, color={col}, err] coordinates {{{pts}}};\n\\addlegendentry{{{name}}}")
    else:
        pts = " ".join(f"({i + 1 + dx:.3f},{get(m):.4f})" for i, m in enumerate(MODELS))
        b.append(f"\\addplot[only marks, mark={mk}, mark size=2.2pt, color={col}] coordinates {{{pts}}};\n\\addlegendentry{{{name}}}")
write("fig_holds", r"""\begin{tikzpicture}
\begin{groupplot}[group style={group size=2 by 1, horizontal sep=1.6cm}, width=7.6cm, height=5.4cm, """ + XT + r"""]
\nextgroupplot[title={(a) Word trials with the yes-score above clean}, ymin=-0.03, ymax=1.08, ylabel={share of word trials},
  legend style={at={(0.5,-0.14)}, anchor=north, legend columns=2}, legend cell align=left]
\draw[cGrey, dashed, line width=0.5pt] (axis cs:0.5,0.5) -- (axis cs:4.5,0.5);
""" + "\n".join(a) + r"""
\nextgroupplot[title={(b) Adding $H$ alone, plain question}, ymin=-6, ymax=26, ylabel={change in $\log P(\mathrm{Yes})-\log P(\mathrm{No})$},
  legend style={at={(0.5,-0.14)}, anchor=north, legend columns=2}, legend cell align=left]
\draw[cGrey, line width=0.5pt] (axis cs:0.5,0) -- (axis cs:4.5,0);
""" + "\n".join(b) + r"""
\end{groupplot}
\end{tikzpicture}
""")

# ------------------------------------------------------------------ Figure: stages
S = P["stages"]
pa, pb, pc = [], [], []
for m in MODELS:
    s = S[m]
    L, full = s["L"], s["full_word"][0]
    pts = " ".join(f"({int(l) / L:.4f},{v[0] / full:.4f})" for l, v in sorted(s["cut_word"].items(), key=lambda kv: int(kv[0])))
    pa.append(f"\\addplot[color={MCOL[m]}, mark={MMARK[m]}, mark size=1.3pt, line width=0.9pt] coordinates {{{pts}}};\n\\addlegendentry{{{LABEL[m]}}}")
    for cond, dash in (("answer position | untouched", "solid"), ("answer position | H held", "dashed")):
        c = s["decode_curve"][cond]
        pts = " ".join(f"({int(l) / L:.4f},{v:.4f})" for l, v in sorted(c.items(), key=lambda kv: int(kv[0])))
        pb.append(f"\\addplot[color={MCOL[m]}, {dash}, line width=0.9pt] coordinates {{{pts}}};")
    holds = D[m]["holds"]
    wins = sorted([k for k in holds if k.startswith("H held, layers")], key=lambda k: int(k.split("layers ")[1].split("-")[0]))
    pts = " ".join(ci_coord(i + 1 + (MODELS.index(m) - 1.5) * 0.07, holds[k]["above_clean"]) for i, k in enumerate(wins))
    pc.append(f"\\addplot[color={MCOL[m]}, mark={MMARK[m]}, mark size=1.6pt, line width=0.9pt, err] coordinates {{{pts}}};")
write("fig_stages", r"""\begin{tikzpicture}
\begin{groupplot}[group style={group size=3 by 1, horizontal sep=1.25cm}, width=5.6cm, height=5.0cm]
\nextgroupplot[title={(a) Effect gone from the question}, xlabel={layer reset to clean / depth}, ylabel={share of the full effect},
  ymin=-0.15, ymax=1.25, xmin=0.35, xmax=1.0, legend pos=south east]
""" + "\n".join(pa) + r"""
\nextgroupplot[title={(b) Word decodable at the answer}, xlabel={layer / depth}, ylabel={40-way accuracy},
  ymin=-0.03, ymax=1.05, xmin=0.35, xmax=1.0]
\draw[cGrey, dotted, line width=0.6pt] (axis cs:0.35,0.025) -- (axis cs:1,0.025);
""" + "\n".join(pb) + r"""
\nextgroupplot[title={(c) $H$ held in one window only}, xlabel={window of layers (early to late)}, ylabel={word trials above clean},
  xtick={1,2,3,4}, xmin=0.6, xmax=4.4, ymin=-0.05, ymax=1.08]
\draw[cGrey, dashed, line width=0.5pt] (axis cs:0.6,0.5) -- (axis cs:4.4,0.5);
""" + "\n".join(pc) + r"""
\end{groupplot}
\end{tikzpicture}
""")

# ------------------------------------------------------------------ Figure: workspace dose-response
W = P["workspace"]
blocks = []
for k, m in enumerate(MODELS):
    w = W[m]
    wp = " ".join(ci_coord(e["word_kl_mean"], e["word_change"]) for s, e in sorted(w["strengths"].items(), key=lambda kv: float(kv[0])))
    rp = " ".join(ci_coord(e["kl_mean"], e["change"]) for s, e in sorted(w["random_curve"].items(), key=lambda kv: float(kv[0])))
    labels = "\n".join(f"\\node[font=\\tiny, color=cA, anchor=south east] at (axis cs:{e['word_kl_mean']:.4f},{e['word_change'][0]:.4f}) {{{s}}};"
                       for s, e in w["strengths"].items() if float(s) <= 0.03)
    last = k == len(MODELS) - 1
    legend = "legend pos=north west, ymax=37," if last else ""  # headroom keeps the legend clear of the data
    blocks.append(r"\nextgroupplot[title={Qwen3-" + LABEL[m] + r"}, " + legend + r"""]
\draw[cGrey, line width=0.5pt] (axis cs:0.001,0) -- (axis cs:100,0);
\addplot[color=cLight, mark=triangle*, mark size=1.8pt, line width=0.8pt, err] coordinates {""" + rp + r"""};
""" + (r"\addlegendentry{random directions}" if last else "") + r"""
\addplot[color=cA, mark=*, mark size=1.8pt, line width=0.8pt, err] coordinates {""" + wp + r"""};
""" + (r"\addlegendentry{word direction}" if last else "") + "\n" + labels)
write("fig_workspace", r"""\begin{tikzpicture}
\begin{groupplot}[group style={group size=4 by 1, horizontal sep=0.9cm}, width=4.6cm, height=4.8cm, xmode=log,
  xmin=0.001, xmax=100, xtick={0.01,0.1,1,10}, xticklabels={0.01,0.1,1,10}, xlabel={KL from clean (log)}, legend cell align=left]
""" + "\n".join(blocks).replace(r"\nextgroupplot[title={Qwen3-1.7B}, ", r"\nextgroupplot[title={Qwen3-1.7B}, ylabel={change in yes-score}, ", 1) + r"""
\end{groupplot}
\end{tikzpicture}
""")

# ------------------------------------------------------------------ Figure: thinking
T = P["thinking"]
ta, tb = [], []
for i, m in enumerate(MODELS):
    t0, t1 = T[m]["none_0"], T[m]["none_256"]
    x0, x1 = i + 1 - 0.15, i + 1 + 0.15
    ta.append(f"\\addplot[color={MCOL[m]}, mark={MMARK[m]}, mark size=1.8pt, line width=0.9pt, err] coordinates {{{ci_coord(x0, t0['auroc'])} {ci_coord(x1, t1['auroc'])}}};")
    for key, col, mk, dx in (("clean", "cGrey", "square*", -0.06), ("word", "cA", "*", 0.06)):
        tb.append(f"\\addplot[color={col}, mark={mk}, mark size=1.6pt, line width=0.8pt, err{', forget plot' if i else ''}] coordinates "
                  f"{{{ci_coord(x0 + dx, t0[key])} {ci_coord(x1 + dx, t1[key])}}};" + (f"\\addlegendentry{{{'nothing injected' if key == 'clean' else 'word injected'}}}" if i == 0 else ""))
write("fig_thinking", r"""\begin{tikzpicture}
\begin{groupplot}[group style={group size=2 by 1, horizontal sep=1.6cm}, width=7.6cm, height=5.2cm, """ + XT + r"""]
\nextgroupplot[title={(a) Injected vs clean, AUROC (thinking off $\to$ 256 tokens)}, ymin=0.45, ymax=1.03, ylabel={AUROC}]
\draw[cGrey, dashed, line width=0.5pt] (axis cs:0.5,0.5) -- (axis cs:4.5,0.5);
""" + "\n".join(ta) + r"""
\nextgroupplot[title={(b) Mean yes-score (thinking off $\to$ 256 tokens)}, ymin=-30, ymax=8, ylabel={$\log P(\mathrm{Yes})-\log P(\mathrm{No})$},
  legend pos=south west, legend cell align=left]
\draw[cGrey, line width=0.5pt] (axis cs:0.5,0) -- (axis cs:4.5,0);
""" + "\n".join(tb) + r"""
\end{groupplot}
\end{tikzpicture}
""")

# ------------------------------------------------------------------ Figure: deliberate gibberish
G = P["gibberish"]
ga, gb, gc = [], [], []
for i, m in enumerate(MODELS):
    c = G[m]["control"]["model's own gibberish"]
    ga.append(f"\\addplot[color=cB, mark=*, mark size=1.8pt, err{', forget plot' if i else ''}] coordinates {{{ci_coord(i + 1 - 0.12, c['gibberish instruction']['occ_any'])}}};"
              + ("\\addlegendentry{gibberish instruction}" if i == 0 else ""))
    ga.append(f"\\addplot[color=cGrey, mark=square*, mark size=1.8pt, err{', forget plot' if i else ''}] coordinates {{{ci_coord(i + 1 + 0.12, c['coherent instruction']['occ_any'])}}};"
              + ("\\addlegendentry{coherent instruction}" if i == 0 else ""))
    r = G[m]["remove"]["G x0.5"]["vs_random"]
    gb.append(f"\\addplot[color={MCOL[m]}, mark={MMARK[m]}, mark size=2pt, err] coordinates {{{ci_coord(i + 1, r['chars'])}}};")
    gc.append(f"\\addplot[color={MCOL[m]}, mark={MMARK[m]}, mark size=2pt, err] coordinates {{{ci_coord(i + 1, r['alpha_run'])}}};")
write("fig_gibberish", r"""\begin{tikzpicture}
\begin{groupplot}[group style={group size=3 by 1, horizontal sep=1.75cm}, width=5.2cm, height=4.9cm, """ + XT + r""", xticklabel style={font=\scriptsize},
  yticklabel style={/pgf/number format/fixed}]
\nextgroupplot[title={(a) Target words in the J-space}, ylabel={share of reply positions}, ymin=-0.01, ymax=0.2, legend pos=north east,
  legend cell align=left]
""" + "\n".join(ga) + r"""
\nextgroupplot[title={(b) Remove $G$: reply length}, ylabel={characters, $G$ minus random}, ymin=-130, ymax=10]
\draw[cGrey, line width=0.5pt] (axis cs:0.5,0) -- (axis cs:4.5,0);
""" + "\n".join(gb) + r"""
\nextgroupplot[title={(c) Remove $G$: letters in order}, ylabel={alphabetical steps, $G$ minus random}, ymin=-0.15, ymax=1.05]
\draw[cGrey, line width=0.5pt] (axis cs:0.5,0) -- (axis cs:4.5,0);
""" + "\n".join(gc) + r"""
\end{groupplot}
\end{tikzpicture}
""")
