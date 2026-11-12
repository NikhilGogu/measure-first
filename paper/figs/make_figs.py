"""Figure 2: runs flagged by the loop screen, by adapter. The counts are the ones the paper states."""
import matplotlib
matplotlib.use("pgf")  # text is set by pdflatex in the paper's own fonts
import matplotlib.pyplot as plt
plt.rcParams.update({"pgf.texsystem": "pdflatex", "pgf.rcfonts": False, "text.usetex": True, "font.family": "serif", "font.size": 9,
                     "pgf.preamble": r"\usepackage[T1]{fontenc}\usepackage{newtxtext,newtxmath}", "axes.spines.top": False, "axes.spines.right": False})
rows = [("Baseline", 1, 86, "#8A93A0"), ("Pilot\n(101 teacher tasks)", 33, 36, "#B04828"), ("Round 2\n(272 teacher tasks)", 4, 8, "#B04828"),
        ("Round 3\n(short teacher runs)", 6, 8, "#B04828"), ("Round 4\n(362 cleaned)", 5, 8, "#B04828"), ("Probe\n(39 own tasks)", 1, 8, "#1F4E9E")]
fig, ax = plt.subplots(figsize=(6.5, 2.5))
xs = range(len(rows))
vals = [100.0 * a / b for _, a, b, _ in rows]
ax.bar(xs, vals, color=[c for *_, c in rows], width=0.62)
for x, (name, a, b, c), v in zip(xs, rows, vals):
    ax.text(x, v + 2.5, "%d of %d" % (a, b), ha="center", va="bottom", fontsize=8.5)
ax.set_xticks(list(xs)); ax.set_xticklabels([r[0] for r in rows], fontsize=8)
ax.set_ylabel(r"Runs flagged (\%)"); ax.set_ylim(0, 108); ax.set_yticks([0, 25, 50, 75, 100])
ax.tick_params(axis="x", length=0)
import matplotlib.patches as mp
ax.legend(handles=[mp.Patch(color="#8A93A0", label="no adapter"), mp.Patch(color="#B04828", label="trained on 31B teacher runs"), mp.Patch(color="#1F4E9E", label="trained on E4B's own runs")],
          frameon=False, fontsize=8, loc="upper right", bbox_to_anchor=(1.0, 1.04), ncol=1)
fig.tight_layout(pad=0.4)
fig.savefig("screen.pdf")
