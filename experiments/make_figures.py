"""Phase 7: paper figures (300 dpi PNG + vector PDF) and summary tables from results/.

    python -m experiments.make_figures

Every figure is built from the CSVs in results/ and skipped (with a note) if its inputs are not
there yet, so this can be re-run as experiments finish.

Colours: the reference categorical palette in fixed slot order (blue, orange, aqua, yellow);
scatter/box figures use at most the first three slots. Grids are hairline and solid; stacked
segments are separated by a 2 px white gap and also carry 45/135-degree hatching so they survive
grayscale printing.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import load_config, project_path  # noqa: E402

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # blue, orange, aqua, yellow
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e2e1dc"
COL_W, PAGE_W = 3.5, 7.16                              # IEEE column / page width, inches
HATCH = ["", "////", "\\\\\\\\", "xx"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-",
    "axes.axisbelow": True, "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight",
    "lines.linewidth": 1.5, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    "hatch.linewidth": 0.5, "hatch.color": "#ffffff",
})

RES = project_path(load_config()["paths"]["results_dir"])
FIG = project_path(load_config()["paths"]["figures_dir"])
METRICS = ["GA", "MLA", "ED"]
NAMES = {"GA": "Group Accuracy", "MLA": "Message-Level Accuracy", "ED": "Edit-Distance Score"}


def _read(name):
    p = RES / name
    if not p.exists():
        print(f"  skip: {name} not found yet")
        return None
    df = pd.read_csv(p)
    return df[df["Dataset"] != "Average"] if "Dataset" in df else df


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote figures/{name}.png/.pdf")


def fig_robustness():
    """Box plots of GA / MLA / ED across the 16 datasets (cf. paper Fig. 6)."""
    t = _read("E1_main.csv")
    if t is None or len(t) < 16:
        print("  skip robustness: E1 not complete")
        return
    parsers = [("Drain", "Drain"), ("Ours", "Ours"), ("Paper", "Paper\n(reported)")]
    fig, axes = plt.subplots(1, 3, figsize=(PAGE_W, 2.1), sharey=False)
    for ax, m in zip(axes, METRICS):
        data = [t[f"{p}_{m}"].values for p, _ in parsers]
        bp = ax.boxplot(data, widths=0.45, patch_artist=True, showfliers=True,
                        medianprops={"color": INK, "linewidth": 1.2},
                        whiskerprops={"color": INK2, "linewidth": 0.8}, capprops={"color": INK2, "linewidth": 0.8},
                        flierprops={"marker": "o", "markersize": 3, "markerfacecolor": INK2, "markeredgecolor": "white"})
        for patch, c in zip(bp["boxes"], SLOTS):
            patch.set_facecolor(c)
            patch.set_alpha(0.85)
            patch.set_edgecolor("white")
        ax.set_xticks([1, 2, 3], [lab for _, lab in parsers])
        ax.tick_params(axis="x", length=0)
        ax.set_title(NAMES[m])
        ax.set_ylim(0, 1.02)
        ax.grid(axis="x", visible=False)
    _save(fig, "fig_robustness_boxplots")


def fig_per_dataset_ga():
    """Dot plot: per-dataset GA for Drain, ours and the paper's LogBatcher."""
    t = _read("E1_main.csv")
    if t is None:
        return
    t = t.iloc[::-1]
    y = np.arange(len(t))
    fig, axes = plt.subplots(1, 2, figsize=(PAGE_W, 0.18 * len(t) + 0.8), sharey=True)
    for ax, m in zip(axes, ["GA", "MLA"]):
        for (col, lab), c, dy in zip([("Drain", "Drain"), ("Ours", "LogBatcher (ours)"),
                                      ("Paper", "LogBatcher (paper)")], SLOTS, (-0.22, 0, 0.22)):
            ax.scatter(t[f"{col}_{m}"], y + dy, s=16, color=c, edgecolor="white", linewidth=0.8, label=lab, zorder=3)
        ax.set_yticks(y, t["Dataset"])
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(0, 1.02)
        ax.set_title(NAMES[m])
        ax.grid(axis="y", visible=False)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.06), ncol=3, handletextpad=0.2, columnspacing=1)
    _save(fig, "fig_per_dataset_accuracy")


def fig_batch_size(model="claude-haiku-4-5"):
    """Accuracy and tokens/call vs batch size (cf. paper Fig. 8): two panels, never a dual axis."""
    s = _read(f"E5_batchsize_{model}.csv")
    if s is None:
        return
    s = s.assign(k=s["Variant"].str.split("=").str[1].astype(int)).sort_values("k")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(PAGE_W, 2.0))
    for m, c in zip(METRICS, SLOTS):
        a1.plot(s["k"], s[m], color=c, marker="o", markersize=4, markeredgecolor="white", label=m)
        a1.annotate(m, (s["k"].iloc[-1], s[m].iloc[-1]), xytext=(4, 0), textcoords="offset points",
                    va="center", color=INK2, fontsize=7)
    a1.set_xticks(s["k"])
    a1.set_xlabel("Batch size (logs per prompt)")
    a1.set_title("Average accuracy")
    a1.legend(loc="center right", ncol=3, handlelength=1.2, columnspacing=0.8)
    a2.plot(s["k"], s["T_invoc"], color=SLOTS[0], marker="o", markersize=4, markeredgecolor="white")
    a2.set_xticks(s["k"])
    a2.set_xlabel("Batch size (logs per prompt)")
    a2.set_title("Tokens per LLM call")
    a2.set_ylim(0, None)
    _save(fig, f"fig_batch_size_{model}")


def fig_cost_accuracy():
    """Extension: accuracy vs cost per model (max three models, so all pairs stay distinguishable)."""
    e = _read("E6_extension.csv")
    if e is None or e["Cost_USD"].isna().all():
        return
    e = e.dropna(subset=["Cost_USD"]).head(3)
    fig, axes = plt.subplots(1, 2, figsize=(PAGE_W, 2.1))
    for ax, m in zip(axes, ["GA", "MLA"]):
        for (_, r), c in zip(e.iterrows(), SLOTS):
            ax.scatter(r["Cost_USD"], r[m], s=40, color=c, edgecolor="white", linewidth=1.5, zorder=3, label=r["Model"])
            ax.annotate(r["Model"].split(" (")[0], (r["Cost_USD"], r[m]), xytext=(5, 4),
                        textcoords="offset points", fontsize=7, color=INK2)
        ax.set_xlabel("Cost of parsing all 16 datasets (USD, list price)")
        ax.set_title(f"Average {NAMES[m]}")
        ax.set_xlim(0, e["Cost_USD"].max() * 1.6)
        ax.set_ylim(0.70, 1.00)            # fixed range so small gaps are not exaggerated
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.08), ncol=3, handletextpad=0.2)
    _save(fig, "fig_cost_vs_accuracy")


def fig_errors(experiment="E1_logbatcher"):
    """Stacked horizontal bars of template-error categories per dataset."""
    e = _read("E6_error_analysis.csv")
    if e is None:
        return
    e = e[e["Experiment"] == experiment]
    if e.empty:
        return
    cats = [("under_parsing", "Under-parsing"), ("over_parsing", "Over-parsing"),
            ("mixed", "Mixed"), ("label_style", "Label inconsistency")]
    e = e.set_index("Dataset").loc[lambda d: d[[c for c, _ in cats]].sum(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(COL_W, 0.17 * len(e) + 0.8))
    left = np.zeros(len(e))
    for (col, lab), c, h in zip(cats, SLOTS, HATCH):
        ax.barh(e.index, e[col], left=left, height=0.62, color=c, edgecolor="white", linewidth=1.0,
                hatch=h, label=lab)
        left += e[col].values
    ax.set_xlim(0, left.max() * 1.04)
    ax.set_xlabel("Wrongly parsed log lines (of 2,000)")
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=6.5)
    _save(fig, f"fig_error_categories_{experiment}")


def tables():
    """Print the headline tables (they are already saved as CSV by the experiment scripts)."""
    for name in ["E1_main.csv", "E2_efficiency.csv", "E3_ablation_claude-haiku-4-5.csv",
                 "E4_sampling_claude-haiku-4-5.csv", "E5_batchsize_claude-haiku-4-5.csv", "E6_extension.csv"]:
        p = RES / name
        if p.exists():
            print(f"\n== {name}\n" + pd.read_csv(p).round(3).to_string(index=False))


if __name__ == "__main__":
    for f in (fig_robustness, fig_per_dataset_ga, fig_batch_size, fig_cost_accuracy, fig_errors):
        print(f.__name__)
        f()
    tables()
