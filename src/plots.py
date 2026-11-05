"""Figures (matplotlib only). Severity is ordinal, so it is drawn with one blue
ramp light->dark; model identity uses a fixed categorical order."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from . import config  # noqa: E402

INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SEV_COLORS = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]   # Low -> Critical (sequential blue)
BLUE = "#2a78d6"
SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", ["#fcfcfb", "#cde2fb", "#6da7ec", "#256abf", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": INK2,
    "text.color": INK, "axes.titleweight": "bold", "axes.titlesize": 12, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
})


def _save(fig, name):
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIG_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def class_distribution(df):
    counts = df["severity"].value_counts().reindex(config.LABELS)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    bars = ax.bar(config.LABELS, counts.values, color=SEV_COLORS, width=0.6, edgecolor=SURFACE, linewidth=2)
    for b, c in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, c + counts.max() * 0.015, f"{c:,}\n({c / counts.sum():.1%})",
                ha="center", va="bottom", fontsize=9, color=INK2)
    ax.set_ylabel("Records")
    ax.set_ylim(0, counts.max() * 1.22)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(f"Severity class distribution (n = {counts.sum():,})", loc="left")
    _save(fig, "class_distribution.png")


def confusion_matrix_plot(cm, title, fname):
    """Row-normalised confusion matrix with raw counts annotated."""
    norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(norm, cmap=SEQ_CMAP, vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.0%}", ha="center", va="center", fontsize=9,
                    color="#ffffff" if norm[i, j] > 0.55 else INK)
    ax.set_xticks(range(4), config.LABELS)
    ax.set_yticks(range(4), config.LABELS)
    ax.set_xlabel("Predicted severity")
    ax.set_ylabel("True severity")
    ax.set_title(title, loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Row-normalised recall", color=INK2)
    cb.outline.set_visible(False)
    _save(fig, fname)


def top_terms(coef_df, fname="top_terms_per_class.png", k=12):
    """coef_df: DataFrame indexed by term with one column per class (LR coefficients)."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.8))
    for ax, label, color in zip(axes, config.LABELS, SEV_COLORS):
        top = coef_df[label].sort_values(ascending=False).head(k)[::-1]
        ax.barh(top.index, top.values, color=color, height=0.65, edgecolor=SURFACE, linewidth=1)
        ax.set_title(label, loc="left")
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=9)
        ax.set_xlabel("LR coefficient")
    fig.suptitle("Most predictive TF-IDF terms per severity class (Logistic Regression)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    _save(fig, fname)


def model_comparison(test_df, fname="model_comparison.png"):
    """Grouped horizontal bars: accuracy and macro-F1 per model with 95% bootstrap CIs."""
    d = test_df.iloc[::-1]
    y = np.arange(len(d))
    h = 0.36
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    specs = [("accuracy", "Accuracy", "#2a78d6", +h / 2), ("macro_f1", "Macro-F1", "#eb6834", -h / 2)]
    for col, lab, color, off in specs:
        lo, hi = d[f"{col}_ci_low"], d[f"{col}_ci_high"]
        ax.barh(y + off, d[col], height=h - 0.04, color=color, label=lab, edgecolor=SURFACE, linewidth=1)
        ax.errorbar(d[col], y + off, xerr=[d[col] - lo, hi - d[col]], fmt="none", ecolor=INK2, elinewidth=1,
                    capsize=2)
        for yy, v, top in zip(y + off, d[col], hi):
            ax.text(top + 0.01, yy, f"{v:.3f}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(y, d["model"])
    ax.set_xlim(0, 1.05)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False)
    ax.set_title("Held-out test performance (95% bootstrap CI)", loc="left")
    _save(fig, fname)
