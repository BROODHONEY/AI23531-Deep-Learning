"""Aggregate ``results/runs`` into tables and figures.

    python -m src.report

Writes ``results/summary.json``, ``results/summary.md`` and ``results/figures/*.png``.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
from scipy.stats import binomtest

from .data import CLASS_NAMES, load_raw

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = RES / "figures"

ORDER = ["logreg", "mlp", "cnn", "resnet", "vit", "resnet_noaug"]
LABEL = {
    "logreg": "Logistic regression (baseline)", "mlp": "MLP", "cnn": "Simple CNN",
    "resnet": "ResNet-lite", "vit": "Tiny ViT", "resnet_noaug": "ResNet-lite, no augmentation",
}
SHORT = {"logreg": "Logistic reg.", "mlp": "MLP", "cnn": "Simple CNN", "resnet": "ResNet-lite",
         "vit": "Tiny ViT", "resnet_noaug": "ResNet-lite (no aug)"}
# Fixed categorical order (validated palette, light mode); colour follows the model, never its rank.
COLOR = {"logreg": "#2a78d6", "mlp": "#eb6834", "cnn": "#1baf7a", "resnet": "#eda100",
         "vit": "#e87ba4", "resnet_noaug": "#008300"}
MARKER = {"logreg": "o", "mlp": "s", "cnn": "^", "resnet": "D", "vit": "v", "resnet_noaug": "P"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "figure.dpi": 110,
})


def load_runs():
    runs = {}
    for p in sorted((RES / "runs").glob("*_seed*.json")):
        r = json.loads(p.read_text())
        npz = np.load(p.with_name(p.stem + "_preds.npz"))
        r["pred"], r["conf"] = npz["pred"].astype(int), npz["conf"].astype(float)
        runs.setdefault(r["model"], []).append(r)
    return runs


def ms(vals, scale=100.0, nd=2):
    v = np.asarray(vals, dtype=float) * scale
    return f"{v.mean():.{nd}f} ± {v.std(ddof=1):.{nd}f}" if len(v) > 1 else f"{v.mean():.{nd}f}"


def mcnemar(pred_a, pred_b, y):
    """Exact McNemar test on paired test-set predictions."""
    a_ok, b_ok = pred_a == y, pred_b == y
    only_a, only_b = int((a_ok & ~b_ok).sum()), int((~a_ok & b_ok).sum())
    n = only_a + only_b
    p = binomtest(min(only_a, only_b), n, 0.5).pvalue if n else 1.0
    return only_a, only_b, p


def savefig(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- figures
def fig_samples(x_test, y_test):
    fig, axes = plt.subplots(2, 5, figsize=(9, 4.1))
    for c, ax in enumerate(axes.ravel()):
        i = int(np.where(y_test == c)[0][0])
        ax.imshow(x_test[i], cmap="gray", vmin=0, vmax=255)
        ax.set_title(CLASS_NAMES[c], fontsize=9, fontweight="normal", loc="center")
        ax.axis("off")
    fig.suptitle("Fashion-MNIST: one test example per class (28×28 grayscale)", x=0.01, ha="left",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "01_samples.png")


def fig_curves(runs, models):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1))
    for key, ax, title in (("val_f1", axes[0], "Validation macro-F1"), ("val_loss", axes[1], "Validation cross-entropy")):
        for m in models:
            h = np.array([r["history"][key] for r in runs[m]])
            x = np.arange(1, h.shape[1] + 1)
            ax.plot(x, h.mean(0), color=COLOR[m], lw=2, marker=MARKER[m], ms=4.5,
                    markeredgecolor=SURFACE, markeredgewidth=0.8, label=SHORT[m])
            if len(h) > 1:
                ax.fill_between(x, h.min(0), h.max(0), color=COLOR[m], alpha=0.15, lw=0)
        ax.set_xlabel("Epoch")
        ax.set_title(title)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].set_ylim(0.82, 0.95)
    axes[1].set_ylim(0.15, 0.70)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    savefig(fig, "02_learning_curves.png")


def fig_comparison(runs, models):
    fig, ax = plt.subplots(figsize=(9, 0.62 * len(models) + 1.5))
    ypos = {m: i for i, m in enumerate(models)}
    for m in models:
        f1 = np.array([r["test_macro_f1"] for r in runs[m]]) * 100
        y = ypos[m]
        ax.scatter(f1, [y] * len(f1), s=26, color=COLOR[m], alpha=0.55, marker=MARKER[m], zorder=3)
        ax.scatter([f1.mean()], [y], s=95, color=COLOR[m], marker=MARKER[m], edgecolor=INK, linewidth=1, zorder=4)
        ax.text(f1.max() + 0.25, y, f"{f1.mean():.2f}" + (f"  (n={len(f1)})"), va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([LABEL[m] for m in models])
    ax.invert_yaxis()
    ax.set_xlabel("Test macro-F1 (%)  —  large marker = mean over seeds, small = individual seeds")
    ax.set_title("Test performance by model")
    ax.grid(axis="y", visible=False)
    lo = min(np.array([r["test_macro_f1"] for r in runs[m]]).min() for m in models) * 100
    ax.set_xlim(lo - 1.0, 97.5)
    fig.tight_layout()
    savefig(fig, "03_model_comparison.png")


def fig_confusion(runs, best):
    cm = np.sum([np.array(r["confusion_matrix"]) for r in runs[best]], axis=0).astype(float)
    pct = cm / cm.sum(1, keepdims=True) * 100
    cmap = LinearSegmentedColormap.from_list("seq", [SURFACE, "#9ec1ee", "#2a78d6", "#0d2f5c"])
    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    ax.imshow(pct, cmap=cmap, vmin=0, vmax=100)
    for i in range(10):
        for j in range(10):
            if pct[i, j] >= 0.5:
                ax.text(j, i, f"{pct[i, j]:.0f}" if pct[i, j] >= 1 else "<1", ha="center", va="center", fontsize=8,
                        color="white" if pct[i, j] > 55 else INK)
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xticklabels(CLASS_NAMES, rotation=40, ha="right"); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xticks(np.arange(-.5, 10, 1), minor=True); ax.set_yticks(np.arange(-.5, 10, 1), minor=True)
    ax.grid(which="minor", color=SURFACE, lw=2); ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
    ax.set_title(f"{LABEL[best]}: % of each true class (test set, summed over seeds)")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    savefig(fig, "04_confusion_matrix.png")
    return cm


def fig_per_class(runs, base, best):
    def f1s(m):
        return np.array([[r["per_class"][c]["f1"] for c in CLASS_NAMES] for r in runs[m]]).mean(0) * 100
    fb, fm = f1s(base), f1s(best)
    order = np.argsort(fm)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for k, c in enumerate(order):
        ax.plot([fb[c], fm[c]], [k, k], color=GRID, lw=2.5, zorder=1)
    ax.scatter(fb[order], range(10), s=70, color=COLOR[base], marker=MARKER[base], edgecolor=SURFACE, zorder=3, label=SHORT[base])
    ax.scatter(fm[order], range(10), s=70, color=COLOR[best], marker=MARKER[best], edgecolor=SURFACE, zorder=3, label=SHORT[best])
    ax.set_yticks(range(10)); ax.set_yticklabels([CLASS_NAMES[c] for c in order])
    ax.set_xlabel("Test F1 per class (%)")
    ax.set_title("Per-class F1: baseline vs. best model")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    savefig(fig, "05_per_class_f1.png")


def fig_errors(run, x_test, y_test, n=10):
    wrong = np.where(run["pred"] != y_test)[0]
    top = wrong[np.argsort(-run["conf"][wrong])[:n]]
    fig, axes = plt.subplots(2, 5, figsize=(10, 5.6))
    for ax, i in zip(axes.ravel(), top):
        ax.imshow(x_test[i], cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"true: {CLASS_NAMES[y_test[i]]}\npred: {CLASS_NAMES[run['pred'][i]]} ({run['conf'][i]:.2f})",
                     fontsize=8.5, fontweight="normal", loc="center")
        ax.axis("off")
    fig.suptitle(f"Most confident mistakes of {LABEL[run['model']]} (seed {run['seed']}, test set; predicted probability in brackets)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(h_pad=2.2)
    savefig(fig, "06_confident_errors.png")


# --------------------------------------------------------------------------- main
def main():
    runs = load_runs()
    models = [m for m in ORDER if m in runs]
    _, _, x_test, y_test = load_raw()

    # best model = highest mean *validation* macro-F1 over all candidates (test set is not used for selection)
    best = max(models, key=lambda m: np.mean([r["val_macro_f1"] for r in runs[m]]))
    base = "logreg"

    rows = []
    summary = {"best_model_by_validation": best, "models": {}}
    for m in models:
        rs = runs[m]
        acc, f1 = [r["test_accuracy"] for r in rs], [r["test_macro_f1"] for r in rs]
        summary["models"][m] = {
            "n_seeds": len(rs), "params": rs[0]["params"], "epochs": rs[0]["config"]["epochs"],
            "test_accuracy_mean": float(np.mean(acc)), "test_accuracy_std": float(np.std(acc, ddof=1)) if len(rs) > 1 else None,
            "test_macro_f1_mean": float(np.mean(f1)), "test_macro_f1_std": float(np.std(f1, ddof=1)) if len(rs) > 1 else None,
            "val_macro_f1_mean": float(np.mean([r["val_macro_f1"] for r in rs])),
            "best_epoch_mean": float(np.mean([r["best_epoch"] for r in rs])),
            "train_time_min_mean": float(np.mean([r["train_time_s"] for r in rs]) / 60),
            "infer_ms_per_image_mean": float(np.mean([r["infer_ms_per_image"] for r in rs])),
        }
        gap = [(r["history"]["train_acc"][r["best_epoch"] - 1] - r["history"]["val_acc"][r["best_epoch"] - 1]) * 100
               for r in rs]
        summary["models"][m]["train_minus_val_acc_pts"] = float(np.mean(gap))
        rows.append(
            f"| {LABEL[m]} | {rs[0]['params']:,} | {rs[0]['config']['epochs']} | {len(rs)} | "
            f"{ms(acc)} | {ms(f1)} | {np.mean([r['best_epoch'] for r in rs]):.1f} | {np.mean(gap):+.1f} | "
            f"{np.mean([r['train_time_s'] for r in rs]) / 60:.1f} | {np.mean([r['infer_ms_per_image'] for r in rs]):.3f} |")

    lines = ["# Results summary", "",
             "Test metrics are for the checkpoint with the best validation macro-F1 (mean ± sample std over seeds; "
             "std omitted when n=1). 'Gap' = training accuracy minus validation accuracy at the selected epoch "
             "(for augmented models, training accuracy is measured on augmented batches, so it is not comparable). "
             "Train time is wall-clock per run on 1 CPU thread (two jobs shared a 2-core machine).", "",
             "| Model | Params | Epochs | Seeds | Test acc (%) | Test macro-F1 (%) | Best epoch | Gap (pts) | Train time (min) | Inference (ms/img) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|", *rows, "",
             f"Best model by mean validation macro-F1: **{LABEL[best]}**.", ""]

    # McNemar tests (seed 0 predictions, paired on the 10k test images)
    y = y_test.astype(int)
    pairs = []
    ranked = sorted(models, key=lambda m: -summary["models"][m]["test_macro_f1_mean"])
    pairs += [(ranked[0], ranked[1])] if len(ranked) > 1 else []
    for a, b in (("logreg", best), ("mlp", "cnn"), ("cnn", "resnet"), ("resnet", "vit"), ("resnet_noaug", "resnet")):
        if a in runs and b in runs and (a, b) not in pairs and (b, a) not in pairs:
            pairs.append((a, b))
    lines += ["## Paired significance (exact McNemar test, seed 0, 10,000 test images)", "",
              "| A | B | only A correct | only B correct | p-value |", "|---|---|---:|---:|---:|"]
    summary["mcnemar"] = []
    for a, b in pairs:
        pa = next(r for r in runs[a] if r["seed"] == 0)["pred"]
        pb = next(r for r in runs[b] if r["seed"] == 0)["pred"]
        oa, ob, p = mcnemar(pa, pb, y)
        lines.append(f"| {LABEL[a]} | {LABEL[b]} | {oa} | {ob} | {p:.2g} |")
        summary["mcnemar"].append({"a": a, "b": b, "only_a": oa, "only_b": ob, "p": p})
    lines.append("")

    # figures
    fig_samples(x_test, y_test)
    fig_curves(runs, models)
    fig_comparison(runs, models)
    cm = fig_confusion(runs, best)
    fig_per_class(runs, base, best)
    fig_errors(next(r for r in runs[best] if r["seed"] == 0), x_test, y_test)

    # most confused class pairs of the best model
    off = [(cm[i, j], i, j) for i in range(10) for j in range(10) if i != j]
    off.sort(reverse=True)
    lines += [f"## Most frequent confusions — {LABEL[best]} (summed over seeds)", "",
              "| True class | Predicted as | Count | % of true class |", "|---|---|---:|---:|"]
    summary["top_confusions"] = []
    for cnt, i, j in off[:6]:
        pct = cnt / cm[i].sum() * 100
        lines.append(f"| {CLASS_NAMES[i]} | {CLASS_NAMES[j]} | {int(cnt)} | {pct:.1f} |")
        summary["top_confusions"].append({"true": CLASS_NAMES[i], "pred": CLASS_NAMES[j], "count": int(cnt), "pct_of_true": pct})
    lines.append("")
    per_class = {c: float(np.mean([r["per_class"][c]["f1"] for r in runs[best]])) for c in CLASS_NAMES}
    lines += [f"## Per-class F1 — {LABEL[best]} (mean over seeds)", "", "| Class | F1 (%) |", "|---|---:|",
              *[f"| {c} | {v * 100:.2f} |" for c, v in sorted(per_class.items(), key=lambda kv: kv[1])], ""]
    summary["per_class_f1_best"] = per_class

    (RES / "summary.md").write_text("\n".join(lines))
    (RES / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
