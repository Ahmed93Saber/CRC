"""
Publication figures for comparing several models across several metrics, showing
95% confidence intervals rather than a raw scatter of fold scores.

Two figures
-----------
panel_figure()  : your existing layout (one panel per metric), but each model is
                  drawn as ENSEMBLE point estimate + patient-bootstrap 95% CI.
                  Faint fold dots are kept so training stability is still visible.
                  Asterisks mark paired significance vs the baseline ("Ours").

forest_figure() : the significance-faithful companion. For each metric it plots the
                  paired difference (baseline - other) with its bootstrap 95% CI.
                  A CI that does not cross 0 is a significant paired difference --
                  this is the correct way to read significance off a plot, because
                  it uses the PAIRED interval, not two marginal ones.

Data contract
-------------
results[model][metric] = {
    "point" : float,            # ensemble metric on the external cohort
    "lo"    : float, "hi": float,   # patient-bootstrap 95% CI on that ensemble
    "folds" : 1-D array or None,    # the individual fold-model scores (optional)
}
diffs[model][metric] = {"point":..., "lo":..., "hi":..., "p":...}
    # baseline - model, its bootstrap 95% CI, and the paired permutation p-value

compute_results() / compute_diffs() below build these from raw probability arrays
using the companion multiclass_paired_comparison module, so you can go straight
from (y, probs per model) to the figures.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _stars(p):
    if p is None or not np.isfinite(p):
        return ""
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"


def panel_figure(results, metrics, models, baseline=None, sig=None,
                 ncols=2, figsize=None, title="Comparison of metrics across models",
                 point_color="#c0392b", ci_color="#7fa8c9",
                 shared_ylim=None, row_shared_ylim=False, savepath=None):
    dot_color = "#34495e"
    n = len(metrics)
    nrows = int(np.ceil(n / ncols))
    if figsize is None:
        figsize = (5.2 * ncols, 3.4 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()

    def _range_for(metric_list):
        vals = []
        for metric in metric_list:
            for model in models:
                r = results[model][metric]
                vals += [r["lo"], r["hi"]]
                f = r.get("folds")
                if f is not None and len(f):
                    vals += list(np.asarray(f))
        lo, hi = min(vals), max(vals)
        rng = hi - lo or 1.0
        return (lo - rng * 0.08, hi + rng * 0.16)   # bottom pad + star headroom

    # y-limit strategy, in priority order:
    #   row_shared_ylim : each ROW shares one range (group similar metrics per row)
    #   shared_ylim=True: every panel shares one range
    #   shared_ylim=list: the named metrics share one range
    #   else            : per-panel autoscale
    if shared_ylim is True:
        shared_metrics = list(metrics)
    elif shared_ylim:
        shared_metrics = list(shared_ylim)
    else:
        shared_metrics = []
    ylim_shared = _range_for(shared_metrics) if shared_metrics else None

    row_ylim = {}
    if row_shared_ylim:
        for row in range(nrows):
            row_metrics = metrics[row * ncols:(row + 1) * ncols]
            if row_metrics:
                row_ylim[row] = _range_for(row_metrics)

    x = np.arange(len(models))
    for ai, metric in enumerate(metrics):
        ax = axes[ai]
        for xi, model in zip(x, models):
            r = results[model][metric]
            folds = r.get("folds")
            if folds is not None and len(folds):
                jit = (np.random.default_rng(xi).random(len(folds)) - 0.5) * 0.16
                ax.scatter(np.full(len(folds), xi) + jit, folds, s=22,
                           color=dot_color, alpha=0.55, zorder=2, edgecolors="none")
            lo, hi, pt = r["lo"], r["hi"], r["point"]
            ax.plot([xi, xi], [lo, hi], color=ci_color, lw=2.2, zorder=3,
                    solid_capstyle="round")
            ax.plot([xi - 0.08, xi + 0.08], [lo, lo], color=ci_color, lw=2.2, zorder=3)
            ax.plot([xi - 0.08, xi + 0.08], [hi, hi], color=ci_color, lw=2.2, zorder=3)
            ax.scatter([xi], [pt], marker="_", s=95, color=point_color, zorder=5)
            if sig is not None and baseline is not None and model != baseline:
                p = sig.get(model, {}).get(metric)
                s = _stars(p)
                if s:
                    ax.annotate(s, (xi, hi), textcoords="offset points",
                                xytext=(0, 5), ha="center", fontsize=9, color="#2c3e50")

        ax.set_title(metric, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha="right", fontsize=9)
        ax.grid(axis="y", ls="--", alpha=0.35)
        ax.margins(x=0.12)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

        # y-limits: row-shared > shared_metrics > per-panel autoscale
        row = ai // ncols
        if row_shared_ylim and row in row_ylim:
            ax.set_ylim(*row_ylim[row])
            if ai % ncols != 0:
                ax.tick_params(axis="y", labelleft=False)
        elif metric in shared_metrics and ylim_shared is not None:
            ax.set_ylim(*ylim_shared)
            if ai % ncols != 0:
                ax.tick_params(axis="y", labelleft=False)
        else:
            ax.set_ylim(*_range_for([metric]))

    for j in range(n, len(axes)):
        axes[j].axis("off")

    handles = [
        Line2D([0], [0], marker="_", color="none", markerfacecolor=point_color,
               markeredgecolor=point_color, markersize=11, label="Ensemble (point est.)"),
        Line2D([0], [0], color=ci_color, lw=2.2, label="95% CI (patient bootstrap)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=dot_color,
               markersize=7, label="Individual fold scores", alpha=0.7),
    ]
    if n < len(axes):
        lax = axes[n]
        lax.axis("off")
        lax.legend(handles=handles, loc="center", frameon=False, fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
    else:
        fig.tight_layout(rect=[0, 0.06, 1, 0.98])
        fig.legend(handles=handles, loc="lower center", ncol=3,
                   frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.0))
    if savepath:
        fig.savefig(savepath, dpi=170, bbox_inches="tight")
    return fig


def forest_figure(diffs, metrics, others, baseline_name="Ours",
                  figsize=None, savepath=None,
                  title=None):
    """Forest plot of paired differences (baseline - other) with 95% CI.

    A CI not crossing the dashed zero line = significant paired difference.
    This is the CORRECT way to read significance off a figure for paired tests;
    positive means the baseline is better on that metric.
    """
    if title is None:
        title = f"Paired difference ({baseline_name} \u2212 baseline), 95% CI"
    n = len(metrics)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    if figsize is None:
        figsize = (4.6 * ncols, 2.6 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.ravel()

    y = np.arange(len(others))[::-1]
    for ai, metric in enumerate(metrics):
        ax = axes[ai]
        ax.axvline(0, color="#888", ls="--", lw=1)
        for yi, model in zip(y, others):
            d = diffs[model][metric]
            sig = (d["lo"] > 0) or (d["hi"] < 0)
            col = "#c0392b" if sig else "#95a5a6"
            ax.plot([d["lo"], d["hi"]], [yi, yi], color=col, lw=2.2,
                    solid_capstyle="round")
            ax.scatter([d["point"]], [yi], color=col, s=45, zorder=5)
            star = _stars(d.get("p"))
            if star and star != "ns":
                ax.annotate(star, (d["hi"], yi), textcoords="offset points",
                            xytext=(6, 0), va="center", fontsize=9, color=col)
        ax.set_yticks(y)
        ax.set_yticklabels(others, fontsize=9)
        ax.set_title(metric, fontsize=11)
        ax.grid(axis="x", ls="--", alpha=0.3)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)

    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle(title, fontsize=13, y=1.0)
    fig.text(0.5, -0.01, f"positive \u2192 {baseline_name} better   |   "
             "CI clear of 0 = significant paired difference",
             ha="center", fontsize=9, color="#555")
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=170, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Building the data from raw probabilities (uses multiclass_paired_comparison)
# ---------------------------------------------------------------------------

def _multi_arm_ci(y, P, metric_fns, n_boot, seed, alpha=0.05):
    """Patient-bootstrap CIs for SEVERAL metrics on one model at once.

    Draws each resample once and scores every metric on it, so the expensive part
    (index generation) is shared. Returns {metric_name: (lo, hi)}.
    """
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    blocks = [np.flatnonzero(y == k) for k in np.unique(y)]
    names = list(metric_fns.keys())
    vals = {nm: np.empty(n_boot) for nm in names}
    for i in range(n_boot):
        idx = np.concatenate([b[rng.integers(0, len(b), len(b))] for b in blocks])
        yi, Pi = y[idx], P[idx]
        for nm in names:
            vals[nm][i] = metric_fns[nm](yi, Pi)
    return {nm: tuple(np.percentile(vals[nm],
                      [100 * alpha / 2, 100 * (1 - alpha / 2)])) for nm in names}


def _results_for_model(args):
    (model, yy, ens, probs, metric_items, n_boot, seed) = args
    import src.utils_stats as mpc  # noqa: F401 (kept for parity)
    metric_fns = dict(metric_items)
    cis = _multi_arm_ci(yy, ens, metric_fns, n_boot, seed)
    res = {}
    for name, fn in metric_items:
        folds = np.array([fn(yy, probs[f]) for f in range(probs.shape[0])])
        lo, hi = cis[name]
        res[name] = {"point": float(fn(yy, ens)), "lo": float(lo),
                     "hi": float(hi), "folds": folds}
    return model, res


def compute_results(y, probs_by_model, metrics_by_name, ordinal_order=None,
                    n_boot=10000, ensemble_method="mean", seed=0, n_jobs=1):
    """Build the results dict for panel_figure from raw arrays.

    y               : (n,) labels in ORIGINAL coding
    probs_by_model  : {model_name: (n_folds, n, K) probabilities}
    metrics_by_name : {display_name: metric_fn(y, P)->float}
    ordinal_order   : e.g. [3,0,1,2] to remap into rank space, or None
    n_jobs          : >1 parallelises across models.
    """
    import src.utils_stats as mpc

    metric_items = list(metrics_by_name.items())
    jobs = []
    for mi, (model, probs) in enumerate(probs_by_model.items()):
        probs = np.asarray(probs, float)
        if ordinal_order is not None:
            yy, probs = mpc.remap_to_ordinal(y, probs, ordinal_order)
        else:
            yy = np.asarray(y)
        ens = mpc.ensemble_folds(probs, ensemble_method)
        jobs.append((model, yy, ens, probs, metric_items, n_boot, seed + 1000 * mi))

    out = {}
    if n_jobs and n_jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            for model, res in ex.map(_results_for_model, jobs):
                out[model] = res
    else:
        for job in jobs:
            model, res = _results_for_model(job)
            out[model] = res
    return out


def _diff_for_model(args):
    (model, yy, base_ens, oth_ens, metric_items, n_boot, n_perm, seed) = args
    import src.utils_stats as mpc
    res = {}
    for di, (name, fn) in enumerate(metric_items):
        b = mpc.paired_bootstrap(yy, base_ens, oth_ens, metric=fn,
                                 n_boot=n_boot, seed=seed + di)
        perm = mpc.paired_permutation(yy, base_ens, oth_ens, metric=fn,
                                      n_perm=n_perm, seed=seed + di)
        res[name] = {"point": b["difference"], "lo": b["ci_95"][0],
                     "hi": b["ci_95"][1], "p": perm["p_value"]}
    return model, res


def compute_diffs(y, probs_by_model, metrics_by_name, baseline,
                  ordinal_order=None, n_boot=10000, n_perm=10000,
                  ensemble_method="mean", seed=0, n_jobs=1):
    """Build the diffs dict (baseline - other) for forest_figure.

    n_jobs > 1 runs the per-model work in parallel processes. Each model is
    independent, so this scales almost linearly up to the number of non-baseline
    models. n_jobs=1 keeps it single-process (easier to debug / no import quirks).
    """
    import src.utils_stats as mpc

    probs_by_model = {k: np.asarray(v, float) for k, v in probs_by_model.items()}
    if ordinal_order is not None:
        yy, base = mpc.remap_to_ordinal(y, probs_by_model[baseline], ordinal_order)
    else:
        yy, base = np.asarray(y), probs_by_model[baseline]
    base_ens = mpc.ensemble_folds(base, ensemble_method)

    metric_items = list(metrics_by_name.items())
    jobs = []
    for mi, (model, probs) in enumerate(probs_by_model.items()):
        if model == baseline:
            continue
        if ordinal_order is not None:
            _, pr = mpc.remap_to_ordinal(y, probs, ordinal_order)
        else:
            pr = probs
        oth_ens = mpc.ensemble_folds(pr, ensemble_method)
        jobs.append((model, yy, base_ens, oth_ens, metric_items,
                     n_boot, n_perm, seed + 7 * mi))

    out = {}
    if n_jobs and n_jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            for model, res in ex.map(_diff_for_model, jobs):
                out[model] = res
    else:
        for job in jobs:
            model, res = _diff_for_model(job)
            out[model] = res
    return out


# ---------------------------------------------------------------------------
# One-call wrapper: raw probabilities -> both figures
# ---------------------------------------------------------------------------

def default_metrics():
    """The five metrics from the reference figure, as {display_name: fn}."""
    import src.utils_stats as mpc
    return {"Accuracy": mpc.accuracy,
            "Balanced Accuracy": mpc.balanced_accuracy,
            "F1": mpc.weighted_f1,
            "QWK": mpc.quadratic_kappa,
            "AUC": mpc.macro_auroc_ovr,
            "Recall": mpc.macro_recall,
            "Precision": mpc.macro_precision,}


def figures_from_predictions(y, probs_by_model, baseline,
                             metrics=None, ordinal_order=None,
                             ensemble_method="mean", n_boot=10000, n_perm=10000,
                             seed=0, save_prefix=None, n_jobs=1, shared_ylim=None):
    """Everything from raw predictions to both figures in one call.

    Parameters
    ----------
    y               : (n,) ground-truth labels in ORIGINAL coding.
    probs_by_model  : dict {model_name: array of shape (n_folds, n, K)}.
                      Insertion order sets the left-to-right order on the plots,
                      so put your own method first.
    baseline        : the model name every paired comparison is made against
                      (e.g. "Ours"); it is the reference in the forest plot and
                      the anchor for the significance stars on the panel plot.
    metrics         : {display_name: metric_fn(y, P)->float}; defaults to the five
                      in default_metrics(). For ordinal metrics (QWK) pass
                      ordinal_order so the remap happens.
    ordinal_order   : e.g. [3, 0, 1, 2], or None for nominal classes.
    n_boot, n_perm  : bootstrap / permutation replicate counts (use >=10000 for a
                      paper; lower only for a quick look).
    save_prefix     : if given, writes '<prefix>_panel.png' and '<prefix>_forest.png'.

    Returns
    -------
    dict with keys: results, diffs, sig, fig_panel, fig_forest.
    """
    y = np.asarray(y)
    models = list(probs_by_model.keys())
    if baseline not in models:
        raise ValueError(f"baseline {baseline!r} not among models {models}")

    shapes = {m: np.asarray(p).shape for m, p in probs_by_model.items()}
    ndims = {len(s) for s in shapes.values()}
    if ndims != {3}:
        raise ValueError(f"every model needs a 3-D (n_folds, n, K) array; got {shapes}")
    ns = {s[1] for s in shapes.values()}
    if ns != {len(y)}:
        raise ValueError(f"patient axis must equal len(y)={len(y)}; got {shapes}")

    if metrics is None:
        metrics = default_metrics()
    metric_names = list(metrics.keys())

    results = compute_results(y, probs_by_model, metrics,
                              ordinal_order=ordinal_order, n_boot=n_boot,
                              ensemble_method=ensemble_method, seed=seed,
                              n_jobs=n_jobs)
    diffs = compute_diffs(y, probs_by_model, metrics, baseline=baseline,
                          ordinal_order=ordinal_order, n_boot=n_boot,
                          n_perm=n_perm, ensemble_method=ensemble_method,
                          seed=seed, n_jobs=n_jobs)

    # significance dict for the panel stars, read straight off the paired diffs
    sig = {m: {} for m in models}
    for m in models:
        if m == baseline:
            continue
        for name in metric_names:
            sig[m][name] = diffs[m][name]["p"]

    fig_panel = panel_figure(
        results, metric_names, models, baseline=baseline, sig=sig,
        shared_ylim=shared_ylim,
        savepath=(f"{save_prefix}_panel.png" if save_prefix else None))
    others = [m for m in models if m != baseline]
    fig_forest = forest_figure(
        diffs, metric_names, others, baseline_name=baseline,
        savepath=(f"{save_prefix}_forest.png" if save_prefix else None))

    return {"results": results, "diffs": diffs, "sig": sig,
            "fig_panel": fig_panel, "fig_forest": fig_forest}


def _bin_edges(scores, n_bins, strategy):
    if strategy == "uniform":
        return np.linspace(0.0, 1.0, n_bins + 1)
    if strategy == "quantile":
        q = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.unique(np.quantile(scores, q))
        edges[0], edges[-1] = 0.0, 1.0 + 1e-9  # cover the ends
        return edges
    raise ValueError("strategy must be 'uniform' or 'quantile'")


def _binned(scores, correct, n_bins, strategy):
    """Return per-bin (mean_score, mean_correct, count) dropping empty bins."""
    edges = _bin_edges(scores, n_bins, strategy)
    idx = np.clip(np.digitize(scores, edges[1:-1], right=False), 0, len(edges) - 2)
    conf, acc, cnt = [], [], []
    for b in range(len(edges) - 1):
        m = idx == b
        c = int(m.sum())
        if c:
            conf.append(scores[m].mean())
            acc.append(correct[m].mean())
            cnt.append(c)
    return np.array(conf), np.array(acc), np.array(cnt)


# ---------------------------------------------------------------------------
# ECE metrics  (lower is better)
# ---------------------------------------------------------------------------

def confidence_ece(y, P, n_bins=10, strategy="quantile") -> float:
    """Top-label expected calibration error. Lower is better; 0 is perfect."""
    y = np.asarray(y);
    P = np.asarray(P, float)
    conf = P.max(axis=1)
    correct = (P.argmax(axis=1) == y).astype(float)
    c, a, n = _binned(conf, correct, n_bins, strategy)
    return float(np.sum(n / n.sum() * np.abs(a - c))) if len(n) else np.nan


def confidence_mce(y, P, n_bins=10, strategy="quantile") -> float:
    """Maximum calibration error: the worst single bin's gap. Lower is better."""
    y = np.asarray(y);
    P = np.asarray(P, float)
    conf = P.max(axis=1)
    correct = (P.argmax(axis=1) == y).astype(float)
    c, a, n = _binned(conf, correct, n_bins, strategy)
    return float(np.max(np.abs(a - c))) if len(n) else np.nan


def classwise_ece(y, P, n_bins=10, strategy="quantile", weighted=False) -> float:
    """Mean over classes of one-vs-rest calibration error. Lower is better.
    weighted=True weights classes by prevalence instead of equally."""
    y = np.asarray(y);
    P = np.asarray(P, float)
    K = P.shape[1]
    eces, support = [], []
    for k in range(K):
        pk = P[:, k]
        yk = (y == k).astype(float)
        c, a, n = _binned(pk, yk, n_bins, strategy)
        if len(n):
            eces.append(np.sum(n / n.sum() * np.abs(a - c)))
            support.append((y == k).sum())
    if not eces:
        return np.nan
    eces = np.array(eces);
    support = np.array(support, float)
    if weighted and support.sum() > 0:
        return float(np.average(eces, weights=support))
    return float(eces.mean())


def neg_confidence_ece(y, P, **kw) -> float:
    """Higher-is-better wrapper, for feeding the paired-comparison framework
    (which treats larger differences as favouring the first model). A positive
    difference then means the first model is BETTER calibrated (lower ECE)."""
    return -confidence_ece(y, P, **kw)


# ---------------------------------------------------------------------------
# Reliability curve data + bootstrap CI on ECE
# ---------------------------------------------------------------------------

def reliability_curve(y, P, n_bins=10, strategy="quantile"):
    """Per-bin data for a top-label reliability curve.
    Returns dict: mean_conf, mean_acc, count, ece."""
    y = np.asarray(y);
    P = np.asarray(P, float)
    conf = P.max(axis=1)
    correct = (P.argmax(axis=1) == y).astype(float)
    c, a, n = _binned(conf, correct, n_bins, strategy)
    ece = float(np.sum(n / n.sum() * np.abs(a - c))) if len(n) else np.nan
    return {"mean_conf": c, "mean_acc": a, "count": n, "ece": ece}


def ece_ci(y, P, metric=confidence_ece, n_boot=10000, seed=0, alpha=0.05, **kw):
    """Patient-bootstrap 95% CI for an ECE metric, stratified by class (matches the
    rest of the pipeline). Returns (point, lo, hi)."""
    y = np.asarray(y);
    P = np.asarray(P, float)
    rng = np.random.default_rng(seed)
    blocks = [np.flatnonzero(y == k) for k in np.unique(y)]
    vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([b[rng.integers(0, len(b), len(b))] for b in blocks])
        vals[i] = metric(y[idx], P[idx], **kw)
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(metric(y, P, **kw)), float(lo), float(hi)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def reliability_diagram(y, probs_by_model, ensemble_method="mean",
                        n_bins=10, strategy="quantile", title="Reliability",
                        colors=None, savepath=None):
    """Overlay reliability curves for several models, with a confidence histogram.

    probs_by_model : {model_name: (n_folds, n, K) OR (n, K)}. 3-D arrays are
                     ensembled (mean prob across folds) first.
    Returns the matplotlib figure. ECE (top-label) is shown per model in the legend.
    """
    import src.utils_stats as mpc

    y = np.asarray(y)
    if colors is None:
        colors = ["#c0392b", "#2980b9", "#27ae60", "#8e44ad", "#e67e22", "#16a085"]

    fig, (ax, axh) = plt.subplots(
        2, 1, figsize=(5.4, 6.2), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
        constrained_layout=True)

    ax.plot([0, 1], [0, 1], ls="--", color="#888", lw=1, label="Perfect calibration")

    for i, (name, probs) in enumerate(probs_by_model.items()):
        probs = np.asarray(probs, float)
        P = mpc.ensemble_folds(probs, ensemble_method) if probs.ndim == 3 else probs
        rc = reliability_curve(y, P, n_bins=n_bins, strategy=strategy)
        col = colors[i % len(colors)]
        ax.plot(rc["mean_conf"], rc["mean_acc"], "-o", color=col, lw=1.8, ms=5,
                label=f"{name} (ECE={rc['ece']:.3f})", zorder=3)
        # confidence histogram (step outline so overlaps stay readable)
        conf = P.max(axis=1)
        axh.hist(conf, bins=np.linspace(conf.min(), 1.0, n_bins + 1),
                 histtype="step", color=col, lw=1.5, alpha=0.8)

    ax.set_ylabel("Observed accuracy")
    ax.set_xlim(0, 1);
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(ls="--", alpha=0.35)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5)
    ax.set_title(title, fontsize=12)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    axh.set_xlabel("Predicted confidence (max probability)")
    axh.set_ylabel("Count")
    axh.grid(axis="y", ls="--", alpha=0.35)
    for sp in ("top", "right"):
        axh.spines[sp].set_visible(False)

    if savepath:
        fig.savefig(savepath, dpi=170, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Demo: generate raw (5, 200, 4) predictions per model, then ONE call
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n_folds, n, K = 5, 200, 4
    ORDINAL_ORDER = [3, 0, 1, 2]          # scale runs 3 < 0 < 1 < 2

    # shared ground truth + latent severity, so models are comparable
    ranks = rng.choice(K, size=n, p=[0.30, 0.30, 0.25, 0.15])
    y = np.array(ORDINAL_ORDER)[ranks]
    latent = ranks + rng.normal(0, 0.8, size=n)

    def make_model(noise):
        """A model = 5 fold prediction arrays of shape (n, K), noisier => worse."""
        d = np.abs(latent[None, :, None] - np.arange(K)[None, None, :])
        z = -d ** 2 / 1.2 + rng.normal(0, noise, size=(n_folds, n, K))
        e = np.exp(z - z.max(axis=2, keepdims=True))
        p = e / e.sum(axis=2, keepdims=True)
        return p[..., np.argsort(ORDINAL_ORDER)]     # back to original coding

    # different noise levels -> a clear ranking, "Ours" best
    probs_by_model = {
        "Ours":         make_model(0.8),
        "MoE+ABMIL+CE": make_model(1.0),
        "MoE+CLAM+CE":  make_model(1.3),
        "ABMIL+CE":     make_model(1.5),
        "CLAM+CE":      make_model(1.7),
    }

    out = figures_from_predictions(
        y, probs_by_model, baseline="Ours",
        ordinal_order=ORDINAL_ORDER,
        n_boot=2000, n_perm=2000,          # bump to >=10000 for the paper
        n_jobs=1,                          # set to n(models) on a multi-core machine
        save_prefix="/home/claude/from_preds")

    # quick text summary so the numbers are visible too
    print(f"{'model':<15}{'Accuracy':>20}{'QWK':>20}{'AUC':>20}")
    for m in probs_by_model:
        row = out["results"][m]
        def cell(k):
            r = row[k]; return f"{r['point']:.3f} [{r['lo']:.3f},{r['hi']:.3f}]"
        print(f"{m:<15}{cell('Accuracy'):>20}{cell('QWK'):>20}{cell('AUC'):>20}")
    print("\nwrote from_preds_panel.png and from_preds_forest.png")