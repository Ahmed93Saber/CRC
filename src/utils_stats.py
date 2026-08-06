"""
Paired comparison of two modelling methodologies on a single external cohort.
MULTICLASS version (K classes; written and tested for K = 4).

Setup this assumes
------------------
Each methodology was trained with 5-fold CV internally, giving 5 fold-models.
All 5 score every patient in the SAME external cohort.
The deployable predictor is the ensemble (mean class-probability across folds).

Inputs
------
y     : (n,)          integer labels in {0, ..., K-1}
probs : (n_folds, n, K)   class probabilities, rows summing to 1
        e.g. (5, 200, 4)

What changes vs the binary case
-------------------------------
1. DeLong has no multiclass form. It is applied one-vs-rest, once per class, and
   the K p-values are then corrected for multiplicity. There is NO closed-form
   test for macro-averaged AUC -- use the bootstrap for that.
2. The bootstrap is stratified across all K classes, so every replicate keeps the
   exact class counts and no class can vanish (which would make OvR AUC undefined).
3. Metrics are macro-averaged by default. Macro treats all classes equally and is
   the honest choice under imbalance; weighted/micro variants are provided too.

Dependencies: numpy, scipy, scikit-learn
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    cohen_kappa_score,
    f1_score,
    balanced_accuracy_score,
)
from sklearn.preprocessing import label_binarize


# ---------------------------------------------------------------------------
# 1. Ensembling the folds
# ---------------------------------------------------------------------------

def ensemble_folds(probs: np.ndarray, method: str = "mean") -> np.ndarray:
    """(n_folds, n, K) -> (n, K), renormalised so each row sums to 1.

    method="mean"      arithmetic mean of probabilities. The default, and what you
                       would actually deploy. Preserves calibration reasonably.
    method="geometric" mean in log space (equivalently, geometric mean renormalised).
                       Sharper than the arithmetic mean and less forgiving of a
                       single confident-but-wrong fold. Use if the folds disagree a
                       lot; it is the natural "average of logits" for K > 2.

    Whichever you pick, use the SAME one for both methodologies.
    """
    P = np.asarray(probs, dtype=float)
    if P.ndim != 3:
        raise ValueError("probs must be 3-D with shape (n_folds, n_samples, n_classes)")

    if method == "mean":
        out = P.mean(axis=0)
    elif method == "geometric":
        eps = 1e-12
        out = np.exp(np.log(np.clip(P, eps, 1.0)).mean(axis=0))
    else:
        raise ValueError(f"unknown method: {method}")

    return out / out.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# 2. Metrics.  Signature is metric(y, P) -> float, higher = better.
#    y is (n,) integer labels; P is (n, K) probabilities.
# ---------------------------------------------------------------------------

def _fast_binary_auc(y_bin: np.ndarray, s: np.ndarray) -> float:
    """Mann-Whitney AUC with tie correction. Much faster than sklearn in a loop."""
    n1 = int(y_bin.sum())
    n0 = len(y_bin) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = stats.rankdata(s)
    return (r[y_bin == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def macro_auroc_ovr(y, P) -> float:
    """Mean over classes of the one-vs-rest AUROC. The usual headline number."""
    y = np.asarray(y)
    P = np.asarray(P, float)
    return float(np.nanmean([_fast_binary_auc((y == k).astype(int), P[:, k])
                             for k in range(P.shape[1])]))


def weighted_auroc_ovr(y, P) -> float:
    """OvR AUROC weighted by class prevalence."""
    y = np.asarray(y)
    P = np.asarray(P, float)
    K = P.shape[1]
    aucs = np.array([_fast_binary_auc((y == k).astype(int), P[:, k]) for k in range(K)])
    w = np.array([(y == k).sum() for k in range(K)], dtype=float)
    ok = ~np.isnan(aucs)
    return float(np.average(aucs[ok], weights=w[ok]))


def macro_auroc_ovo(y, P) -> float:
    """Hand & Till one-vs-one macro AUC. Insensitive to class prevalence, so it is
    the better choice if the external cohort's class mix differs from the internal
    one. Hand-rolled because sklearn's multi_class='ovo' path is ~12x slower, which
    matters when it is called 10,000 times inside a bootstrap."""
    y = np.asarray(y)
    P = np.asarray(P, float)
    K = P.shape[1]
    vals = []
    for i in range(K):
        for j in range(i + 1, K):
            mask = (y == i) | (y == j)
            if not mask.any():
                continue
            yi = (y[mask] == i).astype(int)
            a_ij = _fast_binary_auc(yi, P[mask, i])          # i vs j using column i
            a_ji = _fast_binary_auc(1 - yi, P[mask, j])      # j vs i using column j
            vals.append(np.nanmean([a_ij, a_ji]))
    return float(np.nanmean(vals))


def per_class_auroc(k: int):
    """Factory: OvR AUROC for one specific class."""
    def _f(y, P):
        return _fast_binary_auc((np.asarray(y) == k).astype(int), np.asarray(P)[:, k])
    _f.__name__ = f"auroc_class{k}"
    return _f


def macro_auprc(y, P) -> float:
    P = np.asarray(P, float)
    Y = label_binarize(np.asarray(y), classes=np.arange(P.shape[1]))
    return float(average_precision_score(Y, P, average="macro"))


def accuracy(y, P) -> float:
    return float((np.asarray(P).argmax(axis=1) == np.asarray(y)).mean())


def _confusion(y, pred, K):
    """K x K confusion matrix via a single bincount; far faster than sklearn in a
    bootstrap loop. Rows = true class, cols = predicted class."""
    return np.bincount(y * K + pred, minlength=K * K).reshape(K, K)


def balanced_accuracy(y, P) -> float:
    y = np.asarray(y); K = np.asarray(P).shape[1]
    C = _confusion(y, np.asarray(P).argmax(axis=1), K).astype(float)
    per_class_recall = np.divide(np.diag(C), C.sum(axis=1),
                                 out=np.zeros(K), where=C.sum(axis=1) > 0)
    present = C.sum(axis=1) > 0
    return float(per_class_recall[present].mean())


def macro_f1(y, P) -> float:
    y = np.asarray(y); K = np.asarray(P).shape[1]
    C = _confusion(y, np.asarray(P).argmax(axis=1), K).astype(float)
    tp = np.diag(C)
    fp = C.sum(axis=0) - tp
    fn = C.sum(axis=1) - tp
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros(K), where=denom > 0)
    return float(f1.mean())


def weighted_f1(y, P) -> float:
    """Per-class F1 averaged with weights equal to each class's true-count (support).
    This is sklearn's average='weighted'. Under class imbalance it tracks the common
    classes and, unlike macro F1, is not dragged down by a rare class the model
    handles poorly -- so report whichever matches the claim you want to make."""
    y = np.asarray(y); K = np.asarray(P).shape[1]
    C = _confusion(y, np.asarray(P).argmax(axis=1), K).astype(float)
    tp = np.diag(C)
    fp = C.sum(axis=0) - tp
    fn = C.sum(axis=1) - tp
    support = C.sum(axis=1)
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros(K), where=denom > 0)
    total = support.sum()
    return float((f1 * support).sum() / total) if total > 0 else np.nan


def weighted_f1(y, P) -> float:
    """Per-class F1 averaged weighted by each class's support (true count). Matches
    sklearn f1_score(average='weighted'). Prevalence-dominated, so majority classes
    drive it -- report it next to macro F1, which weights every class equally."""
    y = np.asarray(y); K = np.asarray(P).shape[1]
    C = _confusion(y, np.asarray(P).argmax(axis=1), K).astype(float)
    tp = np.diag(C)
    fp = C.sum(axis=0) - tp
    fn = C.sum(axis=1) - tp
    support = C.sum(axis=1)
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros(K), where=denom > 0)
    total = support.sum()
    return float((f1 * support).sum() / total) if total > 0 else np.nan


def _weighted_kappa(y, P, power) -> float:
    y = np.asarray(y); K = np.asarray(P).shape[1]
    C = _confusion(y, np.asarray(P).argmax(axis=1), K).astype(float)
    N = C.sum()
    if N == 0:
        return np.nan
    idx = np.arange(K)
    W = (np.abs(idx[:, None] - idx[None, :]) ** power).astype(float)
    row = C.sum(axis=1); col = C.sum(axis=0)
    E = np.outer(row, col) / N
    num = (W * C).sum()
    den = (W * E).sum()
    return float(1 - num / den) if den > 0 else np.nan


def quadratic_kappa(y, P) -> float:
    """Quadratic-weighted Cohen's kappa. Penalises an error by the SQUARED distance
    between class indices, so it is only meaningful once the integer coding matches
    the clinical order -- see remap_to_ordinal() below. Applied to an arbitrary
    nominal coding it silently returns a wrong number rather than an error."""
    return _weighted_kappa(y, P, power=2)


def linear_kappa(y, P) -> float:
    """Linearly-weighted kappa. Less harsh on distant errors than quadratic."""
    return _weighted_kappa(y, P, power=1)


# ---------------------------------------------------------------------------
# 2b. ORDINAL support
#
# These require the integer coding to run lowest -> highest along the clinical
# scale. If your saved labels do not (e.g. the order is 3 < 0 < 1 < 2), call
# remap_to_ordinal() FIRST and use its output everywhere afterwards.
# ---------------------------------------------------------------------------

def remap_to_ordinal(y, probs, order):
    """Recode labels and reorder probability columns into ordinal rank space.

    order : the ORIGINAL class ids listed from lowest to highest on the scale.
            e.g. order=[3, 0, 1, 2] means original class 3 is the lowest grade,
            then 0, then 1, then 2 is the highest.

    Returns (y_ranked, probs_reordered) where labels are 0..K-1 in true order and
    column j of the probability array is the probability of rank j.

    Works for probs of shape (n, K) or (n_folds, n, K).

    Note this changes NOTHING for accuracy, macro F1, macro AUROC/AUPRC or the
    Brier score -- all of those are invariant to a permutation of the class labels.
    It only matters for the ordinal metrics, which is exactly why the bug is easy
    to miss: the rest of your table looks fine while kappa is quietly wrong.
    """
    order = np.asarray(order, dtype=int)
    K = len(order)
    if len(np.unique(order)) != K:
        raise ValueError("order must list each class exactly once")

    rank_of = np.full(int(order.max()) + 1, -1, dtype=int)
    rank_of[order] = np.arange(K)

    y = np.asarray(y, dtype=int)
    if np.any(rank_of[y] < 0):
        raise ValueError("y contains a class not listed in order")

    P = np.asarray(probs, dtype=float)
    if P.shape[-1] != K:
        raise ValueError(f"probs last axis is {P.shape[-1]}, order has {K} classes")

    return rank_of[y], P[..., order]


def _expected_rank(P) -> np.ndarray:
    """E[rank] under the predicted distribution. Uses the whole distribution rather
    than just the argmax, so it is far more sensitive than a hard prediction."""
    P = np.asarray(P, float)
    return P @ np.arange(P.shape[1], dtype=float)


def cumulative_auc(y, P) -> float:
    """Mean AUC over the K-1 cumulative dichotomisations: for each cut t, does the
    model separate (rank >= t) from (rank < t), scored by the cumulative probability
    P(rank >= t)?

    This is the right discrimination metric for an ordinal outcome. One-vs-rest
    macro AUROC asks a question that makes no sense on a scale -- it treats
    "grade 1 vs everything else" as a target, lumping grade 0 and grade 3 together
    as a single negative class.
    """
    y = np.asarray(y)
    P = np.asarray(P, float)
    K = P.shape[1]
    vals = [_fast_binary_auc((y >= t).astype(int), P[:, t:].sum(axis=1))
            for t in range(1, K)]
    return float(np.nanmean(vals))


def cumulative_auc_at(t: int):
    """Factory: the single cumulative AUC at cut t, i.e. (rank >= t) vs (rank < t)."""
    def _f(y, P):
        P = np.asarray(P, float)
        return _fast_binary_auc((np.asarray(y) >= t).astype(int), P[:, t:].sum(axis=1))
    _f.__name__ = f"cum_auc_ge{t}"
    return _f


def neg_mae_argmax(y, P) -> float:
    """Negated mean absolute error in ranks, using the argmax prediction.
    Directly interpretable: 0.4 means the average prediction is off by 0.4 grades."""
    return float(-np.mean(np.abs(np.asarray(P).argmax(axis=1) - np.asarray(y))))


def neg_mae_expected(y, P) -> float:
    """Negated MAE using E[rank] instead of the argmax. Continuous, so it responds
    to changes in confidence that never flip the predicted class."""
    return float(-np.mean(np.abs(_expected_rank(P) - np.asarray(y))))


def kendall_tau(y, P) -> float:
    """Kendall's tau-b between E[rank] and the true rank. A rank-correlation view of
    ordinal agreement; equivalent in spirit to Somers' D and insensitive to
    calibration, only to ordering."""
    tau = stats.kendalltau(_expected_rank(P), np.asarray(y)).statistic
    return float(tau) if np.isfinite(tau) else np.nan


def neg_multiclass_brier(y, P) -> float:
    """Negated multiclass Brier score (range 0..2 before negation), so higher is
    better like the other metrics. This is the proper scoring rule -- it is the one
    that punishes a model for being confidently wrong, which AUROC does not."""
    P = np.asarray(P, float)
    Y = np.zeros_like(P)
    Y[np.arange(len(P)), np.asarray(y).astype(int)] = 1.0
    return float(-np.mean(((P - Y) ** 2).sum(axis=1)))


def neg_log_loss(y, P, eps: float = 1e-12) -> float:
    P = np.clip(np.asarray(P, float), eps, 1.0)
    return float(np.mean(np.log(P[np.arange(len(P)), np.asarray(y).astype(int)])))


# ---------------------------------------------------------------------------
# 3. DeLong, applied one-vs-rest per class
# ---------------------------------------------------------------------------

def _midrank(x: np.ndarray) -> np.ndarray:
    J = np.argsort(x, kind="mergesort")
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T + 1
    return T2


def _fast_delong(preds_sorted: np.ndarray, m: int):
    """preds_sorted: (k_models, n), positives in the first m columns."""
    k, total = preds_sorted.shape
    n = total - m
    tx = np.empty((k, m)); ty = np.empty((k, n)); tz = np.empty((k, total))
    for r in range(k):
        tx[r] = _midrank(preds_sorted[r, :m])
        ty[r] = _midrank(preds_sorted[r, m:])
        tz[r] = _midrank(preds_sorted[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.atleast_2d(np.cov(v01)) / m + np.atleast_2d(np.cov(v10)) / n
    return aucs, cov


def delong_binary(y_bin, score_a, score_b) -> dict:
    y_bin = np.asarray(y_bin).astype(int)
    order = np.argsort(-y_bin, kind="mergesort")
    m = int(y_bin.sum())
    if m == 0 or m == len(y_bin):
        raise ValueError("class has no positives (or no negatives) in this cohort")
    preds = np.vstack([np.asarray(score_a, float),
                       np.asarray(score_b, float)])[:, order]
    aucs, cov = _fast_delong(preds, m)
    diff = aucs[0] - aucs[1]
    se = np.sqrt(max(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1], 0.0))
    z = diff / se if se > 0 else 0.0
    return {"auc_a": aucs[0], "auc_b": aucs[1], "difference": diff, "se": se,
            "ci_95": (diff - 1.96 * se, diff + 1.96 * se),
            "z": z, "p_value": 2 * stats.norm.sf(abs(z))}


def delong_ovr(y, P_a, P_b, correction: str = "holm") -> dict:
    """Run DeLong once per class (class k vs rest) and correct across the K tests.

    Returns per-class rows plus the adjusted p-values. Note there is no closed-form
    DeLong for the MACRO AUC -- for that, use paired_bootstrap(metric=macro_auroc_ovr).
    """
    y = np.asarray(y)
    P_a = np.asarray(P_a, float); P_b = np.asarray(P_b, float)
    K = P_a.shape[1]

    rows = []
    for k in range(K):
        r = delong_binary((y == k).astype(int), P_a[:, k], P_b[:, k])
        r["class"] = k
        r["n_pos"] = int((y == k).sum())
        rows.append(r)

    raw = np.array([r["p_value"] for r in rows])
    adj = holm(raw) if correction == "holm" else benjamini_hochberg(raw)
    for r, a in zip(rows, adj):
        r["p_adjusted"] = float(a)
    return {"per_class": rows, "correction": correction}


# ---------------------------------------------------------------------------
# 4. Paired stratified bootstrap (the general-purpose workhorse)
# ---------------------------------------------------------------------------

def _bootstrap_indices(y, n_boot, stratified, rng):
    y = np.asarray(y)
    n = len(y)
    if not stratified:
        return rng.integers(0, n, size=(n_boot, n))
    blocks = []
    for k in np.unique(y):
        idx_k = np.flatnonzero(y == k)
        blocks.append(idx_k[rng.integers(0, len(idx_k), size=(n_boot, len(idx_k)))])
    return np.hstack(blocks)


def paired_bootstrap(y, P_a, P_b, metric=macro_auroc_ovr, n_boot=10000,
                     stratified=True, seed=0, alpha=0.05) -> dict:
    """Paired patient-level bootstrap for metric(A) - metric(B).

    The SAME resampled patients are used for both methodologies in every replicate.
    That pairing is what cancels the cohort-level noise and makes the CI on the
    DIFFERENCE far tighter than the two individual CIs -- which is also why you must
    not judge significance by whether the two per-model CIs overlap.

    stratified=True resamples each of the K classes separately, fixing the class
    counts at their observed values. Keep it True: with unstratified resampling a
    small class can disappear from a replicate and its OvR AUC becomes undefined.

    Two-sided percentile p-value with a +1 continuity correction, so the smallest
    reportable value is 2/(n_boot+1). Report the CI as the headline result.
    """
    y = np.asarray(y)
    A = np.asarray(P_a, float); B = np.asarray(P_b, float)
    rng = np.random.default_rng(seed)

    observed = metric(y, A) - metric(y, B)
    idx = _bootstrap_indices(y, n_boot, stratified, rng)

    deltas = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        j = idx[i]
        yj = y[j]
        deltas[i] = metric(yj, A[j]) - metric(yj, B[j])

    lo, hi = np.percentile(deltas, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p_left = (1 + np.sum(deltas <= 0)) / (n_boot + 1)
    p_right = (1 + np.sum(deltas >= 0)) / (n_boot + 1)
    return {"metric_a": metric(y, A), "metric_b": metric(y, B),
            "difference": observed, "ci_95": (lo, hi),
            "p_value": min(1.0, 2 * min(p_left, p_right)),
            "boot_sd": deltas.std(ddof=1), "replicates": deltas}


# ---------------------------------------------------------------------------
# 4b. Paired permutation test (builds an actual null distribution)
# ---------------------------------------------------------------------------

def paired_permutation(y, P_a, P_b, metric=macro_auroc_ovr, n_perm=10000,
                       seed=0) -> dict:
    """Paired permutation test for metric(A) - metric(B).

    How this differs from paired_bootstrap
    --------------------------------------
    The bootstrap resamples PATIENTS and builds the sampling distribution of the
    difference, centred on the OBSERVED value; its p-value comes from asking whether
    0 falls in the tail. That is a confidence-interval inversion, not a null
    distribution.

    This builds the actual NULL distribution. Under H0 the two methodologies are
    exchangeable, so for each patient independently we flip a fair coin and swap that
    patient's predicted probability vector between the two arms, then recompute the
    difference. The reference distribution is centred on 0 by construction and the
    p-value is the proportion of permuted differences at least as extreme as observed.

    Which to report
    ---------------
    Usually both: the effect size and CI from the bootstrap (that is what belongs in
    the abstract), and the p-value from the permutation test (that is what a careful
    reviewer recognises as a hypothesis test). They should broadly agree; if they
    disagree sharply, the metric is behaving badly under resampling and you should
    look at why before reporting either.

    Caveat: the null here is per-patient exchangeability of the two prediction
    vectors, which is slightly stronger than "the two metrics are equal".
    """
    y = np.asarray(y)
    A = np.asarray(P_a, float)
    B = np.asarray(P_b, float)
    n = len(y)
    rng = np.random.default_rng(seed)

    observed = metric(y, A) - metric(y, B)

    null = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        swap = rng.random(n) < 0.5
        Ai = np.where(swap[:, None], B, A)
        Bi = np.where(swap[:, None], A, B)
        null[i] = metric(y, Ai) - metric(y, Bi)

    p = (1 + np.sum(np.abs(null) >= abs(observed))) / (n_perm + 1)
    return {"metric_a": metric(y, A), "metric_b": metric(y, B),
            "difference": observed, "p_value": float(p),
            "null_sd": float(null.std(ddof=1)),
            "null_mean": float(null.mean()), "null": null}


# ---------------------------------------------------------------------------
# 5. McNemar on argmax predictions (fixed operating point)
# ---------------------------------------------------------------------------

def mcnemar_test(y, P_a, P_b, exact_below: int = 25) -> dict:
    """Compares overall accuracy. Unchanged by K: it only asks, per patient,
    whether each methodology's argmax was right, and counts the discordances.

    Always returns the same keys, including when there are no discordant pairs.
    n_discordant == 0 means the two methodologies gave the identical argmax class
    for every patient -- McNemar has nothing to test, and you should check whether
    the two probability arrays are actually different before reading anything else.
    """
    y = np.asarray(y)
    ca = np.asarray(P_a).argmax(axis=1) == y
    cb = np.asarray(P_b).argmax(axis=1) == y
    b = int(np.sum(ca & ~cb))   # A right, B wrong
    c = int(np.sum(~ca & cb))   # A wrong, B right

    out = {"b_a_correct_b_wrong": b,
           "c_a_wrong_b_correct": c,
           "n_discordant": b + c,
           "acc_a": float(ca.mean()),
           "acc_b": float(cb.mean())}

    if b + c == 0:
        out.update({"p_value": 1.0, "test": "none (no discordant pairs)"})
    elif b + c < exact_below:
        out.update({"p_value": float(stats.binomtest(b, b + c, 0.5).pvalue),
                    "test": "exact binomial"})
    else:
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        out.update({"p_value": float(stats.chi2.sf(chi2, df=1)),
                    "test": "chi-square, continuity corrected"})
    return out


def mcnemar_per_class(y, P_a, P_b) -> list:
    """Where does the difference come from? McNemar restricted to the patients whose
    TRUE class is k, i.e. per-class recall. Descriptive -- correct the K p-values,
    and don't promote a per-class win to the headline claim."""
    y = np.asarray(y)
    out = []
    for k in np.unique(y):
        mask = y == k
        r = mcnemar_test(y[mask], np.asarray(P_a)[mask], np.asarray(P_b)[mask])
        r["class"] = int(k)
        r["n"] = int(mask.sum())
        out.append(r)
    raw = np.array([r["p_value"] for r in out])
    for r, a in zip(out, holm(raw)):
        r["p_adjusted"] = float(a)
    return out


def stuart_maxwell_test(y, P_a, P_b) -> dict:
    """Generalised McNemar (Stuart-Maxwell) on the K x K table of A's prediction vs
    B's prediction. Tests whether the two methodologies distribute their predictions
    across classes differently AT ALL -- not whether one is more accurate. Useful as
    a sanity check that the models genuinely differ; not a performance test."""
    K = np.asarray(P_a).shape[1]
    pa = np.asarray(P_a).argmax(axis=1)
    pb = np.asarray(P_b).argmax(axis=1)
    N = np.zeros((K, K), dtype=float)
    for i, j in zip(pa, pb):
        N[i, j] += 1

    d = (N.sum(axis=1) - N.sum(axis=0))[:-1]
    S = np.zeros((K - 1, K - 1))
    for i in range(K - 1):
        S[i, i] = N[i, :].sum() + N[:, i].sum() - 2 * N[i, i]
        for j in range(K - 1):
            if i != j:
                S[i, j] = -(N[i, j] + N[j, i])
    try:
        chi2 = float(d @ np.linalg.pinv(S) @ d)
        return {"chi2": chi2, "df": K - 1, "p_value": float(stats.chi2.sf(chi2, K - 1))}
    except np.linalg.LinAlgError:
        return {"chi2": np.nan, "df": K - 1, "p_value": np.nan}


# ---------------------------------------------------------------------------
# 6. Hierarchical bootstrap: patients AND folds
# ---------------------------------------------------------------------------

def hierarchical_bootstrap(y, probs_a, probs_b, metric=macro_auroc_ovr, n_boot=5000,
                           stratified=True, seed=0, alpha=0.05,
                           ensemble_method="mean") -> dict:
    """Each replicate resamples patients AND resamples the 5 fold-models with
    replacement before ensembling, so training variability enters the interval.
    This is what a claim about the METHODOLOGY (rather than about these particular
    trained weights) requires.

    probs_a, probs_b : (n_folds, n, K)

    Caveat: 5 folds estimates the fold-level variance very crudely, and the fold
    resample also changes the ensemble size composition. Treat the widening as
    indicative. Repeated CV (5 folds x 5 seeds = 25 models) makes it trustworthy.
    """
    y = np.asarray(y)
    A = np.asarray(probs_a, float); B = np.asarray(probs_b, float)
    ka, kb = A.shape[0], B.shape[0]
    rng = np.random.default_rng(seed)

    observed = (metric(y, ensemble_folds(A, ensemble_method))
                - metric(y, ensemble_folds(B, ensemble_method)))

    idx = _bootstrap_indices(y, n_boot, stratified, rng)
    fa = rng.integers(0, ka, size=(n_boot, ka))
    fb = rng.integers(0, kb, size=(n_boot, kb))

    deltas = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        j = idx[i]
        yj = y[j]
        sa = ensemble_folds(A[fa[i]][:, j, :], ensemble_method)
        sb = ensemble_folds(B[fb[i]][:, j, :], ensemble_method)
        deltas[i] = metric(yj, sa) - metric(yj, sb)

    lo, hi = np.percentile(deltas, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p_left = (1 + np.sum(deltas <= 0)) / (n_boot + 1)
    p_right = (1 + np.sum(deltas >= 0)) / (n_boot + 1)
    return {"difference": observed, "ci_95": (lo, hi),
            "p_value": min(1.0, 2 * min(p_left, p_right)),
            "boot_sd": deltas.std(ddof=1), "replicates": deltas}


# ---------------------------------------------------------------------------
# 7. Multiplicity
# ---------------------------------------------------------------------------

def benjamini_hochberg(pvals) -> np.ndarray:
    p = np.asarray(pvals, float); n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n); out[order] = np.clip(ranked, 0, 1)
    return out


def holm(pvals) -> np.ndarray:
    p = np.asarray(pvals, float); n = len(p)
    order = np.argsort(p)
    ranked = np.maximum.accumulate(p[order] * (n - np.arange(n)))
    out = np.empty(n); out[order] = np.clip(ranked, 0, 1)
    return out


# ---------------------------------------------------------------------------
# 8. One-call report
# ---------------------------------------------------------------------------

def compare(y, probs_a, probs_b, name_a="A", name_b="B", n_boot=5000,
            ensemble_method="mean", ordinal_order=None, seed=0,
            metric_set=None, primary=None):
    """Primary + secondary analysis in one go.

    ordinal_order : None for nominal classes.
                    For an ordinal outcome, give the ORIGINAL class ids from lowest
                    to highest on the scale, e.g. ordinal_order=[3, 0, 1, 2].
                    Labels and probability columns are remapped internally, so pass
                    y and probs in their original coding and let this handle it.

    metric_set    : None runs the full diagnostic panel. To match a published
                    protocol, pass the subset you pre-specified, e.g.
                        metric_set=["macro AUROC (OvR)", "quadratic kappa",
                                    "accuracy (exact)", "macro F1"]

    primary       : name of the ONE pre-specified primary metric. Flagged in the
                    output; the rest are then reportable as secondary without a
                    multiplicity correction. Leaving this None prints a warning,
                    because a table of p-values with no declared primary is a
                    multiple-testing problem whether it is labeled one.
    """
    y = np.asarray(y)
    labels = None

    if ordinal_order is not None:
        y_new, probs_a = remap_to_ordinal(y, probs_a, ordinal_order)
        _, probs_b = remap_to_ordinal(y, probs_b, ordinal_order)
        y = y_new
        labels = [str(c) for c in ordinal_order]
        print("ORDINAL mode. Scale (low -> high): "
              + " < ".join(f"class {c}" for c in ordinal_order))
        print("  labels and probability columns remapped to rank space 0..%d\n"
              % (len(ordinal_order) - 1))

    A = ensemble_folds(probs_a, ensemble_method)
    B = ensemble_folds(probs_b, ensemble_method)
    K = A.shape[1]
    if labels is None:
        labels = [str(k) for k in range(K)]

    print(f"n = {len(y)}   classes = {K}   "
          f"counts = {np.bincount(y, minlength=K).tolist()}")
    print(f"{name_a} vs {name_b}   (positive difference favours {name_a})")

    # --- sanity check: are the two ensembles actually different? -------------
    max_abs = float(np.abs(A - B).max())
    agree = float((A.argmax(axis=1) == B.argmax(axis=1)).mean())
    print(f"ensemble divergence: max |p_{name_a} - p_{name_b}| = {max_abs:.3g}   "
          f"argmax agreement = {agree:.1%}")
    if max_abs < 1e-9:
        print(f"  !! The two probability arrays are IDENTICAL. Every difference "
              f"below will be exactly 0.\n"
              f"     Check that you passed two different models' predictions, "
              f"and that they are\n"
              f"     indexed in the same patient order.")
    elif agree == 1.0:
        print(f"  !! The ensembles differ in probability but give the SAME argmax "
              f"class for all\n"
              f"     {len(y)} patients. Threshold-based tests (McNemar, accuracy, "
              f"F1) cannot detect\n"
              f"     any difference; only the ranking/calibration metrics can.")
    print()

    if ordinal_order is not None:
        metrics = [("cumulative AUC", cumulative_auc),
                   ("quadratic kappa", quadratic_kappa),
                   ("linear kappa", linear_kappa),
                   ("-MAE (expected)", neg_mae_expected),
                   ("-MAE (argmax)", neg_mae_argmax),
                   ("Kendall tau-b", kendall_tau),
                   ("-Brier (multiclass)", neg_multiclass_brier),
                   ("accuracy (exact)", accuracy),
                   ("F1", weighted_f1),
                   ("macro AUROC (OvR)", macro_auroc_ovr)]
    else:
        metrics = [("macro AUROC (OvR)", macro_auroc_ovr),
                   ("macro AUROC (OvO)", macro_auroc_ovo),
                   ("macro AUPRC", macro_auprc),
                   ("accuracy", accuracy),
                   ("balanced accuracy", balanced_accuracy),
                   ("F1", weighted_f1),
                   ("-Brier (multiclass)", neg_multiclass_brier)]

    if metric_set is not None:
        keep = {m.lower() for m in metric_set}
        chosen = [(l, m) for l, m in metrics if l.lower() in keep
                  or m.__name__.lower() in keep]
        missing = keep - {l.lower() for l, _ in metrics} - \
                  {m.__name__.lower() for _, m in metrics}
        if missing:
            raise ValueError(f"unknown metric name(s): {sorted(missing)}")
        metrics = chosen

    print(f"{'metric':<22}{name_a:>9}{name_b:>9}{'diff':>10}{'95% CI':>22}{'p':>9}")
    print("-" * 81)
    results = {}
    for label, m in metrics:
        r = paired_bootstrap(y, A, B, metric=m, n_boot=n_boot, seed=seed)
        results[label] = r
        flag = " <- PRIMARY" if primary is not None and (
            label.lower() == primary.lower() or m.__name__.lower() == primary.lower()
        ) else ""
        print(f"{label:<22}{r['metric_a']:>9.4f}{r['metric_b']:>9.4f}"
              f"{r['difference']:>+10.4f}"
              f"   [{r['ci_95'][0]:+.4f}, {r['ci_95'][1]:+.4f}]"
              f"{r['p_value']:>9.4f}{flag}")

    if primary is None:
        print("\n  NOTE: no primary metric declared. Every p-value above is one of "
              f"{len(metrics)} tests\n"
              "  on the same cohort. Pre-specify ONE as primary and label the rest\n"
              "  secondary/descriptive, or correct across the whole family.")
    else:
        others = [l for l, _ in metrics if l.lower() != primary.lower()]
        print(f"\n  PRIMARY = {primary}. The other {len(others)} rows are secondary "
              "and descriptive:\n"
              "  report them with CIs, do not promote one to the headline claim.")

    if ordinal_order is not None:
        print("\nCumulative AUC by cut point (DeLong, Holm-adjusted)")
        print(f"{'cut':<22}{'n>=cut':>8}{name_a:>9}{name_b:>9}{'diff':>10}"
              f"{'p_raw':>9}{'p_adj':>9}")
        print("-" * 76)
        rows = []
        for t in range(1, K):
            yb = (y >= t).astype(int)
            rows.append(delong_binary(yb, A[:, t:].sum(axis=1), B[:, t:].sum(axis=1)))
        for t, r, adj in zip(range(1, K), rows,
                             holm([r["p_value"] for r in rows])):
            cut = f"{'/'.join(labels[:t])} vs {'/'.join(labels[t:])}"
            print(f"{cut:<22}{int((y >= t).sum()):>8}{r['auc_a']:>9.4f}"
                  f"{r['auc_b']:>9.4f}{r['difference']:>+10.4f}"
                  f"{r['p_value']:>9.4f}{adj:>9.4f}")
    else:
        print("\nPer-class OvR AUROC (DeLong, Holm-adjusted across classes)")
        print(f"{'class':<8}{'n_pos':>7}{name_a:>9}{name_b:>9}{'diff':>10}"
              f"{'p_raw':>9}{'p_adj':>9}")
        print("-" * 61)
        for r in delong_ovr(y, A, B)["per_class"]:
            print(f"{labels[r['class']]:<8}{r['n_pos']:>7}{r['auc_a']:>9.4f}"
                  f"{r['auc_b']:>9.4f}{r['difference']:>+10.4f}"
                  f"{r['p_value']:>9.4f}{r['p_adjusted']:>9.4f}")

    mc = mcnemar_test(y, A, B)
    print(f"\nMcNemar (exact accuracy)   {mc['acc_a']:.4f} vs {mc['acc_b']:.4f}   "
          f"b={mc['b_a_correct_b_wrong']} c={mc['c_a_wrong_b_correct']} "
          f"(n_discordant={mc['n_discordant']})   "
          f"p={mc['p_value']:.4f}   ({mc['test']})")
    sm = stuart_maxwell_test(y, A, B)
    print(f"Stuart-Maxwell             chi2={sm['chi2']:.2f}  df={sm['df']}  "
          f"p={sm['p_value']:.4g}   (do the models differ at all?)")

    head = cumulative_auc if ordinal_order is not None else macro_auroc_ovr
    h = hierarchical_bootstrap(y, probs_a, probs_b, metric=head,
                               n_boot=max(1000, n_boot // 5), seed=seed + 1,
                               ensemble_method=ensemble_method)
    print(f"\nHierarchical (patients + folds), {head.__name__}: "
          f"{h['difference']:+.4f}  95% CI [{h['ci_95'][0]:+.4f}, "
          f"{h['ci_95'][1]:+.4f}]  p={h['p_value']:.4f}")


# ---------------------------------------------------------------------------
# Worked example: (5, 200, 4)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n_folds, n, K = 5, 200, 4

    # Scale runs 3 < 0 < 1 < 2, i.e. original class 3 is the LOWEST grade.
    ORDINAL_ORDER = [3, 0, 1, 2]

    ranks = rng.choice(K, size=n, p=[0.30, 0.30, 0.25, 0.15])   # true rank 0..3
    y = np.array(ORDINAL_ORDER)[ranks]                          # original coding

    # a latent severity signal, so that adjacent grades are genuinely confusable
    latent = ranks + rng.normal(0, 0.8, size=n)

    def make(noise):
        d = np.abs(latent[None, :, None]
                   - np.arange(K)[None, None, :])               # rank space
        z = -d ** 2 / 1.2 + rng.normal(0, noise, size=(n_folds, n, K))
        e = np.exp(z - z.max(axis=2, keepdims=True))
        p = e / e.sum(axis=2, keepdims=True)
        inv = np.argsort(ORDINAL_ORDER)                         # back to original
        return p[..., inv]

    probs_A = make(0.9)
    probs_B = make(1.5)

    PAPER_METRICS = ["macro AUROC (OvR)", "quadratic kappa",
                     "accuracy (exact)", "F1"]

    compare(y, probs_A, probs_B, name_a="MethodA", name_b="MethodB",
            n_boot=3000, ordinal_order=ORDINAL_ORDER,
            metric_set=PAPER_METRICS, primary="quadratic kappa")