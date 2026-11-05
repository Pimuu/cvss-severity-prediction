"""Evaluation utilities: headline metrics, per-class report, ordinal errors, bootstrap CIs."""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support,
                             precision_score, recall_score)

from . import config


def summary_metrics(y_true, y_pred):
    """Accuracy, macro P/R/F1, weighted F1 and ordinal error statistics."""
    idx = {l: i for i, l in enumerate(config.LABELS)}
    t = np.array([idx[v] for v in y_true])
    p = np.array([idx[v] for v in y_pred])
    dist = np.abs(t - p)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        # Severity is ordinal: how far off are the mistakes?
        "ordinal_mae": dist.mean(),                       # mean |class index error|
        "off_by_one_share_of_errors": (dist == 1).sum() / max((dist > 0).sum(), 1),
        "severe_error_rate": (dist >= 2).mean(),          # e.g. Critical predicted as Medium/Low
    }


def per_class(y_true, y_pred):
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=config.LABELS, zero_division=0)
    return pd.DataFrame({"class": config.LABELS, "precision": p, "recall": r, "f1": f, "support": s})


def confusion(y_true, y_pred):
    return confusion_matrix(y_true, y_pred, labels=config.LABELS)


def bootstrap_ci(y_true, y_pred, n=config.N_BOOTSTRAP, seed=config.SEED, alpha=0.05):
    """Percentile bootstrap 95% CI of accuracy and macro-F1 on the test set."""
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    accs, f1s = [], []
    for _ in range(n):
        i = rng.integers(0, len(y_true), len(y_true))
        accs.append(accuracy_score(y_true[i], y_pred[i]))
        f1s.append(f1_score(y_true[i], y_pred[i], average="macro", zero_division=0))
    q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    return {"accuracy_ci": tuple(np.percentile(accs, q)), "macro_f1_ci": tuple(np.percentile(f1s, q))}


def paired_bootstrap(y_true, pred_a, pred_b, n=config.N_BOOTSTRAP, seed=config.SEED):
    """Share of bootstrap resamples where model A's macro-F1 exceeds model B's."""
    rng = np.random.default_rng(seed)
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    wins = 0
    for _ in range(n):
        i = rng.integers(0, len(y_true), len(y_true))
        wins += f1_score(y_true[i], pred_a[i], average="macro") > f1_score(y_true[i], pred_b[i], average="macro")
    return wins / n
