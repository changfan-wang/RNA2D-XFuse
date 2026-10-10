import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def safe_auc(y_true, y_score):
    y_true = np.asarray(y_true).reshape(-1)
    y_score = np.asarray(y_score).reshape(-1)

    if len(np.unique(y_true)) < 2:
        return float("nan")

    return float(roc_auc_score(y_true, y_score))


def safe_ap(y_true, y_score):
    y_true = np.asarray(y_true).reshape(-1)
    y_score = np.asarray(y_score).reshape(-1)

    if len(np.unique(y_true)) < 2:
        return float("nan")

    return float(average_precision_score(y_true, y_score))


def pooled_binding_site_metrics(
    labels,
    scores,
    valid_mask,
    interaction_labels=None,
):
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    valid_mask = np.asarray(valid_mask).astype(bool)

    if interaction_labels is not None:
        interaction_labels = np.asarray(
            interaction_labels
        ).reshape(-1)
        valid_mask = valid_mask & (
            interaction_labels[:, None] > 0.5
        )

    y = labels[valid_mask]
    p = scores[valid_mask]

    return {
        "auc": safe_auc(y, p),
        "ap": safe_ap(y, p),
        "n_positions": int(y.size),
    }
