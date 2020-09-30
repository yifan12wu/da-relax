import logging
import numpy as np


def reaches_percentile(i, n):
    """Return True if i just reaches 10%,...,100% percentile of n."""
    lower_nearest_percentile = np.floor((i + 1) * 10 / n)
    threshold = np.ceil(lower_nearest_percentile * n / 10)
    return i + 1 == threshold


def evaluate(preds, scores, labels):
    """
    Scores represent prediction confidence.
    Label=-1 for out of support data.
    Return acccuracies and AUC score.
    """
    corrects = (preds == labels).astype(np.int64)
    n = scores.shape[0]
    idx = np.argsort(- scores)
    corrects = corrects[idx]
    res = np.zeros(n)
    cumu = 0
    accuracies = []
    for i in range(n):
        cumu += corrects[i]
        res[i] = cumu / (i + 1)
        if reaches_percentile(i, n):
            accuracies.append(res[i])
    assert len(accuracies) == 10
    auc = np.mean(res)
    return auc, accuracies, res


def print_evaluation(auc, accuracies):
    assert len(accuracies) == 10
    for i in range(10):
        logging.info('Accuracy for first {}0%% data: {:.4g}.'
                .format(i + 1, accuracies[i]))
    logging.info('AUC: {:.4g}'.format(auc))
