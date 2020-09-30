import logging
import numpy as np


def reaches_percentile(i, n):
    """Return True if i just reaches 10%,...,100% percentile of n."""
    lower_nearest_percentile = np.floor((i + 1) * 10 / n)
    threshold = np.ceil(lower_nearest_percentile * n / 10)
    return i + 1 == threshold
    

def evaluate(scores, preds, labels, res_file=None):
    """
    Scores represent prediction confidence.
    Label=-1 for out of support data.
    Return acccuracies and AUC score.
    """
    corrects = (preds == labels).astype(np.int64)
    n = scores.shape[0]
    idx = np.argsort(- scores)
    corrects = corrects[idx]
    acc = np.sum(corrects) / n
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


def evaluate_(scores, preds, labels, res_file=None):
    corrects = (preds == labels).astype(np.int64)
    outputs = np.stack([labels, preds, corrects], axis=-1).astype(np.int64)
    if res_file is not None:
        np.savetxt(res_file+'_outputs.csv', outputs, delimiter=',')
        np.savetxt(res_file+'_scores.csv', 
                scores, fmt='%.4g', delimiter=',')
    n = scores.shape[0]
    idx = np.argsort(- scores)
    corrects = corrects[idx]
    acc = np.sum(corrects) / n
    logging.info('Total accuracy: {:.4g}'.format(acc))
    res = np.zeros(n)
    cumu = 0
    for i in range(n):
        cumu += corrects[i]
        res[i] = cumu / (i + 1)
        if (i + 1) % int(n // 10) == 0:
            logging.info('Accuracy for first {}0%% data: {:.4g}.'
                    .format(int((i+1) // int(n//10)), res[i]))
    auc = np.mean(res)
    logging.info('AUC: {:.4g}'.format(auc))   
    if res_file is not None:
        np.savetxt(res_file+'.csv', res, fmt='%.4g', delimiter=',')

