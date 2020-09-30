import logging

import numpy as np
from predprob import distance_evaluator



EPS = 1e-10

class Evaluator(distance_evaluator.Evaluator):

    def __init__(self, mode='pos', kernel_variance=None, *args, **kwargs):
        self._kernel_variance = kernel_variance
        super(Evaluator, self).__init__(*args, **kwargs)
        assert mode in ['pos', 'neg']
        self._mode = mode  # pos or neg
        self._name = 'kde_{}_layer{}_per{:.2g}'.format(
                self._mode, self._layer, self._percentile)

    def _get_batch_score(self, tr_fs, tr_ys, fs, ps):
        tr_norms = np.sum(np.square(tr_fs), axis=-1)
        te_norms = np.sum(np.square(fs), axis=-1)
        inprods = fs @ (tr_fs.T)
        dists = tr_norms[None, :] + te_norms[:, None] - 2 * inprods
        # print testing distance statistics
        logging.info('Statistics for distances between training and '
                'test samples:')
        for i in range(11):
            percentile = i * 10
            distance_at_percentile = np.percentile(dists, percentile)
            logging.info('Percentile {}: distance {:.4g}.'
                    .format(percentile, distance_at_percentile))
        # set kernel variance
        if self._kernel_variance is not None:
            w = self._kernel_variance
        else:
            w = np.percentile(dists, self._percentile)
            w = max(w, EPS)
        logging.info('Selected kernel variance: {:.4g}.'.format(w))
        dists /= w
        scores = np.mean(np.exp(-dists), axis=-1) 
        # compute scores for a random number of training samples
        ntr = tr_fs.shape[0]
        n_sampled = int(ntr * self._train_score_propo)
        rand = np.random.RandomState(0)
        sampled_idx = rand.choice(ntr, n_sampled, replace=False)
        sampled_fs = tr_fs[sampled_idx]
        sampled_norms = tr_norms[sampled_idx]
        inprods_tr = sampled_fs @ (tr_fs.T)
        dists_tr = (tr_norms[None, :] + sampled_norms[:, None] 
                - 2 * inprods_tr)
        logging.info('Statistics for distances among training samples:')
        for i in range(11):
            percentile = i * 10
            distance_at_percentile = np.percentile(dists_tr, percentile)
            logging.info('Percentile {}: distance {:.4g}.'
                    .format(percentile, distance_at_percentile))
        dists_tr /= w
        train_scores = np.mean(np.exp(-dists_tr), axis=-1) 
        if self._mode == 'neg':
            scores = - scores
            train_scores = - train_scores
        return scores, train_scores


