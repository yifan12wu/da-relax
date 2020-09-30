import logging
import numpy as np

from . import utils
from ..tools import timer_tools


class Evaluator:

    def _get_batch_outputs(self, batch, iid=True):
        """Return predictions, scores(confidence), labels."""
        raise NotImplementedError

    def evaluate(self, test_batch_iid, test_batch_oos=None, result_file=None):
        global_timer = timer_tools.Timer()
        logging.info('Evaluation starts...')
        logging.info('Getting outputs for iid samples...')
        outputs_iid = self._get_batch_outputs(test_batch_iid, iid=True)
        preds_iid, scores_iid, labels_iid = outputs_iid
        if test_batch_oos is None:
            preds, scores, labels = preds_iid, scores_iid, labels_iid
        else:
            logging.info('Getting outputs for oos samples...')
            outputs_oos = self._get_batch_outputs(test_batch_oos, iid=False)
            preds_oos, scores_oos, _ = outputs_oos
            preds = np.concatenate([preds_iid, preds_oos], axis=0)
            scores = np.concatenate([scores_iid, scores_oos], axis=0)
            labels_oos = np.zeros(scores_oos.shape[0]).astype(np.int64) - 1
            labels = np.concatenate([labels_iid, labels_oos], axis=0)
        logging.info('Evaluating...')
        auc, accuracies, res = utils.evaluate(
                preds, scores, labels)
        utils.print_evaluation(auc, accuracies)
        # maybe save results to file
        if result_file is not None:
            np.savetxt(result_file, res, delimiter=',')
        time_cost = global_timer.time_cost()
        logging.info('Evaluation finished, time cost {:.4g}s.'
                .format(time_cost))
        return auc


