import numpy as np

from . import evaluator
from . import model_loader


def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits, axis=-1)[:, None])
    return exp_logits / np.sum(exp_logits, axis=-1)[:, None]


class SoftmaxEvaluator(evaluator.Evaluator):

    def __init__(self,
            model,
            model_path=None,
            batch_size=128,
            device=None,
            temp=1.0,
            ):
        self._model_loader = model_loader.ModelLoader(
                model=model,
                model_path=model_path,
                batch_size=batch_size,
                device=device)
        self._temp = temp

    def _get_batch_outputs(self, batch, iid=True):
        _, logits, labels = self._model_loader.get_outputs(
                data=batch, layer=-1)
        predprobs = softmax(logits)
        scores = np.max(predprobs, axis=-1)
        preds = np.argmax(predprobs, axis=-1)
        return preds, scores, labels

