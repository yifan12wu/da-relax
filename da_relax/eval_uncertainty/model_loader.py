"""Load a trained model and get intermediate representations."""
import logging
import numpy as np

import torch


class ModelLoader(object):

    def __init__(self,
            model,
            model_path=None, 
            batch_size=128,
            device=None,
            ):
        self._model = model
        self._model_path = model_path
        self._batch_size = batch_size
        self._device = device
        # device
        if self._device is None:
            self._device = torch.device(
                    'cuda' if torch.cuda.is_available() else 'cpu')
            logging.info('Device: {}.'.format(self._device))
        self._model.to(device=self._device)
        if self._model_path is not None:
            logging.info('Loading model from {}...'.format(self._model_path))
            self._model.load_state_dict(torch.load(self._model_path))

    def _tensor(self, x):
        if x.dtype in [np.float32, np.float64]:
            return torch.tensor(x, dtype=torch.float32, device=self._device)
        elif x.dtype in [np.int32, np.int64, np.uint8]:
            return torch.tensor(x, dtype=torch.int64, device=self._device)
        else:
            raise ValueError('Unknown dtype {} for tensor.'
                    .format(str(x.dtype)))

    def get_outputs(self, data, layer, print_accuracy=True):
        """Use print accuracy only when the number of classes are the same
                for both model and data.
        """
        features = []
        logits = []
        labels = []
        if print_accuracy:
            total_acc = 0.0
            total_size = 0
        #
        data_iter = data.get_one_shot_iterator(self._batch_size)
        i = 0
        for batch, size in data_iter:
            x = self._tensor(batch.x)
            output_head = self._model(x, is_training=False)
            if i == 0:
                # print
                n_layers = len(output_head.features)
                if layer < n_layers and layer >= -n_layers:
                    layer = layer % n_layers
                    logging.info('Using the {}/{} layer output as features'
                            .format(layer + 1, n_layers))
                else:
                    raise ValueError('Layer {} exceeds limit {}.'
                            .format(layer, n_layers))
            features.append(
                    output_head.features[layer][:size]
                    .detach().cpu().numpy())
            logits.append(output_head.logits[:size]
                    .detach().cpu().numpy())
            labels.append(batch.y[:size])
            i += 1
            if print_accuracy:
                y = self._tensor(batch.y)
                output_head.get_label(y)
                acc = output_head.accs[:size].sum().item()
                total_size += size
                total_acc += acc
        if print_accuracy:
            logging.info('Prediction accuracy {}.'.format(
                total_acc / total_size))
        features = np.concatenate(features, axis=0)
        features = features.reshape([features.shape[0], -1])
        logits = np.concatenate(logits, axis=0)
        labels = np.concatenate(labels, axis=0)
        return features, logits, labels






