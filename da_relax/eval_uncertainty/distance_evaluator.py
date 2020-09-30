import os
import logging
import importlib
import numpy as np
import tensorflow as tf
from predprob import evaluation
from predprob import model
from predprob.tf_tools import utils as tf_utils
from predprob.tools import timer

tf.logging.set_verbosity(tf.logging.ERROR)
init_from_checkpoint = tf.contrib.framework.init_from_checkpoint

EPS = 1e-10

########### flags ############

DEFAULT_FLAGS = dict(
        config_dir='predprob.configs',
        config_file='cifar10_config',
        log_dir='/media/yw4/hdd/log/predprob/tmp',
        model_name='model',
        batch_size=128,
        print_freq=1000,
        )


class Evaluator(object):

    def __init__(self, model_factory, dataset_factory, flags,
            layer=-1, percentile=50, train_score_propo=0.1):
        # update flags:
        for key, val in DEFAULT_FLAGS.items():
            if not hasattr(flags, key):
                setattr(flags, key, val)
        # load config, flags may be updated
        config_module = importlib.import_module(
                flags.config_dir+'.'+flags.config_file)
        config = config_module.Config(flags=flags)
        if not os.path.exists(flags.log_dir):
            raise ValueError('log_dir {} does not exist.'
                    .format(flags.log_dir))
        self._model_path = os.path.join(flags.log_dir, flags.model_name)
        ########## setup tf graph
        tf.reset_default_graph()
        # model and dataset
        with tf.variable_scope('net'):
            model_ = model_factory(config)
            dataset = dataset_factory(config)
        # setup attributes
        self._config = config
        self._flags = flags
        self._model = model.Model(model_)
        self._dataset = dataset
        self._percentile = percentile
        self._layer = layer  # might be changed later
        self._train_score_propo = train_score_propo
        # setup training computation graph
        self._build()
        self._name = 'distance_layer{}_per{:.2g}'.format(
                self._layer, self._percentile)


    def _build(self):
        self._b = self._dataset.batch_size
        self._x_shape = list(self._dataset.x_shape)
        self._y_shape = list(self._dataset.y_shape)
        self._x_holder = tf.placeholder(tf.float32,
                [self._b]+self._x_shape)
        self._y_holder = tf.placeholder(tf.int64,
                [self._b]+self._y_shape)
        self._h = self._model(self._x_holder, is_training=False)
        self._h.get_y(self._y_holder)
        n_layers = len(self._h.features)
        if self._layer < n_layers and self._layer >= -n_layers:
            self._layer = self._layer % n_layers
            logging.info('Using the {}/{} layer output as features'
                    .format(self._layer+1, n_layers))
        else:
            raise ValueError('Layer {} exceeds limit {}.'
                    .format(self._layer, n_layers))
        self._features = tf.contrib.layers.flatten(
                self._h.features[self._layer])
        self._d = self._features.shape.as_list()[-1]
        init_from_checkpoint(self._model_path, {'net/': 'net/'})


    def _get_outputs(self, data):
        iter_ = data.one_shot_iterator
        n = data.n
        d = self._d
        features = np.zeros([n, d])
        preds = np.zeros(n)
        labels = np.zeros(n)
        with tf_utils.tf_session() as sess:
            i = 0
            for x, y, m in iter_:
                fd = {self._x_holder: x, self._y_holder: y}
                run_ops = [self._features, self._h.predictions]
                feature, pred = sess.run(run_ops, feed_dict=fd)
                features[i:i+m] = feature[:m]
                preds[i:i+m] = pred[:m]
                labels[i:i+m] = y[:m]
                i += m
        return features, preds, labels


    def _get_batch_score(self, tr_fs, tr_ys, fs, ps):
        tr_norms = np.sum(np.square(tr_fs), axis=-1)
        te_norms = np.sum(np.square(fs), axis=-1)
        inprods = fs @ (tr_fs.T)
        dists = tr_norms[None, :] + te_norms[:, None] - 2 * inprods
        w = np.percentile(dists, self._percentile)
        w = max(w, EPS)
        dists /= w
        # or / np.median(dists)
        mask = (tr_ys[None, :] == ps[:, None])
        scores = (np.sum(np.exp(-dists) * mask, axis=-1) 
                / (np.sum(np.exp(-dists), axis=-1) + 1e-10))
        # compute scores for a random number of training samples
        ntr = tr_fs.shape[0]
        n_sampled = int(ntr * self._train_score_propo)
        rand = np.random.RandomState(0)
        sampled_idx = rand.choice(ntr, n_sampled, replace=False)
        sampled_fs = tr_fs[sampled_idx]
        sampled_ps = tr_ys[sampled_idx]
        sampled_norms = tr_norms[sampled_idx]
        inprods_tr = sampled_fs @ (tr_fs.T)
        dists_tr = (tr_norms[None, :] + sampled_norms[:, None] 
                - 2 * inprods_tr)
        dists_tr /= w
        mask_tr = (tr_ys[None, :] == sampled_ps[:, None])
        train_scores = (np.sum(np.exp(-dists_tr) * mask_tr, axis=-1) 
                / (np.sum(np.exp(-dists_tr), axis=-1) + EPS))
        return scores, train_scores


    def evaluate(self):
        sttt = timer.get_time()
        tr_features, tr_preds, tr_labels = self._get_outputs(
                self._dataset.train)
        te_features, te_preds, te_labels = self._get_outputs(
                self._dataset.test)
        d = tr_features.shape[1]
        logging.info('Latent feature has {} dimensions.'.format(d))
        # compute distance based scores
        scores, train_scores = self._get_batch_score(tr_features, tr_labels,
                te_features, te_preds)
        #else:
        #    nte = te_features.shape[0]
        #    scores = np.zeros(nte)
        #    for i in range(nte):
        #        scores[i] = self._get_score(tr_features, tr_labels,
        #                te_features[i], te_preds[i])
        #        if (i+1) % self._flags.print_freq == 0:
        #            logging.info('Scores computed for {}/{} test samples'
        #                    .format(i+1, nte))
        # save scores
        res_file = os.path.join(
                self._flags.log_dir,
                'res_{}'
                .format(self._name))
        # save train scores
        np.savetxt(res_file+'_train_scores.csv', 
                train_scores, delimiter=',', fmt='%.4g')
        evaluation.evaluate(scores, te_preds, te_labels, res_file)
        edtt = timer.get_time()
        logging.info('Test finished, time cost {:.4g}s'
                .format(edtt-sttt))
