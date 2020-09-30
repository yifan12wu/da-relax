import tensorflow as tf
from predprob.configs import base


class Config(base.BaseConfig):

    def _default_flags(self):
        flags = super(Config, self)._default_flags()
        flags.batch_size = 128
        flags.optimizer = 'mm'
        flags.network = 'lenet'
        flags.learning_rate = 0.001
        flags.weight_decay = 2e-3
        return flags

    def _setup_flags(self):
        if self._flags.network == 'lenet':
            self._flags.learning_rate = [
                0.01, 
                (0.001, 20000), 
                (0.0002, 40000),
                ]
        if self._flags.network == 'resnet18':
            self._flags.learning_rate = [
                0.1, 
                (0.01, 10000), 
                (0.001, 20000),
                (0.0002, 30000)
                ]
            #self._flags.learning_rate = 0.001

    def _build(self):
        optimizers = dict(
                adam=tf.train.AdamOptimizer,
                mm=lambda lr: tf.train.MomentumOptimizer(lr, 0.9))
        self._optimizer_cls = optimizers[self._flags.optimizer]

    @property
    def batch_size(self):
        return self._flags.batch_size

    @property
    def weight_decay(self):
        return self._flags.weight_decay

    def optimizer_factory(self, step):
        # step is a global step op
        lr = self._flags.learning_rate
        if isinstance(lr, float):
            return self._optimizer_cls(lr)
        elif isinstance(lr, list):
            lr_op = tf.constant(lr[0], dtype=tf.float32)
            for i in range(len(lr) - 1):
                lr_op = (lr_op 
                        + tf.to_float(tf.greater(step, lr[i+1][1]))
                        * (lr[i+1][0] - lr_op))
            tf.summary.scalar('learning_rate', lr_op)
            return self._optimizer_cls(lr_op)
