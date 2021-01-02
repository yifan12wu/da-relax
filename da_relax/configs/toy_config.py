import os

from ..train import da_learner
from ..train import networks
from ..data import toy


DEFAULT_LOG_DIR = os.path.join(os.getenv('HOME', '/'), 'tmp/da-relax/test')


class Config(da_learner.DALearnerConfig):

    def _set_default_flags(self):
        super()._set_default_flags()
        flags = self._flags
        # 
        flags.batch_size = 128
        flags.d_loss_w = 1.0
        flags.d_relax = 0.0
        flags.d_loss_name = 'js'
        flags.d_grad_penalty = 0.0
        #
        flags.log_dir = DEFAULT_LOG_DIR
        flags.total_train_steps = 5000
        flags.print_freq = 500
        flags.summary_freq = 100
        flags.save_freq = 1000
        flags.eval_freq = 1000
        # optimizer
        # name, learning rate, weight decay
        flags.opt_args.name_f = 'Adam2'
        flags.opt_args.lr_f = 1e-4
        flags.opt_args.wd_f = 1e-3
        flags.opt_args.name_d = 'Adam2'
        flags.opt_args.lr_d = 1e-4
        flags.opt_args.wd_d = 1e-5

    def _source_dataset_factory(self):
        return toy.TwoGaussianSource()

    def _target_dataset_factory(self):
        return toy.TwoGaussianTarget()

    def _f_model_factory(self):
        return networks.MLP(input_shape=(2,), n_units=(50, 50, 2, 2))

    def _d_model_factory(self):
        return networks.MLP(input_shape=(2,), n_units=(50, 50, 1))

