import os
from ..tools import flag_tools
from . import data
from . import net


class Config(flag_tools.ConfigBase):

    def _set_default_flags(self):
        flags = self._flags
        # training
        flags.total_train_steps = 10000
        flags.log_dir = os.path.join(
                os.getenv('HOME', '/'), 'tmp/toynn/train_model/tmp')
        flags.device = 'cpu'
        # logging
        flags.print_freq = 500
        flags.summary_freq = 100000
        flags.save_freq = 100000
        flags.test_freq = 500
        # dataset
        flags.dataset = 'cls'
        flags.dim = 100
        flags.n_train = 30
        flags.seed = 0
        # model
        flags.linear = True
        flags.filter_size = 5
        flags.n_conv_layers = 0
        flags.n_fc_layers = 0
        flags.use_bias = False
        flags.init_mul = 1.0
        # optimization
        flags.loss = 'ce'  # ce or l2
        flags.batch_size = 0  # 0: use full batch
        flags.optimizer_name = 'SGD'
        flags.init_lr = 0.01
        flags.scheduled_lrs = ()
        flags.init_wd = 0.0
        flags.scheduled_wds = ()

    def _build(self):
        if self._flags.batch_size == 0:
            self._flags.batch_size = self._flags.n_train
        self._dataset = self._build_dataset()
        # build model after dataset so that it can use e.g. num of classes
        self._model = self._build_model()
        self._optimizer_config = self._build_optimizer_config()
        self._trainer_args = self._build_trainer_args()

    def _build_dataset(self):
        return data.get_dataset(
                name=self._flags.dataset,
                dim=self._flags.dim,
                n_train=self._flags.n_train,
                seed=self._flags.seed)

    def _build_model(self):
        if self._flags.linear:
            activation = 'none'
        else:
            activation = 'relu'
        return net.ModelWrapper(
                loss=self._flags.loss,
                d_in=self._flags.dim,
                k=self._flags.filter_size,
                n_conv_layers=self._flags.n_conv_layers,
                n_fc_layers=self._flags.n_fc_layers,
                activation=activation,
                use_bias=self._flags.use_bias,
                init_mul=self._flags.init_mul,
                )

    def _build_optimizer_config(self):
        opt_cfg = flag_tools.Flags()
        opt_cfg.name = self._flags.optimizer_name

        def _get_lr(step):
            lr = self._flags.init_lr
            for _lr, _step in self._flags.scheduled_lrs:
                if step >= _step:
                    lr = _lr
            return lr

        def _get_wd(step):
            wd = self._flags.init_wd
            for _wd, _step in self._flags.scheduled_wds:
                if step >= _step:
                    wd = _wd
            return wd
        opt_cfg.get_lr = _get_lr
        opt_cfg.get_wd = _get_wd
        return opt_cfg

    @property
    def trainer_args(self):
        return vars(self._trainer_args)

    def _build_trainer_args(self):
        args = flag_tools.Flags()
        args.model = self._model
        args.dataset = self._dataset
        args.optimizer_config = self._optimizer_config
        # copy from flags
        args.log_dir = self._flags.log_dir
        args.batch_size = self._flags.batch_size
        args.total_train_steps = self._flags.total_train_steps
        args.print_freq = self._flags.print_freq
        args.summary_freq = self._flags.summary_freq
        args.save_freq = self._flags.save_freq
        args.test_freq = self._flags.test_freq
        args.device = self._flags.device
        return args
