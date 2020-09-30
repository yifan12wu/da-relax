from .. import trainer


class Config(trainer.TrainerConfig):

    def _set_default_flags(self):
        super()._set_default_flags()
        flags = self._flags
        flags.total_train_steps = 30000
        flags.batch_size = 128
        flags.optimizer_name = 'Momentum'
        flags.init_lr = 0.1
        flags.scheduled_lrs = [(0.01, 10000), (0.001, 20000), (0.0002, 30000)]
        flags.weight_decay = 5e-4

