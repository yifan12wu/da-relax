from .. import trainer


class Config(trainer.TrainerConfig):

    def _set_default_flags(self):
        super()._set_default_flags()
        flags = self._flags
        flags.total_train_steps = 40000
        flags.batch_size = 128
        flags.optimizer_name = 'Momentum'
        flags.init_lr = 0.001
        flags.scheduled_lrs = []
        flags.weight_decay = 2e-3
