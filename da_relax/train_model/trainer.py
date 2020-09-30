import os
import logging
import collections

import numpy as np
import torch
from torch import optim
from torch.utils import tensorboard

from ..data import datasets
from ..networks import nets

from ..tools import timer_tools
from ..tools import py_tools
from ..tools import flag_tools
from ..tools import summary_tools


def get_optimizer(name):
    if name == 'SGD':
        return optim.SGD
    if name == 'Momentum':
        def _opt_fn(params, lr=0.001, weight_decay=0.0):
            return optim.SGD(
                    params, 
                    lr=lr, 
                    weight_decay=weight_decay, 
                    momentum=0.9)
        return _opt_fn
    if name == 'Adam':
        return optim.Adam
    else:
        raise ValueError('Unknown optimizer {}.'.format(name))


class Trainer:

    @py_tools.store_args
    def __init__(self,
            log_dir=os.path.join(os.getenv('HOME', '/'), 'tmp'),
            # model training args
            model=None,
            dataset=None,
            optimizer_config=None,
            batch_size=128,
            # training loop args
            total_train_steps=30000,
            print_freq=2000,
            summary_freq=100,
            save_freq=10000,
            test_freq=10000,
            # others
            device=None
            ):
        self._build()

    @property
    def model(self):
        return self._model

    def _build(self):
        # device
        if self._device is None:
            self._device = torch.device(
                    'cuda' if torch.cuda.is_available() else 'cpu')
            logging.info('Device: {}.'.format(self._device))
        self._model.to(device=self._device)
        self._build_optimizer()
        self._global_step = 0
        self._train_info = collections.OrderedDict()
        self._summary_writer = tensorboard.SummaryWriter(
                log_dir=self._log_dir)

    def _build_optimizer(self):
        cfg = self._optimizer_config
        opt_fn = get_optimizer(cfg.name)
        self._optimizer = opt_fn(self._model.parameters(),
                lr=cfg.get_lr(0), weight_decay=cfg.get_wd(0))

    def _maybe_update_optimizer(self):
        lr = self._optimizer_config.get_lr(self._global_step)
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
        wd = self._optimizer_config.get_wd(self._global_step)
        for param_group in self._optimizer.param_groups:
            param_group['weight_decay'] = wd

    def _get_current_lr(self):
        return self._optimizer.param_groups[0]['lr']

    def _get_current_wd(self):
        return self._optimizer.param_groups[0]['weight_decay']

    def _tensor(self, x):
        if x.dtype in [np.float32, np.float64]:
            return torch.tensor(x, dtype=torch.float32, device=self._device)
        elif x.dtype in [np.int32, np.int64, np.uint8]:
            return torch.tensor(x, dtype=torch.int64, device=self._device)
        else:
            raise ValueError('Unknown dtype {} for tensor.'
                    .format(str(x.dtype)))

    def _train_step(self, batch):
        info = collections.OrderedDict()
        x = self._tensor(batch.x)
        y = self._tensor(batch.y)
        output_head = self._model(x, is_training=True)
        output_head.get_label(y)
        self._optimizer.zero_grad()
        output_head.loss.backward()
        self._optimizer.step()
        self._global_step += 1
        self._maybe_update_optimizer()
        info['train_acc'] = output_head.acc.item()
        info['train_loss'] = output_head.loss.item()
        info['lr'] = self._get_current_lr()
        info['wd'] = self._get_current_wd()
        return info

    def _get_valid_info(self, batch):
        info = collections.OrderedDict()
        x = self._tensor(batch.x)
        y = self._tensor(batch.y)
        output_head = self._model(x, is_training=False)
        output_head.get_label(y)
        info['valid_acc'] = output_head.acc.item()
        info['valid_loss'] = output_head.loss.item()
        return info

    def train(self):
        dtr = self._dataset.train
        dva = self._dataset.valid
        train_iter = dtr.get_random_iterator(self._batch_size)
        valid_iter = dva.get_random_iterator(self._batch_size)
        # dte = self._dataset.test
        global_timer = timer_tools.Timer()
        local_timer = timer_tools.Timer()
        final_eval_done = False
        # -- start training
        while self._global_step < self._total_train_steps:
            train_batch, _ = next(train_iter)
            # make sure that step is incr. by 1 in training_step
            train_info = self._train_step(train_batch)
            self._train_info.update(train_info)
            # write summary to tensorboard
            if self._global_step % self._summary_freq == 0:
                # get validation loss and acc
                valid_batch, _ = next(valid_iter)
                valid_info = self._get_valid_info(valid_batch)
                self._train_info.update(valid_info)
                summary_tools.write_summary(
                        writer=self._summary_writer,
                        info=self._train_info,
                        step=self._global_step)
            # print info
            if self._global_step % self._print_freq == 0:
                summary_str = summary_tools.get_summary_str(
                        step=self._global_step,
                        info=self._train_info)
                logging.info(summary_str)
                steps_per_sec = local_timer.steps_per_sec_and_refresh(
                        self._global_step)
                logging.info('Steps per second: {:.4g}.'
                        .format(steps_per_sec))
            # save ckpt
            if self._global_step % self._save_freq == 0:
                self._save_model()
            # run through validation set
            if self._global_step % self._test_freq == 0:
                logging.info('Testing at step {}/{}'.
                        format(self._global_step, self._total_train_steps))
                valid_eval_iter = dva.get_one_shot_iterator(self._batch_size)
                eval_info = self._evaluate(valid_eval_iter)
                logging.info(summary_tools.get_summary_str(info=eval_info))
                if self._global_step == self._total_train_steps:
                    final_eval_done = True
        # save ckpt
        self._save_model()
        if not final_eval_done:
            # run through validation set
            logging.info('Testing at step {}/{}'.
                    format(self._global_step, self._total_train_steps))
            valid_eval_iter = dva.get_one_shot_iterator(self._batch_size)
            eval_info = self._evaluate(valid_eval_iter)
            logging.info(summary_tools.get_summary_str(info=eval_info))
        time_cost = global_timer.time_cost()
        logging.info('Training finished, time cost {:.4g}s.'
                .format(time_cost))

    def _save_model(self):
        model_path = os.path.join(self._log_dir, 'model.ckpt')
        torch.save(self._model.state_dict(), model_path)

    def _evaluate(self, data_iter):
        info = collections.OrderedDict()
        total_loss = 0.0
        total_acc = 0.0
        total_size = 0
        for batch, size in data_iter:
            x = self._tensor(batch.x)
            y = self._tensor(batch.y)
            output_head = self._model(x, is_training=False)
            output_head.get_label(y)
            loss = output_head.losses[:size].sum().item()
            acc = output_head.accs[:size].sum().item()
            total_loss += loss
            total_acc += acc
            total_size += size
        info['acc'] = total_acc / total_size
        info['loss'] = total_loss / total_size
        return info


class TrainerConfig(flag_tools.ConfigBase):

    def _set_default_flags(self):
        flags = self._flags
        flags.total_train_steps = 30000
        flags.log_dir = os.path.join(
                os.getenv('HOME', '/'), 'tmp/uncertainty/train_model/tmp')
        flags.model = 'lenet'
        flags.data_dir = None
        flags.dataset = 'cifar5'
        flags.batch_size = 128
        # logging
        flags.print_freq = 2000
        flags.summary_freq = 100
        flags.save_freq = 10000
        flags.test_freq = 10000
        # optimizer
        flags.optimizer_name = 'Momentum'
        flags.init_lr = 0.001
        flags.scheduled_lrs = ()
        flags.init_wd = 0.0
        flags.scheduled_wds = ()
        #
        flags.device = None

    def _build(self):
        self._dataset = self._build_dataset()
        # build model after dataset so that it can use e.g. num of classes
        self._model = self._build_model()
        self._optimizer_config = self._build_optimizer_config()
        self._trainer_args = self._build_trainer_args()

    def _build_dataset(self):
        args = {}
        if self._flags.data_dir is not None:
            args['data_dir'] = self._flags.data_dir
        return datasets.get_dataset(self._flags.dataset, **args)

    def _build_model(self):
        n_classes = self._dataset.info.n_classes
        return nets.get_model(self._flags.model, n_classes=n_classes)

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


