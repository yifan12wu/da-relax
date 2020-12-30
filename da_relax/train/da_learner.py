import os
import logging
import collections

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils import tensorboard

from ..tools import timer_tools
from ..tools import py_tools
from ..tools import flag_tools
from ..tools import summary_tools
from ..tools import torch_tools


DEFAULT_LOG_DIR = os.path.join(os.getenv('HOME', '/'), 'tmp/da-relax/test')


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


class DALearner:

    @py_tools.store_args
    def __init__(self,
            # pytorch
            device=None,
            # data args
            source_dataset=None,
            target_dataset=None,
            # model training args
            model_cfg=None,
            optimizer_cfg=None,
            batch_size=128,
            # training loop args
            log_dir=DEFAULT_LOG_DIR,
            total_train_steps=30000,
            print_freq=2000,
            summary_freq=100,
            save_freq=10000,
            eval_freq=10000,
            ):
        self._build()

    def _build(self):
        # device
        if self._device is None:
            self._device = torch.device(
                    'cuda' if torch.cuda.is_available() else 'cpu')
        logging.info('Device: {}.'.format(self._device))
        self._build_model()
        self._build_optimizer()
        self._build_dataset()
        # information maintained during training and evaluation
        self._global_step = 0
        self._train_info = collections.OrderedDict()
        self._eval_info = collections.OederedDict()
        self._result_info = collections.OederedDict()
        self._summary_writer = tensorboard.SummaryWriter(
                log_dir=self._log_dir)

    def _build_model(self):
        self._model.to(device=self._device)

    def _build_optimizer(self):
        cfg = self._optimizer_config
        opt_fn = get_optimizer(cfg.name)
        self._optimizer = opt_fn(self._model.parameters(),
                lr=cfg.get_lr(0), weight_decay=cfg.get_wd(0))

    def _build_dataset(self):
        source_train = self._source_dataset.train
        source_valid = self._source_dataset.valid
        target_train = self._target_dataset.train
        target_valid = self._target_dataset.valid
        target_test = self._target_dataset.test
        # setup data iteraters for training
        self._source_train_iter = source_train.get_random_iterator(
                self._batch_size)
        self._target_train_iter = target_train.get_random_iterator(
                self._batch_size)
        self._source_valid_iter = source_valid.get_random_iterator(
                self._batch_size)
        self._target_valid_iter = target_valid.get_random_iterator(
                self._batch_size)
        def _get_eval_valid_iter():
            return target_valid.get_one_shot_iterator(self._batch_size)
        def _get_eval_test_iter():
            return target_test.get_one_shot_iterator(self._batch_size)
        self._get_eval_valid_iter = _get_eval_valid_iter
        self._get_eval_test_iter = _get_eval_test_iter

    def _save_model(self, step=None):
        if step is not None:
            filename = 'model-{}.ckpt'.format(step)
        else:
            filename = 'model.ckpt'
        model_path = os.path.join(self._log_dir, filename)
        torch.save(self._model.state_dict(), model_path)

    def _get_train_batch(self):
        source_batch, _ = next(self._source_train_iter)
        target_batch, _ = next(self._target_train_iter)
        batch = [source_batch.x, source_batch.y, target_batch.x]
        batch = [torch_tools.to_tensor(x, self._device) for x in batch]
        return batch

    def _build_train_loss(self, batch):
        info = collections.OrderedDict()
        output_head = self._model(x, is_training=True)
        output_head.get_label(y)
        info['train_acc'] = output_head.acc.item()
        info['train_loss'] = output_head.loss.item()
        return output_head.loss, info

    def _train_step(self):
        batch = self._get_train_batch()
        loss, info = self._build_train_loss(batch)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        self._train_info.update(info)
        self._global_step += 1

    def _get_valid_batch(self):
        source_batch, _ = next(self._source_valid_iter)
        target_batch, _ = next(self._target_valid_iter)
        batch = [source_batch.x, source_batch.y, 
                target_batch.x, target_batch.y]
        batch = [torch_tools.to_tensor(x, self._device) for x in batch]
        return batch

    def _get_valid_info(self, batch):
        info = collections.OrderedDict()
        output_head = self._model(x, is_training=False)
        output_head.get_label(y)
        info['valid_acc'] = output_head.acc.item()
        info['valid_loss'] = output_head.loss.item()
        return info

    def _valid_step(self):
        batch = self._get_valid_batch()
        info = self._get_valid_info(batch)
        self._train_info.update(info)

    def train(self):
        # 
        timer = timer_tools.Timer()
        # -- start training
        for step in range(self._total_train_steps):
            assert step == self._global_step
            self._train_step()
            ## 
            # write summary to tensorboard
            if step == 0 or (step + 1) % self._summary_freq == 0:
                # get validation info on a minibatch
                self._valid_step()
                steps_per_sec = timer.steps_per_sec(self._global_step)
                self._train_info['steps_per_sec'] = steps_per_sec
                summary_tools.write_summary(
                        writer=self._summary_writer,
                        info=self._train_info,
                        step=self._global_step)
            # print info
            if step == 0 or (step + 1) % self._print_freq == 0:
                summary_str = summary_tools.get_summary_str(
                        step=self._global_step,
                        info=self._train_info)
                logging.info(summary_str)
            # save ckpt
            if (step + 1) % self._save_freq == 0:
                self._save_model()
            # run through the whole validation set
            if (step + 1) % self._eval_freq == 0:
                logging.info('Evaluating on validation at step {}/{}.'.
                        format(self._global_step, self._total_train_steps))
                self._eval_on_valid()
                # print and write summary for evaluation
                logging.info(summary_tools.get_summary_str(
                        info=self._eval_info))
                summary_tools.write_summary(
                        writer=self._summary_writer,
                        info=self._eval_info,
                        step=self._global_step)
        # print out total training time
        time_cost = timer.time_cost()
        logging.info('Training finished, time cost {:.4g}s.'
                .format(time_cost))
        # save ckpt
        self._save_model()
        # run through the test set
        logging.info('Evaluating on test.')
        self._eval_on_test()
        logging.info(summary_tools.get_summary_str(info=self._result_info))
        self._save_result()

    def _evaluate(self, data_iter):
        total_loss = 0.0
        total_acc = 0.0
        total_size = 0
        for batch, size in data_iter:
            x = torch_tools.to_tensor(batch.x, self._device)
            y = torch_tools.to_tensor(batch.y, self._device)
            output_head = self._f_fn(x, is_training=False)
            output_head.get_label(y)
            loss = output_head.losses[:size].sum().item()
            acc = output_head.accs[:size].sum().item()
            total_loss += loss
            total_acc += acc
            total_size += size
        info = collections.OrderedDict()
        info['eval_acc'] = total_acc / total_size
        info['eval_loss'] = total_loss / total_size
        return info

    def _eval_on_valid(self):
        data_iter = self._get_eval_valid_iter()
        self._eval_info.update(self._evaluate(data_iter))

    def _eval_on_test(self):
        data_iter = self._get_eval_test_iter()
        self._result_info.update(self._evaluate(data_iter))

    def _save_result(self):
        result_file = os.path.join(self._log_dir, 'result.csv')
        result = np.array(self._result_info['eval_acc'])
        with open(result_file, 'w') as f:
            np.savetxt(f, result, fmt='%.4g', delimiter=',')


class DAModel(nn.Module):

    def __init__(self,
            f_model_factory,
            d_model_factory,
            ):
        super().__init__()
        self.f_fn = f_model_factory()
        self.d_fn = d_model_factory()


class DALearnerConfig(flag_tools.ConfigBase):

    def _set_default_flags(self):
        flags = self._flags
        # 
        flags.device = None
        flags.total_train_steps = 20000
        flags.log_dir = DEFAULT_LOG_DIR
        # 
        flags.batch_size = 128
        # logging
        flags.print_freq = 2000
        flags.summary_freq = 100
        flags.save_freq = 5000
        flags.eval_freq = 10000
        # optimizer
        flags.opt_args = flag_tools.Flags(
                name_f='Adam', lr_f=1e-3, wd_f=0.0,
                name_d='Adam', lr_d=1e-3, wd_d=0.0,
        )

    def _build(self):
        self._build_dataset()
        self._build_model()
        self._build_optimizer()
        self._build_args()

    def _build_dataset(self):
        self._source_dataset = self._source_dataset_factory()
        self._target_dataset = self._target_dataset_factory()

    def _build_model(self):
        self._model_cfg = flag_tools.Flags(
            f_model_factory=self._f_model_factory,
            d_model_factory=self._d_model_factory,
        )

    def _build_optimizer(self):
        self._optimizer_cfg = flag_tools.Flags(
            f_optimizer_factory=self._f_optimizer_factory,
            d_optimizer_factory=self._d_optimizer_factory,
        )

    def _source_dataset_factory(self):
        raise NotImplementedError

    def _target_dataset_factory(self):
        raise NotImplementedError

    def _f_model_factory(self):
        raise NotImplementedError

    def _d_model_factory(self):
        raise NotImplementedError

    def _f_optimizer_factory(self, parameters):
        raise NotImplementedError

    def _d_optimizer_factory(self, parameters):
        raise NotImplementedError

    def _build_args(self):
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

    @property
    def args(self):
        return vars(self._args)

