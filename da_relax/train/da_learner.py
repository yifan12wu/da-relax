import os
import logging
import collections

import numpy as np
import torch
from torch import nn
from torch import autograd
from torch import optim
from torch.utils import tensorboard

from ..tools import timer_tools
from ..tools import py_tools
from ..tools import flag_tools
from ..tools import summary_tools
from ..tools import torch_tools

from . import utils


DEFAULT_LOG_DIR = os.path.join(os.getenv('HOME', '/'), 'tmp/da-relax/test')


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
            d_loss_w=1.0,  # weight for distribution matching loss
            d_relax=0.0,  # relaxation for distribution matching
            d_loss_name='js_beta',  # divergence name
            d_grad_penalty=0.0,  # gradient penalty for regularizing d
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
        self._eval_info = collections.OrderedDict()
        self._results = []
        self._summary_writer = tensorboard.SummaryWriter(
                log_dir=self._log_dir)

    def _build_model(self):
        self._model = self._model_cfg.model_factory()
        self._model.to(device=self._device)
        self._f_fn = self._model.f_fn
        self._d_fn = self._model.d_fn
        self._div_fn = utils.get_div_fn(self._d_loss_name, self._d_relax)

    def _build_optimizer(self):
        cfg = self._optimizer_cfg
        self._f_optimizer = cfg.f_optimizer_factory(self._f_fn.parameters())
        self._d_optimizer = cfg.d_optimizer_factory(self._d_fn.parameters())

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

    def _build_f_loss(self, batch):
        source_x, source_y, target_x = batch
        source_output = self._f_fn(source_x)
        source_output.get_label(source_y)
        f_loss = source_output.loss
        target_output = self._f_fn(target_x)
        # taking the last hidden layer as representations
        source_z = source_output.features
        target_z = target_output.features
        z = torch.cat([source_z, target_z], 0)
        dz = self._d_fn(z)
        n_source = source_z.shape[0]
        source_dz = dz[:n_source]
        target_dz = dz[n_source:]
        fd_loss = self._div_fn(source_dz, target_dz)
        total_f_loss = f_loss + self._d_loss_w * fd_loss
        info = collections.OrderedDict()
        info['train_source_acc'] = source_output.acc.item()
        info['train_f_loss'] = f_loss.item()
        info['train_fd_loss'] = fd_loss.item()
        info['train_total_f_loss'] = total_f_loss.item()
        return total_f_loss, info

    def _build_d_grad_loss(self, z1, z2):
        """gradient penalty on random interpolations between z1 and z2."""
        alpha = torch.rand(z1.shape[0], 1, device=self._device)
        z_intp = alpha * z1 + (1 - alpha) * z2
        z_intp.requires_grad = True
        dz_intp = self._d_fn(z_intp)
        z_intp_grad = autograd.grad(outputs=dz_intp, inputs=z_intp,
                grad_outputs=torch.ones(dz_intp.shape, device=self._device),
                create_graph=True, retain_graph=True)[0]
        z_grad_norm = (z_intp_grad.square().sum(-1) + 1e-10).sqrt()
        d_grad_loss = (z_grad_norm - 1.0).square().mean()
        return d_grad_loss

    def _build_d_loss(self, batch):
        source_x, _, target_x = batch
        with torch.no_grad():
            source_output = self._f_fn(source_x)
            target_output = self._f_fn(target_x)
        source_z = source_output.features
        target_z = target_output.features
        z = torch.cat([source_z, target_z], 0)
        dz = self._d_fn(z)
        n_source = source_z.shape[0]
        source_dz = dz[:n_source]
        target_dz = dz[n_source:]
        d_loss = - self._div_fn(source_dz, target_dz)
        info = collections.OrderedDict()
        if self._d_grad_penalty > 0:
            # gradient penalty
            d_grad_loss = self._build_d_grad_loss(source_z, target_z)
            total_d_loss = d_loss + self._d_grad_penalty * d_grad_loss
            info['train_d_grad_loss'] = d_grad_loss.item()
        else:
            total_d_loss = d_loss
        info['train_d_loss'] = d_loss.item()
        info['train_total_d_loss'] = total_d_loss.item()
        return total_d_loss, info

    def _train_step(self):
        batch = self._get_train_batch()
        # update f
        f_loss, f_info = self._build_f_loss(batch)
        self._f_optimizer.zero_grad()
        f_loss.backward()
        self._f_optimizer.step()
        self._train_info.update(f_info)
        # update d
        d_loss, d_info = self._build_d_loss(batch)
        self._d_optimizer.zero_grad()
        d_loss.backward()
        self._d_optimizer.step()
        self._train_info.update(d_info)
        # step += 1
        self._global_step += 1

    def _get_valid_batch(self):
        source_batch, _ = next(self._source_valid_iter)
        target_batch, _ = next(self._target_valid_iter)
        batch = [source_batch.x, source_batch.y, 
                target_batch.x, target_batch.y]
        batch = [torch_tools.to_tensor(x, self._device) for x in batch]
        return batch

    def _get_valid_info(self, batch):
        source_x, source_y, target_x, target_y = batch
        with torch.no_grad():
            source_output = self._f_fn(source_x)
            source_output.get_label(source_y)
            target_output = self._f_fn(target_x)
            target_output.get_label(target_y)
        info = collections.OrderedDict()
        info['valid_source_acc'] = source_output.acc.item()
        info['valid_source_loss'] = source_output.loss.item()
        info['valid_target_acc'] = target_output.acc.item()
        info['valid_target_loss'] = target_output.loss.item()
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
        logging.info('Accuracy: {:.4g}.'.format(self._results[-1]))
        self._save_result()

    def _evaluate(self, data_iter):
        total_loss = 0.0
        total_acc = 0.0
        total_size = 0
        for batch, size in data_iter:
            x = torch_tools.to_tensor(batch.x, self._device)
            y = torch_tools.to_tensor(batch.y, self._device)
            with torch.no_grad():
                output_head = self._f_fn(x)
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
        self._results.append(self._evaluate(data_iter)['eval_acc'])

    def _save_result(self):
        result_file = os.path.join(self._log_dir, 'result.csv')
        with open(result_file, 'w') as f:
            np.savetxt(f, self._results, fmt='%.4g', delimiter=',')


class DAModel(nn.Module):

    def __init__(self,
            f_model_factory,
            d_model_factory,
            ):
        super().__init__()
        self.f_fn = utils.FModelWrapper(f_model_factory, repr_layer=-2)
        self.d_fn = utils.DModelWrapper(d_model_factory)


class DALearnerConfig(flag_tools.ConfigBase):

    def _set_default_flags(self):
        flags = self._flags
        #
        flags.device = None
        # 
        flags.batch_size = 128
        flags.d_loss_w = 1.0
        flags.d_relax = 0.0
        flags.d_loss_name = 'js'
        flags.d_grad_penalty = 0.0
        #
        flags.log_dir = DEFAULT_LOG_DIR
        flags.total_train_steps = 20000
        flags.print_freq = 2000
        flags.summary_freq = 100
        flags.save_freq = 5000
        flags.eval_freq = 10000
        # optimizer
        # name, learning rate, weight decay
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
        def _model_factory():
            return DAModel(self._f_model_factory, self._d_model_factory)
        self._model_cfg = flag_tools.Flags(model_factory=_model_factory)

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

    def _get_optimizer(self, parameters, name, lr, wd):
        if name == 'Adam2':
            optimizer = optim.Adam(parameters, betas=(0.5, 0.999),
                    lr=lr, weight_decay=wd)
        else:
            opt_cls = getattr(optim, name)
            optimizer = opt_cls(parameters, lr=lr, weight_decay=wd)
        return optimizer

    def _f_optimizer_factory(self, parameters):
        return self._get_optimizer(parameters, self._flags.opt_args.name_f,
                self._flags.opt_args.lr_f, self._flags.opt_args.wd_f)

    def _d_optimizer_factory(self, parameters):
        return self._get_optimizer(parameters, self._flags.opt_args.name_d,
                self._flags.opt_args.lr_d, self._flags.opt_args.wd_d)

    def _build_args(self):
        args = flag_tools.Flags()
        # non-flag args
        args.source_dataset = self._source_dataset
        args.target_dataset = self._target_dataset
        args.model_cfg = self._model_cfg
        args.optimizer_cfg = self._optimizer_cfg
        # copy from flags
        keys = [
                'device', 
                'batch_size', 
                'd_loss_w', 
                'd_relax', 
                'd_loss_name', 
                'd_grad_penalty', 
                'log_dir', 
                'total_train_steps',
                'print_freq',
                'summary_freq',
                'save_freq',
                'eval_freq',
                ]
        for key in keys:
            setattr(args, key, getattr(self._flags, key))
        #
        self._args = args

    @property
    def args(self):
        return vars(self._args)

