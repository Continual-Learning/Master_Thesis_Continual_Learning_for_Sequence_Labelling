import abc
import math
import typing as t

import torch.optim as optim_module
import torch.optim.lr_scheduler as scheduler_module

from model.model import ClassificationModel


class Optimizer(abc.ABC):
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.optimizers: t.List[optim_module.Optimizer] = []

    def zero_grad(self):
        for opt in self.optimizers:
            opt.zero_grad()

    def step(self):
        for opt in self.optimizers:
            opt.step()

    def state_dict(self):
        return [opt.state_dict() for opt in self.optimizers]

    def load_state_dict(self, state_dicts):
        for opt, state_dict in zip(self.optimizers, state_dicts):
            opt.load_state_dict(state_dict)

    def reset(self):
        return self.__class__(self.model, self.config)


class ClassificationOptimizer(Optimizer):
    def __init__(self, model: ClassificationModel, opt_configs):
        super().__init__(model, opt_configs)
        lr_config = []
        for config in opt_configs:
            trainable_params = []
            for lr_config in config["lr_opts"]:
                trainable_params.extend([
                    {"params": p, "lr": lr_config["lr"]}
                    for n, p in model.named_parameters() if lr_config["layer_type"] in n
                ])
            self.optimizers.append(getattr(optim_module, config['type'])(trainable_params, **config['args']))

        self.model = model


class ClassificationScheduler(abc.ABC):
    def __init__(self, optimizer: Optimizer, config):
        self.schedulers = []
        self.config = config
        for optimizer in optimizer.optimizers:
            self.schedulers.append(getattr(scheduler_module, config['type'])(optimizer, **config['args']))

    def step(self):
        for scheduler in self.schedulers:
            scheduler.step()

    def reset(self, optimizer: optim_module.Optimizer):
        return self.__class__(optimizer, self.config)


class LinearSchedulerWithWarmup():
    def __init__(self, optimizer: Optimizer, lr_scale: float, warmup_steps: int) -> None:
        self.lr_scale = lr_scale
        self.warmup_steps = warmup_steps
        self.optimizers = optimizer.optimizers

        # Store initial learning rates for all optimizers and parameter groups
        # We assume that parameter groups are fixed
        self.start_lrs = []
        for opt in self.optimizers:
            self.start_lrs.append([grp['lr'] for grp in opt.param_groups])

        self.warmup_delta = [
            [(lr * self.lr_scale - lr) / self.warmup_steps for lr in grp_lrs] for grp_lrs in self.start_lrs
        ]
        self.steps = 0

    def __set_lr(self, optimizer: optim_module.Optimizer, group: int, lr: float) -> None:
        optimizer.param_groups[group]['lr'] = lr

    def step(self) -> t.List[float]:
        self.steps += 1
        for i, optimizer in enumerate(self.optimizers):
            for j, (lr, delta) in enumerate(zip(self.start_lrs[i], self.warmup_delta[i])):
                if self.steps <= self.warmup_steps:
                    self.__set_lr(optimizer, j, lr + self.steps * delta)
                else:
                    self.__set_lr(optimizer, j, lr * self.lr_scale / math.sqrt(self.steps))
