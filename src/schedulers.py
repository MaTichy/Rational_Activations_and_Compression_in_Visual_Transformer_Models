"""Custom learning rate schedulers with warmup support."""

from __future__ import annotations

import math
from torch.optim.lr_scheduler import StepLR, LRScheduler


class WarmupStepLR(LRScheduler):
    """Linear warmup followed by step decay.

    Args:
        optimizer: Wrapped optimizer.
        warmup_epochs: Number of warmup epochs with linear ramp.
        step_size: Period of learning rate decay after warmup.
        start_lr_warmup: Starting learning rate for warmup phase.
        gamma: Multiplicative factor of learning rate decay.
    """

    def __init__(self, optimizer, warmup_epochs, step_size, start_lr_warmup, gamma, last_epoch=-1) -> None:
        self.warmup_epochs = warmup_epochs
        self.step_size = step_size
        self.start_lr_warmup = start_lr_warmup
        self.gamma = gamma
        self.lr_decay = StepLR(optimizer, step_size=step_size, gamma=gamma, last_epoch=last_epoch)
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        if self.last_epoch < self.warmup_epochs:
            return [
                self.start_lr_warmup + (base_lr - self.start_lr_warmup) / self.warmup_epochs * self.last_epoch
                for base_lr in self.base_lrs
            ]
        return self.lr_decay.get_last_lr()

    def step(self, epoch: int | None = None) -> None:
        if self.last_epoch < self.warmup_epochs:
            super().step(epoch)
        else:
            self.lr_decay.step(epoch)


class WarmupCosineLR(LRScheduler):
    """Linear warmup followed by cosine annealing.

    Args:
        optimizer: Wrapped optimizer.
        warmup_epochs: Number of warmup epochs.
        cycle_length: Length of cosine annealing cycle.
        start_lr_warmup: Starting learning rate for warmup phase.
    """

    def __init__(self, optimizer, warmup_epochs, cycle_length, start_lr_warmup=0, last_epoch=-1) -> None:
        self.warmup_epochs = warmup_epochs
        self.cycle_length = cycle_length
        self.start_lr_warmup = start_lr_warmup
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        if self.last_epoch < self.warmup_epochs:
            return [
                self.start_lr_warmup + (base_lr - self.start_lr_warmup) / self.warmup_epochs * self.last_epoch
                for base_lr in self.base_lrs
            ]
        step = (self.last_epoch - self.warmup_epochs) % self.cycle_length
        return [
            0.5 * (1.0 + math.cos(math.pi * step / (self.cycle_length - 1))) * base_lr
            for base_lr in self.base_lrs
        ]

    def step(self, epoch: int | None = None) -> None:
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr
