import math
import warnings
from typing import Optional, cast, List

from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR, _enable_get_lr_call


class Lin_incr_LRScheduler(_LRScheduler):
    def __init__(self, optimizer, max_lr: float, max_steps: int, current_step: int = None):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.max_steps = max_steps
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        new_lr = self.max_lr / self.max_steps * (1 + current_step)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr


class Lin_incr_offset_LRScheduler(_LRScheduler):
    def __init__(self, optimizer, max_lr: float, max_steps: int, start_step: int, current_step: int = None):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.max_steps = max_steps
        self.start_step = start_step
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        new_lr = self.max_lr / self.max_steps * (1 + current_step - self.start_step)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr


class PolyLRScheduler_offset(_LRScheduler):
    def __init__(
        self,
        optimizer,
        initial_lr: float,
        max_steps: int,
        start_step: int,
        exponent: float = 0.9,
        current_step: int = None,
    ):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max_steps - start_step
        self.start_step = start_step
        self.exponent = exponent
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        current_step = current_step - self.start_step
        if current_step <= 0:
            current_step = 0

        new_lr = self.initial_lr * (1 - current_step / self.max_steps) ** self.exponent
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr


class CosineAnnealingLR_offset(CosineAnnealingLR):
    def __init__(
        self, optimizer: Optimizer, T_max: int, eta_min=0, last_epoch=-1, verbose="deprecated", offset: int = 0
    ):
        self.offset = offset
        super().__init__(
            optimizer,
            T_max,
            eta_min,
            last_epoch,
            verbose,
        )

    def _get_closed_form_lr(self):
        return [
            self.eta_min
            + (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * (self.last_epoch - self.offset) / (self.T_max - self.offset)))
            / 2
            for base_lr in self.base_lrs
        ]

    def step(self, epoch: Optional[int] = None):

        # Raise a warning if old pattern is detected
        # https://github.com/pytorch/pytorch/issues/20124
        if self._step_count == 1:
            if not hasattr(self.optimizer.step, "_wrapped_by_lr_sched"):
                warnings.warn(
                    "Seems like `optimizer.step()` has been overridden after learning rate scheduler "
                    "initialization. Please, make sure to call `optimizer.step()` before "
                    "`lr_scheduler.step()`. See more details at "
                    "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate",
                    UserWarning,
                )

            # Just check if there were two first lr_scheduler.step() calls before optimizer.step()
            elif not getattr(self.optimizer, "_opt_called", False):
                warnings.warn(
                    "Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
                    "In PyTorch 1.1.0 and later, you should call them in the opposite order: "
                    "`optimizer.step()` before `lr_scheduler.step()`.  Failure to do this "
                    "will result in PyTorch skipping the first value of the learning rate schedule. "
                    "See more details at "
                    "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate",
                    UserWarning,
                )
        self._step_count += 1

        with _enable_get_lr_call(self):
            if epoch is None:
                self.last_epoch += 1
            else:
                self.last_epoch = epoch
            values = cast(List[float], self._get_closed_form_lr())

        for i, data in enumerate(zip(self.optimizer.param_groups, values)):
            param_group, lr = data
            if isinstance(param_group["lr"], Tensor):
                lr_val = lr.item() if isinstance(lr, Tensor) else lr  # type: ignore[attr-defined]
                param_group["lr"].fill_(lr_val)
            else:
                param_group["lr"] = lr

        self._last_lr: List[float] = [group["lr"] for group in self.optimizer.param_groups]