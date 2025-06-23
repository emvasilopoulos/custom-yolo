import abc
import collections
import logging
import math
import random
from typing import Dict, List

import numpy as np
import torch

import custom_yolo_lib.logging

# TODO: make all scheduler implement warmup steps option
# TODO #2: make all inherit from 'torch.optim.lr_scheduler.LRScheduler'


class BaseLRScheduler:

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        logging_level: int = logging.WARNING,
    ):
        self.logger = custom_yolo_lib.logging.get_logger(
            __name__, loglevel=logging_level
        )
        self.set_optimizer(optimizer)

    def set_optimizer(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer

    @abc.abstractmethod
    def step(self, loss: torch.nn.Module = None):
        pass


class StepLRScheduler(BaseLRScheduler):

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        update_step_size: int,
        logging_level: int = logging.WARNING,
    ):
        super().__init__(optimizer, logging_level)
        if update_step_size <= 0:
            raise ValueError(
                f"Step size must be greater than 0, but got {update_step_size}"
            )
        self.__step_size = update_step_size
        self.__current_step = 0

        self.param_groups_initial_lrs = [
            param_group["lr"] for param_group in self.optimizer.param_groups
        ]

    def step(self, loss: torch.nn.Module = None):
        self.__current_step += 1
        if self.__current_step % self.__step_size == 0:
            perc = (
                torch.rand(1).item() * 0.1 + 0.85
            )  # equivalent with random.uniform(0.85, 0.95)
            for i, param_group in enumerate(self.optimizer.param_groups):
                if param_group["lr"] > 0.000001:
                    temp = param_group["lr"]
                    param_group["lr"] *= perc
                    self.logger.info(
                        f"Updating 'lr' for param_group-{i} from '{temp:.6f}' to {param_group['lr']:.6f} "
                    )
                else:
                    param_group["lr"] = self.param_groups_initial_lrs[i] / 10
                    self.logger.info(
                        f"Resetting 'lr' for param_group-{i} to {param_group['lr']:.6f} "
                    )

    def get_lr(self):
        return [param_group["lr"] for param_group in self.optimizer.param_groups]


class MyLRScheduler(BaseLRScheduler):

    def __init__(
        self, optimizer: torch.optim.Optimizer, logging_level: int = logging.WARNING
    ):
        super().__init__(optimizer, logging_level)

        self.param_groups_initial_lrs = []
        for param_group in self.optimizer.param_groups:
            self.param_groups_initial_lrs.append(param_group["lr"])

        self.max_len = 100
        self.losses = collections.deque(maxlen=self.max_len)
        self.loss_moving_average = collections.deque(maxlen=self.max_len)
        self.logger = custom_yolo_lib.logging.logging.get_logger("MyLRScheduler")

    def _reset_moving_average(self):
        self.loss_moving_average = collections.deque(maxlen=self.max_len)

    def _fit_line(self, data_points: List[float]):
        x = [i for i in range(len(data_points))]
        y = data_points
        m, b = np.polyfit(x, y, 1)
        return m, b

    def step(self, loss: torch.nn.Module = None):
        self.losses.append(loss.item())
        current_mean = torch.Tensor(self.losses).mean()
        self.loss_moving_average.append(current_mean)
        if len(self.loss_moving_average) == self.loss_moving_average.maxlen:
            line_angle, b = self._fit_line(self.loss_moving_average)
            if line_angle > 0:
                for i, param_group in enumerate(self.optimizer.param_groups):
                    temp = param_group["lr"]

                    # NOTE: This here could be a big mistake. Maybe all LRs should be multiplied with the same value
                    perc = random.uniform(0.85, 0.95)

                    param_group["lr"] *= perc
                    if param_group["lr"] < 0.000001:
                        param_group["lr"] = self.param_groups_initial_lrs[i] / 10
                    self.logger.info(
                        f"Updating 'lr' for param_group-{i} from '{temp:.7f}' to {param_group['lr']:.7f} "
                    )
            self._reset_moving_average()


class WarmupLRScheduler(BaseLRScheduler):
    """
    Warmup learning rate scheduler.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        update_step_size: int,
        warmup_steps: int,
        logging_level: int = logging.WARNING,
    ):
        super().__init__(optimizer, logging_level)

        if update_step_size <= 0:
            raise ValueError(
                f"Step size must be greater than 0, but got {update_step_size}"
            )
        if warmup_steps <= 0:
            raise ValueError(
                f"Warmup steps must be greater than 0, but got {warmup_steps}"
            )

        self.warmup_steps = warmup_steps
        self.current_step = 0
        self.initial_lrs = [
            param_group["lr"] for param_group in self.optimizer.param_groups
        ]
        self.upate_step_size = update_step_size

    def step(self, loss: torch.nn.Module):
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            for i, param_group in enumerate(self.optimizer.param_groups):
                warmup_lr = self.initial_lrs[i] * (
                    self.current_step / self.warmup_steps
                )
                param_group["lr"] = warmup_lr
        else:
            if self.current_step % self.upate_step_size == 0:
                perc = random.uniform(0.90, 0.95)
                for i, param_group in enumerate(self.optimizer.param_groups):
                    param_group["lr"] *= perc
                    if param_group["lr"] < 0.000001:
                        param_group["lr"] = self.initial_lrs[i] / 10

    def get_lr(self):
        return [param_group["lr"] for param_group in self.optimizer.param_groups]


class WarmupCosineScheduler(torch.optim.lr_scheduler.LRScheduler):
    """Cosine decay with linear warm-up.

    Args:
        optimizer: Wrapped optimizer.
        warmup_steps: Number of **initial** steps for which the learning-rate
            increases linearly from 0 to the base LR.
        max_steps: Total number of training steps.  After this many calls to
            :py:meth:`step`, the LR remains fixed at ``min_factor * base_lr``.
        cycles: Number of cosine cycles to complete after warm-up.  The default
            (``0.5``) is a *half* cosine cycle, giving a single smooth decay.
        min_factor: Multiplicative factor applied to the base LR at the end of
            decay (must be in ``[0, 1]``).  For example, ``min_factor=0.1``
            means the LR will never fall below 10 % of the initial LR.
        last_step: The index of the last step.  Default: ``-1`` (scheduler
            starts from step 0 on first call to :py:meth:`step`).

    Note:
        ``max_steps`` **must** be strictly greater than ``warmup_steps``.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        max_steps: int,
        cycles: float = 0.5,
        min_factor: float = 0.0,
        last_step: int = -1,
    ) -> None:
        if warmup_steps < 0 or max_steps <= 0:
            raise ValueError("warmup_steps and max_steps must be positive")
        if warmup_steps >= max_steps:
            raise ValueError("warmup_steps must be < max_steps")
        if not 0.0 <= min_factor <= 1.0:
            raise ValueError("min_factor must be in [0, 1]")
        if cycles <= 0:
            raise ValueError("cycles must be > 0")

        self.warmup_steps: int = warmup_steps
        self.max_steps: int = max_steps
        self.cycles: float = cycles
        self.min_factor: float = min_factor

        super().__init__(optimizer, last_step)

    def _get_warmup_factor(self, step: int) -> float:
        """Return LR factor for a warm-up step (0 ≤ step < warmup_steps)."""
        return (step + 1) / float(self.warmup_steps)

    def _get_cosine_factor(self, step: int) -> float:
        """Return LR factor for a decay step (warmup_steps ≤ step ≤ max_steps)."""
        progress: float = (step - self.warmup_steps) / float(
            max(1, self.max_steps - self.warmup_steps)
        )
        # progress ∈ [0, 1]; cosine cycles spiral downwards
        cosine_decay: float = 0.5 * (
            1.0 + math.cos(math.pi * self.cycles * 2.0 * progress)
        )
        # Scale to [min_factor, 1]
        return self.min_factor + (1.0 - self.min_factor) * cosine_decay

    def get_lr(self) -> List[float]:  # noqa: D401  # (Returns a list, not a rate)
        current_step: int = (
            self.last_epoch
        )  # PyTorch uses ``last_epoch`` for step index
        if current_step < self.warmup_steps:
            factor: float = self._get_warmup_factor(current_step)
        else:
            factor: float = self._get_cosine_factor(current_step)
        return [base_lr * factor for base_lr in self.base_lrs]

    def state_dict(self) -> Dict[str, float]:
        return {
            "last_step": self.last_epoch,
            "warmup_steps": self.warmup_steps,
            "max_steps": self.max_steps,
            "cycles": self.cycles,
            "min_factor": self.min_factor,
        }

    def load_state_dict(self, state_dict: Dict[str, float]) -> None:
        """Load the scheduler state (compat w/ :py:meth:`state_dict`)."""
        self.last_epoch = state_dict["last_step"]
        self.warmup_steps = int(state_dict["warmup_steps"])
        self.max_steps = int(state_dict["max_steps"])
        self.cycles = float(state_dict["cycles"])
        self.min_factor = float(state_dict["min_factor"])
