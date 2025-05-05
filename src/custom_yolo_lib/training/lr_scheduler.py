import abc
import collections
import logging
import random
from typing import List

import numpy as np
import torch

import custom_yolo_lib.logging


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
    def update_loss(self, loss: torch.nn.Module):
        pass


class StepLRScheduler(BaseLRScheduler):

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        update_step_size: int,
        logging_level: int = logging.WARNING,
    ):
        super().__init__(optimizer, logging_level)
        self.__step_size = update_step_size
        self.__current_step = 0

        self.param_groups_initial_lrs = [
            param_group["lr"] for param_group in self.optimizer.param_groups
        ]

    def update_loss(self, loss: torch.nn.Module):
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

    def update_loss(self, loss: torch.nn.Module):
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
        self.warmup_steps = warmup_steps
        self.current_step = 0
        self.initial_lrs = [
            param_group["lr"] for param_group in self.optimizer.param_groups
        ]
        self.upate_step_size = update_step_size

    def update_loss(self, loss: torch.nn.Module):
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
