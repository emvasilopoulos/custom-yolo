import enum
import logging

import torch

import custom_yolo_lib.training.lr_scheduler


class SchedulerType(enum.Enum):
    """
    Enum for different types of schedulers.
    """

    STEP = enum.auto()
    WARMUP_STEP = enum.auto()
    WARMUP_COSINE = enum.auto()
    CUSTOM = enum.auto()


_DEFAULT_STEP_SIZE = -1
_DEFAULT_WARMUP_STEPS = -1
_DEFAULT_MAX_STEPS = -1
_DEFAULT_CYCLES = -1.0
_DEFAULT_MIN_FACTOR = -1.0
_DEFAULT_LAST_STEP = -1


def init_scheduler(
    scheduler_type: SchedulerType,
    optimizer: torch.optim.Optimizer,
    logging_level: int = logging.WARNING,
    update_step_size: int = _DEFAULT_STEP_SIZE,
    warmup_steps: int = _DEFAULT_WARMUP_STEPS,
    max_steps: int = _DEFAULT_MAX_STEPS,
    cycles: float = _DEFAULT_CYCLES,
    min_factor: float = _DEFAULT_MIN_FACTOR,
    last_step: int = _DEFAULT_LAST_STEP,
) -> object:
    """
    Initialize a scheduler based on the type and parameters provided.

    Args:
        scheduler_type (SchedulersType): The type of scheduler to initialize.

    Returns:
        object: An instance of the specified scheduler.
    """
    if scheduler_type == SchedulerType.STEP:
        if warmup_steps != _DEFAULT_WARMUP_STEPS:
            logging.warning(
                f"warmup_steps with value {warmup_steps} is ignored for STEP scheduler"
            )
        if max_steps != _DEFAULT_MAX_STEPS:
            logging.warning(
                f"max_steps with value {max_steps} is ignored for STEP scheduler"
            )
        if cycles != _DEFAULT_CYCLES:
            logging.warning(f"cycles with value {cycles} is ignored for STEP scheduler")
        if min_factor != _DEFAULT_MIN_FACTOR:
            logging.warning(
                f"min_factor with value {min_factor} is ignored for STEP scheduler"
            )
        if last_step != _DEFAULT_LAST_STEP:
            logging.warning(
                f"last_step with value {last_step} is ignored for STEP scheduler"
            )
        return custom_yolo_lib.training.lr_scheduler.StepLRScheduler(
            optimizer=optimizer,
            update_step_size=update_step_size,
            logging_level=logging_level,
        )
    elif scheduler_type == SchedulerType.WARMUP_STEP:
        if max_steps != _DEFAULT_MAX_STEPS:
            logging.warning(
                f"max_steps with value {max_steps} is ignored for WARMUP_STEP scheduler"
            )
        if cycles != _DEFAULT_CYCLES:
            logging.warning(
                f"cycles with value {cycles} is ignored for WARMUP_STEP scheduler"
            )
        if min_factor != _DEFAULT_MIN_FACTOR:
            logging.warning(
                f"min_factor with value {min_factor} is ignored for WARMUP_STEP scheduler"
            )
        if last_step != _DEFAULT_LAST_STEP:
            logging.warning(
                f"last_step with value {last_step} is ignored for WARMUP_STEP scheduler"
            )
        return custom_yolo_lib.training.lr_scheduler.WarmupLRScheduler(
            optimizer=optimizer,
            update_step_size=update_step_size,
            warmup_steps=warmup_steps,
            logging_level=logging_level,
        )
    elif scheduler_type == SchedulerType.WARMUP_COSINE:
        if update_step_size != _DEFAULT_STEP_SIZE:
            logging.warning(
                f"update_step_size with value {update_step_size} is ignored for WARMUP_COSINE scheduler"
            )
        return custom_yolo_lib.training.lr_scheduler.WarmupCosineScheduler(
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            cycles=cycles,
            min_factor=min_factor,
            last_step=last_step,
        )
    elif scheduler_type == SchedulerType.CUSTOM:
        if update_step_size != _DEFAULT_STEP_SIZE:
            logging.warning(
                f"update_step_size with value {update_step_size} is ignored for CUSTOM scheduler"
            )
        if warmup_steps != _DEFAULT_WARMUP_STEPS:
            logging.warning(
                f"warmup_steps with value {warmup_steps} is ignored for CUSTOM scheduler"
            )
        if max_steps != _DEFAULT_MAX_STEPS:
            logging.warning(
                f"max_steps with value {max_steps} is ignored for CUSTOM scheduler"
            )
        if cycles != _DEFAULT_CYCLES:
            logging.warning(
                f"cycles with value {cycles} is ignored for CUSTOM scheduler"
            )
        if min_factor != _DEFAULT_MIN_FACTOR:
            logging.warning(
                f"min_factor with value {min_factor} is ignored for CUSTOM scheduler"
            )
        if last_step != _DEFAULT_LAST_STEP:
            logging.warning(
                f"last_step with value {last_step} is ignored for CUSTOM scheduler"
            )
        return custom_yolo_lib.training.lr_scheduler.MyLRScheduler(
            optimizer=optimizer, logging_level=logging_level
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
