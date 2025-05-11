import enum
import logging

import torch

import custom_yolo_lib.training.utils
import custom_yolo_lib.logging

LOGGER = custom_yolo_lib.logging.get_logger(__name__, loglevel=logging.WARNING)


class OptimizerType(enum.Enum):
    VANILLA = enum.auto()
    SPLIT_GROUPS_ADAMW = enum.auto()


def init_optimizer(
    optimizer_type: OptimizerType,
    model: torch.nn.Module,
    initial_lr: float,
    momentum: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    if optimizer_type == OptimizerType.VANILLA:
        LOGGER.warning("Vanilla optimizer ignores weight decay & momentum")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=initial_lr,
        )
    elif optimizer_type == OptimizerType.SPLIT_GROUPS_ADAMW:
        parameters_grouped = custom_yolo_lib.training.utils.get_params_grouped(model)
        optimizer = torch.optim.AdamW(
            parameters_grouped.with_weight_decay,
            lr=initial_lr,
            betas=(momentum, 0.999),
            weight_decay=weight_decay,
        )
        optimizer.add_param_group(
            {"params": parameters_grouped.bias, "weight_decay": weight_decay}
        )
        optimizer.add_param_group(
            {"params": parameters_grouped.no_weight_decay, "weight_decay": 0.0}
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    return optimizer
