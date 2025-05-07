# Copied from ultralytics
import dataclasses
from typing import List
import torch


def _get_pytorch_norm_layers():
    return tuple(v for k, v in torch.nn.__dict__.items() if "Norm" in k)


@dataclasses.dataclass
class GroupedParams:
    with_weight_decay: List[torch.nn.Parameter] = dataclasses.field(
        default_factory=List[torch.nn.Parameter]
    )
    no_weight_decay: List[torch.nn.Parameter] = dataclasses.field(
        default_factory=List[torch.nn.Parameter]
    )
    bias: List[torch.nn.Parameter] = dataclasses.field(
        default_factory=List[torch.nn.Parameter]
    )


def get_params_grouped(model: torch.nn.Module) -> GroupedParams:
    bn = _get_pytorch_norm_layers()

    parameters_grouped = GroupedParams(with_weight_decay=[], no_weight_decay=[], bias=[])
    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            fullname = f"{module_name}.{param_name}" if module_name else param_name
            if "bias" in fullname:  # bias (no decay)
                parameters_grouped.bias.append(param)
            elif isinstance(module, bn):  # weight (no decay)
                parameters_grouped.no_weight_decay.append(param)
            else:  # weight (with decay)
                parameters_grouped.with_weight_decay.append(param)
    return parameters_grouped
