import torch


def initialize_weights(tensor_shape: torch.Size):
    return torch.rand(tensor_shape)


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
