import torch
import torchvision


class SimpleImageNormalizer(torch.nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / 255.0


class ImageNetNormalizer(torch.nn.Module):

    def __init__(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self._normalizer = torchvision.transforms.Normalize(mean, std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._normalizer(x)
