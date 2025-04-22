import torch

from custom_yolo_lib.model.building_blocks.conv_block import ConvBlock


class Backbone(torch.nn.Module):
    def __init__(self) -> None:
        super(Backbone, self).__init__()
        self.layers = torch.nn.Sequential(
            ConvBlock(3, 32, 3, 1, 1),  # Input channels = 3 (RGB)
            torch.nn.MaxPool2d(2, 2),
            ConvBlock(32, 64, 3, 1, 1),
            torch.nn.MaxPool2d(2, 2),
            ConvBlock(64, 128, 3, 1, 1),
            torch.nn.MaxPool2d(2, 2),
            ConvBlock(128, 256, 3, 1, 1),
            torch.nn.MaxPool2d(2, 2),
            ConvBlock(256, 512, 3, 1, 1),
            torch.nn.MaxPool2d(2, 2),
            ConvBlock(512, 1024, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
