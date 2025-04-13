import torch

from custom_yolo_lib.model.backbones.vanilla_cnn import ConvBlock


class AdditiveNeck(torch.nn.Module):

    def __init__(self) -> None:
        super(AdditiveNeck, self).__init__()
        self.conv_large = ConvBlock(1024, 512, 1, 1, 0)
        self.upsample1 = torch.nn.Upsample(
            scale_factor=2, mode="nearest"
        )  # Upsample by 2

        self.conv_medium = ConvBlock(512, 256, 1, 1, 0)
        self.upsample2 = torch.nn.Upsample(
            scale_factor=2, mode="nearest"
        )  # Upsample by 2

    def forward(self, small: torch.Tensor, medium: torch.Tensor, large: torch.Tensor):
        small = small + self.upsample2(self.conv_medium(medium))
        medium = medium + self.upsample1(self.conv_large(large))

        return small, medium, large
