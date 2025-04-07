import torch

from custom_yolo_lib.model.backbones.vanilla_cnn import ConvBlock


class NeckConcat(torch.nn.Module):
    # TODO - add expected input channels for each scale

    def __init__(self):
        super(NeckConcat, self).__init__()
        self.conv_large = ConvBlock(1024, 512, 1, 1, 0)
        self.upsample1 = torch.nn.Upsample(scale_factor=2, mode="nearest")

        self.conv_medium = ConvBlock(512, 256, 1, 1, 0)
        self.upsample2 = torch.nn.Upsample(scale_factor=2, mode="nearest")

        self.conv_small = ConvBlock(256, 128, 1, 1, 0)

        self.fusion_medium = ConvBlock(768, 256, 1, 1, 0)
        self.fusion_small = ConvBlock(384, 128, 1, 1, 0)

    def forward(self, small, medium, large):
        large = self.conv_large(large)
        medium = self.conv_medium(medium)
        small = self.conv_small(small)

        medium = torch.cat([medium, self.upsample1(large)], dim=1)
        medium = self.fusion_medium(medium)

        small = torch.cat([small, self.upsample2(medium)], dim=1)
        small = self.fusion_small(small)

        return small, medium, large
