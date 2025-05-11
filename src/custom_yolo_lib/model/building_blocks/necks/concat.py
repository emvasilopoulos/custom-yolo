import torch

from custom_yolo_lib.model.building_blocks.conv_block import ConvBlock


class ConcatNeck(torch.nn.Module):

    def __init__(
        self, small_in_channels: int, medium_in_channels: int, large_in_channels: int
    ) -> None:
        super(ConcatNeck, self).__init__()
        self.small_output_channels = 128
        self.medium_output_channels = 256
        self.large_output_channels = 512

        _HIDDEN_MEDIUM = 256
        _HIDDEN_SMALL = 128

        self.conv_large = ConvBlock(
            large_in_channels, self.large_output_channels, 1, 1, 0
        )
        self.upsample1 = torch.nn.Upsample(scale_factor=2, mode="nearest")

        self.conv_medium = ConvBlock(medium_in_channels, _HIDDEN_MEDIUM, 1, 1, 0)
        self.upsample2 = torch.nn.Upsample(scale_factor=2, mode="nearest")

        self.conv_small = ConvBlock(small_in_channels, _HIDDEN_SMALL, 1, 1, 0)

        self.fusion_medium = ConvBlock(
            self.large_output_channels + _HIDDEN_MEDIUM,
            self.medium_output_channels,
            1,
            1,
            0,
        )
        self.fusion_small = ConvBlock(
            self.medium_output_channels + _HIDDEN_SMALL,
            self.small_output_channels,
            1,
            1,
            0,
        )

    def forward(
        self, small: torch.Tensor, medium: torch.Tensor, large: torch.Tensor
    ) -> torch.Tensor:
        large = self.conv_large(large)
        medium = self.conv_medium(medium)
        small = self.conv_small(small)

        medium = torch.cat([medium, self.upsample1(large)], dim=1)
        medium = self.fusion_medium(medium)

        small = torch.cat([small, self.upsample2(medium)], dim=1)
        small = self.fusion_small(small)

        return small, medium, large


class NeckConcat(torch.nn.Module):
    # TODO - add expected input channels for each scale

    def __init__(self) -> None:
        super(NeckConcat, self).__init__()
        self.small_output_channels = 128
        self.medium_output_channels = 256
        self.large_output_channels = 512

        self.conv_large = ConvBlock(1024, 512, 1, 1, 0)
        self.upsample1 = torch.nn.Upsample(scale_factor=2, mode="nearest")

        self.conv_medium = ConvBlock(512, 256, 1, 1, 0)
        self.upsample2 = torch.nn.Upsample(scale_factor=2, mode="nearest")

        self.conv_small = ConvBlock(256, 128, 1, 1, 0)

        self.fusion_medium = ConvBlock(768, self.medium_output_channels, 1, 1, 0)
        self.fusion_small = ConvBlock(384, self.small_output_channels, 1, 1, 0)

    def forward(
        self, small: torch.Tensor, medium: torch.Tensor, large: torch.Tensor
    ) -> torch.Tensor:
        large = self.conv_large(large)
        medium = self.conv_medium(medium)
        small = self.conv_small(small)

        medium = torch.cat([medium, self.upsample1(large)], dim=1)
        medium = self.fusion_medium(medium)

        small = torch.cat([small, self.upsample2(medium)], dim=1)
        small = self.fusion_small(small)

        return small, medium, large
