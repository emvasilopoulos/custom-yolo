import torch

from custom_yolo_lib.model.backbones.vanilla_cnn import ConvBlock
from custom_yolo_lib.model.building_blocks.conv_block import ResidualBlock
from custom_yolo_lib.model.necks.additive import AdditiveNeck


class ThreeScalesFeatures(torch.nn.Module):
    def __init__(self) -> None:
        super(ThreeScalesFeatures, self).__init__()
        self.initial_layer = torch.nn.Sequential(
            ConvBlock(3, 32, 3, 1, 1),
        )

        self.layer1 = self._make_layer(32, 64, 1)
        self.layer2 = self._make_layer(64, 128, 2)
        self.layer3 = self._make_layer(128, 256, 8)
        self.layer4 = self._make_layer(256, 512, 8)
        self.layer5 = self._make_layer(512, 1024, 4)

    def _make_layer(
        self, in_channels: int, out_channels: int, num_blocks: int
    ) -> torch.nn.Sequential:
        layers = [ConvBlock(in_channels, out_channels, 3, 2, 1)]  # Downsample
        for _ in range(num_blocks):
            layers.append(ResidualBlock(out_channels))
        return torch.nn.Sequential(*layers)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.initial_layer(x)
        x = self.layer1(x)
        x = self.layer2(x)
        small_feature = self.layer3(x)
        medium_feature = self.layer4(small_feature)
        large_feature = self.layer5(medium_feature)
        return small_feature, medium_feature, large_feature


# Example usage
if __name__ == "__main__":
    model = ThreeScalesFeatures()
    x = torch.randn((1, 3, 416, 416))
    small, medium, large = model(x)
    print(f"Small feature map shape: {small.shape}")  # Should be (1, 256, 52, 52)
    print(f"Medium feature map shape: {medium.shape}")  # Should be (1, 512, 26, 26)
    print(f"Large feature map shape: {large.shape}")  # Should be (1, 1024, 13, 13)

    neck = AdditiveNeck()
    small, medium, large = neck(small, medium, large)
    print(
        f"Small feature map shape after neck: {small.shape}"
    )  # Should be (1, 128, 52, 52)
    print(
        f"Medium feature map shape after neck: {medium.shape}"
    )  # Should be (1, 256, 26, 26)
    print(
        f"Large feature map shape after neck: {large.shape}"
    )  # Should be (1, 512, 13, 13)
