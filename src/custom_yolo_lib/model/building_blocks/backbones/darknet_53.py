import torch

from custom_yolo_lib.model.building_blocks.backbones.three_scales import ResidualBlock
from custom_yolo_lib.model.building_blocks.backbones.vanilla_cnn import ConvBlock


class Darknet53Backbone(torch.nn.Module):
    def __init__(self) -> None:
        super(Darknet53Backbone, self).__init__()
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial_layer(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x


# Example usage
if __name__ == "__main__":
    model = Darknet53Backbone()
    x = torch.randn((1, 3, 416, 416))  # output is (1, 1024, 13, 13)
    output = model(x)
    print(output.shape)  # Should print feature map size after backbone
