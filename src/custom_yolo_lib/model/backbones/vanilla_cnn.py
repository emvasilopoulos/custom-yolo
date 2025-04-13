import torch


class ConvBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ) -> None:
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False
        )
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.leaky_relu = torch.nn.LeakyReLU(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.leaky_relu(self.bn(self.conv(x)))


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


if __name__ == "__main__":
    # Example usage
    model = Backbone()
    x = torch.randn(1, 3, 416, 416)  # Batch size of 1, 3 channels, 416x416 image
    output = model(x)
    print(output.shape)  # Should be (1, 1024, 13, 13) for a YOLO-like architecture

    x = torch.randn(1, 3, 640, 640)  # Batch size of 1, 3 channels, 640x640 image
    output = model(x)
    print(output.shape)  # Should be (1, 1024, 20, 20) for a YOLO-like architecture
