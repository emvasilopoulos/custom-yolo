import torch


class ConvBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        activation: torch.nn.Module = torch.nn.LeakyReLU(0.1),
    ):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False
        )
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.activation = activation

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels: int):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBlock(channels, channels // 2, 1, 1, 0)
        self.conv2 = ConvBlock(channels // 2, channels, 3, 1, 1)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        return identity + out
