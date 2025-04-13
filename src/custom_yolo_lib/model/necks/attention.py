import torch

from custom_yolo_lib.model.backbones.vanilla_cnn import ConvBlock


class AttentionFusion(torch.nn.Module):
    def __init__(self, channels: int) -> None:
        super(AttentionFusion, self).__init__()
        self.fc1 = torch.nn.Linear(channels, channels // 2)
        self.fc2 = torch.nn.Linear(channels // 2, channels)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.mean(x, dim=[2, 3])  # Global Average Pooling
        w = self.fc1(w)
        w = self.fc2(w)
        w = self.sigmoid(w).unsqueeze(-1).unsqueeze(-1)
        return x * w


class AttentionNeck(torch.nn.Module):
    # By ChatGPT
    def __init__(self) -> None:
        super(AttentionNeck, self).__init__()
        self.conv_large = ConvBlock(1024, 512, 1, 1, 0)
        self.upsample1 = torch.nn.Upsample(scale_factor=2, mode="nearest")

        self.conv_medium = ConvBlock(512, 256, 1, 1, 0)
        self.upsample2 = torch.nn.Upsample(scale_factor=2, mode="nearest")

        self.conv_small = ConvBlock(256, 128, 1, 1, 0)

        self.attention_medium = AttentionFusion(256)
        self.attention_small = AttentionFusion(128)

    def forward(
        self, small: torch.Tensor, medium: torch.Tensor, large: torch.Tensor
    ) -> torch.Tensor:
        large = self.conv_large(large)
        medium = self.conv_medium(medium)
        small = self.conv_small(small)

        medium = self.attention_medium(medium + self.conv_medium(self.upsample1(large)))
        small = self.attention_small(small + self.conv_small(self.upsample2(medium)))

        return small, medium, large


class AttentionNeck2(torch.nn.Module):
    # By Github Copilot
    def __init__(
        self, channels_l: int = 512, channels_m: int = 256, channels_s: int = 128
    ) -> None:
        super(AttentionNeck2, self).__init__()
        self.attention_large = AttentionFusion(channels_l)
        self.attention_medium = AttentionFusion(channels_m)
        self.attention_small = AttentionFusion(channels_s)

    def forward(
        self, small: torch.Tensor, medium: torch.Tensor, large: torch.Tensor
    ) -> torch.Tensor:
        large = self.attention_large(large)
        medium = self.attention_medium(medium)
        small = self.attention_small(small)

        return small, medium, large
