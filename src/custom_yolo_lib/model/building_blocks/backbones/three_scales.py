import torch

from custom_yolo_lib.model.building_blocks.conv_block import ConvBlock, ResidualBlock


class ThreeScalesFeatures(torch.nn.Module):
    def __init__(self) -> None:
        super(ThreeScalesFeatures, self).__init__()
        self.initial_layer = torch.nn.Sequential(
            ConvBlock(3, 32, 3, 1, 1),
        )

        self.small_features_filters = 256
        self.medium_features_filters = 512
        self.large_features_filters = 1024
        self.layer1 = self._make_layer(32, 64, 1)
        self.layer2 = self._make_layer(64, 128, 1)
        self.layer3 = self._make_layer(128, self.small_features_filters, 1)
        self.layer4 = self._make_layer(self.small_features_filters, self.medium_features_filters, 1)
        self.layer5 = self._make_layer(self.medium_features_filters, self.large_features_filters, 1)

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



class ThreeScalesFeaturesAndFPN(torch.nn.Module):
    def __init__(self) -> None:
        super(ThreeScalesFeaturesAndFPN, self).__init__()
        self.three_scales = ThreeScalesFeatures()

        self.upsampler = torch.nn.Upsample(scale_factor=2, mode="nearest")
        self.res_small = ResidualBlock(self.three_scales.small_features_filters)
        self.res_small_new = ResidualBlock(
            self.three_scales.medium_features_filters
            + self.three_scales.large_features_filters
        )
        self.res_medium = ResidualBlock(self.three_scales.medium_features_filters)
        self.res_medium_new = ResidualBlock(self.three_scales.large_features_filters)
        self.res_large = ResidualBlock(self.three_scales.large_features_filters)

        self.small_features_filters = (
            self.three_scales.small_features_filters
            + self.three_scales.medium_features_filters
            + self.three_scales.large_features_filters
        )
        self.medium_features_filters = (
            self.three_scales.medium_features_filters
            + self.three_scales.large_features_filters
        )
        self.large_features_filters = (
            self.three_scales.large_features_filters
        )  # no concat here

    def _upsample_large_features_to_medium(
        self, large_feature: torch.Tensor
    ) -> torch.Tensor:
        return self.upsampler(large_feature)

    def _upsample_medium_features_to_small(
        self, medium_feature: torch.Tensor
    ) -> torch.Tensor:
        return self.upsampler(medium_feature)

    def _concat_medium_features(
        self, medium_features: torch.Tensor, new_medium_features: torch.Tensor
    ) -> torch.Tensor:
        m1 = self.res_medium_new(new_medium_features)
        m2 = self.res_medium(medium_features)
        return torch.cat((m1, m2), dim=1)

    def _concat_small_features(
        self, small_features: torch.Tensor, new_small_features: torch.Tensor
    ) -> torch.Tensor:
        s1 = self.res_small_new(new_small_features)
        s2 = self.res_small(small_features)
        return torch.cat((s1, s2), dim=1)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        small_features, medium_features, large_features = self.three_scales(x)

        large_features = self.res_large(large_features)

        new_medium_features = self._upsample_large_features_to_medium(large_features)
        medium_features = self._concat_medium_features(
            medium_features, new_medium_features
        )
        new_small_features = self._upsample_medium_features_to_small(medium_features)
        small_features = self._concat_small_features(small_features, new_small_features)

        return small_features, medium_features, large_features
