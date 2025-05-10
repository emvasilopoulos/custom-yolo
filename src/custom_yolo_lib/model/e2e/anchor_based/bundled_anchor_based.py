from typing import List, Tuple
import torch

from custom_yolo_lib.model.building_blocks.backbones.three_scales import (
    ThreeScalesFeatures,
    ThreeScalesFeaturesAndFPN,
)
from custom_yolo_lib.model.building_blocks.heads.detections_3_anchors import (
    DetectionHead,
    DetectionHeadOutput,
    FeatureMapType,
)
from custom_yolo_lib.model.building_blocks.necks.concat import NeckConcat, ConcatNeck


class YOLOModel(torch.nn.Module):

    def __init__(self, num_classes: int, training: bool) -> None:
        super(YOLOModel, self).__init__()
        self.__training = training
        self.backbone = ThreeScalesFeaturesAndFPN()
        self.neck = ConcatNeck(
            self.backbone.small_features_filters,
            self.backbone.medium_features_filters,
            self.backbone.large_features_filters,
        )
        self.detect_small = DetectionHead(
            self.neck.small_output_channels,
            num_classes,
            feature_map_type=FeatureMapType.SMALL,
        )
        self.detect_medium = DetectionHead(
            self.neck.medium_output_channels,
            num_classes,
            feature_map_type=FeatureMapType.MEDIUM,
        )
        self.detect_large = DetectionHead(
            self.neck.large_output_channels,
            num_classes,
            feature_map_type=FeatureMapType.LARGE,
        )

        self.anchors_per_head = self.detect_small.num_anchors
        self.feats_per_head = self.detect_small.feats_per_anchor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.__training:
            return self.train_forward2(x)
        small, medium, large = self.backbone(x)
        small, medium, large = self.neck(small, medium, large)
        out_small = self.detect_small(small, self.__training)
        out_medium = self.detect_medium(medium, self.__training)
        out_large = self.detect_large(large, self.__training)
        return torch.cat(
            [
                out_small.view(
                    out_small.shape[0], self.anchors_per_head, self.feats_per_head, -1
                ),
                out_medium.view(
                    out_medium.shape[0], self.anchors_per_head, self.feats_per_head, -1
                ),
                out_large.view(
                    out_large.shape[0], self.anchors_per_head, self.feats_per_head, -1
                ),
            ],
            dim=3,
        )  # (batch_size, 3, 85, grid_x1*grid_y1 + grid_x2*grid_y2 + grid_x3*grid_y3)

    def train_forward(self, x: torch.Tensor) -> List[DetectionHeadOutput]:
        small, medium, large = self.backbone(x)
        small, medium, large = self.neck(small, medium, large)
        out_small = self.detect_small.train_forward(small, self.__training)
        out_medium = self.detect_medium.train_forward(medium, self.__training)
        out_large = self.detect_large.train_forward(large, self.__training)
        return out_small, out_medium, out_large

    def train_forward2(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        small, medium, large = self.backbone(x)
        small, medium, large = self.neck(small, medium, large)
        out_small = self.detect_small(small, self.__training)
        out_medium = self.detect_medium(medium, self.__training)
        out_large = self.detect_large(large, self.__training)
        return out_small, out_medium, out_large


class YOLOModel_FAILURE(torch.nn.Module):
    """
    It fails because I believe it does not upsample enough times.
    Hear me out:
    A Conv2D layer outputs information at a lower resolution than the input and each new
    cell has "neighbouring" information.
    If you only downsample each cell will know nothing about the cells far from it I guess.
    Upsampling may provide a similar mechanism to a fully connected layer.
    To understand more read this:
    'Fully Convolutional Networks for Semantic Segmentation'
    """

    def __init__(self, num_classes: int, training: bool) -> None:
        super(YOLOModel_FAILURE, self).__init__()
        self.__training = training
        self.backbone = ThreeScalesFeatures()
        self.neck = NeckConcat()
        self.detect_small = DetectionHead(
            128, num_classes, feature_map_type=FeatureMapType.SMALL
        )
        self.detect_medium = DetectionHead(
            256, num_classes, feature_map_type=FeatureMapType.MEDIUM
        )
        self.detect_large = DetectionHead(
            512, num_classes, feature_map_type=FeatureMapType.LARGE
        )

        self.anchors_per_head = self.detect_small.num_anchors
        self.feats_per_head = self.detect_small.feats_per_anchor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        small, medium, large = self.backbone(x)
        small, medium, large = self.neck(small, medium, large)
        out_small = self.detect_small(small, self.__training)
        out_medium = self.detect_medium(medium, self.__training)
        out_large = self.detect_large(large, self.__training)
        return torch.cat(
            [
                out_small.view(
                    out_small.shape[0], self.anchors_per_head, self.feats_per_head, -1
                ),
                out_medium.view(
                    out_medium.shape[0], self.anchors_per_head, self.feats_per_head, -1
                ),
                out_large.view(
                    out_large.shape[0], self.anchors_per_head, self.feats_per_head, -1
                ),
            ],
            dim=3,
        )  # (batch_size, 3, 85, grid_x1*grid_y1 + grid_x2*grid_y2 + grid_x3*grid_y3)

    def train_forward(self, x: torch.Tensor) -> List[DetectionHeadOutput]:
        small, medium, large = self.backbone(x)
        small, medium, large = self.neck(small, medium, large)
        out_small = self.detect_small.train_forward(small, self.__training)
        out_medium = self.detect_medium.train_forward(medium, self.__training)
        out_large = self.detect_large.train_forward(large, self.__training)
        return out_small, out_medium, out_large

    def train_forward2(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        small, medium, large = self.backbone(x)
        small, medium, large = self.neck(small, medium, large)
        out_small = self.detect_small(small, self.__training)
        out_medium = self.detect_medium(medium, self.__training)
        out_large = self.detect_large(large, self.__training)
        return out_small, out_medium, out_large
