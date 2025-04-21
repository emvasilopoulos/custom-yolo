import math
from typing import List, Tuple
import torch

from custom_yolo_lib.model.building_blocks.backbones.three_scales import (
    ThreeScalesFeatures,
)
from custom_yolo_lib.model.building_blocks.heads.detections_3_anchors import (
    DetectionHead,
    DetectionHeadOutput,
    FeatureMapType,
)
from custom_yolo_lib.model.building_blocks.necks.additive import AdditiveNeck


class YOLOModel(torch.nn.Module):

    def __init__(self, num_classes: int, training: bool) -> None:
        super(YOLOModel, self).__init__()
        self.__training = training
        self.backbone = ThreeScalesFeatures()
        self.neck = AdditiveNeck()
        self.detect_small = DetectionHead(
            256, num_classes, feature_map_type=FeatureMapType.SMALL
        )
        self.detect_medium = DetectionHead(
            512, num_classes, feature_map_type=FeatureMapType.MEDIUM
        )
        self.detect_large = DetectionHead(
            1024, num_classes, feature_map_type=FeatureMapType.LARGE
        )

    def forward(self, x: torch.Tensor) -> List[DetectionHeadOutput]:
        small, medium, large = self.backbone(x)
        small, medium, large = self.neck(small, medium, large)
        out_small = self.detect_small(small, self.__training)
        out_medium = self.detect_medium(medium, self.__training)
        out_large = self.detect_large(large, self.__training)
        return out_small, out_medium, out_large
