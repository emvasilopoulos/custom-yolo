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
        self.training = training
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
        out_small = self.detect_small(small, self.training)
        out_medium = self.detect_medium(medium, self.training)
        out_large = self.detect_large(large, self.training)
        return out_small, out_medium, out_large


if __name__ == "__main__":

    model = YOLOModel(num_classes=80, training=False)
    x = torch.randn((1, 3, 640, 640))
    out_small, out_medium, out_large = model(x)
    print(
        f"Output small shape: {out_small.anchor1_output.shape}"
    )  # Should be (1, num_classes + 5, 52, 52)
    print(
        f"Output medium shape: {out_medium.anchor2_output.shape}"
    )  # Should be (1, num_classes + 5, 26, 26)
    print(
        f"Output large shape: {out_large.anchor3_output.shape}"
    )  # Should be (1, num_classes + 5, 13, 13)
