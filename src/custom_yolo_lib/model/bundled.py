import math
from typing import List
import torch

from custom_yolo_lib.model.backbones.three_scales import ThreeScalesFeatures
from custom_yolo_lib.model.heads.detections_3_anchors import (
    DetectionHead,
    DetectionHeadOutput,
    FeatureMapType,
)
import custom_yolo_lib.model.heads.anchors.anchors_3_coco
from custom_yolo_lib.model.necks.additive import AdditiveNeck
import custom_yolo_lib.process.bbox.utils

_ANCHORS_AS_BBOX_TENSORS = {
    FeatureMapType.SMALL: torch.tensor(
        custom_yolo_lib.model.heads.anchors.anchors_3_coco.SMALL_MAP_FEATS_ANCHORS_LIST
    ),
    FeatureMapType.MEDIUM: torch.tensor(
        custom_yolo_lib.model.heads.anchors.anchors_3_coco.MEDIUM_MAP_FEATS_ANCHORS_LIST
    ),
    FeatureMapType.LARGE: torch.tensor(
        custom_yolo_lib.model.heads.anchors.anchors_3_coco.LARGE_MAP_FEATS_ANCHORS_LIST
    ),
}


class YOLOModel(torch.nn.Module):
    def __init__(self, num_classes: int, training: bool):
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


def build_feature_map_targets(
    annotations: torch.Tensor,
    feature_map_type: FeatureMapType,
    grid_size_h: int,
    grid_size_w: int,
    num_classes: int,
    check_values: bool = False,
):
    anchors = _ANCHORS_AS_BBOX_TENSORS[feature_map_type]
    num_anchors = anchors.shape[0]
    assert num_anchors == 3, "Only 3 anchors are supported for each feature map type"
    targets = torch.zeros((num_anchors, grid_size_h, grid_size_w, 5 + num_classes))

    for obj in annotations:
        class_label, bx, by, bw, bh = obj  # YOLOv2 naming convention
        grid_x = int(bx * grid_size_w)
        grid_y = int(by * grid_size_h)
        box = torch.tensor([0, 0, bw, bh])
        ious = custom_yolo_lib.process.bbox.utils.calculate_iou_tensors(box, anchors)
        best_anchor = torch.argmax(ious)
        # Fill the target
        targets[best_anchor, grid_y, grid_x, 0] = (
            bx * grid_size_w - grid_x
        )  # x offset inside cell | example if x=0.5, grid_size=13, grid_x=6 then in grid cell x offset is 0.5*13-6=0.5
        targets[best_anchor, grid_y, grid_x, 1] = (
            by * grid_size_h - grid_y
        )  # y offset inside cell

        if check_values:
            assert (targets[best_anchor, grid_y, grid_x, 0] <= 1).all()
            assert (0 <= targets[best_anchor, grid_y, grid_x, 0]).all()
            assert (targets[best_anchor, grid_y, grid_x, 1] <= 1).all()
            assert (0 <= targets[best_anchor, grid_y, grid_x, 1]).all()

        t_w = math.log(bw / anchors[best_anchor, 2])
        t_h = math.log(bh / anchors[best_anchor, 3])
        targets[best_anchor, grid_y, grid_x, 2] = t_w
        targets[best_anchor, grid_y, grid_x, 3] = t_h
        targets[best_anchor, grid_y, grid_x, 4] = 1  # objectness
        targets[best_anchor, grid_y, grid_x, 5 + class_label] = 1  # one-hot class label
    return targets


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
