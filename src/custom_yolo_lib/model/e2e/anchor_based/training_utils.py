import math
from typing import Tuple
import torch

import custom_yolo_lib.process.bbox.utils
import custom_yolo_lib.model.building_blocks.heads.anchors.anchors_3_coco
from custom_yolo_lib.model.building_blocks.heads.detections_3_anchors import (
    FeatureMapType,
)

_ANCHORS_AS_BBOX_TENSORS = {
    FeatureMapType.SMALL: torch.tensor(
        custom_yolo_lib.model.building_blocks.heads.anchors.anchors_3_coco.SMALL_MAP_FEATS_ANCHORS_LIST
    ),
    FeatureMapType.MEDIUM: torch.tensor(
        custom_yolo_lib.model.building_blocks.heads.anchors.anchors_3_coco.MEDIUM_MAP_FEATS_ANCHORS_LIST
    ),
    FeatureMapType.LARGE: torch.tensor(
        custom_yolo_lib.model.building_blocks.heads.anchors.anchors_3_coco.LARGE_MAP_FEATS_ANCHORS_LIST
    ),
}


def build_feature_map_targets(
    annotations: torch.Tensor,
    feature_map_type: FeatureMapType,
    grid_size_h: int,
    grid_size_w: int,
    num_classes: int,
    check_values: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    anchors = _ANCHORS_AS_BBOX_TENSORS[feature_map_type]
    num_anchors = anchors.shape[0]
    assert num_anchors == 3, "Only 3 anchors are supported for each feature map type"

    targets = torch.zeros((num_anchors, 5 + num_classes, grid_size_h, grid_size_w))
    targets_mask = torch.zeros(
        (num_anchors, grid_size_h, grid_size_w), dtype=torch.bool
    )

    # targets_masks2 = torch.zeros(num_anchors, grid_size_h, grid_size_w) """ EXPERIMENTAL """
    for obj in annotations:
        bx, by, bw, bh, class_id = obj  # YOLOv2 naming convention | zero based class_id
        class_id = int(class_id)
        grid_x = int(bx * grid_size_w)
        grid_y = int(by * grid_size_h)
        box = torch.tensor([0, 0, bw, bh])
        ious = custom_yolo_lib.process.bbox.utils.calculate_iou_tensors(box, anchors)
        best_anchor = torch.argmax(ious)
        # Fill the target
        targets[best_anchor, 0, grid_y, grid_x] = (
            bx * grid_size_w - grid_x
        )  # x offset inside cell | example if x=0.5, grid_size=13, grid_x=6 then in grid cell x offset is 0.5*13-6=0.5
        targets[best_anchor, 1, grid_y, grid_x] = (
            by * grid_size_h - grid_y
        )  # y offset inside cell
        t_w = math.log(bw / anchors[best_anchor, 2])
        t_h = math.log(bh / anchors[best_anchor, 3])
        if check_values:
            assert (targets[best_anchor, 0, grid_y, grid_x] <= 1).all()
            assert (0 <= targets[best_anchor, 0, grid_y, grid_x]).all()
            assert (targets[best_anchor, 1, grid_y, grid_x] <= 1).all()
            assert (0 <= targets[best_anchor, 1, grid_y, grid_x]).all()
            assert 0 <= t_w <= 1
            assert 0 <= t_h <= 1

        targets[best_anchor, 2, grid_y, grid_x] = t_w
        targets[best_anchor, 3, grid_y, grid_x] = t_h
        targets[best_anchor, 4, grid_y, grid_x] = 1.0  # objectness
        targets[best_anchor, 5 + class_id, grid_y, grid_x] = 1.0  # class score

        targets_mask[best_anchor, grid_y, grid_x] = True

        """ EXPERIMENTAL """
        # # where the object is & neighbouring grids, to account for one use case:
        # # target is in grid cell x, y & x offset is ~0.9 while prediction is in grid cell x+1, y & x offset is ~0.1
        # targets_masks2[
        #     best_anchor,
        #     max(grid_y - 1, 0) : min(grid_y + 1, grid_size_h),
        #     max(grid_x, 0) : min(grid_x + 1, grid_size_w),
        # ] = 1.0

    return targets, targets_mask
