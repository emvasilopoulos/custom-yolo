import dataclasses
from typing import Tuple
import torch

import custom_yolo_lib.process.bbox.utils
import custom_yolo_lib.model.building_blocks.heads.anchors.anchors_3_coco
from custom_yolo_lib.model.building_blocks.heads.detections_3_anchors import (
    FeatureMapType,
)


@dataclasses.dataclass
class AnchorsTensor:
    anchors: torch.Tensor
    feature_map_type: FeatureMapType


def get_anchors_as_bbox_tensors(
    device: torch.device = torch.device("cpu"),
) -> Tuple[AnchorsTensor, AnchorsTensor, AnchorsTensor]:
    """
    Get anchors as bounding box tensors for the specified feature map type.

    Args:
        feature_map_type (FeatureMapType): The feature map type.

    Returns:
        torch.Tensor: Anchors as bounding box tensors.
    """
    small_anchors = AnchorsTensor(
        anchors=torch.tensor(
            custom_yolo_lib.model.building_blocks.heads.anchors.anchors_3_coco.SMALL_MAP_FEATS_ANCHORS_LIST,
            device=device,
        ),
        feature_map_type=FeatureMapType.SMALL,
    )
    medium_anchors = AnchorsTensor(
        anchors=torch.tensor(
            custom_yolo_lib.model.building_blocks.heads.anchors.anchors_3_coco.MEDIUM_MAP_FEATS_ANCHORS_LIST,
            device=device,
        ),
        feature_map_type=FeatureMapType.MEDIUM,
    )
    large_anchors = AnchorsTensor(
        anchors=torch.tensor(
            custom_yolo_lib.model.building_blocks.heads.anchors.anchors_3_coco.LARGE_MAP_FEATS_ANCHORS_LIST,
            device=device,
        ),
        feature_map_type=FeatureMapType.LARGE,
    )
    return small_anchors, medium_anchors, large_anchors


def build_feature_map_targets(
    annotations: torch.Tensor,
    anchor_tensor: AnchorsTensor,
    grid_size_h: int,
    grid_size_w: int,
    num_classes: int,
    check_values: bool = False,
    positive_sample_iou_thershold: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    anchors = anchor_tensor.anchors
    num_anchors = anchors.shape[0]
    assert num_anchors == 3, "Only 3 anchors are supported for each feature map type"

    targets = torch.zeros(
        (num_anchors, 5 + num_classes, grid_size_h, grid_size_w),
        device=annotations.device,
    )
    targets_mask = torch.zeros(
        (num_anchors, grid_size_h, grid_size_w),
        dtype=torch.bool,
        device=annotations.device,
    )

    for obj in annotations:
        bx, by, bw, bh, class_id = obj  # YOLOv2 naming convention | zero based class_id
        box = torch.tensor([0, 0, bw, bh], device=annotations.device)
        ious = custom_yolo_lib.process.bbox.utils.calculate_iou_tensors(box, anchors)
        for anchor_i in range(num_anchors):
            # Filter anchors
            if ious[anchor_i] < positive_sample_iou_thershold:
                continue
            potential_anchor = anchor_i
            """ NOTE: matches with custom_yolo_lib.model.building_blocks.heads.detections_3_anchors.decode_output """
            t_w = torch.sqrt(
                bw / anchors[potential_anchor, 2]
            )  # Encode with sqrt instead of log to ensure more anchors in range [0, 1]
            t_h = torch.sqrt(bh / anchors[potential_anchor, 3])
            if t_w > 1 or t_h > 1:
                # This anchor is not good enough
                continue

            class_id = int(class_id)
            grid_x = int(bx * grid_size_w)
            grid_y = int(by * grid_size_h)
            objectness = ious[potential_anchor]

            # Fill the target
            targets[potential_anchor, 0, grid_y, grid_x] = (
                bx * grid_size_w - grid_x
            )  # x offset inside cell | example if x=0.5, grid_size=13, grid_x=6 then in grid cell x offset is 0.5*13-6=0.5
            targets[potential_anchor, 1, grid_y, grid_x] = (
                by * grid_size_h - grid_y
            )  # y offset inside cell
            if check_values:
                assert (targets[potential_anchor, 0, grid_y, grid_x] <= 1).all()
                assert (0 <= targets[potential_anchor, 0, grid_y, grid_x]).all()
                assert (targets[potential_anchor, 1, grid_y, grid_x] <= 1).all()
                assert (0 <= targets[potential_anchor, 1, grid_y, grid_x]).all()
                assert 0 <= t_w <= 1  # fails with bad anchor
                assert 0 <= t_h <= 1  # fails with bad anchor

            targets[potential_anchor, 2, grid_y, grid_x] = t_w
            targets[potential_anchor, 3, grid_y, grid_x] = t_h
            targets[potential_anchor, 4, grid_y, grid_x] = objectness  # objectness
            targets[potential_anchor, 5 + class_id, grid_y, grid_x] = 1.0  # class score

            targets_mask[potential_anchor, grid_y, grid_x] = True
    return targets, targets_mask
