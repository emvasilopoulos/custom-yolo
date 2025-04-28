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
    positive_sample_iou_thershold: float = 0.15,  # don't care if iou is low as long as I skip t_w and t_h if > 1
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

    for i, obj in enumerate(annotations):
        bx, by, bw, bh, class_id = obj  # YOLOv2 naming convention | zero based class_id
        box = torch.tensor([0, 0, bw, bh], device=annotations.device)
        ious = custom_yolo_lib.process.bbox.utils.calculate_iou_tensors(box, anchors)
        anchor_i = torch.argmax(ious)

        if ious[anchor_i] < positive_sample_iou_thershold:
            continue

        grid_x = int(bx * grid_size_w)
        grid_y = int(by * grid_size_h)
        objectness_score = ious[anchor_i]
        if targets[anchor_i, 4, grid_y, grid_x] < objectness_score:
            potential_anchor = anchor_i
            """ NOTE: matches with custom_yolo_lib.model.building_blocks.heads.detections_3_anchors.decode_output """
            t_w = torch.sqrt(bw / anchors[potential_anchor, 2]) / 2
            t_h = torch.sqrt(bh / anchors[potential_anchor, 3]) / 2
            if t_w > 1 or t_h > 1:
                print("Warning: large t_w or t_h, skipping")
                continue

            class_id = int(class_id)

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
            targets[potential_anchor, 4, grid_y, grid_x] = (
                objectness_score  # objectness
            )
            targets[potential_anchor, 5 + class_id, grid_y, grid_x] = 1.0  # class score

            targets_mask[potential_anchor, grid_y, grid_x] = True
    return targets, targets_mask
