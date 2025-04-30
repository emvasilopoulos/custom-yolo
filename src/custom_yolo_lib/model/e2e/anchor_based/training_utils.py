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
    positive_sample_iou_thershold: float = 0.3,  # don't care if iou is low as long as I skip t_w and t_h if > 1
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

        # replace with better object
        if targets[anchor_i, 4, grid_y, grid_x] < objectness_score:

            # bbox
            # x offset inside cell | example if x=0.5, grid_size=13, grid_x=6 then in grid cell x offset is 0.5*13-6=0.5
            targets[anchor_i, 0, grid_y, grid_x] = bx * grid_size_w - grid_x
            # y offset inside cell
            targets[anchor_i, 1, grid_y, grid_x] = by * grid_size_h - grid_y
            targets[anchor_i, 2, grid_y, grid_x] = bw
            targets[anchor_i, 3, grid_y, grid_x] = bh
            if check_values:
                assert (targets[anchor_i, 0, grid_y, grid_x] <= 1).all()
                assert (0 <= targets[anchor_i, 0, grid_y, grid_x]).all()
                assert (targets[anchor_i, 1, grid_y, grid_x] <= 1).all()
                assert (0 <= targets[anchor_i, 1, grid_y, grid_x]).all()
                assert 0 <= bw <= 1  # fails with bad anchor
                assert 0 <= bh <= 1  # fails with bad anchor

            # objectness
            targets[anchor_i, 4, grid_y, grid_x] = objectness_score
            # _bump_objectness(
            #     targets, anchor_i, grid_y, grid_x, max_value=objectness_score
            # )

            # class
            targets[anchor_i, 5 + int(class_id), grid_y, grid_x] = 1.0  # class score

            targets_mask[anchor_i, grid_y, grid_x] = True
    return targets, targets_mask


def _bump_objectness(
    targets: torch.Tensor,
    potential_anchor: int,
    grid_y: int,
    grid_x: int,
    max_value: float = 1.0,
) -> None:
    """
    In-place: raises objectness scores around (grid_y, grid_x)
    for the given anchor in `targets`.
    """
    device = targets.device
    _, _, H, W = targets.shape

    # 8×2 tensors of offsets for 1-pixel and 2-pixel neighbors
    offsets1 = torch.tensor(
        [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]],
        device=device,
    )
    offsets2 = torch.tensor(
        [
            [-2, -2],
            [-2, -1],
            [-2, 0],
            [-2, 1],
            [-2, 2],
            [-1, -2],
            [-1, 2],
            [0, -2],
            [0, 2],
            [1, -2],
            [1, 2],
            [2, -2],
            [2, -1],
            [2, 0],
            [2, 1],
            [2, 2],
        ],
        device=device,
    )

    # helper to bump a whole neighborhood
    def bump(offsets: torch.Tensor, thresh: float) -> None:
        # compute absolute neighbor coords: (8,2)
        base = torch.tensor([grid_y, grid_x], device=device)
        neigh = offsets + base  # shape (8,2)
        ys, xs = neigh[:, 0], neigh[:, 1]

        # mask in‐bounds
        valid = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
        if not valid.any():
            return

        ys, xs = ys[valid], xs[valid]
        # gather current objectness scores
        curr = targets[potential_anchor, 4, ys, xs]
        # find which need raising
        to_raise = curr < thresh
        if to_raise.any():
            targets[potential_anchor, 4, ys[to_raise], xs[to_raise]] = thresh

    # apply to 1-pixel neighbors (→0.75 * max_value) and 2-pixel neighbors (→0.50 * max_value)
    bump(offsets1, thresh=max_value * 0.75)
    bump(offsets2, thresh=max_value * 0.50)
