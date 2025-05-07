import dataclasses
from typing import Tuple
import torch

from custom_yolo_lib.model.e2e.anchor_based.constants import ANCHOR_GAIN
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
    positive_sample_iou_thershold: float = 0.3,
    epsilon: float = 1e-4,
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

    if len(annotations) == 0:
        return targets, targets_mask

    # Extract annotation data
    bx, by, bw, bh, class_ids = annotations.unbind(1)
    class_ids = class_ids.long()

    # Reshape for broadcasting
    bw_expanded = bw.unsqueeze(1)  # [N, 1]
    bh_expanded = bh.unsqueeze(1)  # [N, 1]
    anchor_w = anchors[:, 2].unsqueeze(0)  # [1, num_anchors]
    anchor_h = anchors[:, 3].unsqueeze(0)  # [1, num_anchors]

    # Check which anchors are valid for each annotation
    valid_mask = (anchor_w * (ANCHOR_GAIN**2) - epsilon >= bw_expanded) & (
        anchor_h * (ANCHOR_GAIN**2) - epsilon >= bh_expanded
    )  # sigmoid never reaches 0 or 1 and we don't want too large values for its input

    # Get indices of valid pairs
    ann_idx, anchor_idx = valid_mask.nonzero(as_tuple=True)

    if len(ann_idx) == 0:
        return targets, targets_mask

    # Get corresponding data for valid pairs
    sel_bx = bx[ann_idx]
    sel_by = by[ann_idx]
    sel_bw = bw[ann_idx]
    sel_bh = bh[ann_idx]
    sel_class_ids = class_ids[ann_idx]

    # Calculate grid positions and offsets
    grid_x = (sel_bx * grid_size_w).long()
    grid_y = (sel_by * grid_size_h).long()
    x_offset = sel_bx * grid_size_w - grid_x.float()
    y_offset = sel_by * grid_size_h - grid_y.float()

    # Still need a loop for conditional update and _bump_objectness
    objectness_score = 1.0
    for i in range(len(ann_idx)):
        a_idx = anchor_idx[i]
        g_y = grid_y[i]
        g_x = grid_x[i]

        if targets[a_idx, 4, g_y, g_x] < objectness_score:
            # Update bbox
            targets[a_idx, 0, g_y, g_x] = x_offset[i]
            targets[a_idx, 1, g_y, g_x] = y_offset[i]
            targets[a_idx, 2, g_y, g_x] = sel_bw[i]
            targets[a_idx, 3, g_y, g_x] = sel_bh[i]

            if check_values:
                assert 0 <= x_offset[i] <= 1
                assert 0 <= y_offset[i] <= 1
                assert 0 <= sel_bw[i] <= 1
                assert 0 <= sel_bh[i] <= 1

            # Update objectness
            targets[a_idx, 4, g_y, g_x] = objectness_score
            # _bump_objectness(targets, a_idx, g_y, g_x, max_value=objectness_score)

            # Update class
            # if 5 + sel_class_ids[i]-3 >= 0:
            #     targets[a_idx, 5 + sel_class_ids[i]-3, g_y, g_x] = 0.25
            # if 5 + sel_class_ids[i]-2 >= 0:
            #     targets[a_idx, 5 + sel_class_ids[i]-2, g_y, g_x] = 0.5
            # if 5 + sel_class_ids[i]-1 >= 0:
            #     targets[a_idx, 5 + sel_class_ids[i]-1, g_y, g_x] = 0.75
            targets[a_idx, 5 + sel_class_ids[i], g_y, g_x] = 1.0
            # if sel_class_ids[i]+1 < num_classes:
            #     targets[a_idx, 5 + sel_class_ids[i]+1, g_y, g_x] = 0.75
            # if sel_class_ids[i]+2 < num_classes:
            #     targets[a_idx, 5 + sel_class_ids[i]+2, g_y, g_x] = 0.5
            # if sel_class_ids[i]+3 < num_classes:
            #     targets[a_idx, 5 + sel_class_ids[i]+3, g_y, g_x] = 0.25

            targets_mask[a_idx, g_y, g_x] = True

    return targets, targets_mask


def build_feature_map_targets_backup(
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
        """
        NOTE:
        I BELIVE without the following condition an experiment will not be well established because we try to predict big bboxes
        with small anchors. This means the predictions should have very large values thus exploding the gradient possibly.
        """
        # box = torch.tensor([0, 0, bw, bh], device=annotations.device)
        for anchor_i in range(num_anchors):
            if (
                anchors[anchor_i, 2] * (ANCHOR_GAIN**2) < bw
                or anchors[anchor_i, 2] * (ANCHOR_GAIN**2) < bh
            ):
                # too big bbox for anchor
                continue
            # ious = custom_yolo_lib.process.bbox.utils.calculate_iou_tensors(box, anchors)
            # anchor_i = torch.argmax(ious)
            # if ious[anchor_i] < positive_sample_iou_thershold:
            #     continue

            grid_x = int(bx * grid_size_w)
            grid_y = int(by * grid_size_h)
            objectness_score = 1.0

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
                targets[anchor_i, 5 + int(class_id), grid_y, grid_x] = (
                    1.0  # class score
                )

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
