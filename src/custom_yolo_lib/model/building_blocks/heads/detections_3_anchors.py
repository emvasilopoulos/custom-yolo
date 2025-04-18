import dataclasses
import enum
from typing import Dict, List, Optional

import torch

import custom_yolo_lib.process.bbox
import custom_yolo_lib.process.bbox.utils
import custom_yolo_lib.model.building_blocks.heads.anchors.anchors_3_coco


class FeatureMapType(enum.Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


_ANCHORS_AS_BBOX = {
    FeatureMapType.SMALL: custom_yolo_lib.model.heads.anchors.anchors_3_coco.SMALL_MAP_FEATS_ANCHORS,
    FeatureMapType.MEDIUM: custom_yolo_lib.model.heads.anchors.anchors_3_coco.MEDIUM_MAP_FEATS_ANCHORS,
    FeatureMapType.LARGE: custom_yolo_lib.model.heads.anchors.anchors_3_coco.LARGE_MAP_FEATS_ANCHORS,
}

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


def match_norm_bbox_with_anchor(
    feature_map_type: FeatureMapType,
    bbox: custom_yolo_lib.process.bbox.Bbox,
) -> custom_yolo_lib.model.heads.anchors.anchors_3_coco.ThreeAnchorCoco:
    if not bbox.is_normalized:
        raise ValueError("Bounding box must be normalized.")

    max_iou = 0
    best_anchor = custom_yolo_lib.model.heads.anchors.anchors_3_coco.ThreeAnchorCoco.ONE
    for anchor in custom_yolo_lib.model.heads.anchors.anchors_3_coco.ThreeAnchorCoco:
        iou = custom_yolo_lib.process.bbox.utils.calculate_iou(
            bbox, _ANCHORS_AS_BBOX[feature_map_type][anchor]
        )
        if iou > max_iou:
            max_iou = iou
            best_anchor = anchor
    return best_anchor


"""
For each feature map, we have 3 anchors. Each anchor is represented by a tensor of 4 values:
- The first two values are the center coordinates of the anchor box (x, y).
- The last two values are the width and height of the anchor box (w, h).
"""


@dataclasses.dataclass
class DetectionHeadOutput:
    anchor1_output: torch.Tensor
    anchor2_output: torch.Tensor
    anchor3_output: torch.Tensor

    def __iter__(self) -> iter:
        return iter((self.anchor1_output, self.anchor2_output, self.anchor3_output))


class DetectionHead(torch.nn.Module):
    def __init__(
        self, in_channels: int, num_classes: int, feature_map_type: FeatureMapType
    ) -> None:
        super(DetectionHead, self).__init__()
        self.num_anchors = 3
        self.feats_per_anchor = 5 + num_classes
        self.conv = torch.nn.Conv2d(
            in_channels, self.num_anchors * self.feats_per_anchor, 1, 1, 0
        )
        self.anchors = _ANCHORS_AS_BBOX_TENSORS[feature_map_type]

        """
        https://arxiv.org/pdf/2004.10934 
        "..S: Eliminate grid sensitivity the equation bx = sigmoid(tx)+
        cx, by = sigmoid(ty) + cy , where cx and cy are always whole
        numbers, is used in YOLOv3 for evaluating the ob-
        ject coordinates, therefore, extremely high tx absolute
        values are required for the bx value approaching the
        cx or cx + 1 values. We solve this problem through
        multiplying the sigmoid by a factor exceeding 1.0, so
        eliminating the effect of grid on which the object is
        undetectable.."
        """
        self._multiplier = 2.0

        self.meshgrids: Dict[str, List[torch.Tensor]] = {}

    def forward(
        self, x: torch.Tensor, training: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = torch.sigmoid(self.conv(x))
        batch_size, out_feats, grid_size_h, grid_size_w = out.shape
        out = out.view(
            batch_size,
            self.num_anchors,
            self.feats_per_anchor,
            grid_size_h,
            grid_size_w,
        )  # to operate on all anchors at once (see next line)

        if training:
            return DetectionHeadOutput(
                out[:, 0], out[:, 1], out[:, 2]
            )  # we want values between 0 and 1 for training???

        # Create grid tensors for x and y coordinates
        grid_size_h = out.shape[3]
        grid_size_w = out.shape[4]
        grids = self.meshgrids.get(f"{grid_size_h}_{grid_size_w}")
        if grids is None:
            grids = _make_grids(grid_size_h, grid_size_w)
            self.meshgrids[f"{grid_size_h}_{grid_size_w}"] = grids
        return decode_output(out, self._multiplier, grids)


def _make_grids(grid_size_h: int, grid_size_w: int) -> List[torch.Tensor]:
    grid_y, grid_x = torch.meshgrid(
        torch.arange(grid_size_h, dtype=torch.float32),
        torch.arange(grid_size_w, dtype=torch.float32),
        indexing="ij",
    )
    grid_x = grid_x.unsqueeze(0).unsqueeze(0)  # (1, grid_size_h, grid_size_h)
    grid_y = grid_y.unsqueeze(0).unsqueeze(0)  # (1, grid_size_w, grid_size_w)
    return [grid_y, grid_x]


def decode_output(
    out: torch.Tensor, multiplier: int, grids: Optional[List[torch.Tensor]] = None
) -> DetectionHeadOutput:
    # https://paperswithcode.com/method/grid-sensitive
    # apply in x,y,w,h
    out[:, :, :4, :, :].mul_(multiplier).sub_((multiplier - 1) / 2)

    if grids is None:
        # Create grid tensors for x and y coordinates
        grid_size_h = out.shape[3]
        grid_size_w = out.shape[4]
        grids = _make_grids(grid_size_h, grid_size_w)
    grid_y, grid_x = grids

    # Add grid offsets to x coordinates (index 0)
    out[:, :, 0, :, :].add_(grid_x)

    # Add grid offsets to y coordinates (index 1)
    out[:, :, 1, :, :].add_(grid_y)

    # wh
    out[:, :, 2:4, :, :].exp_()

    return DetectionHeadOutput(out[:, 0], out[:, 1], out[:, 2])
