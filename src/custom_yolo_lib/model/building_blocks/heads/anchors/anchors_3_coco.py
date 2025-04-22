# Anchors, AKA reference bounding boxes. Predictions are computed relative to these.
import enum

import custom_yolo_lib.process.bbox


class ThreeAnchorCoco(enum.Enum):
    ONE = 0
    TWO = 1
    THREE = 2


### Values
# xywh
LARGE_MAP_FEATS_ANCHORS_LIST = [
    [0, 0, 0.29, 0.23],
    [0, 0, 0.39, 0.49],
    [0, 0, 0.9, 0.80],
]
_L_A1 = custom_yolo_lib.process.bbox.Bbox(
    *LARGE_MAP_FEATS_ANCHORS_LIST[0], is_normalized=True, is_top_left=True
)
_L_A2 = custom_yolo_lib.process.bbox.Bbox(
    *LARGE_MAP_FEATS_ANCHORS_LIST[1], is_normalized=True, is_top_left=True
)
_L_A3 = custom_yolo_lib.process.bbox.Bbox(
    *LARGE_MAP_FEATS_ANCHORS_LIST[2], is_normalized=True, is_top_left=True
)

MEDIUM_MAP_FEATS_ANCHORS_LIST = [
    [0.0, 0.0, 0.08, 0.16],
    [0.0, 0.0, 0.16, 0.12],
    [0.0, 0.0, 0.16, 0.30],
]
_M_A1 = custom_yolo_lib.process.bbox.Bbox(
    *MEDIUM_MAP_FEATS_ANCHORS_LIST[0], is_normalized=True, is_top_left=True
)
_M_A2 = custom_yolo_lib.process.bbox.Bbox(
    *MEDIUM_MAP_FEATS_ANCHORS_LIST[1], is_normalized=True, is_top_left=True
)
_M_A3 = custom_yolo_lib.process.bbox.Bbox(
    *MEDIUM_MAP_FEATS_ANCHORS_LIST[2], is_normalized=True, is_top_left=True
)

SMALL_MAP_FEATS_ANCHORS_LIST = [
    [0, 0, 0.03, 0.04],
    [0, 0, 0.05, 0.09],
    [0, 0, 0.09, 0.07],
]
_S_A1 = custom_yolo_lib.process.bbox.Bbox(
    *SMALL_MAP_FEATS_ANCHORS_LIST[0], is_normalized=True, is_top_left=True
)
_S_A2 = custom_yolo_lib.process.bbox.Bbox(
    *SMALL_MAP_FEATS_ANCHORS_LIST[1], is_normalized=True, is_top_left=True
)
_S_A3 = custom_yolo_lib.process.bbox.Bbox(
    *SMALL_MAP_FEATS_ANCHORS_LIST[2], is_normalized=True, is_top_left=True
)
### END of Values

LARGE_MAP_FEATS_ANCHORS = {
    ThreeAnchorCoco.ONE: _L_A1,
    ThreeAnchorCoco.TWO: _L_A2,
    ThreeAnchorCoco.THREE: _L_A3,
}

MEDIUM_MAP_FEATS_ANCHORS = {
    ThreeAnchorCoco.ONE: _M_A1,
    ThreeAnchorCoco.TWO: _M_A2,
    ThreeAnchorCoco.THREE: _M_A3,
}

SMALL_MAP_FEATS_ANCHORS = {
    ThreeAnchorCoco.ONE: _S_A1,
    ThreeAnchorCoco.TWO: _S_A2,
    ThreeAnchorCoco.THREE: _S_A3,
}
