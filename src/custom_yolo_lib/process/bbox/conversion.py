import torch

import custom_yolo_lib.process.bbox


def tl_xywh_to_center_xywh(
    tlbr: custom_yolo_lib.process.bbox.Bbox,
) -> custom_yolo_lib.process.bbox.Bbox:
    x1, y1, w, h = tlbr
    return x1 + w / 2, y1 + h / 2, w, h


def center_xywh_to_tl_xywh(
    center_xywh: custom_yolo_lib.process.bbox.Bbox,
) -> custom_yolo_lib.process.bbox.Bbox:
    x, y, w, h = center_xywh
    x_ = x - w / 2
    if x_ < 0:
        w_out_of_bounds = -x_
        x_ = 0
        w -= w_out_of_bounds

    y_ = y - h / 2
    if y_ < 0:
        h_out_of_bounds = -y_
        y_ = 0
        h -= h_out_of_bounds
    return x_, y_, w, h


def center_xywh_to_tl_xywh_tensor(center_xywh: torch.Tensor) -> torch.Tensor:
    center_xywh[0] = center_xywh[0] - center_xywh[2] / 2
    if center_xywh[0] < 0:
        w_out_of_bounds = -center_xywh[0]
        center_xywh[0] = 0
        center_xywh[2] -= w_out_of_bounds

    center_xywh[1] = center_xywh[1] - center_xywh[3] / 2
    if center_xywh[1] < 0:
        h_out_of_bounds = -center_xywh[1]
        center_xywh[1] = 0
        center_xywh[3] -= h_out_of_bounds

    return center_xywh


def tl_xywh_to_tlbr(
    tlwh: custom_yolo_lib.process.bbox.Bbox,
) -> custom_yolo_lib.process.bbox.Bbox:
    x, y, w, h = tlwh
    return x, y, x + w, y + h


def tl_xywh_to_tlbr_tensor(tlwh: torch.Tensor) -> torch.Tensor:
    tlwh[:, 2] = tlwh[:, 0] + tlwh[:, 2]
    tlwh[:, 3] = tlwh[:, 1] + tlwh[:, 3]
    return tlwh
