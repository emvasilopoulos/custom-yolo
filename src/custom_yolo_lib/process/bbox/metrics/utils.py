from typing import Tuple
import torch


def extract_coords(
    box: torch.Tensor, xywh=True, eps=1e-7
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1) = box.chunk(4, -1)
        w1_, h1_ = w1 / 2, h1 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    return b1_x1, b1_y1, b1_x2, b1_y2, w1, h1


def intersection(
    b1_x1: torch.Tensor,
    b1_y1: torch.Tensor,
    b1_x2: torch.Tensor,
    b1_y2: torch.Tensor,
    b2_x1: torch.Tensor,
    b2_y1: torch.Tensor,
    b2_x2: torch.Tensor,
    b2_y2: torch.Tensor,
) -> torch.Tensor:
    # Intersection area
    return (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp_(0)


def union(
    w1: torch.Tensor,
    h1: torch.Tensor,
    w2: torch.Tensor,
    h2: torch.Tensor,
    inter: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    # Union Area
    return w1 * h1 + w2 * h2 - inter + eps


def convex_smallest_enclosing_box(
    b1_x1: torch.Tensor,
    b1_y1: torch.Tensor,
    b1_x2: torch.Tensor,
    b1_y2: torch.Tensor,
    b2_x1: torch.Tensor,
    b2_y1: torch.Tensor,
    b2_x2: torch.Tensor,
    b2_y2: torch.Tensor,
) -> torch.Tensor:
    # convex (smallest enclosing box)
    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex width
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
    return cw, ch
