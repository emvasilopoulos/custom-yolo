import dataclasses

import torch

import custom_yolo_lib.image_size
import custom_yolo_lib.process.image.resize
import custom_yolo_lib.process.image.pad


def resize_image(x: torch.Tensor, new_height: int, new_width: int) -> torch.Tensor:
    if len(x.shape) < 3 or len(x.shape) > 4:
        raise ValueError(
            f"Input tensor must have 3 (HWC) or 4 (NCHW) dimensions, but got {len(x.shape)}."
        )
    has_batch_dimension = len(x.shape) == 4
    if not has_batch_dimension:
        x = x.unsqueeze(0)
    x = torch.nn.functional.interpolate(
        x,
        size=(new_height, new_width),
        mode="bilinear",
        align_corners=False,
    )
    if not has_batch_dimension:
        x = x.squeeze(0)
    return x


@dataclasses.dataclass
class ResizeImageSize:
    current_image_size: custom_yolo_lib.image_size.ImageSize
    resized_image_size: custom_yolo_lib.image_size.ImageSize
