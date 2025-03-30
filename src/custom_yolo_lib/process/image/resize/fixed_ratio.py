import dataclasses
from typing import Tuple

import torch

import custom_yolo_lib.image_size
import custom_yolo_lib.process.image.resize
import custom_yolo_lib.process.image.pad


def _calculate_new_tensor_dimensions(
    current_image_size: custom_yolo_lib.image_size.ImageSize,
    expected_image_size: custom_yolo_lib.image_size.ImageSize,
):
    x_w = current_image_size.width
    x_h = current_image_size.height
    expected_image_width = expected_image_size.width
    expected_image_height = expected_image_size.height
    if x_w <= expected_image_width and x_h <= expected_image_height:
        width_ratio = x_w / expected_image_width  # less than 1
        height_ratio = x_h / expected_image_height  # less than 1
        if width_ratio > height_ratio:
            new_height = int(x_h / width_ratio)
            resize_height = new_height
            resize_width = expected_image_width
            pad_dimension = custom_yolo_lib.image_size.Dimension.HEIGHT
            expected_dimension_size = expected_image_height
        else:
            new_width = int(x_w / height_ratio)
            resize_height = expected_image_height
            resize_width = new_width
            pad_dimension = custom_yolo_lib.image_size.Dimension.WIDTH
            expected_dimension_size = expected_image_width
    elif x_w <= expected_image_width and x_h > expected_image_height:
        keep_ratio = x_w / x_h
        new_height = expected_image_height
        new_width = int(new_height * keep_ratio)
        resize_height = expected_image_height
        resize_width = new_width
        pad_dimension = custom_yolo_lib.image_size.Dimension.WIDTH
        expected_dimension_size = expected_image_width
    elif x_w > expected_image_width and x_h <= expected_image_height:
        keep_ratio = x_w / x_h
        new_width = expected_image_width
        new_height = int(new_width / keep_ratio)
        resize_height = new_height
        resize_width = expected_image_width
        pad_dimension = custom_yolo_lib.image_size.Dimension.HEIGHT
        expected_dimension_size = expected_image_height
    else:
        width_ratio = x_w / expected_image_width  # greater than 1
        height_ratio = x_h / expected_image_height  # greater than 1
        if width_ratio > height_ratio:
            new_height = int(x_h / width_ratio)
            resize_height = new_height
            resize_width = expected_image_width
            pad_dimension = custom_yolo_lib.image_size.Dimension.HEIGHT
            expected_dimension_size = expected_image_height
        else:
            new_width = int(x_w / height_ratio)
            resize_height = expected_image_height
            resize_width = new_width
            pad_dimension = custom_yolo_lib.image_size.Dimension.WIDTH
            expected_dimension_size = expected_image_width
    return resize_height, resize_width, pad_dimension, expected_dimension_size


def _calculate_new_tensor_dimensions_v2(
    current_image_size: custom_yolo_lib.image_size.ImageSize,
    expected_image_size: custom_yolo_lib.image_size.ImageSize,
) -> Tuple[int, int, int, int]:
    x_w = current_image_size.width
    x_h = current_image_size.height
    expected_image_width = expected_image_size.width
    expected_image_height = expected_image_size.height
    if x_w <= expected_image_width and x_h <= expected_image_height:
        width_ratio = x_w / expected_image_width  # less than 1
        height_ratio = x_h / expected_image_height  # less than 1
        if width_ratio > height_ratio:
            new_height = int(x_h / width_ratio)
            resize_height = new_height
            resize_width = expected_image_width
        else:
            new_width = int(x_w / height_ratio)
            resize_height = expected_image_height
            resize_width = new_width
    elif x_w <= expected_image_width and x_h > expected_image_height:
        keep_ratio = x_w / x_h
        new_height = expected_image_height
        new_width = int(new_height * keep_ratio)
        resize_height = expected_image_height
        resize_width = new_width
    elif x_w > expected_image_width and x_h <= expected_image_height:
        keep_ratio = x_w / x_h
        new_width = expected_image_width
        new_height = int(new_width / keep_ratio)
        resize_height = new_height
        resize_width = expected_image_width
    else:
        width_ratio = x_w / expected_image_width  # greater than 1
        height_ratio = x_h / expected_image_height  # greater than 1
        if width_ratio > height_ratio:
            new_height = int(x_h / width_ratio)
            resize_height = new_height
            resize_width = expected_image_width
        else:
            new_width = int(x_w / height_ratio)
            resize_height = expected_image_height
            resize_width = new_width

    pad_x = expected_image_width - resize_width
    pad_y = expected_image_height - resize_height
    return resize_height, resize_width, pad_y, pad_x


@dataclasses.dataclass
class ResizeFixedRatioComponents_v2(
    custom_yolo_lib.process.image.resize.ResizeImageSize
):

    def __post_init__(self):
        resize_h, resize_w, pad_y, pad_x = _calculate_new_tensor_dimensions_v2(
            self.current_image_size, self.resized_image_size
        )
        self.resize_height = resize_h
        self.resize_width = resize_w
        self.pad_y = pad_y
        self.pad_x = pad_x


@dataclasses.dataclass
class ResizeFixedRatioComponents:
    resize_height: int
    resize_width: int
    pad_dimension: custom_yolo_lib.image_size.Dimension
    expected_dimension_size: int

    def get_pad_amount(self) -> int:
        if self.pad_dimension == custom_yolo_lib.image_size.Dimension.HEIGHT:
            return self.expected_dimension_size - self.resize_height
        elif self.pad_dimension == custom_yolo_lib.image_size.Dimension.WIDTH:
            return self.expected_dimension_size - self.resize_width
        else:
            raise ValueError("Invalid pad dimension")

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (
            self.resize_height,
            self.resize_width,
            self.pad_dimension,
            self.expected_dimension_size,
        )


def resize_image(
    x: torch.Tensor,
    new_height: int,
    new_width: int,
    padding_percent: float,
    padding_value: int,
) -> torch.Tensor:
    """
    Resize the image to the new height and width.

    Args:
        x: The image tensor to resize.
        new_height: The new height of the image.
        new_width: The new width of the image.
        padding_percent: considering the new image should have new_width and new_height, in order to preserve the aspect ratio
        some padding should be applied which is calculated automatically. The 'padding_percent' is the percentage of the total calculated
        padding that should be applied at the start (top or left) of the image.
    """
    if padding_percent < 0 or padding_percent > 1:
        raise ValueError("The padding_percent should be between 0 and 1")

    resize_fixed_ratio_components = ResizeFixedRatioComponents_v2(
        current_image_size=custom_yolo_lib.image_size.ImageSize(
            width=x.shape[2], height=x.shape[1]
        ),
        resized_image_size=custom_yolo_lib.image_size.ImageSize(
            width=new_width, height=new_height
        ),
    )
    x = custom_yolo_lib.process.image.resize.resize_image(
        x,
        resize_fixed_ratio_components.resize_height,
        resize_fixed_ratio_components.resize_width,
    )

    pad_top = int(resize_fixed_ratio_components.pad_y * padding_percent)
    pad_bottom = resize_fixed_ratio_components.pad_y - pad_top
    pad_left = int(resize_fixed_ratio_components.pad_x * padding_percent)
    pad_right = resize_fixed_ratio_components.pad_x - pad_left
    return custom_yolo_lib.process.image.pad.pad_image_v2(
        x,
        pad_top,
        pad_right,
        pad_bottom,
        pad_left,
        pad_value=padding_value,
    )


def calculate_new_tensor_dimensions(
    current_image_size: custom_yolo_lib.image_size.ImageSize,
    expected_image_size: custom_yolo_lib.image_size.ImageSize,
):
    resize_height, resize_width, pad_dimension, expected_dimension_size = (
        _calculate_new_tensor_dimensions(current_image_size, expected_image_size)
    )
    return ResizeFixedRatioComponents(
        resize_height, resize_width, pad_dimension, expected_dimension_size
    )


def fixed_ratio_resize(
    x: torch.Tensor,
    expected_image_size: custom_yolo_lib.image_size.ImageSize,
    padding_percent: float = 0.5,
    pad_value: int = 0,
) -> torch.Tensor:
    """
    Resize the image to the expected image size while keeping the aspect ratio.

    Args:
        x: The tensor to resize.
        expected_image_size: The expected image size.
        padding_percent: The percentage of padding to add to the image.
        pad_value: The value to pad the image with.
    """

    current_image_size = custom_yolo_lib.image_size.ImageSize(
        width=x.shape[2], height=x.shape[1]
    )
    fixed_ratio_components = calculate_new_tensor_dimensions(
        current_image_size, expected_image_size
    )
    x = custom_yolo_lib.process.image.resize.resize_image(
        x, fixed_ratio_components.resize_height, fixed_ratio_components.resize_width
    )
    x = custom_yolo_lib.process.image.pad.pad_image(
        x,
        fixed_ratio_components.pad_dimension,
        fixed_ratio_components.expected_dimension_size,
        padding_percent,
        pad_value,
    )
    return x
