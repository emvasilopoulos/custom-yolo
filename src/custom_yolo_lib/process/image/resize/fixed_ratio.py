import dataclasses
from typing import Tuple

import torch

import custom_yolo_lib.image_size
import custom_yolo_lib.process.image.resize
import custom_yolo_lib.process.image.pad


@dataclasses.dataclass
class ResizeFixedRatioComponents2(custom_yolo_lib.process.image.resize.ResizeImageSize):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        resize_h, resize_w, pad_dim, expected_dim_size = (
            __calculate_new_tensor_dimensions(
                self.current_image_size, self.resized_image_size
            )
        )
        self.pad_amount = self.get_pad_amount()
        self.resize_height = resize_h
        self.resize_width = resize_w
        self.pad_dimension = pad_dim
        self.expected_dimension_size = expected_dim_size

    def get_pad_amount(self) -> int:
        if self.pad_dimension == custom_yolo_lib.image_size.Dimension.HEIGHT:
            return self.expected_dimension_size - self.resize_height
        elif self.pad_dimension == custom_yolo_lib.image_size.Dimension.WIDTH:
            return self.expected_dimension_size - self.resize_width
        else:
            raise ValueError("Invalid pad dimension")


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


def calculate_new_tensor_dimensions_EXPERIMENTAL_TO_BE_TESTED(
    current_image_size: custom_yolo_lib.image_size.ImageSize,
    expected_image_size: custom_yolo_lib.image_size.ImageSize,
) -> ResizeFixedRatioComponents:
    """Re-created with chatGPT"""
    x_w, x_h = current_image_size.width, current_image_size.height
    expected_w, expected_h = expected_image_size.width, expected_image_size.height

    width_ratio = x_w / expected_w
    height_ratio = x_h / expected_h

    # Determine the scaling factor that best fits the expected size
    if width_ratio > height_ratio:
        # Width is the limiting factor
        scale_factor = width_ratio
        resize_width = expected_w
        resize_height = int(x_h / scale_factor)
        pad_dimension = custom_yolo_lib.image_size.Dimension.HEIGHT
        expected_dimension_size = expected_h
    else:
        # Height is the limiting factor
        scale_factor = height_ratio
        resize_height = expected_h
        resize_width = int(x_w / scale_factor)
        pad_dimension = custom_yolo_lib.image_size.Dimension.WIDTH
        expected_dimension_size = expected_w

    return ResizeFixedRatioComponents(
        resize_height=resize_height,
        resize_width=resize_width,
        pad_dimension=pad_dimension,
        expected_dimension_size=expected_dimension_size,
    )


def __calculate_new_tensor_dimensions(
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


def calculate_new_tensor_dimensions(
    current_image_size: custom_yolo_lib.image_size.ImageSize,
    expected_image_size: custom_yolo_lib.image_size.ImageSize,
):
    resize_height, resize_width, pad_dimension, expected_dimension_size = (
        __calculate_new_tensor_dimensions(current_image_size, expected_image_size)
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
