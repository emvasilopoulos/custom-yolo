from typing import Tuple

import custom_yolo_lib.image_size
import custom_yolo_lib.process.bbox
import custom_yolo_lib.process.image.pad
import custom_yolo_lib.process.image.resize.fixed_ratio


def translate_bbox_to_resized_image(
    bbox: custom_yolo_lib.process.bbox.Bbox,
    resize_components: custom_yolo_lib.process.image.resize.ResizeImageSize,
    padding: custom_yolo_lib.process.image.pad.Padding,
) -> custom_yolo_lib.process.bbox.Bbox:
    """
    Supposedly, first you do resizing, then you do padding when you transform an image shape. This translates the bounding box
    in the same regard. The bounding box is first translated to the resized image, then to the padded image.

    Args:
        bbox (custom_yolo_lib.process.bbox.Bbox): _description_
        resize_components (custom_yolo_lib.process.image.resize.ResizeImageSize): _description_
        padding (custom_yolo_lib.process.image.pad.Padding): _description_

    Returns:
        _type_: _description_
    """

    new_width = (
        resize_components.resized_image_size.width + padding.left + padding.right
    )
    # px = padding.left / new_width
    new_height = (
        resize_components.resized_image_size.height + padding.top + padding.bottom
    )
    # py = padding.top / new_height
    if bbox.is_normalized:
        x_norm = (
            (bbox.x * resize_components.resized_image_size.width) + padding.left
        ) / new_width
        y_norm = (
            (bbox.y * resize_components.resized_image_size.height) + padding.top
        ) / new_height
        w_norm = (bbox.w * resize_components.resized_image_size.width) / new_width
        h_norm = (bbox.h * resize_components.resized_image_size.height) / new_height
        return custom_yolo_lib.process.bbox.Bbox(
            x=x_norm,
            y=y_norm,
            w=w_norm,
            h=h_norm,
            is_normalized=True,
            is_top_left=bbox.is_top_left,
        )
    else:
        raise NotImplementedError(
            "There is something wrong with the current implementation. Do not use yet."
        )
        x_norm = bbox.x / resize_components.current_image_size.width
        y_norm = bbox.y / resize_components.current_image_size.height
        w0_norm = bbox.w / resize_components.current_image_size.width
        h0_norm = bbox.h / resize_components.current_image_size.height
        w_norm = (w0_norm * resize_components.resized_image_size.width) / new_width
        h_norm = (h0_norm * resize_components.resized_image_size.height) / new_height
        return custom_yolo_lib.process.bbox.Bbox(
            x=x_norm * new_width,
            y=y_norm * new_height,
            w=w_norm * new_width,
            h=h_norm * new_height,
            is_normalized=False,
        )


def translate_bbox_to_padded_image(
    bbox: custom_yolo_lib.process.bbox.Bbox,
    fixed_ratio_components: custom_yolo_lib.process.image.resize.fixed_ratio.ResizeFixedRatioComponents,
    padding_percentage: float,
    image_size: custom_yolo_lib.image_size.ImageSize,
) -> Tuple[int, int, int, int]:
    if bbox.is_normalized:
        return __translate_norm_bbox_to_padded_image(
            (bbox.x, bbox.y, bbox.w, bbox.h),
            fixed_ratio_components,
            padding_percentage,
            image_size,
        )
    else:
        raise NotImplementedError("Only normalized bounding boxes are supported")


def __map_bbox_to_padded_image(
    x1: float,
    y1: float,
    w: float,
    h: float,
    fixed_ratio_components: custom_yolo_lib.process.image.resize.fixed_ratio.ResizeFixedRatioComponents,
    padding_percent: float,
):
    max_pad_amount = fixed_ratio_components.get_pad_amount()
    pad_amount = int(max_pad_amount * padding_percent)
    if (
        fixed_ratio_components.pad_dimension
        == custom_yolo_lib.image_size.Dimension.HEIGHT
    ):
        y1 += pad_amount
        im_w = fixed_ratio_components.resize_width
        im_h = fixed_ratio_components.expected_dimension_size
    elif (
        fixed_ratio_components.pad_dimension
        == custom_yolo_lib.image_size.Dimension.WIDTH
    ):
        x1 += pad_amount
        im_w = fixed_ratio_components.expected_dimension_size
        im_h = fixed_ratio_components.resize_height
    else:
        raise ValueError("The pad_dimension should be 1 or 2")

    if x1 + w >= im_w:
        w = fixed_ratio_components.resize_width - x1 - 1
    if y1 + h >= im_h:
        h = fixed_ratio_components.resize_height - y1 - 1
    return x1, y1, w, h


def __translate_norm_bbox_to_padded_image(
    normalized_bbox: custom_yolo_lib.process.bbox.Bbox,
    fixed_ratio_components: custom_yolo_lib.process.image.resize.fixed_ratio.ResizeFixedRatioComponents,
    padding_percent: float,
    field_shape: custom_yolo_lib.image_size.ImageSize,
) -> Tuple[int, int, int, int]:
    if any(x > 1 for x in normalized_bbox):
        raise ValueError("Normalized bounding box values should be between 0 and 1")

    x1, y1, w, h = normalized_bbox

    # Scale - Denormalize
    x1 *= fixed_ratio_components.resize_width
    y1 *= fixed_ratio_components.resize_height
    w *= fixed_ratio_components.resize_width
    h *= fixed_ratio_components.resize_height

    # Translate
    x1, y1, w, h = __map_bbox_to_padded_image(
        x1, y1, w, h, fixed_ratio_components, padding_percent
    )

    # Normalize
    return (
        x1 / field_shape.width,
        y1 / field_shape.height,
        w / field_shape.width,
        h / field_shape.height,
    )
