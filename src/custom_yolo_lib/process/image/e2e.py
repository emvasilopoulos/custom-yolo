from typing import Tuple
import torch
import custom_yolo_lib.image_size
import custom_yolo_lib.process.image.pipeline
import custom_yolo_lib.process.tensor
import custom_yolo_lib.process.normalize
import custom_yolo_lib.process.image.resize.fixed_ratio


class E2EPreprocessor:

    input_pipeline = custom_yolo_lib.process.image.pipeline.ImagePipeline(
        dtype_converter=custom_yolo_lib.process.tensor.TensorDtypeConverter(
            torch.float32
        ),
        normalize=custom_yolo_lib.process.normalize.SimpleImageNormalizer(),
    )

    def __init__(self, expected_image_size: custom_yolo_lib.image_size.ImageSize):
        self.expected_image_size = expected_image_size

    def __call__(
        self, raw_img_tensor: torch.Tensor, padding_percent: float, pad_value: int
    ) -> Tuple[
        torch.Tensor,
        custom_yolo_lib.process.image.resize.fixed_ratio.ResizeFixedRatioComponents_v2,
    ]:
        """ """
        img_tensor = self.input_pipeline(raw_img_tensor)
        resize_fixed_ratio_components = custom_yolo_lib.process.image.resize.fixed_ratio.ResizeFixedRatioComponents_v2(
            current_image_size=custom_yolo_lib.image_size.ImageSize(
                width=img_tensor.shape[2], height=img_tensor.shape[1]
            ),
            expected_image_size=self.expected_image_size,
        )
        standard_resized_img_tensor = custom_yolo_lib.process.image.resize.fixed_ratio.resize_image_with_ready_components(
            img_tensor,
            fixed_ratio_components=resize_fixed_ratio_components,
            padding_percent=padding_percent,
            pad_value=pad_value,
        )

        return standard_resized_img_tensor, resize_fixed_ratio_components
