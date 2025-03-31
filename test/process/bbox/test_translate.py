import unittest

import custom_yolo_lib.process.bbox
import custom_yolo_lib.process.bbox.translate
import custom_yolo_lib.process.image.resize
import custom_yolo_lib.process.image.resize.fixed_ratio
import custom_yolo_lib.image_size
import custom_yolo_lib.process.image.pad


class TestBboxTranslate(unittest.TestCase):
    def test_translate_norm_bbox_to_padded_image(self):
        # Create a mock bbox
        bbox = custom_yolo_lib.process.bbox.Bbox(
            x=0.5,
            y=0.5,
            w=0.2,
            h=0.2,
            is_normalized=True,
        )

        expected_image_size = custom_yolo_lib.image_size.ImageSize(
            width=1024, height=1024
        )
        fixed_ratio_components = (
            custom_yolo_lib.process.image.resize.fixed_ratio.ResizeFixedRatioComponents(
                resize_height=768,
                resize_width=1024,
                pad_dimension=custom_yolo_lib.image_size.Dimension.HEIGHT,
                expected_dimension_size=1024,
            )
        )

        # Define padding percentage
        padding_percentage = 0.0

        # Call the function to test
        result_x1, result_y1, result_w, result_h = (
            custom_yolo_lib.process.bbox.translate.translate_bbox_to_padded_image(
                bbox,
                fixed_ratio_components,
                padding_percentage,
                expected_image_size,
            )
        )
        self.assertAlmostEqual(result_x1, 0.5)
        self.assertAlmostEqual(result_y1, 0.375)
        self.assertAlmostEqual(result_w, 0.2)
        self.assertAlmostEqual(result_h, 0.15)

    def test_translate_norm_bbox_to_padded_image_v2(self):
        bbox = custom_yolo_lib.process.bbox.Bbox(
            x=0.5,
            y=0.5,
            w=0.2,
            h=0.2,
            is_normalized=True,
        )
        current_image_size = custom_yolo_lib.image_size.ImageSize(
            width=1024, height=768
        )
        resized_image_size = custom_yolo_lib.image_size.ImageSize(
            width=1024, height=768
        )
        resize_components = custom_yolo_lib.process.image.resize.ResizeImageSize(
            current_image_size=current_image_size,
            resized_image_size=resized_image_size,
        )
        padding = custom_yolo_lib.process.image.pad.Padding(
            top=0, bottom=256, left=0, right=0, pad_value=0
        )

        result = custom_yolo_lib.process.bbox.translate.translate_bbox_to_resized_image(
            bbox,
            resize_components=resize_components,
            padding=padding,
        )

        self.assertAlmostEqual(result.x, 0.5)
        self.assertAlmostEqual(result.y, 0.375)
        self.assertAlmostEqual(result.w, 0.2)
        self.assertAlmostEqual(result.h, 0.15)

    def test_translate_norm_bbox_to_padded_image_v2_2(self):
        bbox = custom_yolo_lib.process.bbox.Bbox(
            x=0.5,
            y=0.5,
            w=0.2,
            h=0.2,
            is_normalized=True,
        )
        current_image_size = custom_yolo_lib.image_size.ImageSize(
            width=1024, height=768
        )
        resized_image_size = custom_yolo_lib.image_size.ImageSize(
            width=1024, height=768
        )
        resize_components = custom_yolo_lib.process.image.resize.ResizeImageSize(
            current_image_size=current_image_size,
            resized_image_size=resized_image_size,
        )
        padding = custom_yolo_lib.process.image.pad.Padding(
            top=256, bottom=0, left=0, right=0, pad_value=0
        )

        result = custom_yolo_lib.process.bbox.translate.translate_bbox_to_resized_image(
            bbox,
            resize_components=resize_components,
            padding=padding,
        )

        self.assertAlmostEqual(result.x, 0.5)
        self.assertAlmostEqual(result.y, 0.625)
        self.assertAlmostEqual(result.w, 0.2)
        self.assertAlmostEqual(result.h, 0.15)

    def test_translate_norm_bbox_to_padded_image_v2_3(self):
        bbox = custom_yolo_lib.process.bbox.Bbox(
            x=0.5,
            y=0.5,
            w=0.2,
            h=0.2,
            is_normalized=True,
        )
        current_image_size = custom_yolo_lib.image_size.ImageSize(
            width=1024, height=768
        )
        expected_image_size = custom_yolo_lib.image_size.ImageSize(
            width=1024, height=1024
        )
        fixed_ratio_components = custom_yolo_lib.process.image.resize.fixed_ratio.ResizeFixedRatioComponents_v2(
            current_image_size=current_image_size,
            expected_image_size=expected_image_size,
        )
        padding_percent = 1.0
        resize_image, padding = fixed_ratio_components.get_translation_components(
            padding_percent, pad_value=0
        )
        result = custom_yolo_lib.process.bbox.translate.translate_bbox_to_resized_image(
            bbox,
            resize_components=resize_image,
            padding=padding,
        )

        self.assertAlmostEqual(result.x, 0.5)
        self.assertAlmostEqual(result.y, 0.625)
        self.assertAlmostEqual(result.w, 0.2)
        self.assertAlmostEqual(result.h, 0.15)


if __name__ == "__main__":
    unittest.main()
