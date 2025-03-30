import unittest

import torch
import custom_yolo_lib.process.image.resize.fixed_ratio
import custom_yolo_lib.image_size


class TestFixedRatioResize(unittest.TestCase):
    PADDING_VALUE = 123

    def _resize_image(
        self,
        image_size: custom_yolo_lib.image_size.ImageSize,
        expected_image_size: custom_yolo_lib.image_size.ImageSize,
        padding_percent: float,
    ):
        # Create a dummy image tensor
        image = torch.randn(
            3, image_size.height, image_size.width
        )  # Example image with 3 channels and 640x640 size

        # Define the target size
        TARGET_WIDTH, TARGET_HEIGHT = (
            expected_image_size.width,
            expected_image_size.height,
        )

        # Create the FixedRatioResize object
        return custom_yolo_lib.process.image.resize.fixed_ratio.resize_image(
            image,
            new_height=TARGET_HEIGHT,
            new_width=TARGET_WIDTH,
            padding_percent=padding_percent,
            padding_value=self.PADDING_VALUE,
        )

    def test_fixed_ratio_resize0(self):
        expected_image_size = custom_yolo_lib.image_size.ImageSize(320, 320)
        resized_image = self._resize_image(
            custom_yolo_lib.image_size.ImageSize(640, 640),
            expected_image_size,
            padding_percent=0.0,
        )

        self.assertFalse(torch.all(resized_image[:, 0, :] == self.PADDING_VALUE))
        self.assertFalse(torch.all(resized_image[:, :, 0] == self.PADDING_VALUE))
        self.assertFalse(
            torch.all(
                resized_image[:, expected_image_size.height - 1, :]
                == self.PADDING_VALUE
            )
        )
        self.assertFalse(
            torch.all(
                resized_image[:, :, expected_image_size.width - 1] == self.PADDING_VALUE
            )
        )

    def test_fixed_ratio_resize1(self):
        expected_image_size = custom_yolo_lib.image_size.ImageSize(640, 641)
        resized_image = self._resize_image(
            custom_yolo_lib.image_size.ImageSize(640, 640),
            expected_image_size,
            padding_percent=0.0,
        )

        self.assertFalse(torch.all(resized_image[:, 0, :] == self.PADDING_VALUE))
        self.assertFalse(torch.all(resized_image[:, :, 0] == self.PADDING_VALUE))
        self.assertTrue(
            torch.all(
                resized_image[:, expected_image_size.height - 1, :]
                == self.PADDING_VALUE
            )
        )
        self.assertFalse(
            torch.all(
                resized_image[:, :, expected_image_size.width - 1] == self.PADDING_VALUE
            )
        )

    def test_fixed_ratio_resize2(self):
        expected_image_size = custom_yolo_lib.image_size.ImageSize(641, 640)
        resized_image = self._resize_image(
            custom_yolo_lib.image_size.ImageSize(640, 640),
            expected_image_size,
            padding_percent=0.0,
        )

        self.assertFalse(torch.all(resized_image[:, :, 0] == self.PADDING_VALUE))
        self.assertTrue(
            torch.all(
                resized_image[:, :, expected_image_size.width - 1] == self.PADDING_VALUE
            )
        )

    def test_fixed_ratio_resize3(self):
        expected_image_size = custom_yolo_lib.image_size.ImageSize(640, 641)
        resized_image = self._resize_image(
            custom_yolo_lib.image_size.ImageSize(640, 640),
            expected_image_size,
            padding_percent=1.0,
        )

        self.assertFalse(
            torch.all(
                resized_image[:, expected_image_size.height - 1, :]
                == self.PADDING_VALUE
            )
        )
        self.assertTrue(torch.all(resized_image[:, 0, :] == self.PADDING_VALUE))

    def test_fixed_ratio_resize4(self):
        expected_image_size = custom_yolo_lib.image_size.ImageSize(641, 640)
        resized_image = self._resize_image(
            custom_yolo_lib.image_size.ImageSize(640, 640),
            expected_image_size,
            padding_percent=0.5,
        )
        self.assertFalse(torch.all(resized_image[:, :, 0] == self.PADDING_VALUE))
        self.assertTrue(
            torch.all(
                resized_image[:, :, expected_image_size.width - 1] == self.PADDING_VALUE
            )
        )

    def test_fixed_ratio_resize5(self):
        expected_image_size = custom_yolo_lib.image_size.ImageSize(641, 641)
        resized_image = self._resize_image(
            custom_yolo_lib.image_size.ImageSize(640, 640),
            expected_image_size,
            padding_percent=0.5,
        )

        # Resizes with no padding is the expected behavior

        ##
        self.assertFalse(torch.all(resized_image[:, :, 0] == self.PADDING_VALUE))
        self.assertFalse(torch.all(resized_image[:, 0, :] == self.PADDING_VALUE))
        self.assertFalse(
            torch.all(
                resized_image[:, expected_image_size.height - 1, :]
                == self.PADDING_VALUE
            )
        )
        self.assertFalse(
            torch.all(
                resized_image[:, :, expected_image_size.width - 1] == self.PADDING_VALUE
            )
        )

    def test_fixed_ratio_resize6(self):
        expected_image_size = custom_yolo_lib.image_size.ImageSize(960, 640)
        resized_image = self._resize_image(
            custom_yolo_lib.image_size.ImageSize(640, 640),
            expected_image_size,
            padding_percent=0.5,
        )
        self.assertTrue(torch.all(resized_image[:, :, 0:160] == self.PADDING_VALUE))
        self.assertTrue(torch.all(resized_image[:, :, 800:] == self.PADDING_VALUE))


if __name__ == "__main__":
    unittest.main()
