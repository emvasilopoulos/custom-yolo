import unittest

import torch
import custom_yolo_lib.process.image.resize


class TestResizeImage(unittest.TestCase):

    def test_resize_image(self):
        # Create a dummy image tensor of size (3, 224, 224)
        x = torch.randn(3, 224, 224)

        # Resize the image to (112, 112)
        new_height, new_width = 112, 112
        resized_image = custom_yolo_lib.process.image.resize.resize_image(
            x, new_height, new_width
        )

        # Check the size of the resized image
        self.assertEqual(resized_image.shape[1], new_height)
        self.assertEqual(resized_image.shape[2], new_width)

    def test_resize_image_batch(self):
        # Create a dummy batch of images tensor of size (2, 3, 224, 224)
        x = torch.randn(2, 3, 224, 224)

        # Resize the batch of images to (112, 112)
        new_height, new_width = 112, 112
        resized_image = custom_yolo_lib.process.image.resize.resize_image(
            x, new_height, new_width
        )

        # Check the size of the resized image
        self.assertEqual(resized_image.shape[2], new_height)
        self.assertEqual(resized_image.shape[3], new_width)

    def test_invalid_input(self):
        # Create a dummy image tensor of size (3, 224)
        x = torch.randn(3, 224)

        # Check if ValueError is raised for invalid input
        with self.assertRaises(ValueError):
            custom_yolo_lib.process.image.resize.resize_image(x, 112, 112)

    def test_invalid_input2(self):
        # Create a dummy image tensor of size (3, 224)
        x = torch.randn(3, 2, 2, 2, 2, 2)

        # Check if ValueError is raised for invalid input
        with self.assertRaises(ValueError):
            custom_yolo_lib.process.image.resize.resize_image(x, 112, 112)


if __name__ == "__main__":
    unittest.main()
