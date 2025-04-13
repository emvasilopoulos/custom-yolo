import unittest

import torch

from custom_yolo_lib.model.backbones.three_scales import ThreeScalesFeatures


class TestThreeScalesFeatures(unittest.TestCase):

    def setUp(self):
        self.model = ThreeScalesFeatures()

    def test_forward(self):
        input_tensor = torch.randn(
            (1, 3, 416, 416)
        )  # Batch size of 1, 3 channels, 416x416 image
        small, medium, large = self.model(input_tensor)
        self.assertEqual(small.shape, (1, 256, 52, 52))
        self.assertEqual(medium.shape, (1, 512, 26, 26))
        self.assertEqual(large.shape, (1, 1024, 13, 13))

    def test_forward2(self):
        input_tensor = torch.randn((1, 3, 640, 640))
        small, medium, large = self.model(input_tensor)

        self.assertEqual(small.shape, (1, 256, 80, 80))
        self.assertEqual(medium.shape, (1, 512, 40, 40))
        self.assertEqual(large.shape, (1, 1024, 20, 20))


if __name__ == "__main__":
    unittest.main()
