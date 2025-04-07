import unittest

import torch

from custom_yolo_lib.model.backbones.darknet_53 import Darknet53Backbone


class TestDarknet53Backbone(unittest.TestCase):

    def setUp(self):
        self.model = Darknet53Backbone()

    def test_forward(self):
        input_tensor = torch.randn(
            (1, 3, 416, 416)
        )  # Batch size of 1, 3 channels, 416x416 image
        y = self.model(input_tensor)
        self.assertEqual(y.shape, (1, 1024, 13, 13))

    def test_forward2(self):
        input_tensor = torch.randn((1, 3, 640, 640))
        y = self.model(input_tensor)

        self.assertEqual(y.shape, (1, 1024, 20, 20))


if __name__ == "__main__":
    unittest.main()
