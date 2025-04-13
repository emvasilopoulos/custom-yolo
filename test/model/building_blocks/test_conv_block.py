import unittest

import torch

from custom_yolo_lib.model.building_blocks.conv_block import ConvBlock


class TestConvBlock(unittest.TestCase):

    def test_forward(self):
        input_tensor = torch.randn((1, 3, 64, 64))
        self.conv_block = ConvBlock(3, 16, 3, 1, 1)
        output_tensor = self.conv_block(input_tensor)
        self.assertEqual(output_tensor.shape, (1, 16, 64, 64))

    def test_forward2(self):
        input_tensor = torch.randn((1, 3, 128, 128))
        self.conv_block = ConvBlock(3, 64, 3, 1, 1)
        output_tensor = self.conv_block(input_tensor)
        self.assertEqual(output_tensor.shape, (1, 64, 128, 128))

    def test_forward3(self):
        input_tensor = torch.randn((1, 15, 256, 256))
        self.conv_block = ConvBlock(15, 128, 3, 1, 1)
        output_tensor = self.conv_block(input_tensor)
        self.assertEqual(output_tensor.shape, (1, 128, 256, 256))


if __name__ == "__main__":
    unittest.main()
