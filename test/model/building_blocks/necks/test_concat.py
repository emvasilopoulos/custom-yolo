import unittest

import torch

from custom_yolo_lib.model.building_blocks.necks.concat import NeckConcat


class TestNeckConcat(unittest.TestCase):

    def setUp(self):
        self.model = NeckConcat()

    def test_forward1(self):
        small = torch.randn(1, 256, 52, 52)
        medium = torch.randn(1, 512, 26, 26)
        large = torch.randn(1, 1024, 13, 13)

        small_out, medium_out, large_out = self.model(small, medium, large)

        # Check output shapes
        self.assertEqual(small_out.shape, (1, 128, 52, 52))
        self.assertEqual(medium_out.shape, (1, 256, 26, 26))
        self.assertEqual(large_out.shape, (1, 512, 13, 13))

    def test_forward2(self):
        small = torch.randn(1, 256, 80, 80)
        medium = torch.randn(1, 512, 40, 40)
        large = torch.randn(1, 1024, 20, 20)

        small_out, medium_out, large_out = self.model(small, medium, large)

        # Check output shapes
        self.assertEqual(small_out.shape, (1, 128, 80, 80))
        self.assertEqual(medium_out.shape, (1, 256, 40, 40))
        self.assertEqual(large_out.shape, (1, 512, 20, 20))


if __name__ == "__main__":
    unittest.main()
