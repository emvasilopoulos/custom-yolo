import unittest

import torch

from custom_yolo_lib.model.necks.attention import AttentionNeck, AttentionNeck2


class TestAttentionNeck(unittest.TestCase):

    def setUp(self):
        self.model = AttentionNeck()
        self.model2 = AttentionNeck2()

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
        small = torch.randn(1, 128, 52, 52)
        medium = torch.randn(1, 256, 26, 26)
        large = torch.randn(1, 512, 13, 13)

        small_out, medium_out, large_out = self.model2(small, medium, large)

        # Check output shapes
        self.assertEqual(small_out.shape, (1, 128, 52, 52))
        self.assertEqual(medium_out.shape, (1, 256, 26, 26))
        self.assertEqual(large_out.shape, (1, 512, 13, 13))


if __name__ == "__main__":
    unittest.main()
