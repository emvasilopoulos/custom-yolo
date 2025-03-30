import unittest

import torch

import custom_yolo_lib.process.image.pad


class TestPad(unittest.TestCase):

    def setUp(self):
        self.x = torch.randn(3, 100, 100)

    def test_pad_image(self):
        pad_dimension = 1
        expected_dimension_size = 200
        padding_percent = 0.5
        pad_value = 0
        y = custom_yolo_lib.process.image.pad.pad_image(
            self.x,
            pad_dimension,
            expected_dimension_size,
            padding_percent,
            pad_value,
        )
        self.assertEqual(y.shape[1], expected_dimension_size)
        self.assertEqual(y.shape[2], self.x.shape[2])
        self.assertEqual(y.shape[0], self.x.shape[0])
        self.assertTrue(torch.all(y[:, :50, :] == pad_value))
        self.assertTrue(torch.all(y[:, 50:150, :] == self.x))
        self.assertTrue(torch.all(y[:, 150:, :] == pad_value))

    def test_pad_image2(self):
        pad_dimension = 2
        expected_dimension_size = 200
        padding_percent = 0.0
        pad_value = 100
        y = custom_yolo_lib.process.image.pad.pad_image(
            self.x,
            pad_dimension,
            expected_dimension_size,
            padding_percent,
            pad_value,
        )
        self.assertEqual(y.shape[2], expected_dimension_size)
        self.assertEqual(y.shape[1], self.x.shape[2])
        self.assertEqual(y.shape[0], self.x.shape[0])
        self.assertTrue(torch.all(y[:, :, :100] == self.x))
        self.assertTrue(torch.all(y[:, :, 100:] == pad_value))

    def test_pad_image3(self):
        pad_dimension = 2
        expected_dimension_size = 200
        padding_percent = 1.0
        pad_value = 100
        y = custom_yolo_lib.process.image.pad.pad_image(
            self.x,
            pad_dimension,
            expected_dimension_size,
            padding_percent,
            pad_value,
        )
        self.assertEqual(y.shape[2], expected_dimension_size)
        self.assertEqual(y.shape[1], self.x.shape[2])
        self.assertEqual(y.shape[0], self.x.shape[0])
        self.assertTrue(torch.all(y[:, :, 100:] == self.x))
        self.assertTrue(torch.all(y[:, :, :100] == pad_value))

    def test_pad_image4(self):
        pad_dimension = 2
        expected_dimension_size = 200
        padding_percent = 1.01
        pad_value = 100
        with self.assertRaises(ValueError):
            custom_yolo_lib.process.image.pad.pad_image(
                self.x,
                pad_dimension,
                expected_dimension_size,
                padding_percent,
                pad_value,
            )

    def test_pad_image_v2(self):
        pad_top = 10
        pad_right = 20
        pad_bottom = 30
        pad_left = 40
        pad_value = 100
        y = custom_yolo_lib.process.image.pad.pad_image_v2(
            self.x, pad_top, pad_right, pad_bottom, pad_left, pad_value
        )
        self.assertEqual(y.shape[1], self.x.shape[1] + pad_top + pad_bottom)
        self.assertEqual(y.shape[2], self.x.shape[2] + pad_left + pad_right)
        self.assertEqual(y.shape[0], self.x.shape[0])
        self.assertTrue(torch.all(y[:, :pad_top, :] == pad_value))
        self.assertTrue(torch.all(y[:, -pad_bottom:, :] == pad_value))
        self.assertTrue(torch.all(y[:, :, :pad_left] == pad_value))
        self.assertTrue(torch.all(y[:, :, -pad_right:] == pad_value))

    def test_pad_image_v3(self):
        pad_top = 10
        pad_right = 20
        pad_bottom = 30
        pad_left = 40
        pad_value = 100
        padding = custom_yolo_lib.process.image.pad.Padding(
            top=pad_top,
            right=pad_right,
            bottom=pad_bottom,
            left=pad_left,
            pad_value=pad_value,
        )
        y = custom_yolo_lib.process.image.pad.pad_image_v3(self.x, padding)
        self.assertEqual(y.shape[1], self.x.shape[1] + pad_top + pad_bottom)
        self.assertEqual(y.shape[2], self.x.shape[2] + pad_left + pad_right)
        self.assertEqual(y.shape[0], self.x.shape[0])
        self.assertTrue(torch.all(y[:, :pad_top, :] == pad_value))
        self.assertTrue(torch.all(y[:, -pad_bottom:, :] == pad_value))
        self.assertTrue(torch.all(y[:, :, :pad_left] == pad_value))
        self.assertTrue(torch.all(y[:, :, -pad_right:] == pad_value))


if __name__ == "__main__":
    unittest.main()
