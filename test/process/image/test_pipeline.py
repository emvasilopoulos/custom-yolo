import unittest

import torch

import custom_yolo_lib.process.image.pipeline
import custom_yolo_lib.process.tensor
import custom_yolo_lib.process.normalize


class ProcessTest(unittest.TestCase):

    def test_process_input_pipeline(self):
        dtype_converter = custom_yolo_lib.process.tensor.TensorDtypeConverter(
            torch.float32
        )
        normalize = custom_yolo_lib.process.normalize.SimpleImageNormalizer()
        pipeline = custom_yolo_lib.process.image.pipeline.ImagePipeline(
            dtype_converter=dtype_converter,
            normalize=normalize,
        )

        x = torch.randint(0, 256, (3, 224, 224), dtype=torch.uint8)

        pipeline_output = pipeline(x)
        deconstructed_pipeline: torch.Tensor = normalize(dtype_converter(x))

        assert pipeline_output.min() == deconstructed_pipeline.min()
        assert pipeline_output.max() == deconstructed_pipeline.max()
        assert torch.allclose(pipeline_output, deconstructed_pipeline)
