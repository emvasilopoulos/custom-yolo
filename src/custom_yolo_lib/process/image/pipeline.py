import pathlib
from typing import List

import torch
import torchvision

import custom_yolo_lib.io.read


class ImagePipeline:

    def __init__(
        self,
        dtype_converter: torch.nn.Module,
        normalize: torch.nn.Module,
        modules_list: List[torch.nn.Module] = [],
    ):
        self.__pipeline = torchvision.transforms.Compose(
            [dtype_converter, normalize] + modules_list
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.process_input(x)

    def process_input(self, x: torch.Tensor) -> torch.Tensor:
        return self.__pipeline(x)

    def read_and_process_input(self, image_path: pathlib.Path) -> torch.Tensor:
        return self.process_input(
            custom_yolo_lib.io.read.read_image_torchvision(image_path)
        )
