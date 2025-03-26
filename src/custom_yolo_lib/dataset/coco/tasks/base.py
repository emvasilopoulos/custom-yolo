import abc
import pathlib
import random
from typing import Any, Dict, List, Tuple
import torch

import custom_yolo_lib.io.read
import custom_yolo_lib.image_size
import custom_yolo_lib.process.image
import custom_yolo_lib.process.normalize
import custom_yolo_lib.process.tensor


class BaseCOCODatasetGrouped(torch.utils.data.Dataset):

    @abc.abstractmethod
    def get_year(self) -> int:
        pass

    @abc.abstractmethod
    def get_type(self) -> str:
        """
        one of:
        - captions
        - instances
        - person_keypoints
        """
        pass

    @abc.abstractmethod
    def get_pair(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        pass

    @abc.abstractmethod
    def _read_annotations(
        self, annotations_dir: pathlib.Path, coco_type: str, split: str, year: str
    ):
        pass

    @abc.abstractmethod
    def _define_length(self) -> int:
        pass

    def __init__(
        self,
        data_dir: pathlib.Path,
        split: str,
        expected_image_size: custom_yolo_lib.image_size.ImageSize,
        classes: List[str] = None,
    ):
        """
        Args:
            data_dir (str): path to the COCO dataset directory.
            split (str): the split to use, either 'train' or 'val'.
            transforms (Compose): a composition of torchvision.transforms to apply to the images.
        """
        super().__init__()

        self.data_dir = data_dir
        self.split = split
        self.expected_image_size = expected_image_size
        year = self.get_year()
        coco_type = self.get_type()

        self.images_dir = data_dir / f"{split}{year}"
        annotations_dir = data_dir / "annotations"

        self.desired_classes = classes

        self.input_pipeline = custom_yolo_lib.process.image.ImagePipeline(
            dtype_converter=custom_yolo_lib.process.tensor.TensorDtypeConverter(
                torch.float32
            ),
            normalize=custom_yolo_lib.process.normalize.SimpleImageNormalizer(),
        )

        self._read_annotations(annotations_dir, coco_type, split, year)

    def _read_image(self, image_id: int) -> torch.Tensor:
        filename = self._image_file_name_from_id(image_id)
        img_tensor = custom_yolo_lib.io.read.read_image_torchvision(
            self.images_dir / filename
        )
        if img_tensor.shape[0] == 1:
            img_tensor = img_tensor.repeat(3, 1, 1)
        return self.input_pipeline(img_tensor)

    def _image_file_name_from_id(self, image_id: int) -> str:
        return f"{image_id:012}.jpg"

    def __len__(self) -> int:
        return self._define_length()

    def _random_percentage(self):
        side = random.randint(0, 1)
        if side == 0:
            return random.random() * 0.14
        else:
            return random.random() * 0.14 + 0.86

    @abc.abstractmethod
    def _get_coco_item(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        self._get_coco_item(idx)


class COCODatasetInstances2017(BaseCOCODatasetGrouped):

    def get_year(self) -> int:
        return 2017

    def get_type(self) -> str:
        return "instances"
