import abc
import pathlib
import random
from typing import List, Tuple
import enum

import torch

import custom_yolo_lib.config
import custom_yolo_lib.dataset.coco.tasks.utils
import custom_yolo_lib.dataset.object
import custom_yolo_lib.io.read
import custom_yolo_lib.image_size
import custom_yolo_lib.process.bbox
import custom_yolo_lib.process.bbox.translate
import custom_yolo_lib.process.image
import custom_yolo_lib.process.image.pipeline
import custom_yolo_lib.process.image.resize
import custom_yolo_lib.process.image.resize.fixed_ratio
import custom_yolo_lib.process.normalize
import custom_yolo_lib.process.tensor

MAX_OBJECTS_PER_IMAGE = 100


class COCOYear(enum.Enum):
    YEAR_2014 = 2014
    YEAR_2017 = 2017


class COCOType(enum.Enum):
    INSTANCES = enum.auto()
    CAPTIONS = enum.auto()
    PERSON_KEYPOINTS = enum.auto()


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
    def _read_annotations(self, annotations_path: pathlib.Path) -> None:
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
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
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
        self.device = device
        self.dtype = dtype
        year = self.get_year()
        coco_type = self.get_type()

        self.images_dir = data_dir / f"{split}{year}"
        annotations_dir = data_dir / "annotations/csv"

        if classes is not None:
            self.desired_classes = classes
        else:
            self.desired_classes = [str(i + 1) for i in range(90)]

        self.input_pipeline = custom_yolo_lib.process.image.pipeline.ImagePipeline(
            dtype_converter=custom_yolo_lib.process.tensor.TensorDtypeConverter(
                torch.float32
            ),
            normalize=custom_yolo_lib.process.normalize.SimpleImageNormalizer(),
        )

        annotations_path = (
            annotations_dir
            / custom_yolo_lib.dataset.coco.tasks.utils.get_task_file(
                coco_type,
                split,
                str(year),
                is_grouped=True,
                filetype=custom_yolo_lib.dataset.coco.tasks.utils.AnnotationsType.csv,
            )
        )

        self._read_annotations(annotations_path)

    def _read_image(self, image_path: pathlib.Path) -> torch.Tensor:
        img_tensor = custom_yolo_lib.io.read.read_image_torchvision(image_path)
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
            return random.random() * 0.45
        else:
            return random.random() * 0.45 + 0.55

    @abc.abstractmethod
    def _get_coco_item(
        self, idx: int
    ) -> Tuple[pathlib.Path, List[custom_yolo_lib.dataset.object.Object]]:
        """
        Args:
            idx (int): index of sample in dataset

        Returns:
            Tuple[pathlib.Path, List[custom_yolo_lib.dataset.object.Object]]:
            1. path to image
            2. list of objects
        """
        pass

    def _prepare_objects_tensor(
        self,
        objects: List[custom_yolo_lib.dataset.object.Object],
        resize_fixed_ratio_components: custom_yolo_lib.process.image.resize.fixed_ratio.ResizeFixedRatioComponents_v2,
        padding_percent: float,
        pad_value: int,
    ) -> torch.Tensor:
        if len(objects) > MAX_OBJECTS_PER_IMAGE:
            raise ValueError(
                f"Number of objects {len(objects)} exceeds maximum {MAX_OBJECTS_PER_IMAGE}."
            )
        # Resize bboxes to match resized image
        resize_components, padding = (
            resize_fixed_ratio_components.get_translation_components(
                padding_percent, pad_value
            )
        )
        objects_tensor = (
            custom_yolo_lib.dataset.coco.tasks.utils.create_empty_coco_object_tensor(
                n_coco_classes=len(self.desired_classes),
                n_objects=MAX_OBJECTS_PER_IMAGE,
                dtype=self.dtype,
                device=self.device,
            )
        )
        for i, object_ in enumerate(objects):
            object_.bbox = (
                custom_yolo_lib.process.bbox.translate.translate_bbox_to_resized_image(
                    bbox=object_.bbox,
                    resize_components=resize_components,
                    padding=padding,
                )
            )
            objects_tensor[i, :] = (
                custom_yolo_lib.dataset.coco.tasks.utils.object_to_tensor(
                    object_=object_,
                    n_coco_classes=len(self.desired_classes),
                    dtype=self.dtype,
                    device=self.device,
                )
            )
        return objects_tensor

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_path, objects = self._get_coco_item(idx)

        # Read image
        img_tensor = self.input_pipeline(self._read_image(image_path))

        # Resize image to configured dimensions for training/validation
        padding_percent = self._random_percentage()
        pad_value = random.randint(0, 255)
        resize_fixed_ratio_components = custom_yolo_lib.process.image.resize.fixed_ratio.ResizeFixedRatioComponents_v2(
            current_image_size=custom_yolo_lib.image_size.ImageSize(
                width=img_tensor.shape[2], height=img_tensor.shape[1]
            ),
            expected_image_size=self.expected_image_size,
        )
        standard_resized_img_tensor = custom_yolo_lib.process.image.resize.fixed_ratio.resize_image_with_ready_components(
            img_tensor,
            fixed_ratio_components=resize_fixed_ratio_components,
            padding_percent=padding_percent,
            pad_value=pad_value,
        )

        standard_resized_objects_tensor = self._prepare_objects_tensor(
            objects=objects,
            resize_fixed_ratio_components=resize_fixed_ratio_components,
            padding_percent=padding_percent,
            pad_value=pad_value,
        )
        return (
            standard_resized_img_tensor,
            standard_resized_objects_tensor,
            torch.Tensor([len(objects)]),
        )


class COCODatasetInstances2017(BaseCOCODatasetGrouped):

    def get_year(self) -> int:
        return COCOYear.YEAR_2017.value

    def get_type(self) -> str:
        return COCOType.INSTANCES.name.lower()
