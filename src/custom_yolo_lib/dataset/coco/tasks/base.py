import abc
import pathlib
import random
from typing import List, Optional, Tuple
import enum

import torch

from custom_yolo_lib.dataset.coco.tasks.sample import (
    COCODatasetSample,
    COCODatasetSampleKeys,
)
import custom_yolo_lib.dataset.coco.tasks.utils
import custom_yolo_lib.dataset.object
import custom_yolo_lib.io.read
import custom_yolo_lib.image_size
import custom_yolo_lib.process.bbox
import custom_yolo_lib.process.bbox.translate
import custom_yolo_lib.process.image
import custom_yolo_lib.process.image.resize.fixed_ratio
import custom_yolo_lib.process.image.e2e
import custom_yolo_lib.dataset.augmentation_types

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
        e2e_preprocessor: custom_yolo_lib.process.image.e2e.E2EPreprocessor,
        classes: List[str] = None,
        dtype: torch.dtype = torch.float32,
        is_sama: bool = True,  # the original is CRAP
        augmentations: Optional[
            List[custom_yolo_lib.dataset.augmentation_types.AugmentationType]
        ] = None,
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
        self.dtype = dtype
        year = self.get_year()
        coco_type = self.get_type()

        self.images_dir = data_dir / f"{split}{year}"
        annotations_dir = data_dir / "annotations/csv"

        if classes is not None:
            self.desired_classes = classes
        else:
            self.desired_classes = [str(i + 1) for i in range(80)]

        self.e2e_preprocessor = e2e_preprocessor

        annotations_path = (
            annotations_dir
            / custom_yolo_lib.dataset.coco.tasks.utils.get_task_file(
                coco_type,
                split,
                str(year),
                is_grouped=True,
                filetype=custom_yolo_lib.dataset.coco.tasks.utils.AnnotationsType.csv,
                is_sama=is_sama,
            )
        )

        self._read_annotations(annotations_path)
        self.augmentations = augmentations
        if self.augmentations is None:
            print(
                "Using default augmentations --> [FLIP_X, FLIP_Y, SLIGHT_COLOR_JITTER, SLIGHT_RESIZE]"
            )
            self.augmentations = [
                custom_yolo_lib.dataset.augmentation_types.AugmentationType.FLIP_X,
                custom_yolo_lib.dataset.augmentation_types.AugmentationType.FLIP_Y,
                custom_yolo_lib.dataset.augmentation_types.AugmentationType.SLIGHT_COLOR_JITTER,
                custom_yolo_lib.dataset.augmentation_types.AugmentationType.SLIGHT_RESIZE,
            ]
        self._do_slight_resize = (
            custom_yolo_lib.dataset.augmentation_types.AugmentationType.SLIGHT_RESIZE
            in self.augmentations
        )

    def _read_image(self, image_path: pathlib.Path) -> torch.Tensor:
        img_tensor = custom_yolo_lib.io.read.read_image_torchvision(image_path)
        if img_tensor.shape[0] == 1:
            img_tensor = img_tensor.repeat(3, 1, 1)
        return img_tensor

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

    def _translate_objects_to_resized_image(
        self,
        objects: List[custom_yolo_lib.dataset.object.Object],
        resize_fixed_ratio_components: custom_yolo_lib.process.image.resize.fixed_ratio.ResizeFixedRatioComponents_v2,
        padding_percent: float,
        pad_value: int,
    ) -> List[custom_yolo_lib.dataset.object.Object]:
        resize_components, padding = (
            resize_fixed_ratio_components.get_translation_components(
                padding_percent, pad_value
            )
        )

        for i, object_ in enumerate(objects):
            if not object_.bbox.is_normalized:
                raise ValueError(f"Object bbox {object_.bbox} is not normalized.")
            object_.bbox = (
                custom_yolo_lib.process.bbox.translate.translate_bbox_to_resized_image(
                    bbox=object_.bbox,
                    resize_components=resize_components,
                    padding=padding,
                )
            )
        return objects

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
            )
        )
        for i, object_ in enumerate(objects):
            if not object_.bbox.is_normalized:
                raise ValueError(f"Object bbox {object_.bbox} is not normalized.")
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
                )
            )
        return objects_tensor

    def __getitem__(self, idx: int) -> COCODatasetSample:
        image_path, objects = self._get_coco_item(idx)

        padding_percent = self._random_percentage()
        pad_value = random.randint(0, 255)

        img_tensor = self._read_image(image_path)
        if self._do_slight_resize and random.random() > 0.5:
            _, h, w = img_tensor.shape
            if h > w:
                new_h = h
                new_w = int(w * random.uniform(0.8, 1.2))
            else:
                new_h = int(h * random.uniform(0.8, 1.2))
                new_w = w
            img_tensor = custom_yolo_lib.process.image.resize.resize_image(
                img_tensor, new_h, new_w
            )
        standard_resized_img_tensor, resize_fixed_ratio_components = (
            self.e2e_preprocessor(
                img_tensor,
                padding_percent=padding_percent,
                pad_value=pad_value,
            )
        )

        translated_objects = self._translate_objects_to_resized_image(
            objects=objects,
            resize_fixed_ratio_components=resize_fixed_ratio_components,
            padding_percent=padding_percent,
            pad_value=pad_value,
        )
        assert len(translated_objects) == len(objects)  # sanity check

        # NOTE: sacrificing "optimization" for readability (meaning I could run this in a previous loop)
        for obj in translated_objects:
            obj.bbox.to_center()

        if not translated_objects:
            objects_tensor = torch.tensor([])
        else:
            objects_tensor = torch.stack(
                [obj.to_tensor() for obj in translated_objects]
            )

        for augmentation in self.augmentations:
            if (
                augmentation
                == custom_yolo_lib.dataset.augmentation_types.AugmentationType.FLIP_X
                and random.random() > 0.5
            ):
                standard_resized_img_tensor = torch.flip(
                    standard_resized_img_tensor, dims=[2]
                )
                for i in range(objects_tensor.shape[0]):
                    objects_tensor[i, 0] = 1 - objects_tensor[i, 0]
            elif (
                augmentation
                == custom_yolo_lib.dataset.augmentation_types.AugmentationType.FLIP_Y
                and random.random() > 0.5
            ):
                standard_resized_img_tensor = torch.flip(
                    standard_resized_img_tensor, dims=[1]
                )
                for i in range(objects_tensor.shape[0]):
                    objects_tensor[i, 1] = 1 - objects_tensor[i, 1]
            elif (
                augmentation
                == custom_yolo_lib.dataset.augmentation_types.AugmentationType.SLIGHT_COLOR_JITTER
                and random.random() > 0.5
            ):
                pass
                # standard_resized_img_tensor = (
                #     custom_yolo_lib.process.image.color_jitter(
                #         standard_resized_img_tensor
                #     )
                # )

        return {
            COCODatasetSampleKeys.IMAGE_TENSOR: standard_resized_img_tensor,
            COCODatasetSampleKeys.OBJECTS_TENSOR: objects_tensor,
            COCODatasetSampleKeys.OBJECTS_COUNT: torch.tensor(
                [len(objects)], dtype=torch.uint16
            ),
        }


class COCODatasetInstances2017(BaseCOCODatasetGrouped):

    def get_year(self) -> int:
        return COCOYear.YEAR_2017.value

    def get_type(self) -> str:
        return COCOType.INSTANCES.name.lower()
