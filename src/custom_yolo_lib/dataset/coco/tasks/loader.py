import dataclasses
from typing import List, Tuple
import torch

from custom_yolo_lib.dataset.coco.tasks.base import BaseCOCODatasetGrouped
from custom_yolo_lib.dataset.coco.tasks.sample import (
    COCODatasetSample,
    COCODatasetSampleKeys,
)
from custom_yolo_lib.dataset.coco.constants import MEDIUM_AREA_RANGE, SMALL_AREA_RANGE


@dataclasses.dataclass
class COCODataLoaderBatch:
    images_batch: torch.Tensor
    objects_batch: List[torch.Tensor]

    def get_sample(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.images_batch[index], self.objects_batch[index]


def _split_target(
    target: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    areas = target[:, 2] * target[:, 3]

    # Small objects
    target_s = target[areas < SMALL_AREA_RANGE[1]]

    # Medium objects
    target_m = target[(areas >= MEDIUM_AREA_RANGE[0]) & (areas < MEDIUM_AREA_RANGE[1])]

    # Large objects
    target_l = target[areas >= MEDIUM_AREA_RANGE[1]]
    return target_s, target_m, target_l


def split_batch_per_bbox_area_size(
    coco_batch: COCODataLoaderBatch,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    targets_s = []  # small objects
    targets_m = []
    targets_l = []
    for target in coco_batch.objects_batch:
        if target.size() == torch.Size([0]):
            # No objects in the image
            targets_s.append(target)
            targets_m.append(target)
            targets_l.append(target)
            continue

        target_s, target_m, target_l = _split_target(target)

        targets_s.append(target_s)
        targets_m.append(target_m)
        targets_l.append(target_l)
    return targets_s, targets_m, targets_l


class COCODataLoader(torch.utils.data.DataLoader):

    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=None,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        collate_fn=None,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
        multiprocessing_context=None,
        generator=None,
        *,
        prefetch_factor=None,
        persistent_workers=False,
        pin_memory_device="",
        in_order=True,
    ):
        super().__init__(
            dataset,
            batch_size,
            shuffle,
            sampler,
            batch_sampler,
            num_workers,
            collate_fn,
            pin_memory,
            drop_last,
            timeout,
            worker_init_fn,
            multiprocessing_context,
            generator,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            pin_memory_device=pin_memory_device,
            in_order=in_order,
        )
        if not isinstance(dataset, BaseCOCODatasetGrouped):
            raise TypeError(
                f"dataset should be an instance of BaseCOCODatasetGrouped, got {type(dataset)}"
            )
        self.collate_fn = self._coco_collate_fn

    def _coco_collate_fn(self, batch: List[COCODatasetSample]) -> COCODataLoaderBatch:
        images_batch = []
        objects_batch = []
        for i in range(len(batch)):
            images_batch.append(batch[i][COCODatasetSampleKeys.IMAGE_TENSOR])
            objects_batch.append(batch[i][COCODatasetSampleKeys.OBJECTS_TENSOR])
        return COCODataLoaderBatch(
            images_batch=torch.stack(images_batch, dim=0),
            objects_batch=objects_batch,
        )


@dataclasses.dataclass
class COCODataLoaderThreeFeatureMapBatch:
    images_batch: torch.Tensor
    small_objects_batch: List[torch.Tensor]
    medium_objects_batch: List[torch.Tensor]
    large_objects_batch: List[torch.Tensor]

    def get_sample(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.images_batch[index],
            self.small_objects_batch[index],
            self.medium_objects_batch[index],
            self.large_objects_batch[index],
        )


class COCODataLoaderThreeFeatureMaps(torch.utils.data.DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=None,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        collate_fn=None,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
        multiprocessing_context=None,
        generator=None,
        *,
        prefetch_factor=None,
        persistent_workers=False,
        pin_memory_device="",
        in_order=True,
    ):
        super().__init__(
            dataset,
            batch_size,
            shuffle,
            sampler,
            batch_sampler,
            num_workers,
            collate_fn,
            pin_memory,
            drop_last,
            timeout,
            worker_init_fn,
            multiprocessing_context,
            generator,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            pin_memory_device=pin_memory_device,
            in_order=in_order,
        )
        if not isinstance(dataset, BaseCOCODatasetGrouped):
            raise TypeError(
                f"dataset should be an instance of BaseCOCODatasetGrouped, got {type(dataset)}"
            )
        self.collate_fn = self._coco_collate_fn

    def _coco_collate_fn(
        self, batch: List[COCODatasetSample]
    ) -> COCODataLoaderThreeFeatureMapBatch:
        images_batch = []
        small_objects_batch = []  # Varying number of objects per image
        medium_objects_batch = []  # Varying number of objects per image
        large_objects_batch = []  # Varying number of objects per image
        for i in range(len(batch)):
            images_batch.append(batch[i][COCODatasetSampleKeys.IMAGE_TENSOR])

            target = batch[i][COCODatasetSampleKeys.OBJECTS_TENSOR]
            if not target.size() == torch.Size([0]):
                target_s, target_m, target_l = _split_target(target)
                small_objects_batch.append(target_s)
                medium_objects_batch.append(target_m)
                large_objects_batch.append(target_l)
            else:
                small_objects_batch.append(target)
                medium_objects_batch.append(target)
                large_objects_batch.append(target)
        return COCODataLoaderThreeFeatureMapBatch(
            images_batch=torch.stack(images_batch, dim=0),
            small_objects_batch=small_objects_batch,
            medium_objects_batch=medium_objects_batch,
            large_objects_batch=large_objects_batch,
        )
