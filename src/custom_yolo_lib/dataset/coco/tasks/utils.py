import enum
import pathlib

import pandas as pd
import torch

import custom_yolo_lib.io.read
import custom_yolo_lib.dataset.object
import custom_yolo_lib.dataset.coco.tasks.instances


class AnnotationsType(enum.Enum):
    json = "json"
    csv = "csv"


def get_task_file(
    task_name: str, split: str, year: str, is_grouped: bool, filetype: AnnotationsType
) -> str:
    """
    Get the task file path based on the task name, split, and year.

    Args:
        task_name (str): The name of the task.
        split (str): The data split (e.g., 'train', 'val').
        year (str): The year of the dataset.

    Returns:
        str: The path to the task file.
    """
    if is_grouped:
        return f"{task_name}_{split}{year}_grouped_by_image_id.{filetype.value}"
    return f"{task_name}_{split}{year}.{filetype.value}"


def create_empty_coco_object_tensor(
    n_coco_classes: int,
    n_objects: int = 1,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Create an empty COCO tensor.

    Args:
        n_coco_classes (int): The number of COCO classes.

    Returns:
        torch.Tensor: An empty COCO tensor.
    """
    n_coords_dims = 4
    n_objectness_dims = 1
    n_classes_dims = n_coco_classes
    vector_size = n_coords_dims + n_objectness_dims + n_classes_dims
    if n_objects > 1:
        return torch.zeros(
            (n_objects, vector_size),
            device=device,
            dtype=dtype,
            requires_grad=False,
        )
    return torch.zeros(
        vector_size,
        device=device,
        dtype=dtype,
        requires_grad=False,
    )


def object_to_tensor(
    object_: custom_yolo_lib.dataset.object.Object,
    n_coco_classes: int = 80,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Convert a bounding box to a tensor.

    Args:
        object_ (custom_yolo_lib.dataset.object.Object): object with bbox + class_id
        n_coco_classes (int): The number of COCO classes.
        Defaults to 80.

    Returns:
        torch.Tensor: The bounding box as a tensor.
    """
    if object_.class_id >= n_coco_classes:
        raise ValueError(
            f"Class ID {object_.class_id} (starting from index 0) is out of range for {n_coco_classes} classes."
        )

    x = create_empty_coco_object_tensor(
        n_coco_classes,
        n_objects=1,
        dtype=dtype,
        device=device,
    )

    x[0] = object_.bbox.x
    x[1] = object_.bbox.y
    x[2] = object_.bbox.w
    x[3] = object_.bbox.h
    x[4] = 1.0  # objectness
    x[5 + object_.class_id - 1] = 1.0  # class_id
    return x


def convert_grouped_instances_json_to_csv(json_path: pathlib.Path):
    annotations = custom_yolo_lib.io.read.read_json(json_path)[
        custom_yolo_lib.dataset.coco.tasks.instances.ANNOTATIONS_GROUPED_KEY
    ]
    annotations_for_df = {
        "image_id": [],
        "category_id": [],
        "x1": [],
        "y1": [],
        "w": [],
        "h": [],
        "area": [],
        "iscrowd": [],
        "id": [],
        "segmentation": [],
    }

    for item in annotations.keys():
        for annotation in annotations[item]:
            annotations_for_df["image_id"].append(item)
            annotations_for_df["category_id"].append(annotation["category_id"])
            annotations_for_df["x1"].append(annotation["bbox"][0])
            annotations_for_df["y1"].append(annotation["bbox"][1])
            annotations_for_df["w"].append(annotation["bbox"][2])
            annotations_for_df["h"].append(annotation["bbox"][3])
            annotations_for_df["area"].append(annotation["area"])
            annotations_for_df["iscrowd"].append(annotation["iscrowd"])
            annotations_for_df["id"].append(annotation["id"])
            annotations_for_df["segmentation"].append(annotation.get("segmentation"))

    df = pd.DataFrame(annotations_for_df)
    df.to_csv(json_path.with_suffix(".csv"), index=False)
