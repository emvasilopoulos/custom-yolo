import torch
import custom_yolo_lib.dataset
import custom_yolo_lib.dataset.object
import custom_yolo_lib.process.bbox


def get_task_file(task_name: str, split: str, year: str) -> str:
    """
    Get the task file path based on the task name, split, and year.

    Args:
        task_name (str): The name of the task.
        split (str): The data split (e.g., 'train', 'val').
        year (str): The year of the dataset.

    Returns:
        str: The path to the task file.
    """
    return f"{task_name}_{split}{year}.json"


def create_empty_coco_object_tensor(
    n_coco_classes: int,
    n_objects: int = 1,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
    pin_memory: bool = False,
    non_blocking: bool = False,
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
            pin_memory=pin_memory,
            non_blocking=non_blocking,
        )


def object_to_tensor(
    object_: custom_yolo_lib.dataset.object.Object,
    n_coco_classes: int = 80,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
    pin_memory: bool = False,
    non_blocking: bool = False,
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
        pin_memory=pin_memory,
        non_blocking=non_blocking,
    )

    x[0] = object_.bbox.x
    x[1] = object_.bbox.y
    x[2] = object_.bbox.w
    x[3] = object_.bbox.h
    x[4] = 1.0  # objectness
    x[5 + object_.class_id] = 1.0  # class_id
    return x
