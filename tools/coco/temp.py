import pathlib

import cv2
import numpy as np
import torch

import custom_yolo_lib.dataset.coco.tasks.instances
import custom_yolo_lib.dataset.coco.downloader
import custom_yolo_lib.image_size

def torch_tensor_to_cv2(img_tensor) -> np.ndarray:
    img_tensor = img_tensor * 255
    img = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

if __name__ == "__main__":
    dataset_dir = pathlib.Path("coco_data/")
    coco_dataset = custom_yolo_lib.dataset.coco.tasks.instances.COCOInstances2017(
        data_dir=dataset_dir,
        split="val",
        expected_image_size=custom_yolo_lib.image_size.ImageSize(640, 640),
        classes=None, # All COCO classes
        device=torch.device("cpu"),
    )

    idx = 3
    img_tensor, objects = coco_dataset[idx]
    img = torch_tensor_to_cv2(img_tensor)
    for obj in objects:
        if obj[0] == 0 and obj[1] == 0 and obj[2] == 0 and obj[3] == 0:
            break
        x1 = obj[0] * img.shape[1]
        y1 = obj[1] * img.shape[0]
        w = obj[2] * img.shape[1]
        h = obj[3] * img.shape[0]
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), (0, 255, 0), 2)
        class_id = torch.argmax(obj[5:]).item()
        cv2.putText(img, str(class_id), (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 30, 220), 2)
    cv2.imwrite("output.jpg", img)