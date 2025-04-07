import pathlib

import torch

import custom_yolo_lib.dataset.coco.tasks.instances
import custom_yolo_lib.dataset.coco.downloader

class TestCocoAnnotations:
    def test_coco_annotations(self):
        dataset_dir = pathlib.Path("coco_data")
        if not pathlib.Path("coco_data").exists():
            print("WARNING: DOWNLOADING COCO DATASET YOU NEED TO UNZIP AND RERUN")
            custom_yolo_lib.dataset.coco.downloader.download_val_images_2017(dataset_dir)
            custom_yolo_lib.dataset.coco.downloader.download_train_val_annotations_2017(dataset_dir)
            raise Exception("Please unzip the dataset.")

        coco_dataset = custom_yolo_lib.dataset.coco.tasks.instances.COCOInstances2017(
            data_dir=dataset_dir,
            split="val",
            expected_image_size=(640, 640),
            classes=None, # All COCO classes
            device=torch.device("cpu"),
        )
        raise NotImplementedError("Test not implemented yet")
    
if __name__ == "__main__":
    pass