import pathlib
import argparse

import cv2
import torch
import tqdm
import pandas as pd

import custom_yolo_lib.dataset.coco.tasks.instances
import custom_yolo_lib.dataset.coco.tasks.loader
import custom_yolo_lib.experiments_utils
import custom_yolo_lib.image_size
import custom_yolo_lib.model.e2e.anchor_based.bundled_anchor_based
import custom_yolo_lib.model.e2e.anchor_based.loss
import custom_yolo_lib.model.e2e.anchor_based.training_utils
import custom_yolo_lib.training.utils
import custom_yolo_lib.training.lr_scheduler

torch.manual_seed(42)

BASE_LR = 0.01 / 64
EXPERIMENT_NAME = "debug"
WARMUP_EPOCHS = 3
EPOCHS = 12
NUM_CLASSES = 80
BATCH_SIZE = 12
LR = BASE_LR * BATCH_SIZE
MOMENTUM = 0.9
DECAY = 0.001
IMAGE_SIZE = custom_yolo_lib.image_size.ImageSize(640, 640)
CLASS_LOSS_GAIN = 1.0
OBJECTNESS_LOSS_GAIN = 1.0
BOX_LOSS_GAIN = 1.0
OBJECTNESS_LOSS_SMALL_MAP_GAIN = 4.0  # bigger grid 80x80 results in smaller loss if BCE
OBJECTNESS_LOSS_MEDIUM_MAP_GAIN = 1.0
OBJECTNESS_LOSS_LARGE_MAP_GAIN = 0.4
# torch.set_anomaly_enabled(True)


def init_model(
    device: torch.device,
) -> custom_yolo_lib.model.e2e.anchor_based.bundled_anchor_based.YOLOModel:
    model = custom_yolo_lib.model.e2e.anchor_based.bundled_anchor_based.YOLOModel(
        num_classes=NUM_CLASSES, training=True
    )
    model.to(device)
    return model


def init_optimizer(
    model: custom_yolo_lib.model.e2e.anchor_based.bundled_anchor_based.YOLOModel,
) -> torch.optim.Optimizer:
    parameters_grouped = custom_yolo_lib.training.utils.get_params_grouped(model)
    optimizer = torch.optim.AdamW(
        parameters_grouped.with_weight_decay, lr=LR, betas=(MOMENTUM, 0.999), weight_decay=DECAY
    )
    optimizer.add_param_group(
        {"params": parameters_grouped.bias, "weight_decay": DECAY}
    )
    optimizer.add_param_group(
        {"params": parameters_grouped.no_weight_decay, "weight_decay": 0.0}
    )
    return optimizer


def init_losses(
    model: custom_yolo_lib.model.e2e.anchor_based.bundled_anchor_based.YOLOModel,
    device: torch.device,
):
    predictions_s, predictions_m, predictions_l = model.train_forward2(
        torch.zeros((1, 3, IMAGE_SIZE.height, IMAGE_SIZE.width)).to(device)
    )

    small_map_anchors, medium_map_anchors, large_map_anchors = (
        custom_yolo_lib.model.e2e.anchor_based.training_utils.get_anchors_as_bbox_tensors(
            device
        )
    )
    loss_s = custom_yolo_lib.model.e2e.anchor_based.loss.YOLOLossPerFeatureMapV2(
        num_classes=NUM_CLASSES,
        feature_map_anchors=small_map_anchors,
        grid_size_h=predictions_s.shape[3],
        grid_size_w=predictions_s.shape[4],
    )
    loss_m = custom_yolo_lib.model.e2e.anchor_based.loss.YOLOLossPerFeatureMapV2(
        num_classes=NUM_CLASSES,
        feature_map_anchors=medium_map_anchors,
        grid_size_h=predictions_m.shape[3],
        grid_size_w=predictions_m.shape[4],
    )
    loss_l = custom_yolo_lib.model.e2e.anchor_based.loss.YOLOLossPerFeatureMapV2(
        num_classes=NUM_CLASSES,
        feature_map_anchors=large_map_anchors,
        grid_size_h=predictions_l.shape[3],
        grid_size_w=predictions_l.shape[4],
    )
    return loss_s, loss_m, loss_l


def init_dataloaders(dataset_path: pathlib.Path):
    classes = [i for i in range(NUM_CLASSES)]
    train_dataset = custom_yolo_lib.dataset.coco.tasks.instances.COCOInstances2017(
        dataset_path,
        "train",
        expected_image_size=IMAGE_SIZE,
        classes=classes,
        is_sama=True,
    )
    training_loader = custom_yolo_lib.dataset.coco.tasks.loader.COCODataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_dataset = custom_yolo_lib.dataset.coco.tasks.instances.COCOInstances2017(
        dataset_path,
        "val",
        expected_image_size=IMAGE_SIZE,
        classes=classes,
        is_sama=True,
    )
    validation_loader = custom_yolo_lib.dataset.coco.tasks.loader.COCODataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    return training_loader, validation_loader
import numpy as np

np.printoptions(threshold=np.inf)

def infe_one_batch(
    model: custom_yolo_lib.model.e2e.anchor_based.bundled_anchor_based.YOLOModel,
    training_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: custom_yolo_lib.training.lr_scheduler.WarmupCosineScheduler,
    loss_s: custom_yolo_lib.model.e2e.anchor_based.loss.YOLOLossPerFeatureMapV2,
    loss_m: custom_yolo_lib.model.e2e.anchor_based.loss.YOLOLossPerFeatureMapV2,
    loss_l: custom_yolo_lib.model.e2e.anchor_based.loss.YOLOLossPerFeatureMapV2,
    epoch: int,
    training_step: int,
    experiment_path: pathlib.Path,
    device: torch.device = torch.device("cuda:0"),
):
    # TRAINING
    training_session_data = {
        "bbox_loss_avg_featmap": [],
        "objectness_loss_avg_featmap": [],
        "class_loss_avg_featmap": [],
        "total_loss_avg_featmap": [],
        "epoch": [],
        "step": [],
    }
    tqdm_obj = tqdm.tqdm(training_loader)
    model.train()
    for i, coco_batch in enumerate(training_loader):

        images = coco_batch.images_batch.to(device)
        targets = [t.to(device) for t in coco_batch.objects_batch]

        predictions_s, predictions_m, predictions_l = model.train_forward2(images)

        _, losses_s, targets_in_grid_s = loss_s.debug_forward(predictions_s, targets)  #
        _, losses_m, targets_in_grid_m = loss_m.debug_forward(predictions_m, targets)
        _, losses_l, targets_in_grid_l = loss_l.debug_forward(predictions_l, targets)
        break

    for i, (image, target, target_in_grid_s, target_in_grid_m, target_in_grid_l) in enumerate(zip(
        images, targets, targets_in_grid_s, targets_in_grid_m, targets_in_grid_l
    )):
        # create sample_i subdir
        sample_dir = experiment_path / f"sample_{i}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        # convert image to opencv format
        image = image.permute(1, 2, 0).cpu().numpy() * 255.0
        image = image.astype("uint8")
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(
            str(sample_dir / f"sample_{i}.jpg"), image
        )
        # write targets to txt file
        with open(str(sample_dir / f"sample_{i}.txt"), "w") as f:
            for obj in target:
                for item in obj:
                    f.write(f"{item.item()}, ")
                f.write("\n")

        # draw targets on image
        for obj in target:
            xc, yc, w, h, class_id = obj
            xc = int(xc.item()* IMAGE_SIZE.width)
            yc = int(yc.item() * IMAGE_SIZE.height)
            w = int(w.item()* IMAGE_SIZE.width)
            h = int(h.item() * IMAGE_SIZE.height)
            x1 = int((xc - w / 2) )
            y1 = int((yc - h / 2))
            x2 = int((xc + w / 2) )
            y2 = int((yc + h / 2))
            class_id = int(class_id.item())
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image,
                str(class_id),
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
                )
        cv2.imwrite(
            str(sample_dir / f"sample_{i}_debug.jpg"), image
        )
        
        # write targets_in_grid_s to... 
        # target_in_grid_s is a tensor of shape (num_anchors, num_classes + 5, grid_h, grid_w)
        with open(str(sample_dir / f"sample_{i}_in_grid_s.txt"), "w") as f:
            f.write(f"target_in_grid_s shape: {target_in_grid_s.shape}\n")
            for i, anchor in enumerate(target_in_grid_s):
                f.write(f"============== ANCHOR {i} ==============\n")
                for grid_y in range(anchor.shape[0]):
                    for grid_x in range(anchor.shape[1]):
                        f.write(f"grid_y: {grid_y}, grid_x: {grid_x}\n")
                        f.write(f"{anchor[grid_y, grid_x, :]}\n")
                        f.write("--------------------------------------------\n")
        # write targets_in_grid_m to...
        with open(str(sample_dir / f"sample_{i}_in_grid_m.txt"), "w") as f:
            f.write(f"target_in_grid_m shape: {target_in_grid_m.shape}\n")
            for k, anchor in enumerate(target_in_grid_m):
                f.write(f"============== ANCHOR {k} ==============\n")
                for grid_y in range(anchor.shape[0]):
                    for grid_x in range(anchor.shape[1]):
                        f.write(f"grid_y: {grid_y}, grid_x: {grid_x}\n")
                        f.write(f"{anchor[grid_y, grid_x, :]}\n")
                        f.write("--------------------------------------------\n")
        # write targets_in_grid_l to...
        with open(str(sample_dir / f"sample_{i}_in_grid_l.txt"), "w") as f:
            f.write(f"target_in_grid_l shape: {target_in_grid_l.shape}\n")
            for k, anchor in enumerate(target_in_grid_l):
                f.write(f"============== ANCHOR {k} ==============\n")
                for grid_y in range(anchor.shape[0]):
                    for grid_x in range(anchor.shape[1]):
                        f.write(f"grid_y: {grid_y}, grid_x: {grid_x}\n")
                        f.write(f"{anchor[grid_y, grid_x, :]}\n")
                        f.write("--------------------------------------------\n")
        pass
    return training_step


def session_loop(
    model: custom_yolo_lib.model.e2e.anchor_based.bundled_anchor_based.YOLOModel,
    training_loader: torch.utils.data.DataLoader,
    validation_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: custom_yolo_lib.training.lr_scheduler.StepLRScheduler,
    loss_s: custom_yolo_lib.model.e2e.anchor_based.loss.YOLOLossPerFeatureMapV2,
    loss_m: custom_yolo_lib.model.e2e.anchor_based.loss.YOLOLossPerFeatureMapV2,
    loss_l: custom_yolo_lib.model.e2e.anchor_based.loss.YOLOLossPerFeatureMapV2,
    experiment_path: pathlib.Path,
    device: torch.device = torch.device("cuda:0"),
):
    # Training loop
    training_step = 0
    validation_step = 0
    min_mean_val_loss = float("inf")

    training_step = infe_one_batch(
        model,
        training_loader,
        optimizer,
        scheduler,
        loss_s,
        loss_m,
        loss_l,
        0,
        training_step,
        experiment_path,
        device,
    )



def main(dataset_path: pathlib.Path, experiment_path: pathlib.Path):
    experiment_path = custom_yolo_lib.experiments_utils.make_experiment_dir(
        EXPERIMENT_NAME, experiment_path
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Why bother bro?.")
    device = torch.device("cpu")

    model = init_model(device)
    optimizer = init_optimizer(model)
    training_loader, validation_loader = init_dataloaders(dataset_path)
    steps_per_epoch = len(training_loader)

    scheduler = custom_yolo_lib.training.lr_scheduler.WarmupCosineScheduler(
        optimizer,
        warmup_steps=steps_per_epoch * WARMUP_EPOCHS,
        max_steps=steps_per_epoch * EPOCHS,
    )

    # loss_s, loss_m, loss_l = init_losses(device)
    loss_s, loss_m, loss_l = init_losses(model, device)

    session_loop(
        model,
        training_loader,
        validation_loader,
        optimizer,
        scheduler,
        loss_s,
        loss_m,
        loss_l,
        experiment_path,
        device,
    )


def parse_args():

    parser = argparse.ArgumentParser(description="Train YOLOv5 model")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/home/manos/custom-yolo/coco_data",
        help="Path to the dataset directory",
    )
    parser.add_argument(
        "--experiment_path",
        type=str,
        # required=True,
        default="/home/manos/custom-yolo/experiments",
        help="Path to the experiment directory",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dataset_path = pathlib.Path(args.dataset_path)
    experiment_path = pathlib.Path(args.experiment_path)
    main(dataset_path, experiment_path)
