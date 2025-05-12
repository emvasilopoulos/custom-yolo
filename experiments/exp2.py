"""
In the end the model failed because of a mistake in the loss function.
Either way considering the experiment a failure.
"""

import pathlib
import argparse

import torch
import tqdm
import pandas as pd

import custom_yolo_lib.experiments.dataloaders_factory
import custom_yolo_lib.experiments.optimizer_factory
import custom_yolo_lib.experiments.schedulers_factory
import custom_yolo_lib.experiments.loss_factory
import custom_yolo_lib.experiments.utils
import custom_yolo_lib.image_size
import custom_yolo_lib.model.e2e.anchor_based.bundled_anchor_based
import custom_yolo_lib.model.e2e.anchor_based.losses.loss
import custom_yolo_lib.training.losses
import custom_yolo_lib.training.lr_scheduler
import custom_yolo_lib.experiments.model_factory

torch.manual_seed(42)

MODEL_TYPE = custom_yolo_lib.experiments.model_factory.ModelType.YOLO
OPTIMIZER_TYPE = (
    custom_yolo_lib.experiments.optimizer_factory.OptimizerType.SPLIT_GROUPS_ADAMW
)
LOSS_TYPE = custom_yolo_lib.experiments.loss_factory.LossType.THREESCALE_YOLO
DATASET_TYPE = custom_yolo_lib.experiments.dataloaders_factory.DatasetType.COCO_SAMA
SCHEDULER_TYPE = (
    custom_yolo_lib.experiments.schedulers_factory.SchedulerType.WARMUP_COSINE
)
BASE_LR = 0.01 / 64
EXPERIMENT_NAME = "exp2"
WARMUP_EPOCHS = 3
EPOCHS = 12
NUM_CLASSES = 80
BATCH_SIZE = 16
LR = BASE_LR * BATCH_SIZE
MOMENTUM = 0.9
DECAY = 5e-4
IMAGE_SIZE = custom_yolo_lib.image_size.ImageSize(640, 640)
CLASS_LOSS_GAIN = 2.0
OBJECTNESS_LOSS_GAIN = 0.65
BOX_LOSS_GAIN = 1.0
OBJECTNESS_LOSS_SMALL_MAP_GAIN = 4.0  # bigger grid 80x80 results in smaller loss if BCE
OBJECTNESS_LOSS_MEDIUM_MAP_GAIN = 1.0
OBJECTNESS_LOSS_LARGE_MAP_GAIN = 0.4


def calculate_loss(
    losses_s: torch.Tensor, losses_m: torch.Tensor, losses_l: torch.Tensor
):
    avg_bbox_loss = (losses_s[0] + losses_m[0] + losses_l[0]) * BOX_LOSS_GAIN
    avg_objectness_loss = (
        losses_s[1] * OBJECTNESS_LOSS_SMALL_MAP_GAIN
        + losses_m[1] * OBJECTNESS_LOSS_MEDIUM_MAP_GAIN
        + losses_l[1] * OBJECTNESS_LOSS_LARGE_MAP_GAIN
    ) * OBJECTNESS_LOSS_GAIN
    avg_class_loss = (losses_s[2] + losses_m[2] + losses_l[2]) * CLASS_LOSS_GAIN
    loss = avg_bbox_loss + avg_objectness_loss + avg_class_loss
    return (avg_bbox_loss, avg_objectness_loss, avg_class_loss), loss


def train_one_epoch(
    model: custom_yolo_lib.model.e2e.anchor_based.bundled_anchor_based.YOLOModel_FAILURE,
    training_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: custom_yolo_lib.training.lr_scheduler.WarmupCosineScheduler,
    loss_s: custom_yolo_lib.model.e2e.anchor_based.losses.loss.YOLOLossPerFeatureMapV2,
    loss_m: custom_yolo_lib.model.e2e.anchor_based.losses.loss.YOLOLossPerFeatureMapV2,
    loss_l: custom_yolo_lib.model.e2e.anchor_based.losses.loss.YOLOLossPerFeatureMapV2,
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
    print(f"Training epoch {epoch + 1}/{EPOCHS} | Losses:")
    for i, coco_batch in enumerate(tqdm_obj):

        images = coco_batch.images_batch.to(device)
        targets = [t.to(device) for t in coco_batch.objects_batch]

        predictions_s, predictions_m, predictions_l = model.train_forward2(images)

        _, losses_s = loss_s(predictions_s, targets)
        _, losses_m = loss_m(predictions_m, targets)
        _, losses_l = loss_l(predictions_l, targets)
        (avg_bbox_loss, avg_objectness_loss, avg_class_loss), loss = calculate_loss(
            losses_s, losses_m, losses_l
        )
        if torch.isnan(avg_bbox_loss):
            print("avg_bbox_loss is NaN, skipping step")
            continue
        if torch.isnan(avg_objectness_loss):
            print("avg_objectness_loss is NaN, skipping step")
            continue
        if torch.isnan(avg_class_loss):
            print("avg_class_loss is NaN, skipping step")
            continue
        if loss == 0:
            print("Loss is zero, skipping step")
            continue
        if not torch.isnan(loss):
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            tqdm_obj.set_description(
                f"Total: {loss.item():.4f} | BBox: {avg_bbox_loss.item():.4f} | Obj: {avg_objectness_loss.item():.4f} | Class: {avg_class_loss.item():.4f}"
            )
            training_session_data["bbox_loss_avg_featmap"].append(avg_bbox_loss.item())
            training_session_data["objectness_loss_avg_featmap"].append(
                avg_objectness_loss.item()
            )
            training_session_data["class_loss_avg_featmap"].append(
                avg_class_loss.item()
            )
            training_session_data["total_loss_avg_featmap"].append(loss.item())
            training_session_data["epoch"].append(epoch)
            training_session_data["step"].append(training_step)
            for i, LR in enumerate(scheduler.get_lr()):
                if f"lr-{i}" not in training_session_data:
                    training_session_data[f"lr-{i}"] = []
                training_session_data[f"lr-{i}"].append(LR)
        else:
            print("Loss is NaN, skipping step")
        training_step += 1

    train_data_path = experiment_path / f"training_session_data_epoch_{epoch}.csv"
    pd.DataFrame(training_session_data).to_csv(train_data_path.as_posix(), index=False)
    return training_step


def validate_one_epoch(
    model: custom_yolo_lib.model.e2e.anchor_based.bundled_anchor_based.YOLOModel_FAILURE,
    validation_loader: torch.utils.data.DataLoader,
    loss_s: custom_yolo_lib.model.e2e.anchor_based.losses.loss.YOLOLossPerFeatureMapV2,
    loss_m: custom_yolo_lib.model.e2e.anchor_based.losses.loss.YOLOLossPerFeatureMapV2,
    loss_l: custom_yolo_lib.model.e2e.anchor_based.losses.loss.YOLOLossPerFeatureMapV2,
    epoch: int,
    validation_step: int,
    experiment_path: pathlib.Path,
    device: torch.device = torch.device("cuda:0"),
):
    validation_session_data = {
        "bbox_loss_avg_featmap": [],
        "objectness_loss_avg_featmap": [],
        "class_loss_avg_featmap": [],
        "total_loss_avg_featmap": [],
        "epoch": [],
        "step": [],
    }
    tqdm_obj = tqdm.tqdm(validation_loader)
    model.eval()
    model.training = True
    print(f"Validation epoch {epoch + 1}/{EPOCHS} | Losses:")
    with torch.no_grad():
        for i, coco_batch in enumerate(tqdm_obj):
            """
            targets.shape = (batch_size, MAX_ALLOWED_OBJECTS, 5 + num_classes) | 5-->(x, y, w, h, objectness)
            targets_n_objects.shape = (batch_size, 1)
            """

            images = coco_batch.images_batch.to(device)
            targets = [t.to(device) for t in coco_batch.objects_batch]
            predictions_s, predictions_m, predictions_l = model.train_forward2(images)

            _, losses_s = loss_s(predictions_s, targets)
            _, losses_m = loss_m(predictions_m, targets)
            _, losses_l = loss_l(predictions_l, targets)
            (avg_bbox_loss, avg_objectness_loss, avg_class_loss), loss = calculate_loss(
                losses_s, losses_m, losses_l
            )

            tqdm_obj.set_description(
                f"Total: {loss.item():.4f} | BBox: {avg_bbox_loss.item():.4f} | Obj: {avg_objectness_loss.item():.4f} | Class: {avg_class_loss.item():.4f}"
            )
            validation_session_data["bbox_loss_avg_featmap"].append(
                avg_bbox_loss.item()
            )
            validation_session_data["objectness_loss_avg_featmap"].append(
                avg_objectness_loss.item()
            )
            validation_session_data["class_loss_avg_featmap"].append(
                avg_class_loss.item()
            )
            validation_session_data["total_loss_avg_featmap"].append(loss.item())
            validation_session_data["epoch"].append(epoch)
            validation_session_data["step"].append(validation_step)
            validation_step += 1
    # Store validation data
    validation_data_path = (
        experiment_path / f"validation_session_data_epoch_{epoch}.csv"
    )
    df = pd.DataFrame(validation_session_data)
    df.to_csv(validation_data_path.as_posix(), index=False)
    return validation_step, df["total_loss_avg_featmap"].mean()


def session_loop(
    model: custom_yolo_lib.model.e2e.anchor_based.bundled_anchor_based.YOLOModel_FAILURE,
    training_loader: torch.utils.data.DataLoader,
    validation_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: custom_yolo_lib.training.lr_scheduler.StepLRScheduler,
    loss_s: custom_yolo_lib.model.e2e.anchor_based.losses.loss.YOLOLossPerFeatureMapV2,
    loss_m: custom_yolo_lib.model.e2e.anchor_based.losses.loss.YOLOLossPerFeatureMapV2,
    loss_l: custom_yolo_lib.model.e2e.anchor_based.losses.loss.YOLOLossPerFeatureMapV2,
    experiment_path: pathlib.Path,
    device: torch.device = torch.device("cuda:0"),
):
    # Training loop
    training_step = 0
    validation_step = 0
    min_mean_val_loss = float("inf")
    for epoch in range(EPOCHS):

        training_step = train_one_epoch(
            model,
            training_loader,
            optimizer,
            scheduler,
            loss_s,
            loss_m,
            loss_l,
            epoch,
            training_step,
            experiment_path,
            device,
        )

        validation_step, mean_val_loss = validate_one_epoch(
            model,
            validation_loader,
            loss_s,
            loss_m,
            loss_l,
            epoch,
            validation_step,
            experiment_path,
            device,
        )

        # Store training data
        model_state = model.state_dict()
        if mean_val_loss < min_mean_val_loss:
            min_mean_val_loss = mean_val_loss
            print(f"Saving model with mean validation loss: {mean_val_loss:.4f}")
            model_path = experiment_path / f"model_best.pth"
            torch.save(model_state, model_path.as_posix())
        model_path = experiment_path / f"model_last.pth"
        torch.save(model_state, model_path.as_posix())

        if epoch < 3:
            loss_s.set_box_loss(custom_yolo_lib.training.losses.BoxLoss.IoUType.IoU)
            loss_m.set_box_loss(custom_yolo_lib.training.losses.BoxLoss.IoUType.IoU)
            loss_l.set_box_loss(custom_yolo_lib.training.losses.BoxLoss.IoUType.IoU)
        elif epoch < 6:
            loss_s.set_box_loss(custom_yolo_lib.training.losses.BoxLoss.IoUType.GIoU)
            loss_m.set_box_loss(custom_yolo_lib.training.losses.BoxLoss.IoUType.GIoU)
            loss_l.set_box_loss(custom_yolo_lib.training.losses.BoxLoss.IoUType.GIoU)
        elif epoch < 9:
            loss_s.set_box_loss(custom_yolo_lib.training.losses.BoxLoss.IoUType.DIoU)
            loss_m.set_box_loss(custom_yolo_lib.training.losses.BoxLoss.IoUType.DIoU)
            loss_l.set_box_loss(custom_yolo_lib.training.losses.BoxLoss.IoUType.DIoU)
        else:
            loss_s.set_box_loss(custom_yolo_lib.training.losses.BoxLoss.IoUType.CIoU)
            loss_m.set_box_loss(custom_yolo_lib.training.losses.BoxLoss.IoUType.CIoU)
            loss_l.set_box_loss(custom_yolo_lib.training.losses.BoxLoss.IoUType.CIoU)


def main(dataset_path: pathlib.Path, experiment_path: pathlib.Path):
    experiment_path = custom_yolo_lib.experiments.utils.make_experiment_dir(
        EXPERIMENT_NAME, experiment_path
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Why bother bro?.")
    device = torch.device("cuda:0")

    model = custom_yolo_lib.experiments.model_factory.init_model(
        model_type=MODEL_TYPE, device=device, num_classes=NUM_CLASSES
    )

    optimizer = custom_yolo_lib.experiments.optimizer_factory.init_optimizer(
        OPTIMIZER_TYPE,
        model,
        initial_lr=LR,
        momentum=MOMENTUM,
        weight_decay=DECAY,
    )
    training_loader, validation_loader = (
        custom_yolo_lib.experiments.dataloaders_factory.init_dataloaders(
            DATASET_TYPE,
            dataset_path,
            num_classes=NUM_CLASSES,
            image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
        )
    )
    steps_per_epoch = len(training_loader)

    scheduler = custom_yolo_lib.experiments.schedulers_factory.init_scheduler(
        SCHEDULER_TYPE,
        optimizer,
        warmup_steps=steps_per_epoch * WARMUP_EPOCHS,
        max_steps=steps_per_epoch * EPOCHS,
    )

    # loss_s, loss_m, loss_l = init_losses(device)
    loss_type = custom_yolo_lib.experiments.loss_factory.LossType.THREESCALE_YOLO
    loss_s, loss_m, loss_l = custom_yolo_lib.experiments.loss_factory.init_loss(
        loss_type,
        model,
        device,
        expected_image_size=IMAGE_SIZE,
        num_classes=NUM_CLASSES,
    )
    loss_s.set_box_loss(custom_yolo_lib.training.losses.BoxLoss.IoUType.IoU)
    loss_m.set_box_loss(custom_yolo_lib.training.losses.BoxLoss.IoUType.IoU)
    loss_l.set_box_loss(custom_yolo_lib.training.losses.BoxLoss.IoUType.IoU)

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
