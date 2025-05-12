import pathlib
import argparse

import torch
import tqdm
import pandas as pd

import custom_yolo_lib.experiments.model_factory
import custom_yolo_lib.experiments.utils
import custom_yolo_lib.experiments.optimizer_factory
import custom_yolo_lib.experiments.schedulers_factory
import custom_yolo_lib.experiments.loss_factory
import custom_yolo_lib.experiments.dataloaders_factory
import custom_yolo_lib.image_size
import custom_yolo_lib.model.e2e.anchor_based.bundled_anchor_based
import custom_yolo_lib.model.e2e.anchor_based.loss
import custom_yolo_lib.training.lr_scheduler

torch.manual_seed(42)

MODEL_TYPE = custom_yolo_lib.experiments.model_factory.ModelType.YOLO
OPTIMIZER_TYPE = (
    custom_yolo_lib.experiments.optimizer_factory.OptimizerType.SPLIT_GROUPS_ADAMW
)
LOSS_TYPE = custom_yolo_lib.experiments.loss_factory.LossType.THREESCALE_YOLO
DATASET_TYPE = (
    custom_yolo_lib.experiments.dataloaders_factory.DatasetType.COCO_ORIGINAL_THREE_FEATURE_MAPS
)
SCHEDULER_TYPE = custom_yolo_lib.experiments.schedulers_factory.SchedulerType.STEP
EXPERIMENT_NAME = "exp1"
EPOCHS = 300
NUM_CLASSES = 80
LR = 0.001
MOMENTUM = 0.937
DECAY = 0.001
BATCH_SIZE = 8
IMAGE_SIZE = custom_yolo_lib.image_size.ImageSize(640, 640)


def train_one_epoch(
    model: custom_yolo_lib.model.e2e.anchor_based.bundled_anchor_based.YOLOModel,
    training_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: custom_yolo_lib.training.lr_scheduler.StepLRScheduler,
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
    print(f"Training epoch {epoch + 1}/{EPOCHS} | Losses:")
    for i, coco_batch in enumerate(tqdm_obj):

        images = coco_batch.images_batch.to(device)
        targets_s = [t.to(device) for t in coco_batch.small_objects_batch]
        targets_m = [t.to(device) for t in coco_batch.medium_objects_batch]
        targets_l = [t.to(device) for t in coco_batch.large_objects_batch]

        optimizer.zero_grad()

        predictions_s, predictions_m, predictions_l = model.train_forward2(images)

        total_loss_s, losses_s = loss_s(predictions_s, targets_s)
        total_loss_m, losses_m = loss_m(predictions_m, targets_m)
        total_loss_l, losses_l = loss_l(predictions_l, targets_l)
        loss = total_loss_s + total_loss_m + total_loss_l
        if loss == 0:
            print("Loss is zero, skipping step")
            continue
        loss.backward()

        scheduler.step(loss)
        optimizer.step()

        avg_bbox_loss = losses_s[0] + losses_m[0] + losses_l[0]
        avg_objectness_loss = losses_s[1] + losses_m[1] + losses_l[1]
        avg_class_loss = losses_s[2] + losses_m[2] + losses_l[2]
        tqdm_obj.set_description(
            f"Total: {loss.item():.4f} | BBox: {avg_bbox_loss.item():.4f} | Obj: {avg_objectness_loss.item():.4f} | Class: {avg_class_loss.item():.4f}"
        )
        training_session_data["bbox_loss_avg_featmap"].append(avg_bbox_loss.item())
        training_session_data["objectness_loss_avg_featmap"].append(
            avg_objectness_loss.item()
        )
        training_session_data["class_loss_avg_featmap"].append(avg_class_loss.item())
        training_session_data["total_loss_avg_featmap"].append(loss.item())
        training_session_data["epoch"].append(epoch)
        training_session_data["step"].append(training_step)
        for i, LR in enumerate(scheduler.get_lr()):
            if f"lr-{i}" not in training_session_data:
                training_session_data[f"lr-{i}"] = []
            training_session_data[f"lr-{i}"].append(LR)
        training_step += 1

    train_data_path = experiment_path / f"training_session_data_epoch_{epoch}.csv"
    pd.DataFrame(training_session_data).to_csv(train_data_path.as_posix(), index=False)
    return training_step


def validate_one_epoch(
    model: custom_yolo_lib.model.e2e.anchor_based.bundled_anchor_based.YOLOModel,
    validation_loader: torch.utils.data.DataLoader,
    loss_s: custom_yolo_lib.model.e2e.anchor_based.loss.YOLOLossPerFeatureMapV2,
    loss_m: custom_yolo_lib.model.e2e.anchor_based.loss.YOLOLossPerFeatureMapV2,
    loss_l: custom_yolo_lib.model.e2e.anchor_based.loss.YOLOLossPerFeatureMapV2,
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
            targets_s = [t.to(device) for t in coco_batch.small_objects_batch]
            targets_m = [t.to(device) for t in coco_batch.medium_objects_batch]
            targets_l = [t.to(device) for t in coco_batch.large_objects_batch]
            predictions_s, predictions_m, predictions_l = model.train_forward2(images)

            total_loss_s, losses_s = loss_s(predictions_s, targets_s)
            total_loss_m, losses_m = loss_m(predictions_m, targets_m)
            total_loss_l, losses_l = loss_l(predictions_l, targets_l)
            loss = total_loss_s + total_loss_m + total_loss_l

            avg_bbox_loss = losses_s[0] + losses_m[0] + losses_l[0]
            avg_objectness_loss = losses_s[1] + losses_m[1] + losses_l[1]
            avg_class_loss = losses_s[2] + losses_m[2] + losses_l[2]
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


def main(dataset_path: pathlib.Path, experiment_path: pathlib.Path):
    experiment_path = custom_yolo_lib.experiments.utils.make_experiment_dir(
        "exp1", experiment_path
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
    scheduler = custom_yolo_lib.experiments.schedulers_factory.init_scheduler(
        SCHEDULER_TYPE,
        optimizer,
        update_step_size=10000,
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

    loss_s, loss_m, loss_l = custom_yolo_lib.experiments.loss_factory.init_loss(
        LOSS_TYPE,
        model,
        device,
        expected_image_size=IMAGE_SIZE,
        num_classes=NUM_CLASSES,
    )

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
