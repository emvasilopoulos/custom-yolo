import pathlib
from typing import Union

import pandas as pd
import torch
import tqdm

import custom_yolo_lib.model.e2e.anchor_based.bundled_anchor_based
import custom_yolo_lib.model.e2e.anchor_based.losses.loss as custom_loss
import custom_yolo_lib.model.e2e.anchor_based.losses.loss_v3 as custom_loss_v3
import custom_yolo_lib.training.lr_scheduler
import custom_yolo_lib.config.hyperparameters
import custom_yolo_lib.experiments.loss_factory
import custom_yolo_lib.experiments.dataloaders_factory
import custom_yolo_lib.experiments.model_factory
import custom_yolo_lib.experiments.optimizer_factory
import custom_yolo_lib.experiments.schedulers_factory
import custom_yolo_lib.experiments.utils


def main(
    dataset_path: pathlib.Path,
    experiment_path: pathlib.Path,
    hyperparameters: custom_yolo_lib.config.hyperparameters.ThreeAnchorsHyperparameters,
):

    experiment_path = custom_yolo_lib.experiments.utils.make_experiment_dir(
        hyperparameters.experiment_name, experiment_path
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Why bother bro?.")
    device = torch.device("cuda:0")

    model = custom_yolo_lib.experiments.model_factory.init_model(
        model_type=hyperparameters.model_type,
        device=device,
        num_classes=hyperparameters.dataset_num_classes,
    )

    optimizer = custom_yolo_lib.experiments.optimizer_factory.init_optimizer(
        hyperparameters.optimizer_type,
        model,
        initial_lr=hyperparameters.learning_rate,
        momentum=hyperparameters.momentum,
        weight_decay=hyperparameters.weight_decay,
    )
    training_loader, validation_loader = (
        custom_yolo_lib.experiments.dataloaders_factory.init_dataloaders(
            hyperparameters.dataset_type,
            dataset_path,
            num_classes=hyperparameters.dataset_num_classes,
            image_size=hyperparameters.image_size,
            batch_size=hyperparameters.batch_size,
        )
    )
    steps_per_epoch = len(training_loader)

    scheduler = custom_yolo_lib.experiments.schedulers_factory.init_scheduler(
        hyperparameters.scheduler_type,
        optimizer,
        update_step_size=steps_per_epoch,
        warmup_steps=steps_per_epoch * hyperparameters.warmup_epochs,
        max_steps=steps_per_epoch * hyperparameters.epochs,
        cycles=0.5,  # TODO: add in hyperparameters config
        min_factor=0.1,  # TODO: add in hyperparameters config
        last_step=-1,  # TODO: learn what this is
    )
    loss_s, loss_m, loss_l = custom_yolo_lib.experiments.loss_factory.init_loss(
        hyperparameters.loss_type,
        model,
        device,
        expected_image_size=hyperparameters.image_size,
        num_classes=hyperparameters.dataset_num_classes,
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
        hyperparameters,
        device,
    )


def train_one_epoch(
    model: custom_yolo_lib.model.e2e.anchor_based.bundled_anchor_based.YOLOModel,
    training_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: custom_yolo_lib.training.lr_scheduler.WarmupCosineScheduler,
    loss_s: Union[
        custom_loss.YOLOLossPerFeatureMapV2, custom_loss_v3.YOLOLossPerFeatureMapV3
    ],
    loss_m: Union[
        custom_loss.YOLOLossPerFeatureMapV2, custom_loss_v3.YOLOLossPerFeatureMapV3
    ],
    loss_l: Union[
        custom_loss.YOLOLossPerFeatureMapV2, custom_loss_v3.YOLOLossPerFeatureMapV3
    ],
    epoch: int,
    training_step: int,
    experiment_path: pathlib.Path,
    hyperparameters: custom_yolo_lib.config.hyperparameters.ThreeAnchorsHyperparameters,
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
    print(f"Training epoch {epoch + 1}/{hyperparameters.epochs} | Losses:")
    for i, coco_batch in enumerate(tqdm_obj):

        images = coco_batch.images_batch.to(device)
        targets = [t.to(device) for t in coco_batch.objects_batch]

        predictions_s, predictions_m, predictions_l = model.train_forward2(images)

        _, losses_s = loss_s(predictions_s, targets)
        _, losses_m = loss_m(predictions_m, targets)
        _, losses_l = loss_l(predictions_l, targets)
        (avg_bbox_loss, avg_objectness_loss, avg_class_loss), loss = (
            custom_yolo_lib.experiments.loss_factory.calculate_three_scale_loss(
                losses_s,
                losses_m,
                losses_l,
                hyperparameters.loss_type,
                hyperparameters.box_loss_gain,
                hyperparameters.objectness_loss_gain,
                hyperparameters.objectness_loss_small_map_gain,
                hyperparameters.objectness_loss_medium_map_gain,
                hyperparameters.objectness_loss_large_map_gain,
                hyperparameters.class_loss_gain,
            )
        )
        if torch.isnan(avg_bbox_loss):
            print("avg_bbox_loss is NaN, EXITING...")
            exit(1)
        if torch.isnan(avg_objectness_loss):
            print("avg_objectness_loss is NaN, EXITING...")
            exit(1)
        if torch.isnan(avg_class_loss):
            print("avg_class_loss is NaN, EXITING...")
            exit(1)
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
    model: custom_yolo_lib.model.e2e.anchor_based.bundled_anchor_based.YOLOModel,
    validation_loader: torch.utils.data.DataLoader,
    loss_s: Union[
        custom_loss.YOLOLossPerFeatureMapV2, custom_loss_v3.YOLOLossPerFeatureMapV3
    ],
    loss_m: Union[
        custom_loss.YOLOLossPerFeatureMapV2, custom_loss_v3.YOLOLossPerFeatureMapV3
    ],
    loss_l: Union[
        custom_loss.YOLOLossPerFeatureMapV2, custom_loss_v3.YOLOLossPerFeatureMapV3
    ],
    epoch: int,
    validation_step: int,
    experiment_path: pathlib.Path,
    hyperparameters: custom_yolo_lib.config.hyperparameters.ThreeAnchorsHyperparameters,
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
    print(f"Validation epoch {epoch + 1}/{hyperparameters.epochs} | Losses:")
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
            (avg_bbox_loss, avg_objectness_loss, avg_class_loss), loss = (
                custom_yolo_lib.experiments.loss_factory.calculate_three_scale_loss(
                    losses_s,
                    losses_m,
                    losses_l,
                    hyperparameters.loss_type,
                    hyperparameters.box_loss_gain,
                    hyperparameters.objectness_loss_gain,
                    hyperparameters.objectness_loss_small_map_gain,
                    hyperparameters.objectness_loss_medium_map_gain,
                    hyperparameters.objectness_loss_large_map_gain,
                    hyperparameters.class_loss_gain,
                )
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
    model: custom_yolo_lib.model.e2e.anchor_based.bundled_anchor_based.YOLOModel,
    training_loader: torch.utils.data.DataLoader,
    validation_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: custom_yolo_lib.training.lr_scheduler.StepLRScheduler,
    loss_s: Union[
        custom_loss.YOLOLossPerFeatureMapV2, custom_loss_v3.YOLOLossPerFeatureMapV3
    ],
    loss_m: Union[
        custom_loss.YOLOLossPerFeatureMapV2, custom_loss_v3.YOLOLossPerFeatureMapV3
    ],
    loss_l: Union[
        custom_loss.YOLOLossPerFeatureMapV2, custom_loss_v3.YOLOLossPerFeatureMapV3
    ],
    experiment_path: pathlib.Path,
    hyperparameters: custom_yolo_lib.config.hyperparameters.ThreeAnchorsHyperparameters,
    device: torch.device = torch.device("cuda:0"),
):
    # Training loop
    training_step = 0
    validation_step = 0
    min_mean_val_loss = float("inf")

    for epoch in range(hyperparameters.epochs):

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
            hyperparameters,
            device,
        )
        print(f"CURRENT LEARNING RATES: {scheduler.get_lr()}")

        validation_step, mean_val_loss = validate_one_epoch(
            model,
            validation_loader,
            loss_s,
            loss_m,
            loss_l,
            epoch,
            validation_step,
            experiment_path,
            hyperparameters,
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
