import math
import sys
import time

import torch
import torchvision.models.detection.mask_rcnn
from faster_rcnn.model.library_model_functions import utils
from faster_rcnn.model.library_model_functions.coco_eval import CocoEvaluator
from faster_rcnn.model.library_model_functions.coco_utils import get_coco_api_from_dataset

from config import HEIGHT, WIDTH

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    total = 0
    pose_correct = 0
    orientation_correct = 0
    combined_correct = 0

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        autocast_device = "cuda" if device.type == "cuda" else "cpu"
        with torch.amp.autocast(autocast_device, enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        with torch.no_grad():
            was_training = model.training
            if was_training:
                model.eval()
            outputs = model(images)
            if was_training:
                model.train()

        for output, target in zip(outputs, targets):
            if len(output.get("scores", [])) == 0:
                total += 1
                continue

            best = output["scores"].argmax()
            pred_center = output["centers"][best]
            pred_orientation = output["orientations"][best].item()

            gt_center = target["centers"].to(pred_center.device)
            gt_orientation = target["orientations"].item()

            center_error = torch.norm(torch.tensor([
                (pred_center[0] - gt_center[0]) * WIDTH,
                (pred_center[1] - gt_center[1]) * HEIGHT,
            ], device=pred_center.device))

            pose_is_correct = center_error <= 10
            orientation_is_correct = pred_orientation == gt_orientation
            combined_is_correct = pose_is_correct and orientation_is_correct

            total += 1
            pose_correct += int(pose_is_correct)
            orientation_correct += int(orientation_is_correct)
            combined_correct += int(combined_is_correct)

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()


        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger, {
        "accuracy": combined_correct / total if total else 0.0,
        "pose_accuracy": pose_correct / total if total else 0.0,
        "orientation_accuracy": orientation_correct / total if total else 0.0,
    }


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.inference_mode()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"]: output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator
