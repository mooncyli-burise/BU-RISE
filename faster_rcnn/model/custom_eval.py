import torch
import time
from faster_rcnn.model.library_model_functions import utils
from config import TEST_SIZE, HEIGHT, WIDTH

def custom_eval(model, data_loader, device):
    model.roi_heads.score_thresh = 0.0
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    all_center_error = []
    all_orientation_error = []
    correct = 0
    total = 0
    pose_correct = 0
    orientation_correct = 0
    combined_correct = 0

    with torch.no_grad():
        for images, targets in metric_logger.log_every(data_loader, 10, header):
            images = list(img.to(device) for img in images)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            model_time = time.time()
            outputs = model(images)
            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
            model_time = time.time() - model_time

            evaluator_time = time.time()
            prediction = outputs[0]
            target = targets[0]

            if len(prediction["scores"]) == 0:
                continue
            best = prediction["scores"].argmax()

            print("Best index:", best.item())
            print("Best score:", prediction["scores"][best].item())

            pred_center = prediction["centers"][best]
            pred_orientation = prediction["orientations"][best]
            # pred_orientation = pred_orientation.argmax()

            gt_center = target["centers"]
            gt_orientation = target["orientations"].item()

            error = torch.tensor([
                (pred_center[0] - gt_center[0]) * WIDTH,
                (pred_center[1] - gt_center[1]) * HEIGHT,
            ], device=pred_center.device)
            center_error = torch.norm(error)
            all_center_error.append(center_error.item())

            orientation_error = abs(pred_orientation - gt_orientation)
            orientation_error = min(orientation_error, 72 - orientation_error)
            all_orientation_error.append(orientation_error.item())

            center_is_correct = center_error <= 10
            orientation_is_correct = pred_orientation == gt_orientation
            combined_is_correct = center_is_correct and orientation_is_correct

            total += 1
            pose_correct += int(center_is_correct)
            orientation_correct += int(orientation_is_correct)
            combined_correct += int(combined_is_correct)

            evaluator_time = time.time() - evaluator_time
            metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

            metric_logger.synchronize_between_processes()
            # print("Averaged stats:", metric_logger)

            print(
                f"predicted pose: {pred_center}, "
                f"predicted angle: {pred_orientation.item()*5}°, "
                f"actual pose: {gt_center}, "
                f"actual angle: {gt_orientation*5}°"
            )

            # print(prediction["orientation_logits"])
            # print(prediction["orientation_logits"].argmax(dim=1))

            print("Center error:", center_error.item())
            print("Orientation bin error:", orientation_error.item())

        if len(all_center_error) != 0 and len(all_orientation_error) != 0:
            mean_center_error = sum(all_center_error) / len(all_center_error)
            mean_orientation_error = sum(all_orientation_error) / len(all_orientation_error)
        else:
            mean_center_error = None
            mean_orientation_error = None

        # print("Validation Results\n")
        # print("------------------")
        # print("Orientation accuracy:", accuracy, "%")
        if mean_center_error is None:
            print("\nMean center error: N/A px")
        else:
            print(f"\nMean center error: {mean_center_error:.3f} px")

        if mean_orientation_error is None:
            print("Mean orientation error: N/A degrees")
        else:
            print(f"Mean orientation error: {mean_orientation_error * 5:.3f} degrees")


    return all_center_error, all_orientation_error, {
        "accuracy": combined_correct / total if total else 0.0,
        "pose_accuracy": pose_correct / total if total else 0.0,
        "orientation_accuracy": orientation_correct / total if total else 0.0,
    }