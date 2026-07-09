import torch
import time
from library_model_functions import utils
from config import TEST_SIZE

def custom_eval(model, data_loader, device):
    model.roi_heads.score_thresh = 0.0
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    all_center_error = []
    all_orientation_error = []
    correct = 0
    total = TEST_SIZE

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

            correct += (pred_orientation == gt_orientation).item()

            center_error = torch.norm(pred_center - gt_center)
            all_center_error.append(center_error.item())

            orientation_error = abs(pred_orientation - gt_orientation)
            orientation_error = min(orientation_error, 72 - orientation_error)
            all_orientation_error.append(orientation_error.item())

            evaluator_time = time.time() - evaluator_time
            metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

            metric_logger.synchronize_between_processes()
            print("Averaged stats:", metric_logger)

            #print(prediction["orientation_logits"])

            print("Pred center:", pred_center)
            print("GT center:", gt_center)

            print("Pred orientation bin:", pred_orientation.item())
            print("GT orientation bin:", gt_orientation)

            print("Center error:", center_error.item())
            print("Orientation bin error:", orientation_error.item())

        if(len(all_center_error) != 0 and len(all_orientation_error) != 0):
            accuracy = correct / total
            mean_center_error = sum(all_center_error) / len(all_center_error)
            mean_orientation_error = sum(all_orientation_error) / len(all_orientation_error)
        else:
            accuracy = "N/A"
            mean_center_error = "N/A"
            mean_orientation_error = "N/A"

        print("Validation Results\n")
        print("------------------")
        print("Orientation accuracy:", accuracy, "%")
        print("Mean center error: ", mean_center_error, "px")
        print("Mean orientation error: ", mean_orientation_error * 5, "degrees")

    return 