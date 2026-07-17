import torch
from config import WIDTH, HEIGHT

def eval(model, data_loader_test, device):
    # evaluation loop
    model.eval()

    correct = 0
    total = 0
    pose_correct = 0
    orientation_correct = 0

    all_center_error = []
    all_orientation_error = []

    with torch.no_grad():
        for images, targets in data_loader_test:
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)

            prediction = logits[0]
            target = targets[0]

            if len(prediction["scores"]) == 0:
                continue
            best = prediction["scores"].argmax()

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

            # print each predictiobn
            for prediction, target in zip(prediction, targets):
                print(
                    f"predicted center: {pred_center}, "
                    f"predicted orientation: {pred_orientation}°, "
                    f"actual center: {gt_center}, "
                    f"actual orientation: {gt_orientation}°"
                )

            correct += (prediction == targets).sum().item()
            total += targets.size(0)

    if len(all_center_error) != 0 and len(all_orientation_error) != 0:
        mean_center_error = sum(all_center_error) / len(all_center_error)
        mean_orientation_error = sum(all_orientation_error) / len(all_orientation_error)
    else:
        mean_center_error = None
        mean_orientation_error = None

    if mean_center_error is None:
        print("\nMean center error: N/A px")
    else:
        print(f"\nMean center error: {mean_center_error:.3f} px")

    if mean_orientation_error is None:
        print("Mean orientation error: N/A degrees")
    else:
        print(f"Mean orientation error: {mean_orientation_error * 5:.3f} degrees")


    accuracy = correct / total if total else 0.0
    pose_accuracy = pose_correct / total if total else 0.0
    orientation_accuracy = orientation_correct / total if total else 0.0

    return accuracy, pose_accuracy, orientation_accuracy, all_center_error, all_orientation_error