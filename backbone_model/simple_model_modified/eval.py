import torch
from config import WIDTH, HEIGHT, CENTER_CORRECT_RANGE

def eval(model, data_loader_test, device):
    # evaluation loop
    model.eval()

    combined_correct = 0
    total = 0
    pose_correct = 0
    orientation_correct = 0

    all_center_error = []
    all_orientation_error = []

    with torch.no_grad():
        for images, targets in data_loader_test:
            images = images.to(device)
            targets = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in targets.items()
            }

            logits = model(images)

            #TODO: only calculate center and orientation losses when robot detected
            pred_center = logits["center"]
            gt_center = targets["center"]

            pred_orientation = logits["orientation"].argmax(dim=1)
            gt_orientation = targets["orientation"]

            scale = torch.tensor([WIDTH, HEIGHT], device=device)

            center_error = torch.norm(
                (pred_center - gt_center) * scale,
                dim=1
            )
            all_center_error.extend(
                center_error.cpu().tolist()
            )

            orientation_error = torch.abs(pred_orientation - gt_orientation)
            orientation_error = torch.minimum(orientation_error, 72 - orientation_error)
            all_orientation_error.extend(
                orientation_error.cpu().tolist()
            )

            # change range for center correct here
            center_is_correct = center_error <= CENTER_CORRECT_RANGE
            orientation_is_correct = pred_orientation == gt_orientation
            combined_is_correct = center_is_correct & orientation_is_correct

            batch_size = images.size(0)
            total += batch_size
            pose_correct += int(center_is_correct.sum().item())
            orientation_correct += int(orientation_is_correct.sum().item())
            combined_correct += int(combined_is_correct.sum().item())

            # print each prediction
            for i in range(images.size(0)):
                print(
                    f"predicted center: {pred_center[i].cpu().numpy()}, "
                    f"predicted orientation: {pred_orientation[i].item() * 5}°, "
                    f"actual center: {gt_center[i].cpu().numpy()}, "
                    f"actual orientation: {gt_orientation[i].item() * 5}°"
                )

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


    accuracy = combined_correct / total if total else 0.0
    pose_accuracy = pose_correct / total if total else 0.0
    orientation_accuracy = orientation_correct / total if total else 0.0

    return accuracy, pose_accuracy, orientation_accuracy, all_center_error, all_orientation_error