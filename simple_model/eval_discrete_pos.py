import torch
from config import ANGLE_CLASSES

def eval(model, data_loader_test, device):
    # evaluation loop
    model.eval()

    correct = 0
    total = 0
    pose_correct = 0
    orientation_correct = 0

    with torch.no_grad():
        for images, targets in data_loader_test:
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            preds = logits.argmax(dim=1)

            # These are tensors containing one pose/orientation value per
            # validation example, so they can be compared and counted across
            # the entire batch.
            predicted_poses = preds // ANGLE_CLASSES
            actual_poses = targets // ANGLE_CLASSES
            predicted_orientation_bins = preds % ANGLE_CLASSES
            actual_orientation_bins = targets % ANGLE_CLASSES

            pose_correct += (predicted_poses == actual_poses).sum().item()
            orientation_correct += (
                predicted_orientation_bins == actual_orientation_bins
            ).sum().item()

            # print each predictiobn
            for prediction, target in zip(preds, targets):
                predicted_pose = prediction.item() // ANGLE_CLASSES
                predicted_orientation = (prediction.item() % ANGLE_CLASSES) * (360/ANGLE_CLASSES)

                actual_pose = target.item() // ANGLE_CLASSES
                actual_orientation = (target.item() % ANGLE_CLASSES) * (360/ANGLE_CLASSES)

                print(
                    f"predicted pose: {predicted_pose}, "
                    f"predicted orientation: {predicted_orientation}°, "
                    f"actual pose: {actual_pose}, "
                    f"actual orientation: {actual_orientation}°"
                )

            correct += (preds == targets).sum().item()
            total += targets.size(0)

    accuracy = correct / total if total else 0.0
    pose_accuracy = pose_correct / total if total else 0.0
    orientation_accuracy = orientation_correct / total if total else 0.0

    return accuracy, pose_accuracy, orientation_accuracy