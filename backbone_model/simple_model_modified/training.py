import torch
from config import ORIENTATION_LOSS_WEIGHT, CENTER_LOSS_WEIGHT, WIDTH, HEIGHT, CENTER_CORRECT_RANGE, CE_LOSS_WEIGHT

def train_one_epoch(model, optimizer, data_loader, device, class_criterion, center_criterion, orientation_criterion):
    # training loop
    model.train()
    total_train_loss = 0.0
    total_center_loss = 0.0
    total_ce_loss = 0.0
    total_orientation_loss = 0.0
    total = 0
    pose_correct = 0
    orientation_correct = 0
    combined_correct = 0


    for images, targets in data_loader:
        images = images.to(device)
        targets = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in targets.items()
        }

        optimizer.zero_grad()

        logits = model(images)

        #TODO: only calculate center and orientation loss if robot detected
        # putting orientation stuff through ce loss instead of class
        center_loss = CENTER_LOSS_WEIGHT * center_criterion(logits["center"], targets["center"])
        orientation_loss = ORIENTATION_LOSS_WEIGHT * orientation_criterion(logits["orientation"], targets["orientation"])
        ce_loss = CE_LOSS_WEIGHT * class_criterion(logits["class"], targets["class"])
        loss = center_loss + orientation_loss + ce_loss

        loss.backward()
        optimizer.step()

        pred_center = logits["center"]
        gt_center = targets["center"]

        pred_orientation = logits["orientation"].argmax(dim=1)
        gt_orientation = targets["orientation"]

        scale = torch.tensor([WIDTH, HEIGHT], device=device)

        center_error = torch.norm(
            (pred_center - gt_center) * scale,
            dim=1
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

        total_train_loss += loss.item()
        total_center_loss += center_loss.item()
        total_orientation_loss += orientation_loss.item()
        total_ce_loss += ce_loss.item()

    train_loss = total_train_loss / len(data_loader)
    train_accuracy = combined_correct / total

    return (
        train_loss, 
        train_accuracy,     
        total_ce_loss / len(data_loader),
        total_center_loss / len(data_loader),
        total_orientation_loss / len(data_loader),
    )