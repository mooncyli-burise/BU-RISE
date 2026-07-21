from config import ORIENTATION_LOSS_WEIGHT, CENTER_LOSS_WEIGHT

def train_one_epoch(model, optimizer, data_loader, device, class_criterion, pose_criterion):
    # training loop
    model.train()
    total_train_loss = 0.0

    train_correct = 0
    train_total = 0

    for images, targets in data_loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        logits = model(images)
        ce_loss = class_criterion(logits, targets)
        custom_loss = pose_criterion(logits, targets)
        loss = ORIENTATION_LOSS_WEIGHT * custom_loss + ce_loss

        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)
        train_correct += (preds == targets).sum().item()
        train_total += targets.size(0)

        total_train_loss += loss.item()

    train_loss = total_train_loss / len(data_loader)
    train_accuracy = train_correct / train_total

    return train_loss, train_accuracy
