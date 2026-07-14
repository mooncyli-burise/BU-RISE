import torch
import matplotlib.pyplot as plt
from simple_testing.simple_model_objects import device, data_loader, data_loader_test
from simple_model.model import GridNet
from simple_model.pose_loss import PoseLossFunction
import torch.nn as nn
from config import POSE_WEIGHT

def train_simple():
    train_model = GridNet().to(device)

    pose_criterion = PoseLossFunction().to(device)
    class_criterion = nn.CrossEntropyLoss()

    #initialize optimizer
    #this is how the model learns by adjusting parameters and weights
    #learning rate is step size for learning,
    #momentum determines magnitude and direction of weight updates
    #weight decay is multiplier for penalty term added to loss, prevents from overfitting by favoring lower weights->simpler models
    optimizer = torch.optim.Adam(
        train_model.parameters(),
        lr=1e-4
    )

    #adjusts learning rate,
    #decays learning rate by gamma every step_size epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=10,
        gamma=0.5
    )

    #number of epochs
    num_epochs = 20
    start_epoch = 0

    best_val_loss = float("inf")

    train_losses = []
    val_losses = []

    # #train from checkpoint
    # checkpoint = torch.load("simple_testing/simple_checkpoint.pth", map_location=device)

    # train_model.load_state_dict(checkpoint["model_state_dict"])
    # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    # lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    # best_val_loss = checkpoint.get("best_val_loss", float("inf"))
    # train_losses = checkpoint.get("train_losses", [])
    # val_losses = checkpoint.get("val_losses", [])

    # start_epoch = checkpoint["epoch"] + 1

    # train_model.to(device)

    # GridNet predicts one of 64 joint classes: 16 grid cells * 4 orientations.
    for epoch in range(start_epoch, num_epochs):
        train_model.train()
        total_train_loss = 0.0

        train_correct = 0
        train_total = 0

        for images, targets in data_loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            logits = train_model(images)
            loss = POSE_WEIGHT * pose_criterion(logits, targets) + class_criterion(logits, targets)

            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)
            train_correct += (preds == targets).sum().item()
            train_total += targets.size(0)

            total_train_loss += loss.item()

        lr_scheduler.step()
        train_loss = total_train_loss / len(data_loader)
        train_losses.append(train_loss)
        train_accuracy = train_correct / train_total

        #save checkpoint of model, optimizer, and lr scheduler states
        torch.save({
            "epoch": epoch,
            "model_state_dict": train_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": lr_scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "train_losses": train_losses,
            "val_losses": val_losses,
        }, "simple_testing/simple_checkpoint.pth")

        train_model.eval()

        correct = 0
        total = 0
        pose_correct = 0
        orientation_correct = 0

        with torch.no_grad():
            for images, targets in data_loader_test:
                images = images.to(device)
                targets = targets.to(device)

                logits = train_model(images)
                preds = logits.argmax(dim=1)

                # These are tensors containing one pose/orientation value per
                # validation example, so they can be compared and counted across
                # the entire batch.
                predicted_poses = preds // 4
                actual_poses = targets // 4
                predicted_orientation_bins = preds % 4
                actual_orientation_bins = targets % 4

                pose_correct += (predicted_poses == actual_poses).sum().item()
                orientation_correct += (
                    predicted_orientation_bins == actual_orientation_bins
                ).sum().item()

                for prediction, target in zip(preds, targets):
                    predicted_pose = prediction.item() // 4
                    predicted_orientation = (prediction.item() % 4) * 90

                    actual_pose = target.item() // 4
                    actual_orientation = (target.item() % 4) * 90

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

        # Validation uses the same loss as training; GridNet is not a Faster R-CNN
        # model, so the Faster R-CNN validation helper cannot be used here.
        total_val_loss = 0.0
        with torch.no_grad():
            for images, targets in data_loader_test:
                logits = train_model(images.to(device))
                total_val_loss += POSE_WEIGHT * pose_criterion(logits, targets.to(device)).item() + class_criterion(logits, targets.to(device)).item()
        val_loss = total_val_loss / len(data_loader_test)
        val_losses.append(val_loss)

        print(
            f"\nEpoch {epoch + 1}/{num_epochs}: "
            f"train loss {train_loss:.4f}, "
            f"train accuracy {train_accuracy:.3f}, "
            f"val loss {val_loss:.4f}, "
            f"val accuracy {accuracy:.3f}, "
            f"Pose accuracy: {pose_accuracy:.3f}, "
            f"Orientation accuracy: {orientation_accuracy:.3f}, "
        )

        #save best weights of model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(train_model.state_dict(), "simple_testing/simple_best_robot_detector.pth")
            print(f"Saved best model (val loss = {val_loss:.4f})")

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(6,4))
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()
