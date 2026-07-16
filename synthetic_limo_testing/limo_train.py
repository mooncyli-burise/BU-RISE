import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.library_model_functions.engine import train_one_epoch
from model.get_val_loss import compute_validation_loss
from model.model import create_model
import torch
import matplotlib.pyplot as plt
from model.custom_eval import custom_eval
from synthetic_limo_testing.limo_objects import device, data_loader, data_loader_test
from config import HEIGHT, WIDTH


def compute_detection_metrics(model, data_loader, device, pose_threshold=5.0):
    model.eval()
    total = 0
    pose_correct = 0
    orientation_correct = 0
    combined_correct = 0

    with torch.no_grad():
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            outputs = model(images)

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

                pose_is_correct = center_error <= pose_threshold
                orientation_is_correct = pred_orientation == gt_orientation
                combined_is_correct = pose_is_correct and orientation_is_correct

                total += 1
                pose_correct += int(pose_is_correct)
                orientation_correct += int(orientation_is_correct)
                combined_correct += int(combined_is_correct)

    return {
        "accuracy": combined_correct / total if total else 0.0,
        "pose_accuracy": pose_correct / total if total else 0.0,
        "orientation_accuracy": orientation_correct / total if total else 0.0,
    }


def train_limo():
    train_model = create_model()

    #move model to right device
    train_model.to(device)

    #initialize optimizer
    #this is how the model learns by adjusting parameters and weights
    #learning rate is step size for learning,
    #momentum determines magnitude and direction of weight updates
    #weight decay is multiplier for penalty term added to loss, prevents from overfitting by favoring lower weights->simpler models
    params = [p for p in train_model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005
    )

    #adjusts learning rate,
    #decays learning rate by gamma every step_size epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.1
    )
    

    #number of epochs
    num_epochs = 2
    start_epoch = 0

    best_val_loss = float("inf")

    train_losses = []
    val_losses = []

    # #train from checkpoint
    # checkpoint = torch.load("synthetic_limo_testing/limo_checkpoint.pth", map_location=device)

    # train_model.load_state_dict(checkpoint["model_state_dict"])
    # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    # lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    # best_val_loss = checkpoint.get("best_val_loss", float("inf"))
    # train_losses = checkpoint.get("train_losses", [])
    # val_losses = checkpoint.get("val_losses", [])

    # start_epoch = checkpoint["epoch"] + 1

    # train_model.to(device)

    #for each epoch train with training data, adjust lr, evaluate losses
    for epoch in range(start_epoch, num_epochs):
        train_loss = train_one_epoch(train_model, optimizer, data_loader, device, epoch, print_freq=10)
        lr_scheduler.step()

        train_loss_value = float(train_loss.meters["loss"].global_avg)
        train_losses.append(train_loss_value)

        train_metrics = compute_detection_metrics(train_model, data_loader, device)
        val_loss = compute_validation_loss(train_model, data_loader_test, device)
        val_losses.append(val_loss)
        val_metrics = compute_detection_metrics(train_model, data_loader_test, device)

        custom_eval(train_model, data_loader_test, device=device)

        print(
            f"\nEpoch {epoch + 1}/{num_epochs}: "
            f"train loss {train_loss_value:.4f}, "
            f"train accuracy {train_metrics['accuracy']:.3f} | "
            f"val loss {val_loss:.4f}, "
            f"val accuracy {val_metrics['accuracy']:.3f}, "
            f"pose accuracy {val_metrics['pose_accuracy']:.3f}, "
            f"orientation accuracy {val_metrics['orientation_accuracy']:.3f}"
        )

        #save checkpoint of model, optimizer, and lr scheduler states
        torch.save({
            "epoch": epoch,
            "model_state_dict": train_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": lr_scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "train_losses": train_losses,
            "val_losses": val_losses,
        }, "synthetic_limo_testing/limo_checkpoint.pth")

        #save best weights of model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(train_model.state_dict(), "synthetic_limo_testing/limo_best_robot_detector.pth")
            print(f"Saved best model (val loss = {val_loss:.4f})")


    best_model = create_model()
    best_model.load_state_dict(
        torch.load("synthetic_limo_testing/limo_best_robot_detector.pth", map_location=device)
    )
    best_model.to(device)

    custom_eval(best_model, data_loader_test, device)

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(6,4))
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()