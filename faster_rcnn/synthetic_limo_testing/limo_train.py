import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from faster_rcnn.model.library_model_functions.engine import train_one_epoch
from faster_rcnn.model.get_val_loss import compute_validation_loss
from faster_rcnn.model.model import create_model
import torch
import matplotlib.pyplot as plt
from faster_rcnn.model.custom_eval import custom_eval
from faster_rcnn.synthetic_limo_testing.limo_objects import device, data_loader, data_loader_test
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

    # for p in train_model.backbone.parameters():
    #     p.requires_grad = False

    # for p in train_model.rpn.parameters():
    #     p.requires_grad = False

    # for p in train_model.roi_heads.box_head.parameters():
    #     p.requires_grad = False

    # for p in train_model.roi_heads.box_predictor.parameters():
    #     p.requires_grad = False

    # # Only train the orientation head
    # for p in train_model.roi_heads.robot_head.orientation.parameters():
    #     p.requires_grad = True

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
    num_epochs = 100
    start_epoch = 0

    best_val_loss = float("inf")

    train_losses = []
    val_losses = []

    all_center_error = []
    all_orientation_error = []

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
        train_loss, train_accuracy = train_one_epoch(train_model, optimizer, data_loader, device, epoch, print_freq=10)
        lr_scheduler.step()

        center_error, orientation_error, val_accuracy = custom_eval(train_model, data_loader_test, device=device)
        all_center_error += center_error
        all_orientation_error += orientation_error

        train_loss_value = float(train_loss.meters["loss"].global_avg)
        train_losses.append(train_loss_value)

        val_loss = compute_validation_loss(train_model, data_loader_test, device)
        val_losses.append(val_loss)

        print(
            f"\nEpoch {epoch+1}/{num_epochs}: "
            f"train loss {train_loss_value:.4f}, "
            f"train accuracy {train_accuracy['accuracy']:.3f} | "
            f"\nval loss {val_loss:.4f}, "
            f"val accuracy {val_accuracy['accuracy']:.3f}, "
            f"\npose accuracy {val_accuracy['pose_accuracy']:.3f}, "
            f"orientation accuracy {val_accuracy['orientation_accuracy']:.3f}\n"
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

    # Create a 1-row, 3-column grid layout
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1. Training vs Validation Losses (First Subplot)
    axes[0].plot(epochs, train_losses, label="Training Loss")
    axes[0].plot(epochs, val_losses, label="Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training vs Val Losses")
    axes[0].legend()
    axes[0].grid(True)

    # 2. Center Error Histogram (Second Subplot)
    if all_center_error:
        axes[1].hist(all_center_error, bins=20, edgecolor='black', color='skyblue')
    else:
        axes[1].text(0.5, 0.5, "No center errors collected", ha="center", va="center")
    axes[1].set_xlabel("Center Error")
    axes[1].set_ylabel("Frequency Count")
    axes[1].set_title("Center Error Distribution")

    # 3. Orientation Error Histogram (Third Subplot)
    if all_orientation_error:
        axes[2].hist(all_orientation_error, bins=72, edgecolor='black', color='lightcoral')
    else:
        axes[2].text(0.5, 0.5, "No orientation errors collected", ha="center", va="center")
    axes[2].set_xlabel("Orientation Error")
    axes[2].set_ylabel("Frequency Count")
    axes[2].set_title("Orientation Error Distribution")

    # Clean up spacing and display the single window
    plt.tight_layout()
    plt.show()