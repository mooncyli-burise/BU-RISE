import torch
import matplotlib.pyplot as plt
from simple_testing.og.simple_model_objects import device, data_loader, data_loader_test
from simple_model.model import GridNet
from simple_model.pose_loss import PoseLossFunction
import torch.nn as nn
from config import POSE_WEIGHT, X_CLASSES, Y_CLASSES, ANGLE_CLASSES
import numpy as np
from simple_model.training import train_one_epoch
from simple_model.eval_discrete_pos import eval

def train_simple():
    model = GridNet().to(device)

    pose_criterion = PoseLossFunction().to(device)
    class_criterion = nn.CrossEntropyLoss()

    #initialize optimizer
    #this is how the model learns by adjusting parameters and weights
    #learning rate is step size for learning,
    #momentum determines magnitude and direction of weight updates
    #weight decay is multiplier for penalty term added to loss, prevents from overfitting by favoring lower weights->simpler models
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    #adjusts learning rate,
    #decays learning rate by gamma every step_size epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=10,
        gamma=0.5
    )

    #number of epochs
    num_epochs = 150
    start_epoch = 0

    best_val_loss = float("inf")

    train_losses = []
    val_losses = []
    ce_losses = []
    custom_losses = []
    all_center_error = []
    all_orientation_error = []

    # #train from checkpoint
    # checkpoint = torch.load("simple_testing/simple_checkpoint.pth", map_location=device)

    # model.load_state_dict(checkpoint["model_state_dict"])
    # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    # lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    # best_val_loss = checkpoint.get("best_val_loss", float("inf"))
    # train_losses = checkpoint.get("train_losses", [])
    # val_losses = checkpoint.get("val_losses", [])

    # start_epoch = checkpoint["epoch"] + 1

    # model.to(device)

    # GridNet predicts one of 64 joint classes: 16 grid cells * 4 orientations.
    for epoch in range(start_epoch, num_epochs):
        train_loss, train_accuracy = train_one_epoch(model, optimizer, data_loader, device, class_criterion, pose_criterion)
        lr_scheduler.step()
        train_losses.append(train_loss)

        accuracy, pose_accuracy, orientation_accuracy, center_error, orientation_error = eval(model, data_loader_test, device)
        all_center_error += center_error
        all_orientation_error += orientation_error
    
        # Validation uses the same loss as training; GridNet is not a Faster R-CNN
        # model, so the Faster R-CNN validation helper cannot be used here.
        total_val_loss = 0.0
        with torch.no_grad():
            for images, targets in data_loader_test:
                logits = model(images.to(device))
                total_val_loss += POSE_WEIGHT * pose_criterion(logits, targets.to(device)).item() + class_criterion(logits, targets.to(device)).item()
        val_loss = total_val_loss / len(data_loader_test)
        val_losses.append(val_loss)

        # print all the data
        print(
            f"\nEpoch {epoch + 1}/{num_epochs}: "
            f"train loss {train_loss:.4f}, "
            f"train accuracy {train_accuracy:.3f} \n"
            f"val loss {val_loss:.4f}, "
            f"val accuracy {accuracy:.3f}, "
            f"Pose accuracy: {pose_accuracy:.3f}, "
            f"Orientation accuracy: {orientation_accuracy:.3f}"
        )

        print(
            f"CE Loss = {np.mean(ce_losses[-len(data_loader):]):.4f}, "
            f"Custom Loss = {np.mean(custom_losses[-len(data_loader):]):.4f}\n"
        )

        #save best weights of model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "simple_testing/simple_best_robot_detector.pth")
            print(f"Saved best model (val loss = {val_loss:.4f})")

        #save checkpoint of model, optimizer, and lr scheduler states
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": lr_scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "train_losses": train_losses,
            "val_losses": val_losses,
        }, "simple_testing/simple_checkpoint.pth")

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
    axes[1].hist(all_center_error, bins=20, edgecolor='black', color='skyblue')
    axes[1].set_xlabel("Center Error")
    axes[1].set_ylabel("Frequency Count")
    axes[1].set_title("Center Error Distribution")

    # 3. Orientation Error Histogram (Third Subplot)
    axes[2].hist(all_orientation_error, bins=72, edgecolor='black', color='lightcoral') # Changed color for distinction
    axes[2].set_xlabel("Orientation Error")
    axes[2].set_ylabel("Frequency Count")
    axes[2].set_title("Orientation Error Distribution")

    # Clean up spacing and display the single window
    plt.tight_layout()
    plt.show()
    # #plot individual losses
    # plt.figure(figsize=(6,4))
    # plt.plot(ce_losses, label="Cross Entropy")
    # plt.plot(custom_losses, label="Custom Loss")
    # plt.xlabel("Training Batch")
    # plt.ylabel("Loss")
    # plt.legend()
    # plt.grid(True)

    plt.show()
