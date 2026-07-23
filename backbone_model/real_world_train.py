import torch
import matplotlib.pyplot as plt
from backbone_model.simple_model_objects_modified import device, data_loader, data_loader_test
from backbone_model.simple_model_modified.model import GridNet
from backbone_model.simple_model_modified.loss_function import CenterLossFunction, OrientationLossFunction
import torch.nn as nn
from config import ORIENTATION_LOSS_WEIGHT, CENTER_LOSS_WEIGHT, CE_LOSS_WEIGHT, CENTER_CORRECT_RANGE
import numpy as np
from backbone_model.simple_model_modified.training import train_one_epoch
from backbone_model.simple_model_modified.eval import eval

def train_simple():
    model = GridNet().to(device)

    model.load_state_dict('backbone_model/best_model_5000imgs.pth')
    model.to(device)

    center_criterion = CenterLossFunction().to(device)
    orientation_criterion = OrientationLossFunction().to(device)
    class_criterion = nn.CrossEntropyLoss()

    #initialize optimizer
    #this is how the model learns by adjusting parameters and weights
    #learning rate is step size for learning,
    #momentum determines magnitude and direction of weight updates
    #weight decay is multiplier for penalty term added to loss, prevents from overfitting by favoring lower weights->simpler models
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4,
    )

    #adjusts learning rate,
    #decays learning rate by gamma every step_size epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=10,
        gamma=0.5
    )

    #number of epochs
    num_epochs = 50 # try 45
    start_epoch = 0

    best_val_loss = float("inf")

    train_losses = []
    val_losses = []
    ce_losses = []
    center_losses = []
    orientation_losses = []
    class_losses = []
    all_center_error = []
    all_orientation_error = []
    ce_train_losses = []
    center_train_losses = []
    orientation_train_losses = []
    class_train_losses = []

    # #train from checkpoint
    # checkpoint = torch.load("backbone_model/finetuning_checkpoint.pth", map_location=device)

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
        train_loss, train_accuracy, ce_loss, center_loss, orientation_loss, class_loss = train_one_epoch(model, optimizer, data_loader, device, class_criterion, center_criterion, orientation_criterion)
        lr_scheduler.step()
        train_losses.append(train_loss)
        ce_train_losses.append(ce_loss)
        center_train_losses.append(center_loss)
        orientation_train_losses.append(orientation_loss)
        class_train_losses.append(class_loss)

        accuracy, pose_accuracy, orientation_accuracy, center_error, orientation_error = eval(model, data_loader_test, device)
        all_center_error += center_error
        all_orientation_error += orientation_error
    
        # calculate avg validation loss and avg loss for each detection head (class, center, orientation)
        total_val_loss = 0.0
        total_ce_loss = 0.0
        total_center_loss = 0.0
        total_orientation_loss = 0.0
        total_class_loss = 0.0
        with torch.no_grad():
            for images, targets in data_loader_test:
                logits = model(images.to(device))
                # TODO: testing putting orientation stuff through ce loss instead of class
                total_ce_loss += CE_LOSS_WEIGHT * class_criterion(logits["orientation"], targets["orientation"])
                total_center_loss += CENTER_LOSS_WEIGHT * center_criterion(logits["center"], targets["center"])
                total_orientation_loss += ORIENTATION_LOSS_WEIGHT * orientation_criterion(logits["orientation"], targets["orientation"])
                total_class_loss += class_criterion(logits["class"], targets["class"])
                total_val_loss += total_center_loss + total_orientation_loss + total_ce_loss + total_class_loss
        val_loss = total_val_loss / len(data_loader_test)
        val_losses.append(val_loss)
        ce_loss = total_ce_loss / len(data_loader_test)
        ce_losses.append(ce_loss)
        center_loss = total_center_loss / len(data_loader_test)
        center_losses.append(center_loss)
        orientation_loss = total_orientation_loss / len(data_loader_test)
        orientation_losses.append(orientation_loss)
        class_loss = total_class_loss / len(data_loader_test)
        class_losses.append(class_loss)

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
            f"Center Loss = {np.mean(center_losses[-len(data_loader):]):.4f}, "
            f"Orientation Loss = {np.mean(orientation_losses[-len(data_loader):]):.4f}, "
            f"Class Loss = {np.mean(class_losses[-len(data_loader):]):.4f}\n"
        )

        #save best weights of model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "backbone_model/best_finetuning_model.pth")
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
        }, "backbone_model/finetuning_checkpoint.pth")

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


    # Add text to the right side of the graph
    axes[2].text(
        1.05,                  # X coordinate: slightly to the right of the plot border
        0.5,                   # Y coordinate: centered vertically
        f"Center Weight: {CENTER_LOSS_WEIGHT}\nOrientation Weight: {ORIENTATION_LOSS_WEIGHT}\nCE Weight: {CE_LOSS_WEIGHT}\nCenter Correct Range: {CENTER_CORRECT_RANGE}",    # The text string
        transform=axes[2].transAxes, # Use axes coordinates (0 to 1 scale)
        va="center",           # Center the text vertically on the Y coordinate
        ha="left"              # Align text to the left of the X coordinate
    )


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

    epochs = range(1, len(ce_train_losses) + 1)

    plt.figure(figsize=(8,5))
    plt.plot(epochs, ce_train_losses, label="Cross Entropy")
    plt.plot(epochs, center_train_losses, label="Center Loss")
    plt.plot(epochs, orientation_train_losses, label="Orientation Loss")
    plt.plot(epochs, class_train_losses, label="Class Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Components")
    plt.grid(True)
    plt.legend()
    plt.show()

