import os

# os.system("wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/engine.py")
# os.system("wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/utils.py")
# os.system("wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_utils.py")
# os.system("wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_eval.py")
# os.system("wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/transforms.py")

from src.engine import train_one_epoch, evaluate
from model.dataset import Dataset
from model.model import model
from model.util_functions import get_transform
from model.get_val_loss import compute_validation_loss
from src import utils
import torch
from config import DATA_DIR
import pandas as pd
from dataset.epfl_processing import calculate_car_orientations, get_centers
import matplotlib.pyplot as plt

#use accelerator (offloading operations to gpu) or cpu if accelerator not available
device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else torch.device('cpu')

#set up ground truth data for training and testing
ground_truth = []
orientations = calculate_car_orientations(os.path.join(DATA_DIR, 'epfl-gims08/tripod-seq'))
centers = get_centers(os.path.join(DATA_DIR, 'epfl-gims08/tripod-seq'))

for i in range(len(centers)):
    ground_truth.append({
        "center": centers[i]["center"],
        "orientation": orientations[i]["orientation"]
    })

#create instance of dataset class, with transformations for training data
dataset = Dataset(os.path.join(DATA_DIR, 'epfl-gims08/tripod-seq'), ground_truth, get_transform(train=True))
#create instance of dataset class, with transformations for test data
dataset_test = Dataset(os.path.join(DATA_DIR, 'epfl-gims08/tripod-seq'), ground_truth, get_transform(train=False))

#make list of same size as dataset and randomize order
indices = torch.randperm(len(dataset)).tolist()
#assign subset from start of list to 50 indexes from the end for training
dataset = torch.utils.data.Subset(dataset, indices[:-50])
#assign subset of last 50 of list for test
dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

#load train and test data
#collate_fn for formatting images of diff sizes to be stacked into same batch
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=utils.collate_fn
)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=1,
    shuffle=False,
    collate_fn=utils.collate_fn
)

#move model to right device
model.to(device)

#initialize optimizer
#this is how the model learns by adjusting parameters and weights
#learning rate is step size for learning,
#momentum determines magnitude and direction of weight updates
#weight decay is multiplier for penalty term added to loss, prevents from overfitting by favoring lower weights->simpler models
params = [p for p in model.parameters() if p.requires_grad]
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

best_val_loss = float("inf")

train_losses = []
val_losses = []

#for each epoch train with training data, adjust lr, evaluate losses
for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    lr_scheduler.step()

    #save checkpoint of model, optimizer, and lr scheduler states
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": lr_scheduler.state_dict(),
    }, "checkpoint.pth")

    train_losses.append(train_loss.meters["loss"].global_avg)

    val_loss = compute_validation_loss(model, data_loader_test, device)
    val_losses.append(val_loss)

    #save best weights of model based on validation loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_robot_detector.pth")
        print(f"Saved best model (val loss = {val_loss:.4f})")

    evaluate(model, data_loader_test, device=device)
    
epochs = range(1, num_epochs + 1)

plt.figure(figsize=(6,4))
plt.plot(epochs, train_losses, label="Training Loss")
plt.plot(epochs, val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()