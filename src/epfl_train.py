import os

# os.system("wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/engine.py")
# os.system("wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/utils.py")
# os.system("wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_utils.py")
# os.system("wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_eval.py")
# os.system("wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/transforms.py")

from engine import train_one_epoch, evaluate
from epfl_dataset import Dataset
import model.model as model
from util_functions import get_transform
import utils
import torch
from config import DATA_DIR
import pandas as pd

#use accelerator (offloading operations to gpu) or cpu if accelerator not available
device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else torch.device('cpu')

#set up ground truth data for training and testing
ground_truth = []
with open('tripod-seq.txt', 'r', encoding='utf-8') as file:
    # 1. Read just the very first line of the file
    first_line = file.readline()
    
    # 2. Split the line at the first space and take the very first piece
    # maxsplit=1 stops Python from wasting time splitting the rest of the text
    seq_num = first_line.split(' ', 1)[0]

for i in range(seq_num):
    # Read the next line from the file
    line = file.readline()
    
    # Split the line into parts
    parts = line.split()
    
    # Create a dictionary for this line's data
    data_dict = {
        "frame": int(parts[0]),
        "center": [float(parts[1]), float(parts[2])],
        "heading": float(parts[3])
    }
    
    # Append the dictionary to the list
    ground_truth.append(data_dict)


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

#for each epoch train with training data, adjust lr, evaluate losses
for epoch in range(num_epochs):
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    lr_scheduler.step()
    evaluate(model, data_loader_test, device=device)