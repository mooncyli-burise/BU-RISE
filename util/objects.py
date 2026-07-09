import torch
from model.dataset import Dataset
from model.transform_functions import get_transform
from library_model_functions import utils
import os
from config import DATA_DIR, TEST_SIZE
from epfl_dataset_functions.epfl_processing import calculate_car_orientations, get_centers

#use accelerator (offloading operations to gpu) or cpu if accelerator not available
device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else torch.device('cpu')

#set up ground truth data for training and testing
ground_truth = []
orientations = calculate_car_orientations(os.path.join(DATA_DIR, 'epfl_dataset/tripod-seq'))
centers = get_centers(os.path.join(DATA_DIR, 'epfl_dataset/tripod-seq'))

for i in range(len(centers)):
    ground_truth.append({
        "center": centers[i]["center"],
        "orientation": orientations[i]["orientation"]
    })

#create instance of dataset class, with transformations for training data
dataset = Dataset(os.path.join(DATA_DIR, 'epfl_dataset/tripod-seq'), ground_truth, get_transform(train=True))
#create instance of dataset class, with transformations for test data
dataset_test = Dataset(os.path.join(DATA_DIR, 'epfl_dataset/tripod-seq'), ground_truth, get_transform(train=False))

#make list of same size as dataset and randomize order
indices = torch.randperm(len(dataset)).tolist()
#assign subset from start of list to 50 indexes from the end for training
dataset = torch.utils.data.Subset(dataset, indices[:-900]) #reduce size of training dataset
#assign subset of last 50 of list for test
dataset_test = torch.utils.data.Subset(dataset_test, indices[-TEST_SIZE:])

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