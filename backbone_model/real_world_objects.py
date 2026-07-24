import torch
from backbone_model.simple_model_modified.dataset import Dataset
import os
from config import TEST_SIZE
import json
from simplified_dataset.simple_model.transforms import get_transforms
from april_tags.get_data import get_apriltag_images
from april_tags.create_ground_truth import create_ground_truth

#use accelerator (offloading operations to gpu) or cpu if accelerator not available
# device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else torch.device('cpu')

device = torch.device('cpu')

#set up ground truth data for training and testing
ground_truth = []
with open("backbone_model/real_world_dataset/ground_truth.json", "r") as file:
    ground_truth = json.load(file)
ground_truth_real_world = create_ground_truth(get_apriltag_images('backbone_model/real_world_dataset/limo'))

# GridNet uses fixed-size images and scalar class targets, so the default
# DataLoader collation produces image batches [B, C, H, W] and targets [B].
dataset = Dataset('backbone_model/real_world_dataset/images', ground_truth, get_transforms())
dataset_test = Dataset('backbone_model/real_world_dataset/images', ground_truth, get_transforms())
dataset_real_world = Dataset('backbone_model/real_world_dataset/limo', ground_truth_real_world, get_transforms())

#make list of same size as dataset and randomize order
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:-TEST_SIZE]) 
#assign subset of last 50 of list for test
dataset_test = torch.utils.data.Subset(dataset_test, indices[-TEST_SIZE:])

# dataset = torch.utils.data.Subset(dataset, indices[:50])
# dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

# # dataset for overfitting below
# dataset = torch.utils.data.Subset(dataset, indices[:50])
# #assign subset of last 50 of list for test
# dataset_test = torch.utils.data.Subset(dataset_test, indices[:50])


#load train and test data
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=32,
    shuffle=True, 
)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=32,
    shuffle=False,
)

