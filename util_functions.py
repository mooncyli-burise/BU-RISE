import torch
from torchvision.transforms import v2 as T

def get_transform(train):
    transforms = []
    #50% chance of random horizontal flip if train
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    #converts datatype to floats and scaled to between 0.0 and 1.0
    transforms.append(T.ToDtype(torch.float, scale=True))
    #converts from tv_tensor to torch.Tensor
    transforms.append(T.ToPureTensor())
    #returns all transformations condensed into single callable sequence
    return T.Compose(transforms)


