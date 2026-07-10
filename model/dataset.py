import os
import torch
#import torchvision
import torchvision.ops as ops
from torchvision.io import read_image
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from util.normalize_pixel_coords import normalize_coords
import re

class Dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir: str, ground_truth: list[dict], transform=None):
        self.root_dir = root_dir
        self.ground_truth = ground_truth
        self.transform = transform
        self.image_files = sorted(
            (f for f in os.listdir(root_dir) if f.endswith((".png", ".jpg"))),
            key=lambda filename: int(
                re.search(r"(\d+)(?=\.[^.]+$)", filename).group(1)
            ),
        )
        # print("Images:", len(self.image_files))
        # print("Ground truth:", len(self.ground_truth), len(self.ground_truth))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = read_image(img_path)

        # bbox_size = normalize_coords(64,64)
        bbox_size = (128,128)

        # boxes should just be the keypoints and small fixed box size
        boxes = torch.tensor([[self.ground_truth[idx]["center"][0], 
                                   self.ground_truth[idx]["center"][1], 
                                   bbox_size[0], 
                                   bbox_size[1]]], 
                                   dtype=torch.float32)
        xyxy_boxes = ops.box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")
        # integer label (1-72 for angles) with tensor for how many objects in frame (1 for one robot for now)
        # remove //72 if using mean-shift algorithm for continuous angle prediction
        orientations = torch.tensor([self.ground_truth[idx]["orientation"]%360//5], dtype=torch.int64)
        image_id = idx

        image = tv_tensors.Image(image)

        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(xyxy_boxes, format="XYXY", canvas_size=F.get_size(image))
        # for classification (robot/no robot), remove later
        target["labels"] = torch.tensor([1], dtype=torch.int64)
        target["orientations"] = orientations
        target["centers"] = torch.tensor(self.ground_truth[idx]["center"], dtype=torch.float32)
        target["image_id"] = torch.tensor([image_id])

        # # TODO: remove these two if i make a custom validation loop
        # target["area"] = (xyxy_boxes[:, 2] - xyxy_boxes[:, 0]) * (xyxy_boxes[:, 3] - xyxy_boxes[:, 1])
        # target["iscrowd"] = torch.zeros((1,), dtype=torch.int64)

        if self.transform:
            image = self.transform(image)
        return image, target