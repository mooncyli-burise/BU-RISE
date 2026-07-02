import os
import torch
import torchvision.ops as ops
from torchvision.io import read_image
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F

class Dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir: str, ground_truth: list[dict], transform=None):
        self.root_dir = root_dir
        self.ground_truth = ground_truth
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.png') or f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = read_image(img_path)

        # boxes should just be the keypoints and small fixed box size
        boxes = torch.tv_tensors([[self.ground_truth[idx]["center"][0], 
                                   self.ground_truth[idx]["center"][1], 
                                   16, 
                                   16]], 
                                   dtype=torch.float32)
        xyxy_boxes = ops.box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")
        # integer label (1-72 for angles) with tensor for how many objects in frame (1 for one robot for now)
        # remove //72 if using mean-shift algorithm for continuous angle prediction
        headings = torch.tensor([self.ground_truth[idx]["heading"]%360//5], dtype=torch.int64)
        image_id = idx

        image = tv_tensors.Image(image)

        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(xyxy_boxes, format="XYXY", canvas_size=F.get_size(image))
        # for classification (robot/no robot), remove later
        target["labels"] = torch.tensor([1], dtype=torch.int64)
        target["headings"] = headings
        target["center"] = torch.tensor(self.ground_truth[idx]["center"], dtype=torch.float32)
        target["image_id"] = torch.tensor([image_id])
        

        if self.transform:
            image = self.transform(image)
        return image, target