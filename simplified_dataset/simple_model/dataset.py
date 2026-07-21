import os
import torch
import cv2
import re
from config import ANGLE_CLASSES, X_CLASSES, Y_CLASSES, TOTAL_CLASSES

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
        self.num_pose_classes = X_CLASSES * Y_CLASSES
        # print("Images:", len(self.image_files))
        # print("Ground truth:", len(self.ground_truth), len(self.ground_truth))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, self.image_files[idx])
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        image = torch.from_numpy(image).permute(2, 0, 1)
        # image = (image - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / \
        #     torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        # A single class represents a pair of values:
        #   class = pose_cell * ANGLE_CLASSES + orientation_bin
        # pose_cell is mapped to a 0-based index over the available pose cells,
        # and orientation_bin is a 0-based bin index in [0, ANGLE_CLASSES).
        pose_cell = int(self.ground_truth[idx]["pose"]) % self.num_pose_classes
        pose_cell = max(0, min(pose_cell, self.num_pose_classes - 1))
        orientation_bin = int(self.ground_truth[idx]["orientation"]) % 360 // (360 / ANGLE_CLASSES)
        orientation_bin = max(0, min(orientation_bin, ANGLE_CLASSES - 1))
        target = torch.tensor(
            (pose_cell * ANGLE_CLASSES + orientation_bin) % TOTAL_CLASSES,
            dtype=torch.long,
        )

        if self.transform:
            image = self.transform(image)

        return image, target
