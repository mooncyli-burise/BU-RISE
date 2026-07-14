import torch
import torch.nn as nn
from config import X_CLASSES, Y_CLASSES, ANGLE_CLASSES, TOTAL_CLASSES

class PoseLossFunction(nn.Module):
    def __init__(self):
        super().__init__()

        # 64 classes = 16 grid cells * 4 right-angle orientation bins.
        A = torch.zeros(TOTAL_CLASSES, TOTAL_CLASSES)

        for i in range(TOTAL_CLASSES):
            for j in range(TOTAL_CLASSES):
                pose_i, orientation_i = divmod(i, ANGLE_CLASSES)
                pose_j, orientation_j = divmod(j, ANGLE_CLASSES)

                xi, yi = pose_i % X_CLASSES, pose_i // X_CLASSES
                xj, yj = pose_j % X_CLASSES, pose_j // X_CLASSES

                position_cost = abs(xi - xj) + abs(yi - yj)
                orientation_cost = min(
                    abs(orientation_i - orientation_j),
                    ANGLE_CLASSES - abs(orientation_i - orientation_j),
                )
                A[i, j] = position_cost + orientation_cost
        self.register_buffer("A", A)
        # print(A)

    def forward(self, predictions, targets):
        # predictions: [batch, 64], targets: [batch] with values 0..63
        probs = torch.softmax(predictions, dim=1)

        losses = []
        for i in range(len(targets)):
            #select column of similarity matrix corresponding to target heading
            cost = self.A[:, targets[i]]
            losses.append(torch.dot(probs[i], cost))
        return torch.stack(losses).mean()
