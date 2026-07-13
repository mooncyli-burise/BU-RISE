import torch
import torch.nn as nn

class PoseLossFunction(nn.Module):
    def __init__(self):
        super().__init__()
        # 64 classes = 16 grid cells * 4 right-angle orientation bins.
        A = torch.zeros(64, 64)

        for i in range(64):
            for j in range(64):
                pose_i, orientation_i = divmod(i, 4)
                pose_j, orientation_j = divmod(j, 4)

                xi, yi = pose_i % 4, pose_i // 4
                xj, yj = pose_j % 4, pose_j // 4

                position_cost = abs(xi - xj) + abs(yi - yj)
                orientation_cost = min(
                    abs(orientation_i - orientation_j),
                    4 - abs(orientation_i - orientation_j),
                )
                A[i, j] = position_cost + orientation_cost
        self.register_buffer("A", A)

    def forward(self, predictions, targets):
        # predictions: [batch, 64], targets: [batch] with values 0..63
        probs = torch.softmax(predictions, dim=1)

        losses = []
        for i in range(len(targets)):
            #select column of similarity matrix corresponding to target heading
            cost = self.A[:, targets[i]]
            losses.append(torch.pow(torch.dot(probs[i], cost), 2))
        return torch.stack(losses).mean()
