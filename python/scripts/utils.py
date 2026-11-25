import torch
import torch.nn.functional as F
from scripts.constantes import GRID_HEIGHT, GRID_WIDTH, MEAN_NUM_ENTITIES_PER_IMAGE
from torch import nn


# https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2
        )
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2
        )
        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2
        )
        self.conv4 = nn.Conv2d(
            in_channels=64, out_channels=8, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        x = x.permute(0, 3, 2, 1)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.conv4(x)
        x = x.permute(0, 3, 2, 1)
        return x


# https://medium.com/@mustafa.kamalzhd/writing-a-custom-loss-in-pytorch-part-1-7dd857934f48
class CustomLoss(nn.Module):
    def __init__(
        self,
        pos_weight=(GRID_WIDTH * GRID_HEIGHT - MEAN_NUM_ENTITIES_PER_IMAGE)
        / MEAN_NUM_ENTITIES_PER_IMAGE,
    ):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()

    def forward(self, predictions, targets):
        return (
            self.bce(predictions[:, :, :, 0], targets[:, :, :, 0])
            + self.mse(predictions[:, :, :, 1:5], targets[:, :, :, 1:5])
            + self.ce(predictions[:, :, :, -3], targets[:, :, :, -3])
        )
