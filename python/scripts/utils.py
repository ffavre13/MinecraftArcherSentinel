import numpy as np
import torch
import torch.nn.functional as F
from scripts.constantes import (
    CLASS_ENTITIES,
    GRID_HEIGHT,
    GRID_WIDTH,
    MEAN_NUM_ENTITIES_PER_IMAGE,
    NUM_LABEL_FEATURE,
)
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
            in_channels=64,
            out_channels=NUM_LABEL_FEATURE + len(CLASS_ENTITIES),
            kernel_size=1,
            stride=1,
            padding=0,
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
        # Apply bce on evry predictions but apply mse and ce only if the target grid have an entity
        haveAnEntity = targets[:, :, :, 0] == 1
        if not haveAnEntity.any():
            return self.bce(predictions[:, :, :, 0], targets[:, :, :, 0])

        return (
            self.bce(predictions[:, :, :, 0], targets[:, :, :, 0])
            + self.mse(
                predictions[:, :, :, 1:NUM_LABEL_FEATURE][haveAnEntity],
                targets[:, :, :, 1:NUM_LABEL_FEATURE][haveAnEntity],
            )
            + self.ce(
                predictions[:, :, :, -len(CLASS_ENTITIES) :][haveAnEntity],
                targets[:, :, :, -len(CLASS_ENTITIES) :][haveAnEntity],
            )
        )


def rotation_y(yaw):
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    return np.array([[cos_yaw, 0, sin_yaw], [0, 1, 0], [-sin_yaw, 0, cos_yaw]])


def rotation_x(pitch):
    cos_pitch = np.cos(pitch)
    sin_pitch = np.sin(pitch)

    return np.array([[1, 0, 0], [0, cos_pitch, -sin_pitch], [0, sin_pitch, cos_pitch]])
