from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor


class BrainAgeCNN(nn.Module):
    """
    The BrainAgeCNN predicts the age given a brain MR-image.
    """
    def __init__(self) -> None:
        super().__init__()

        # Feel free to also add arguments to __init__ if you want.
        # ----------------------- ADD YOUR CODE HERE --------------------------
        num_features = torch.prod(
            torch.div(torch.tensor([96] * 3), 4, rounding_mode='floor') - 3)
        self.conv1 = nn.Conv3d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv3d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * num_features, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        # ------------------------------- END ---------------------------------

    def forward(self, imgs: Tensor) -> Tensor:
        """
        Forward pass of your model.

        :param imgs: Batch of input images. Shape (N, 1, H, W, D)
        :return pred: Batch of predicted ages. Shape (N)
        """
        # ----------------------- ADD YOUR CODE HERE --------------------------
        pred = F.max_pool3d(F.relu(self.conv1(imgs)), 2)
        pred = F.max_pool3d(F.relu(self.conv2(pred)), 2)
        pred = pred.view(pred.size(0), -1)
        pred = F.relu(self.fc1(pred))
        pred = F.relu(self.fc2(pred))
        pred = self.fc3(pred).view(-1)
        # ------------------------------- END ---------------------------------
        return pred

    def train_step(
        self,
        imgs: Tensor,
        labels: Tensor,
        return_prediction: Optional[bool] = False
    ):
        """Perform a training step. Predict the age for a batch of images and
        return the loss.

        :param imgs: Batch of input images (N, 1, H, W, D)
        :param labels: Batch of target labels (N)
        :return loss: The current loss, a single scalar.
        :return pred
        """
        pred = self(imgs)  # (N)

        # ----------------------- ADD YOUR CODE HERE --------------------------
        loss = torch.pow((pred - labels), 2).mean()
        # ------------------------------- END ---------------------------------

        if return_prediction:
            return loss, pred
        else:
            return loss
