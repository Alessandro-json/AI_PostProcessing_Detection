import torch
import torch.nn as nn


class UncertaintyWeightedLoss(nn.Module):
    """
    Learnable uncertainty-based loss weighting for multi-task learning.
    The model learns one log-variance parameter for each task.
    """

    def __init__(self):
        super().__init__()

        # Learnable log-variance for the real/fake task.
        self.log_var_fake = nn.Parameter(torch.zeros(1))

        # Learnable log-variance for the transformation task.
        self.log_var_transform = nn.Parameter(torch.zeros(1))

    def forward(self, fake_loss, transform_loss):
        """
        Combine the two task losses using learnable uncertainty weighting.
        """

        # Convert log-variance parameters into positive task weights.
        weight_fake = torch.exp(-self.log_var_fake)
        weight_transform = torch.exp(-self.log_var_transform)

        # Uncertainty-weighted multi-task loss.
        total_loss = (
            weight_fake * fake_loss + self.log_var_fake
            + weight_transform * transform_loss + self.log_var_transform
        )

        # Values useful for monitoring during training.
        loss_info = {
            "weight_fake": weight_fake.item(),
            "weight_transform": weight_transform.item(),
            "log_var_fake": self.log_var_fake.item(),
            "log_var_transform": self.log_var_transform.item(),
        }

        return total_loss, loss_info