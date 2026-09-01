import torch
import torch.nn as nn

class MultiTaskLoss(nn.Module):
    def __init__(self, stain_weights=None, vessel_pos_weight=None, alpha=1.0, beta=2.0):
        """
        Args:
            stain_weights: Tensor of weights for [Background, White, Grey].
            vessel_pos_weight: Tensor weight for the positive vessel class to prevent "No Vessel" default.
            alpha: Weight for the stain loss.
            beta: Weight for the vessel loss (set higher because vessel detection is the primary, harder goal).
        """
        super(MultiTaskLoss, self).__init__()

        self.alpha = alpha
        self.beta = beta

        # CrossEntropyLOss expected logits and int class indecies (0 , 1, 2)
        self.stain_criterion = nn.CrossEntropyLoss(weight=stain_weights)

        # BCEWithLogitsLoss expected logits and float targets (0.0 or 1.0)
        self.vessel_criterion = nn.BCEWithLogitsLoss(pos_weight=vessel_pos_weight)

    def forward(self, z_stain, z_vessel, y_stain, y_vessel):
        # Calculate individual losses
        loss_stain = self.stain_criterion(z_stain, y_stain)

        # Squeeze z_vessel to match y_vessel shape: [B] vs. [B, 1]
        loss_vessel = self.vessel_criterion(z_vessel.squeeze(), y_vessel.float())

        # Combine losses
        total_loss = (self.alpha * loss_stain) + (self.beta * loss_vessel)

        return total_loss, loss_stain, loss_vessel