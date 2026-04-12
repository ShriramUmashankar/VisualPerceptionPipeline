"""Custom Dice loss for segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """Dice loss for multi-class semantic segmentation.
    """

    def __init__(self, eps: float = 1e-6):
        """
        Initialize the DiceLoss module.

        Args:
            eps: Small value to avoid division by zero.
        """
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Dice loss.
        Args:
            logits: [B, num_classes, H, W] raw network outputs.
            targets: [B, H, W] ground truth labels.

        Returns:
            Scalar dice loss.
        """
        num_classes = logits.shape[1]
        
        probs = F.softmax(logits, dim=1)

        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

        probs = probs.contiguous().view(probs.shape[0], num_classes, -1)
        targets_one_hot = targets_one_hot.contiguous().view(targets_one_hot.shape[0], num_classes, -1)
        

        intersection = torch.sum(probs * targets_one_hot, dim=2)
        cardinality = torch.sum(probs + targets_one_hot, dim=2)
        
        dice_coeff = (2. * intersection + self.eps) / (cardinality + self.eps)

        loss = 1. - dice_coeff.mean()
        
        return loss