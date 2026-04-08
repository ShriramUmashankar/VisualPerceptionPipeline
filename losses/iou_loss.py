"""Custom IoU loss 
"""

"""Custom IoU loss 
"""

import torch
import torch.nn as nn

class IoULoss(nn.Module):
    """IoU loss for bounding box regression.
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        """
        Initialize the IoULoss module.
        Args:
            eps: Small value to avoid division by zero.
            reduction: Specifies the reduction to apply to the output: 'mean' | 'sum'.
        """
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        # TODO: validate reduction in {"none", "mean", "sum"}.
        if self.reduction not in {"none", "mean", "sum"}:
            raise ValueError(f"Invalid reduction: {self.reduction}")

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Compute IoU loss between predicted and target bounding boxes.
        Args:
            pred_boxes: [B, 4] predicted boxes in (x_center, y_center, width, height) format.
            target_boxes: [B, 4] target boxes in (x_center, y_center, width, height) format."""
        # TODO: implement IoU loss.

        # Convert (xc, yc, w, h) -> (x1, y1, x2, y2)
        pred_xc, pred_yc, pred_w, pred_h = pred_boxes.unbind(dim=1)
        tgt_xc, tgt_yc, tgt_w, tgt_h = target_boxes.unbind(dim=1)

        pred_x1 = pred_xc - pred_w / 2
        pred_y1 = pred_yc - pred_h / 2
        pred_x2 = pred_xc + pred_w / 2
        pred_y2 = pred_yc + pred_h / 2

        tgt_x1 = tgt_xc - tgt_w / 2
        tgt_y1 = tgt_yc - tgt_h / 2
        tgt_x2 = tgt_xc + tgt_w / 2
        tgt_y2 = tgt_yc + tgt_h / 2

        # Intersection
        inter_x1 = torch.max(pred_x1, tgt_x1)
        inter_y1 = torch.max(pred_y1, tgt_y1)
        inter_x2 = torch.min(pred_x2, tgt_x2)
        inter_y2 = torch.min(pred_y2, tgt_y2)

        inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
        inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
        inter_area = inter_w * inter_h

        # Areas
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        tgt_area = (tgt_x2 - tgt_x1) * (tgt_y2 - tgt_y1)

        # Union
        union = pred_area + tgt_area - inter_area

        # IoU
        iou = inter_area / (union + self.eps)

        # Loss
        loss = 1 - iou

        # Reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # "none"
            return loss