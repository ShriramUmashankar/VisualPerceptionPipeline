"""Reusable custom layers 
"""

import torch
import torch.nn as nn


class CustomDropout(nn.Module):
    """Channel-wise (spatial) dropout."""

    def __init__(self, p: float = 0.5):
        super().__init__()

        if not 0.0 <= p < 1.0:
            raise ValueError("p must be in [0, 1).")

        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # During evaluation or if p = 0 -- do nothing
        if not self.training or self.p == 0.0:
            return x

        B, C, H, W = x.shape

        # Create channel-wise mask
        # Shape: [B, C, 1, 1] → broadcast over H, W
        mask = (torch.rand(B, C, 1, 1, device=x.device) > self.p).float()

        # Apply mask and scale
        x = x * mask / (1.0 - self.p)

        return x