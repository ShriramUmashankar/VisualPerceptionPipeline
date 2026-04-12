"""Reusable custom layers 
"""

import torch
import torch.nn as nn


class CustomDropout(nn.Module):
    """Channel-wise (spatial) dropout."""

    def __init__(self, p: float = 0.5):
        """
        Initialize the CustomDropout layer.

        Args:
            p: Dropout probability.
        """

        super().__init__()

        if not 0.0 <= p < 1.0:
            raise ValueError("p must be in [0, 1).")

        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the CustomDropout layer.

        Args:
            x: Input tensor for shape [B, C, H, W].

        Returns:
            Output tensor.
        """

        # During evaluation or if p = 0 - do nothing
        if not self.training or self.p == 0.0:
            return x

        if x.dim() in [1, 2]:
            # Standard element-wise dropout (Handles FC layers after Flatten)
            mask = (torch.rand_like(x) > self.p).float()
            return x * mask / (1.0 - self.p)

        elif x.dim() == 4:
            # Conv-style (channel-wise) dropout
            B, C, H, W = x.shape
            mask = (torch.rand(B, C, 1, 1, device=x.device) > self.p).float()
            return x * mask / (1.0 - self.p)

        else:
            raise ValueError(
                f"Unsupported input shape {x.shape}. Expected 1D or 4D tensor."
            )