"""Classification components
"""

import torch
import torch.nn as nn
from models.vgg11 import VGG11Encoder  
from models.layers import CustomDropout

class VGG11Classifier(nn.Module):
    """Full classifier = VGG11Encoder + ClassificationHead."""

    def __init__(self, num_classes: int = 37, in_channels: int = 3, dropout_p: float = 0.5):
        """
        Initialize the VGG11Classifier model.
        Args:
            num_classes: Number of output classes.
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the classifier head.
        """
        super().__init__()
        # Encoder
        self.encoder = VGG11Encoder(in_channels=in_channels, dropout_p= dropout_p)

        # Adaptive pooling to remove spatial dependency
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            CustomDropout(p= 0.5),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            CustomDropout(p= 0.5),
            nn.Linear(128, num_classes),
        )
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for classification model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].
        Returns:
            Classification logits [B, num_classes].
        """
        # Extract features
        x = self.encoder(x)

        # Pool to fixed size
        x = self.pool(x)

        # Flatten
        x = torch.flatten(x, 1)

        # Classification head
        x = self.classifier(x)

        return x
  