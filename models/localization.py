"""Localization modules
"""

import torch
import torch.nn as nn
from models.classification import VGG11Classifier
from models.layers import CustomDropout

class VGG11Localizer(nn.Module):
    """VGG11-based localizer."""

    def __init__(self, in_channels: int = 3, dropout_p: float = 0.5, checkpoint_path=None, freeze: bool = True):
        """
        Initialize the VGG11Localizer model.

        Args:
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the localization head.
        """
        super(VGG11Localizer, self).__init__()
        base_model = VGG11Classifier(dropout_p= 0.0)

        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            base_model.load_state_dict(checkpoint["state_dict"])
            print(f"Loaded Task 1 weights (F1: {checkpoint['best_metric']:.4f})")
        
        # Isolate encoders
        self.encoder = base_model.encoder

        # Freeze Backbone as per requirements
        for param in self.encoder.parameters():
            if freeze:
                param.requires_grad = False
            else:
                param.requires_grad = True


        self.pool = nn.AdaptiveAvgPool2d((7, 7))

        self.localization_head = nn.Sequential(
            nn.Flatten(),

            nn.Linear(512 * 7 * 7, 1024), 
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p), 


            nn.Linear(1024, 512), 
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p), 

            nn.Linear(512, 256), 
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            
            # Final Output [cx, cy, w, h]
            nn.Linear(256, 4)         
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for localization model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Bounding box coordinates [B, 4] in (x_center, y_center, width, height) format in original image pixel space(not normalized values).
        """
        # TODO: Implement forward pass.
        
        x = self.encoder(x)
        x = self.pool(x)
        x= self.localization_head(x)
        

        cxcy = torch.sigmoid(x[:, :2]) * 224.0   # center in image space
        wh   = torch.relu(x[:, 2:])             

        bboxes = torch.cat([cxcy, wh], dim=1)
    
        return bboxes
