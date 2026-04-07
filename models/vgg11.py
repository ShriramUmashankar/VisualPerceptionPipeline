"""VGG11 encoder
"""

from typing import Dict, Tuple, Union
import torch
import torch.nn as nn


class VGG11Encoder(nn.Module):
    """VGG11-style encoder with optional intermediate feature returns."""

    def __init__(self, in_channels: int = 3, bn: bool = True):
        super().__init__()

        # NOTE : Deafult Stride = 1, Padding = 1 ( As per paper)

        self.batch_norm = bn

        def make_conv_layer(in_c, out_c):
            layers = []
            # bias is False if batch_norm is True
            layers.append(nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=not self.batch_norm))
            
            if self.batch_norm:
                layers.append(nn.BatchNorm2d(out_c))
                
            layers.append(nn.ReLU(inplace=True))
            return nn.Sequential(*layers)

        # -------- Block 1 --------
        self.block1 = make_conv_layer(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2, 2)

        # -------- Block 2 --------
        self.block2 = make_conv_layer(64, 128)
        self.pool2 = nn.MaxPool2d(2, 2)

        # -------- Block 3 --------
        self.block3 = nn.Sequential(
            make_conv_layer(128, 256),
            make_conv_layer(256, 256)
        )
        self.pool3 = nn.MaxPool2d(2, 2)

        # -------- Block 4 --------
        self.block4 = nn.Sequential(
            make_conv_layer(256, 512),
            make_conv_layer(512, 512)
        )
        self.pool4 = nn.MaxPool2d(2, 2)

        # -------- Block 5 --------
        self.block5 = nn.Sequential(
            make_conv_layer(512, 512),
            make_conv_layer(512, 512)
        )
        self.pool5 = nn.MaxPool2d(2, 2)


    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:

        features: Dict[str, torch.Tensor] = {}

        # -------- Block 1 --------
        x = self.block1(x)
        if return_features:
            features["block1"] = x
        x = self.pool1(x)

        # -------- Block 2 --------
        x = self.block2(x)
        if return_features:
            features["block2"] = x
        x = self.pool2(x)

        # -------- Block 3 --------
        x = self.block3(x)
        if return_features:
            features["block3"] = x
        x = self.pool3(x)

        # -------- Block 4 --------
        x = self.block4(x)
        if return_features:
            features["block4"] = x
        x = self.pool4(x)

        # -------- Block 5 --------
        x = self.block5(x)
        x = self.dropout(x)  # easy hook point
        if return_features:
            features["block5"] = x
        x = self.pool5(x)

        bottleneck = x  # [B, 512, H/32, W/32]

        if return_features:
            return bottleneck, features
        else:
            return bottleneck    