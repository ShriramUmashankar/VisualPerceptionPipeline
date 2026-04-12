"""Segmentation model
"""

import torch
import torch.nn as nn
from models.classification import VGG11Classifier
from models.layers import CustomDropout

class VGG11UNet(nn.Module):
    """U-Net style segmentation network.
    """

    def __init__(self, num_classes: int = 3, in_channels: int = 3, dropout_p: float = 0.5, 
                 checkpoint_path: str = None, freeze_strategy: str = "strict"):
        """
        Initialize the VGG11UNet model.
        Args:
            freeze_strategy: "strict" (all frozen), "partial" (unfreeze block 4 & 5), "none" (all trainable)
        """
        super(VGG11UNet, self).__init__()

        base_vgg = VGG11Classifier(in_channels=in_channels)
        
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            base_vgg.load_state_dict(checkpoint["state_dict"])

        self.enc1 = base_vgg.encoder.block1
        self.enc2 = base_vgg.encoder.block2
        self.enc3 = base_vgg.encoder.block3
        self.enc4 = base_vgg.encoder.block4
        self.enc5 = base_vgg.encoder.block5
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self._apply_freeze(freeze_strategy)


        self.up4 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec4 = self._make_dec_block(512 + 512, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self._make_dec_block(256 + 256, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self._make_dec_block(128 + 128, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self._make_dec_block(64 + 64, 64)

        # Final Segmentation Head
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        self.dropout = CustomDropout(dropout_p)

    def _apply_freeze(self, strategy):
        blocks = [self.enc1, self.enc2, self.enc3, self.enc4, self.enc5]
        if strategy == "strict":
            for block in blocks:
                for p in block.parameters(): p.requires_grad = False
        elif strategy == "partial":
            # Freeze first 3 blocks, leave block 4 and 5 trainable
            for block in blocks[:3]:
                for p in block.parameters(): p.requires_grad = False
        # "none" leaves everything with requires_grad=True by default

    def _make_dec_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder Path
        s1 = self.enc1(x)
        p1 = self.pool(s1)
        s2 = self.enc2(p1)
        p2 = self.pool(s2)
        s3 = self.enc3(p2)
        p3 = self.pool(s3)
        s4 = self.enc4(p3)
        p4 = self.pool(s4)
        b = self.enc5(p4)

        # Decoder Path with Skip Connections
        d4 = self.up4(b)
        d4 = torch.cat([d4, s4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, s3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, s2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, s1], dim=1)
        d1 = self.dec1(d1)

        out = self.dropout(d1)
        return self.final_conv(out)