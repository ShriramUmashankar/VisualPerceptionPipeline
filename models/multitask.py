"""Unified multi-task model
"""

import torch
import torch.nn as nn

from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet

class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model."""

    def __init__(self, num_breeds: int = 37, seg_classes: int = 3, in_channels: int = 3, 
                 classifier_path: str = "classifier.pth", 
                 localizer_path: str = "localizer.pth", 
                 unet_path: str = "unet.pth"):
        """
        Initialize the shared backbone/heads using these trained weights.
        """
        super(MultiTaskPerceptionModel, self).__init__()

        import gdown
        gdown.download(id="1uEQSg3SMJ_cREqoymNbn3PixBeAAQNly", output=classifier_path, quiet=False)
        gdown.download(id="19Ac6hc6DAhgCIZEiHUZRVLN7UxXHURIM", output=localizer_path, quiet=False)
        gdown.download(id="1cpfsICndDct4h17KdvqaYCjfdzES6j_u", output=unet_path, quiet=False)
        
        # ==========================================
        # 1. Instantiate Sub-Models & Load Weights
        # ==========================================
        
        clf_model = VGG11Classifier(dropout_p=0.0)
        clf_ckpt = torch.load(classifier_path, map_location='cpu')
        clf_model.load_state_dict(clf_ckpt["state_dict"])

        loc_model = VGG11Localizer(dropout_p=0.0)
        loc_ckpt = torch.load(localizer_path, map_location='cpu')
        loc_model.load_state_dict(loc_ckpt["state_dict"])

        unet_model = VGG11UNet(dropout_p=0.0)
        unet_ckpt = torch.load(unet_path, map_location='cpu')
        unet_model.load_state_dict(unet_ckpt["state_dict"])

        # ==========================================
        # 2. Steal Components for Unified Pipeline
        # ==========================================
        
        # A. SHARED BACKBONE (Use UNet's encoder as it's perfectly split)
        self.enc1 = unet_model.enc1
        self.enc2 = unet_model.enc2
        self.enc3 = unet_model.enc3
        self.enc4 = unet_model.enc4
        self.enc5 = unet_model.enc5
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    

        # B. CLASSIFICATION HEAD
        # 1x1 pool so the flattened dimension is exactly 512
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_pool_loc = nn.AdaptiveAvgPool2d((7,7))
        self.classifier_head = clf_model.classifier

        # C. LOCALIZATION HEAD
        self.localization_head = loc_model.localization_head

        # D. SEGMENTATION DECODER
        self.up4 = unet_model.up4
        self.dec4 = unet_model.dec4
        
        self.up3 = unet_model.up3
        self.dec3 = unet_model.dec3
        
        self.up2 = unet_model.up2
        self.dec2 = unet_model.dec2
        
        self.up1 = unet_model.up1
        self.dec1 = unet_model.dec1
        
        self.seg_final_conv = unet_model.final_conv

        self.eval()


    def forward(self, x: torch.Tensor):
        """Forward pass for multi-task model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].
        Returns:
            A dict with keys:
            - 'classification': [B, num_breeds] logits tensor.
            - 'localization': [B, 4] bounding box tensor.
            - 'segmentation': [B, seg_classes, H, W] segmentation logits tensor
        """
        # ==========================================
        # 1. SHARED BACKBONE (Feature Extraction)
        # ==========================================
        
        s1 = self.enc1(x)
        p1 = self.pool(s1)

        s2 = self.enc2(p1)
        p2 = self.pool(s2)

        s3 = self.enc3(p2)
        p3 = self.pool(s3)

        s4 = self.enc4(p3)
        p4 = self.pool(s4)

        # UNet Bottleneck (14x14 resolution)
        b = self.enc5(p4) 
        
        # ==========================================
        # 2. TASK BRANCHING
        # ==========================================

        vgg_bottleneck = self.pool(b)

        # --- Branch 1: Classification ---
        clf_feat = self.global_pool(vgg_bottleneck)
        clf_feat = torch.flatten(clf_feat, 1)
        classification_logits = self.classifier_head(clf_feat)

        # --- Branch 2: Localization ---
        loc_feat = self.global_pool_loc(vgg_bottleneck) 
        loc_feat = torch.flatten(loc_feat, 1)
        loc_raw = self.localization_head(loc_feat)
        # Split outputs
        cxcy = torch.sigmoid(loc_raw[:, :2]) * 224.0   # center in image space
        wh   = torch.relu(loc_raw[:, 2:])              # width/height must be positive

        bboxes = torch.cat([cxcy, wh], dim=1)
        bounding_boxes = torch.clamp(bboxes, 0, 224)


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

        segmentation_logits = self.seg_final_conv(d1)

        pet_logits = segmentation_logits[:, 0:1, :, :]
        bg_logits  = segmentation_logits[:, 1:2, :, :]
        border_logits = segmentation_logits[:, 2:3, :, :]

        # Recombine them in the order the autograder expects (BG, Pet, Border)
        segmentation_logits = torch.cat([bg_logits, pet_logits, border_logits], dim=1)

        # ==========================================
        # 3. RETURN UNIFIED DICTIONARY
        # ==========================================
        return {
            'classification': classification_logits,
            'localization': bounding_boxes,
            'segmentation': segmentation_logits
        }