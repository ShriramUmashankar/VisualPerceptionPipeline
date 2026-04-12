"""Dataset skeleton for Oxford-IIIT Pet.
"""

import torch
from torch.utils.data import Dataset
from torchvision.datasets import OxfordIIITPet
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class OxfordIIITPetDataset(Dataset):
    def __init__(self, root, split="trainval", task="classification"):
        """
        task: 'classification' | 'localization' | 'segmentation'
        """
        self.task = task
        self.split = split

        # Define Albumentations for Training 
        if self.split == "trainval":
            self.transform = A.Compose([
                A.Resize(224, 224),
                A.HorizontalFlip(p=0.5),
                A.Affine(translate_percent=(-0.05, 0.05), scale=(0.95, 1.05), rotate=(-15, 15), p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),   
            ])
        # Basic Transforms for Testing (test)
        elif self.split == "test":
            self.transform = A.Compose([
                A.Resize(224, 224),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        else:
            raise ValueError(f"Invalid split: {split}. Expected 'trainval' or 'test'.")

        # Load base dataset
        self.dataset = OxfordIIITPet(
            root=root,
            split=split,
            target_types=("category", "segmentation"),
            download=True
        )

    def __len__(self):
        return len(self.dataset)

    def _mask_to_bbox(self, mask):
        """Convert segmentation mask → bounding box"""

        pos = mask == 0

        if pos.sum() == 0:
            return torch.tensor([0, 0, 0, 0], dtype=torch.float32)

        coords = pos.nonzero()
        y_min, x_min = coords.min(dim=0)[0]
        y_max, x_max = coords.max(dim=0)[0]

        x_center = (x_min + x_max).float() / 2
        y_center = (y_min + y_max).float() / 2
        width    = (x_max - x_min + 1).float()
        height   = (y_max - y_min + 1).float()

        return torch.tensor([x_center, y_center, width, height], dtype=torch.float32)

    def __getitem__(self, idx):
        image, (label, mask) = self.dataset[idx]

        image_np = np.array(image.convert("RGB"))
        mask_np = np.array(mask)
        augmented = self.transform(image=image_np, mask=mask_np)
        
        # Extract transformed tensors
        image = augmented['image']
        mask = augmented['mask'].long()

        mask = mask - 1 

        label = torch.tensor(label, dtype=torch.long)

        # TASK SWITCHING
        if self.task == "classification":
            return image, label

        elif self.task == "segmentation":
            return image, mask

        elif self.task == "localization":
            bbox = self._mask_to_bbox(mask)
            return image, bbox

        else:
            raise ValueError("Invalid task type")