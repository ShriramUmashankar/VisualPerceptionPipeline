"""Dataset skeleton for Oxford-IIIT Pet.
"""

import torch
from torch.utils.data import Dataset
from torchvision.datasets import OxfordIIITPet
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import numpy as np


class OxfordIIITPetDataset(Dataset):
    def __init__(self, root, split="trainval", task="classification"):
        """
        task: 'classification' | 'localization' | 'segmentation'
        """
        self.task = task

        # Image transform
        self.img_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Mask transform (IMPORTANT: nearest interpolation)
        self.mask_transform = transforms.Resize(
            (224, 224),
            interpolation=InterpolationMode.NEAREST
        )

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


        # foreground = anything not background (usually >0)
        pos = mask == 0

        if pos.sum() == 0:
            return torch.tensor([0, 0, 0, 0], dtype=torch.float32)

        coords = pos.nonzero()
        y_min, x_min = coords.min(dim=0)[0]
        y_max, x_max = coords.max(dim=0)[0]

        x_center = (x_min + x_max).float() / 2
        y_center = (y_min + y_max).float() / 2
        width    = (x_max - x_min).float()
        height   = (y_max - y_min).float()

        return torch.tensor([x_center, y_center, width, height], dtype=torch.float32)

    def __getitem__(self, idx):
        image, (label, mask) = self.dataset[idx]

        # Apply transforms
        image = self.img_transform(image)
        mask = self.mask_transform(mask)

        mask = torch.from_numpy(np.array(mask)).long()

        mask = mask - 1  # convert 1,2,3 → 0,1,2

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