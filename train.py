# """
# Training entrypoint (modular for multi-task extension)
# """

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import f1_score, accuracy_score

import wandb
from sklearn.metrics import f1_score

from models.classification import VGG11Classifier
from models.localization import VGG11Localizer

from losses import IoULoss
from data.pets_dataset import OxfordIIITPetDataset


# =========================
# DATA
# =========================
def get_data_loaders(root, task, batch_size=32):
    train_dataset = OxfordIIITPetDataset(
        root=root,
        split="trainval",
        task=task
    )

    val_dataset =  OxfordIIITPetDataset(
        root=root,
        split="test",
        task=task
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers= 8, pin_memory=True)

    return train_loader, val_loader


# =========================
# MODEL
# =========================
def build_model(task, config):
    if task == "classification":
        return VGG11Classifier(
            dropout_p=config["dropout_p"],
        )

    # Future tasks
    elif task == "localization":
        return VGG11Localizer(
            dropout_p=config["dropout_p"],
            checkpoint_path="checkpoints/classification.pth"
        )

    elif task == "segmentation":
        raise NotImplementedError

    else:
        raise ValueError("Invalid task")


# =========================
# LOSS FACTORY
# =========================
def get_loss(task):
    if task == "classification":
        return nn.CrossEntropyLoss()

    elif task == "localization":
        return IoULoss(reduction="mean")

    elif task == "segmentation":
        raise NotImplementedError

# =========================
# TRAIN STEP
# =========================
def train_one_epoch(model, loader, criterion, optimizer, device, task):
    model.train()
    
    total_loss = 0
    total_iou = 0
    all_preds, all_labels = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        output = model(x)
        loss = criterion(output, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if task == "classification":
            preds = torch.argmax(output, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

        if task == "localization":
            # Preds and targets are [B, 4]
            # Calculate IoU for reporting [cite: 108]
            batch_iou = 1 - loss.item() 
            total_iou += batch_iou     

    avg_loss = total_loss / len(loader)

    if task == "classification":
        f1 = f1_score(all_labels, all_preds, average="macro")
        acc = accuracy_score(all_labels, all_preds)
        return avg_loss, {"f1": f1, "accuracy": acc}

    # Return dict update:
    if task == "localization":
        avg_iou = total_iou / len(loader)
        return avg_loss, {"iou": avg_iou}  

    return avg_loss, {}


# =========================
# VALIDATION
# =========================
def validate(model, loader, criterion, device, task):
    model.eval()

    total_loss = 0
    total_iou = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            output = model(x)
            loss = criterion(output, y)

            total_loss += loss.item()

            if task == "classification":
                preds = torch.argmax(output, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

            if task == "localization":
                # Preds and targets are [B, 4]
                # Calculate IoU for reporting [cite: 108]
                batch_iou = 1 - loss.item() 
                total_iou += batch_iou    

    avg_loss = total_loss / len(loader)

    if task == "classification":
        f1 = f1_score(all_labels, all_preds, average="macro")
        acc = accuracy_score(all_labels, all_preds)
        return avg_loss, {"f1": f1, "accuracy": acc}
    

    # Return dict update:
    if task == "localization":
        avg_iou = total_iou / len(loader)
        return avg_loss, {"iou": avg_iou}

    return avg_loss, {}

# def get_lr_scheduler(optimizer, num_epochs):
#     return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# =========================
# MAIN TRAIN LOOP
# =========================

def train(config):
    wandb.init(
        project="DA6401_Assignment_2",
        group=config["task"],
        name=config["run_name"],
        config=config
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device being used is:", device)

    model = build_model(config["task"], config).to(device)

    train_loader, val_loader = get_data_loaders(
        root=config["data_root"],
        task=config["task"],
        batch_size=config["batch_size"]
    )

    criterion = get_loss(config["task"])

    # filter() ensures we only optimize the head if the backbone is frozen (Task 2) 
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=config["lr"], 
        weight_decay=1e-4 
    )

    best_metric = -1

    for epoch in range(config["epochs"]):
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, config["task"]
        )

        val_loss, val_metrics = validate(
            model, val_loader, criterion, device, config["task"]
        )

        log_dict = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }

        # Add task-specific metrics to log_dict
        for k, v in train_metrics.items():
            log_dict[f"train_{k}"] = v
        for k, v in val_metrics.items():
            log_dict[f"val_{k}"] = v

        wandb.log(log_dict)
        print(f"Epoch {epoch}:", log_dict)

        # =========================
        # TASK-SPECIFIC SAVING LOGIC
        # =========================
        if config["task"] == "classification":
            # Assignment requires Macro F1-Score 
            current_metric = val_metrics.get("f1", -1)
            
        elif config["task"] == "localization":
            # Primary training metric for detection is IoU 
            current_metric = val_metrics.get("iou", -1)
            
        elif config["task"] == "segmentation":
            # Assignment requires Dice Similarity Coefficient [cite: 79]
            current_metric = val_metrics.get("dice", -1)
        else:
            current_metric = -1

        # Save checkpoint if performance improved
        if current_metric > best_metric:
            best_metric = current_metric
            
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "best_metric": best_metric,
                    "config": config
                },
                    
                f"checkpoints/{config['task']}.pth"
            )

    print(f"Training Complete. Best {config['task']} metric: {best_metric}")


# =========================
# ENTRYPOINT
# =========================
if __name__ == "__main__":
    config = {
        "task": "localization",
        "run_name": "test",

        "data_root": "./data",
        "batch_size": 64,
        "epochs": 25,
        "lr": 3e-5,
        
        "dropout_p": 0.4,
    }

    train(config)


# SANITY CHECK (ROUGH)

'''
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from data.pets_dataset import OxfordIIITPetDataset  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def denormalize(img):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    return img * std + mean

def show_sample(task):
    dataset = OxfordIIITPetDataset(root="./data", task=task)

    sample = dataset[10]

    if task == "classification":
        image, label = sample

        img = denormalize(image).permute(1, 2, 0).numpy()

        plt.imshow(img)
        plt.title(f"Class: {label.item()}")
        plt.axis("off")
        plt.show()

    elif task == "segmentation":
        image, mask = sample

        img = denormalize(image).permute(1, 2, 0).numpy()
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title("Image")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(mask.numpy(), cmap="gray")
        plt.title("Segmentation Mask")
        plt.axis("off")

        plt.show()

    elif task == "localization":
        image, bbox = sample

        img = denormalize(image).permute(1, 2, 0).numpy()

        fig, ax = plt.subplots(1)
        ax.imshow(img)

        # bbox is [xc, yc, w, h] (possibly normalized)
        H, W = img.shape[:2]

        xc, yc, w, h = bbox

        # if normalized, convert to pixel space
        if xc <= 1:
            xc *= W
            yc *= H
            w *= W
            h *= H

        x_min = xc - w / 2
        y_min = yc - h / 2

        rect = patches.Rectangle(
            (x_min, y_min),
            w,
            h,
            linewidth=2,
            edgecolor='r',
            facecolor='none'
        )

        ax.add_patch(rect)
        plt.title("Bounding Box")
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    print("Checking classification...")
    show_sample("classification")

    print("Checking segmentation...")
    show_sample("segmentation")

    print("Checking localization...")
    show_sample("localization")
'''

## Some more rought check code 

'''

# # only if retraining old model
    # ckpt = torch.load("checkpoints/classification.pth", map_location=device)
    # # ======================================================================
    # old_sd = ckpt["state_dict"]

    # new_sd = {}

    # mapping = [
    #     # block1
    #     ("encoder.0", "encoder.block1.0"),
    #     ("encoder.1", "encoder.block1.1"),

    #     # block2
    #     ("encoder.4", "encoder.block2.0"),
    #     ("encoder.5", "encoder.block2.1"),

    #     # block3
    #     ("encoder.8",  "encoder.block3.0.0"),
    #     ("encoder.9",  "encoder.block3.0.1"),
    #     ("encoder.11", "encoder.block3.1.0"),
    #     ("encoder.12", "encoder.block3.1.1"),

    #     # block4
    #     ("encoder.15", "encoder.block4.0.0"),
    #     ("encoder.16", "encoder.block4.0.1"),
    #     ("encoder.18", "encoder.block4.1.0"),
    #     ("encoder.19", "encoder.block4.1.1"),

    #     # block5
    #     ("encoder.22", "encoder.block5.0.0"),
    #     ("encoder.23", "encoder.block5.0.1"),
    #     ("encoder.25", "encoder.block5.1.0"),
    #     ("encoder.26", "encoder.block5.1.1"),
    # ]

    # for old_prefix, new_prefix in mapping:
    #     for key in old_sd:
    #         if key == old_prefix or key.startswith(old_prefix + "."):
    #             new_key = key.replace(old_prefix, new_prefix)
    #             new_sd[new_key] = old_sd[key]

    # # also copy classifier weights directly
    # for key in old_sd:
    #     if not key.startswith("encoder."):
    #         new_sd[key] = old_sd[key]

    # missing, unexpected = model.load_state_dict(new_sd, strict=False)

    # print("Missing keys:", missing)
    # print("Length of missing keys:", len(missing))
    # print("Unexpected keys:", unexpected)
    # print("Length of unexpected keys:", len(unexpected))

'''