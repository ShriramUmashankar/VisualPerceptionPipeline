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
from models.segmentation import VGG11UNet

from losses.iou_loss import IoULoss
from losses.dice_loss import DiceLoss
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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

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
        return VGG11UNet(
            dropout_p=config["dropout_p"],
            checkpoint_path="checkpoints/classification.pth",
            freeze_strategy= config["unet_freeze_strategy"],
        )

    else:
        raise ValueError("Invalid task")

def calculate_pixel_accuracy(output, mask):
    with torch.no_grad():
        preds = torch.argmax(output, dim=1)
        correct = (preds == mask).sum().item()
        total = mask.numel()
        return correct / total
    
# =========================
# LOSS FACTORY
# =========================
def get_loss(task):
    if task == "classification":
        return nn.CrossEntropyLoss()

    elif task == "localization":
        return {
            "iou": IoULoss(reduction="mean"),
            "mse": nn.MSELoss(),
            "smooth": nn.SmoothL1Loss()

        }

    elif task == "segmentation":
        return DiceLoss()

# =========================
# TRAIN STEP
# =========================
def train_one_epoch(model, loader, criterion, optimizer, device, task):
    model.train()
    
    total_loss = 0
    total_iou = 0
    total_dice = 0
    total_pix_acc = 0
    all_preds, all_labels = [], []

    for x, y in loader:
        x = x.to(device)

        if task == "segmentation":
            y = y.long().to(device)
            if y.dim() == 4 and y.shape[1] == 1:
                y = y.squeeze(1)

            y = y - 1
            y[y < 0] = 1
            y[y > 2] = 1

        elif task == "classification":
            # CrossEntropy expects long integers
            y = y.long().to(device)
        elif task == "localization":
            # Bounding box coordinates must remain floating point decimals
            y = y.float().to(device)

        optimizer.zero_grad()

        output = model(x)

        if task == "localization":
            # criterion is now our dict from get_loss
            output_norm = output / 224.0
            y_norm = y / 224.0
            loss_iou = criterion["iou"](output, y)
            loss_smooth = criterion["smooth"](output_norm, y_norm)
            # Combine them: 5.0 is a good starting weight for IoU
            loss = (15.0 * loss_smooth) + (1.0 * loss_iou)
        else:
            loss = criterion(output, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if task == "classification":
            preds = torch.argmax(output, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

        elif task == "localization":
            # Since IoULoss = 1 - IoU, then IoU = 1 - loss_iou
            current_iou = 1.0 - loss_iou.item()
            total_iou += current_iou 

        elif task == "segmentation":
            # Metric for segmentation is 1 - DiceLoss
            total_dice += (1.0 - loss.item())    
            total_pix_acc += calculate_pixel_accuracy(output, y)  

    avg_loss = total_loss / len(loader)

    if task == "classification":
        f1 = f1_score(all_labels, all_preds, average="macro")
        acc = accuracy_score(all_labels, all_preds)
        return avg_loss, {"f1": f1, "accuracy": acc}

    # Return dict update:
    elif task == "localization":
        avg_iou = total_iou / len(loader)
        return avg_loss, {"iou": avg_iou}  
    
    elif task == "segmentation":
        return avg_loss, {"dice": total_dice / len(loader), "pixel_acc": total_pix_acc / len(loader)}

    return avg_loss, {}


# =========================
# VALIDATION
# =========================
def validate(model, loader, criterion, device, task):
    model.eval()

    total_loss = 0
    total_iou = 0
    total_dice = 0
    total_pix_acc = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)

            if task == "segmentation":
                y = y.long().to(device)
                if y.dim() == 4 and y.shape[1] == 1:
                    y = y.squeeze(1)

                y = y - 1
                y[y < 0] = 1
                y[y > 2] = 1
                
            elif task == "classification":
                # CrossEntropy expects long integers
                y = y.long().to(device)
            elif task == "localization":
                # Bounding box coordinates must remain floating point decimals
                y = y.float().to(device)

            output = model(x)

            if task == "localization":
                # criterion is now our dict from get_loss
                output_norm = output / 224.0
                y_norm = y / 224.0

                loss_iou = criterion["iou"](output, y)
                loss_smooth = criterion["smooth"](output_norm, y_norm)
                
                loss = (15.0 * loss_smooth) + (1.0 * loss_iou)
            else:
                loss = criterion(output, y)


            total_loss += loss.item()

            if task == "classification":
                preds = torch.argmax(output, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

            elif task == "localization":
                # Since IoULoss = 1 - IoU, then IoU = 1 - loss_iou
                current_iou = 1.0 - loss_iou.item()
                total_iou += current_iou

            elif task == "segmentation":
                total_dice += (1.0 - loss.item())
                total_pix_acc += calculate_pixel_accuracy(output, y)     

    avg_loss = total_loss / len(loader)

    if task == "classification":
        f1 = f1_score(all_labels, all_preds, average="macro")
        acc = accuracy_score(all_labels, all_preds)
        return avg_loss, {"f1": f1, "accuracy": acc}
    

    # Return dict update:
    elif task == "localization":
        avg_iou = total_iou / len(loader)
        return avg_loss, {"iou": avg_iou}
    
    elif task == "segmentation":
        return avg_loss, {"dice": total_dice / len(loader), "pixel_acc": total_pix_acc / len(loader)}

    return avg_loss, {}

# def get_lr_scheduler(optimizer, num_epochs):
#     return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# =========================
# MAIN TRAIN LOOP
# =========================

def train(config):
    wandb.init(project="DA6401_Assignment_2", 
               group=config["task"], 
               name=config["run_name"], 
               config=config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device being used is: ", device)
    
    model = build_model(config["task"], config).to(device)

    checkpoint = torch.load("checkpoints/unet.pth", map_location=device)
    model.load_state_dict(checkpoint["state_dict"])

    train_loader, val_loader = get_data_loaders(root=config["data_root"], 
                                                task=config["task"], 
                                                batch_size=config["batch_size"])
    
    criterion = get_loss(config["task"])
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, 
                                  model.parameters()), 
                                  lr=config["lr"], 
                                  weight_decay=1e-4)

    best_metric = -1
    # Checkpoint name override for segmentation
    ckpt_name = "unet.pth" if config["task"] == "segmentation" else f"{config['task']}.pth"

    for epoch in range(config["epochs"]):
        t_loss, t_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, config["task"])
        v_loss, v_metrics = validate(model, val_loader, criterion, device, config["task"])

        log_dict = {"epoch": epoch, "train_loss": t_loss, "val_loss": v_loss}
        log_dict.update({f"train_{k}": v for k, v in t_metrics.items()})
        log_dict.update({f"val_{k}": v for k, v in v_metrics.items()})

        wandb.log(log_dict)
        print(f"Epoch {epoch}:", log_dict)
        
        # Saving Logic
        if config["task"] == "classification": current_metric = v_metrics.get("f1", -1)
        elif config["task"] == "localization": current_metric = v_metrics.get("iou", -1)
        elif config["task"] == "segmentation": current_metric = v_metrics.get("dice", -1)

        if current_metric > best_metric:
            best_metric = current_metric
            torch.save({
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "best_metric": best_metric,
            }, f"checkpoints/{ckpt_name}")

    print(f"Training Complete. Best {config['task']} metric: {best_metric}")


# =========================
# ENTRYPOINT
# =========================
if __name__ == "__main__":
    config = {
        "task": "segmentation",
        "run_name": "run-2",

        "data_root": "./data",
        "batch_size": 16,
        "epochs": 30,
        "lr": 5e-5,
        
        "dropout_p": 0.2,

        "unet_freeze_strategy": "strict", # "strict", "partial", or "none"
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
