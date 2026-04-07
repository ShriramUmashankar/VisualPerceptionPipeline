# """Training entrypoint
# """










# SANITY CHECK (ROUGH)

'''
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from data.pets_dataset import OxfordIIITPetDataset  # your dataset file

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
