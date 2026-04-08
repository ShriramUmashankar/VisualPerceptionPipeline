# import torch
# from torch.utils.data import DataLoader
# from sklearn.metrics import f1_score, classification_report

# from models.classification import VGG11Classifier
# from data.pets_dataset import OxfordIIITPetDataset


# def evaluate():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Load model
#     model = VGG11Classifier(
#         num_classes=37,
#         in_channels=3,
#         dropout_p=0.0
#     ).to(device)

#     checkpoint = torch.load("checkpoints/classification.pth", map_location=device)
#     model.load_state_dict(checkpoint["state_dict"])
#     model.eval()


#     # Dataset
#     test_dataset = OxfordIIITPetDataset(
#         root="./data",
#         split="test",
#         task="classification"
#     )

#     test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

#     all_preds = []
#     all_labels = []

#     with torch.no_grad():
#         for x, y in test_loader:
#             x, y = x.to(device), y.to(device)

#             logits = model(x)
#             preds = torch.argmax(logits, dim=1)

#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(y.cpu().numpy())

#     # Metrics
#     f1 = f1_score(all_labels, all_preds, average="macro")

#     print(f"\nMacro F1: {f1:.4f}\n")

#     print("Classification Report:\n")
#     print(classification_report(all_labels, all_preds))


# if __name__ == "__main__":
#     evaluate()




import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
from models.localization import VGG11Localizer
from data.pets_dataset import OxfordIIITPetDataset

def draw_bbox(image, bbox, color, label):
    """
    bbox format: [xc, yc, w, h] in 224x224 space
    """
    # Convert back to (x1, y1, x2, y2)
    xc, yc, w, h = bbox
    x1 = int(xc - w/2)
    y1 = int(yc - h/2)
    x2 = int(xc + w/2)
    y2 = int(yc + h/2)
    
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def run_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Setup Model (Ensure dropout_p matches your best training run)
    model = VGG11Localizer(dropout_p=0.0).to(device)
    checkpoint = torch.load("checkpoints/localization.pth", map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # 2. Get Data (using test split for fairness)
    dataset = OxfordIIITPetDataset(root="./data", split="test", task="localization")
    
    # 3. Pick 10 random indices
    indices = random.sample(range(len(dataset)), 10)
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 10))
    axes = axes.flatten()

    with torch.no_grad():
        for i, idx in enumerate(indices):
            image_tensor, gt_bbox = dataset[idx]
            
            # Forward pass
            # Input shape [1, 3, 224, 224]
            pred_bbox = model(image_tensor.unsqueeze(0).to(device)).cpu().squeeze().numpy()
            
            # Prepare image for display
            # Reverse Normalization for visualization (approximate)
            img_disp = image_tensor.permute(1, 2, 0).numpy()
            img_disp = (img_disp * 0.229) + 0.485 # Denormalize mean/std
            img_disp = np.clip(img_disp * 255, 0, 255).astype(np.uint8)
            img_disp = cv2.cvtColor(img_disp, cv2.COLOR_RGB2BGR)

            # Draw boxes
            # Green (0, 255, 0) for Ground Truth
            draw_bbox(img_disp, gt_bbox.numpy(), (0, 255, 0), "GT")
            # Red (0, 0, 255) for Prediction
            draw_bbox(img_disp, pred_bbox, (0, 0, 255), "Pred")

            axes[i].imshow(cv2.cvtColor(img_disp, cv2.COLOR_BGR2RGB))
            axes[i].set_title(f"Image {idx}")
            axes[i].axis('off')

    plt.tight_layout()
    plt.savefig("localization_inference_10.png")
    plt.show()

if __name__ == "__main__":
    run_inference()