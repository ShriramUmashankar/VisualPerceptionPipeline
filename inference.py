import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet
from data.pets_dataset import OxfordIIITPetDataset

class InferenceEngine:
    def __init__(self, task="classification", checkpoint_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.task = task
        self.model = self._load_model(checkpoint_path)
        self.model.eval()
        
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def _load_model(self, path):
        if self.task == "classification":
            model = VGG11Classifier(num_classes=37)
            path = path or "checkpoints/classification.pth"
        elif self.task == "localization":
            model = VGG11Localizer()
            path = path or "checkpoints/localization.pth"
        elif self.task == "segmentation":
            model = VGG11UNet(num_classes=3)
            path = path or "checkpoints/unet.pth"
        
        ckpt = torch.load(path, map_location=self.device)
        model.load_state_dict(ckpt["state_dict"])
        return model.to(self.device)


    def get_iou(self, pred, gt):
        p_x1, p_y1 = pred[0] - pred[2]/2, pred[1] - pred[3]/2
        p_x2, p_y2 = pred[0] + pred[2]/2, pred[1] + pred[3]/2
        g_x1, g_y1 = gt[0] - gt[2]/2, gt[1] - gt[3]/2
        g_x2, g_y2 = gt[0] + gt[2]/2, gt[1] + gt[3]/2
        ix1, iy1 = max(p_x1, g_x1), max(p_y1, g_y1)
        ix2, iy2 = min(p_x2, g_x2), min(p_y2, g_y2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = (pred[2] * pred[3]) + (gt[2] * gt[3]) - inter
        return inter / (union + 1e-6)

    def get_dice(self, pred, gt, num_classes=3):
        scores = []
        for c in range(num_classes):
            p, g = (pred == c), (gt == c)
            inter = (p & g).sum()
            total = p.sum() + g.sum()
            if total > 0: scores.append((2.0 * inter) / total)
        return np.mean(scores) if scores else 0.0

    @torch.no_grad()
    def evaluate(self, num_vis=3):
        dataset = OxfordIIITPetDataset(root="./data", split="test", task=self.task)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)

        all_preds, all_gts = [], []
        ious, dices = [], []

        print(f"Gathering metrics for {self.task}...")
        for images, targets in loader:
            images = images.to(self.device)
            out = self.model(images)

            if self.task == "classification":
                all_preds.extend(torch.argmax(out, dim=1).cpu().numpy())
                all_gts.extend(targets.numpy())

            elif self.task == "localization":
                preds, gts = out.cpu().numpy(), targets.numpy()
                for p, g in zip(preds, gts): ious.append(self.get_iou(p, g))

            elif self.task == "segmentation":
                preds = torch.argmax(out, dim=1).cpu().numpy()
                gts = (targets.squeeze().long() - 1).clamp(0, 2).numpy()
                for p, g in zip(preds, gts): dices.append(self.get_dice(p, g))

        self._print_stats(all_preds, all_gts, ious, dices)
        self._visualize_samples(dataset, num_vis)

    def _print_stats(self, preds, gts, ious, dices):
        print("\n" + "="*40 + f"\n{self.task.upper()} PERFORMANCE STATS\n" + "="*40)
        if self.task == "classification":
            print(classification_report(gts, preds))
        elif self.task == "localization":
            ious = np.array(ious)
            print(f"Mean IoU:          {np.mean(ious):.4f}")
            print(f"Accuracy @IoU=0.5: {(ious >= 0.5).mean()*100:.2f}%")
            print(f"Accuracy @IoU=0.75:{(ious >= 0.75).mean()*100:.2f}%")
        elif self.task == "segmentation":
            print(f"Dataset Mean Dice: {np.mean(dices):.4f}")
        print("="*40)

    def _visualize_samples(self, dataset, num_vis):
        indices = np.random.choice(len(dataset), num_vis, replace=False)
        
        if self.task == "segmentation":
           
            fig, axes = plt.subplots(num_vis, 3, figsize=(12, 4 * num_vis))
            for i, idx in enumerate(indices):
                img_t, gt_mask = dataset[idx]
                out = self.model(img_t.unsqueeze(0).to(self.device))
                pred_mask = torch.argmax(out.squeeze(), dim=0).cpu().numpy()
                
                img = np.clip((img_t.permute(1, 2, 0).numpy() * self.std) + self.mean, 0, 1)
                gt_mask = (gt_mask.squeeze().numpy() - 1)
                
                axes[i, 0].imshow(img); axes[i, 0].set_title("Original Image"); axes[i, 0].axis('off')
                axes[i, 1].imshow(gt_mask, cmap='viridis'); axes[i, 1].set_title("Ground Truth"); axes[i, 1].axis('off')
                axes[i, 2].imshow(pred_mask, cmap='viridis'); axes[i, 2].set_title("Prediction"); axes[i, 2].axis('off')
        else:
            
            fig, axes = plt.subplots(1, num_vis, figsize=(5*num_vis, 5))
            if num_vis == 1: axes = [axes]
            for i, idx in enumerate(indices):
                img_t, target = dataset[idx]
                out = self.model(img_t.unsqueeze(0).to(self.device))
                img = np.clip((img_t.permute(1, 2, 0).numpy() * self.std) + self.mean, 0, 1)
                axes[i].imshow(img); axes[i].axis('off')
                
                if self.task == "localization":
                    p_box = out.squeeze().cpu().numpy()
                    rect = patches.Rectangle((p_box[0]-p_box[2]/2, p_box[1]-p_box[3]/2), 
                                             p_box[2], p_box[3], linewidth=2, edgecolor='red', facecolor='none')
                    axes[i].add_patch(rect)
                elif self.task == "classification":
                    pred = torch.argmax(out, dim=1).item()
                    axes[i].set_title(f"Pred: {pred}")

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Options: "classification", "localization", "segmentation"
    engine = InferenceEngine(task="localization") 
    engine.evaluate(num_vis=3)