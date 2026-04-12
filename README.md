# VisualPerceptionPipeline


This project implements a multi-task deep learning pipeline using a **VGG-11** backbone to perform **Classification**, **Localization** (Bounding Box Regression), and **Semantic Segmentation** on the Oxford-IIIT Pet Dataset.

---

## 📂 Project Structure

```text
.
├── checkpoints/             # Trained model weights (.pth files)
├── data/
│   └── pets_dataset.py      # Oxford-IIIT Pet Dataset class
├── losses/
│   ├── dice_loss.py         # Custom Macro-Dice Loss for Segmentation
│   └── iou_loss.py          # Custom IoU Loss for Localization
├── models/
│   ├── vgg11.py             # Base VGG11 Backbone
│   ├── classification.py    # Classification Head
│   ├── localization.py      # Localization Head
│   ├── segmentation.py      # UNet-based Segmentation Decoder
│   ├── multitask.py         # Unified Multi-Task Model
│   └── layers.py            # Custom layers (Dropout, etc.)
├── train.py                 # Main training script
├── inference.py             # Evaluation and visualization script
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

---


### 1. Installation
Clone the repository and install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Training the Model
To train the model, open `train.py` and choose the specific task you wish to train (e.g., `classification`, `localization`, or `segmentation`). Configure the hyperparameters in the script and run:

```bash
python train.py
```

### 3. Running Inference
To evaluate the model and visualize results, open `inference.py`. Select the task to load the appropriate weights and dataset split.

Run the script using:
```bash
python inference.py
```

---

##  Results and Reports

* **W&B Training Report:** [https://api.wandb.ai/links/ae22b008-indian-institute-of-technology-madras/mqwnsxij]
* **GitHub Repository:** [https://github.com/ShriramUmashankar/VisualPerceptionPipeline]

---

