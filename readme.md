# CIFAR-10 DNN Modular Training Pipeline

This repository provides a modular deep learning pipeline for training and evaluating models on the CIFAR-10 dataset using PyTorch and Albumentations. The code is organized for clarity, extensibility, and ease of experimentation.

## Folder Structure

```
dnn_cifar10/
│
├── config.py          # Configuration: device, hyperparameters, dataset stats
├── engine.py          # Training, testing, misclassification, plotting, utils
├── model.py           # Model architecture (Net class)
├── train.py           # Main script: data loading, training loop, evaluation
├── transforms.py      # Data augmentation and CutMix implementation
└── README.md          # This file
```

## Setup

1. **Install dependencies:**
   - Python 3.8+
   - PyTorch
   - torchvision
   - albumentations
   - matplotlib
   - torchsummary
   - tqdm

   ```
   pip install torch torchvision albumentations matplotlib torchsummary tqdm
   ```

2. **Download CIFAR-10:**  
   The dataset will be downloaded automatically when you run the script.

## Usage

Run the main training script:

```
python train.py
```

This will:
- Load and augment CIFAR-10 data
- Build and summarize the model
- Train and evaluate for the configured number of epochs
- Display misclassified images after training

## Customization

- **Model:** Edit `model.py` to change the architecture.
- **Augmentation:** Edit `transforms.py` for different data transforms or CutMix settings.
- **Hyperparameters:** Change values in `config.py` (epochs, batch size, learning rate, etc.).
- **Training Logic:** Modify `engine.py` for custom training/testing loops or metrics.

## Key Features

- **Modular design:** Easy to swap models, transforms, and training logic.
- **Albumentations:** Powerful image augmentation pipeline.
- **CutMix:** Optional advanced augmentation (see `transforms.py`).
- **Parameter counting:** See total trainable parameters in the model.
- **Misclassification visualization:** Plot misclassified images after training.

## Credits

- [PyTorch](https://pytorch.org/)
- [Albumentations](https://albumentations.ai/)
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

---

**For questions or improvements, feel free to open an issue or pull request.**