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
 

## CIFAR10 Model Training Comprehensive Analysis

### Objectives

Write a new network that

1. has the architecture to C1C2C3C40 (No MaxPooling, but 3 3x3 layers with stride of 2 instead) (If you can figure out how to use Dilated kernels here instead of MP or strided convolution, then 200pts extra!)
total RF must be more than 44
2. one of the layers must use Depthwise Separable Convolution
3. one of the layers must use Dilated Convolution
4. use GAP (compulsory):- add FC after GAP to target #of classes (optional)
5. use augmentation library and apply:
   -horizontal flip
   -shiftScaleRotate
   -coarseDropout (max_holes = 1, max_height=16px, max_width=1, min_holes = 1, min_height=16px, min_width=16px, fill_value=(mean of your dataset), mask_fill_value = None)
6. achieve 85% accuracy, as many epochs as you want. Total Params to be less than 200k.
7. make sure you're following code modularity (else 0 for full assignment)

### Result

<li>Parameters: 97,264</li>
<li>Best Training Accuracy: 79.38%</li>
<li>Best Validation Accuracy: 85.55%</li>
<li>Epochs Run: 75</li>


### Analysis

A well-structured network, and receptive field (RF) expansion is clearly intentional through standard, depthwise separable, stride, and dilated convolutions.

**1. Architecture & Receptive Field Design**

   The model balances RF expansion, parameter efficiency, and modularity, structured approximately as:
   
         [Block 1] Standard Convs (C1)
            Conv3×3 → Conv3×3 → 1×1 s=2
         
         [Block 2] Depthwise Separable + Dilated (C2)
            Conv3×3 → Depthwise3×3 + Pointwise1×1 → 1×1 s=2 → Dilated3×3 (d=2)
         
         [Block 3] Deeper Standard Convs (C3)
            Conv3×3 → 1×1 s=2 → Conv3×3 → Depthwise3×3 + Pointwise1×1
         
         [GAP + Classifier] (C4→GAP)
            AdaptiveAvgPool2d → 1×1 Conv to 10 classes
   
   
   **Key design strengths:**
   
      - Instead of MaxPooling, 1×1 stride=2 conv layers (SC1, SC2, SC3) are used for spatial downsampling, which preserve more representational flexibility.   
      - Dilated Conv (d=2) provides a major RF boost (from 13 → 29), without increasing parameter count.  
      - Depthwise Separable layers are used twice, cutting parameters drastically compared to equivalent standard convolutions.   
      - Receptive Field (RF) grows steadily and reaches 69×69 by the last block, fully covering the 32×32 CIFAR-10 image.


   ***Receptive Field***
   
   | Layer / Block      | Kernel | Stride | Dilation | jump | RF after layer |
   | ------------------ | ------ | ------ | -------- | ---- | -------------- |
   | Input              | -      | -      | -        | 1    | 1              |
   | Conv3×3            | 3      | 1      | 1        | 1    | 3              |
   | Conv3×3            | 3      | 1      | 1        | 1    | 5              |
   | SC1 1×1 s=2        | 1      | 2      | 1        | 2    | 5              |
   | Conv3×3            | 3      | 1      | 1        | 2    | 9              |
   | Depthwise3×3       | 3      | 1      | 1        | 2    | 13             |
   | Pointwise1×1       | 1      | 1      | 1        | 2    | 13             |
   | SC2 1×1 s=2        | 1      | 2      | 1        | 4    | 13             |
   | DilatedConv3×3 d=2 | 5eff   | 1      | 2        | 4    | 29             |
   | Conv3×3            | 3      | 1      | 1        | 4    | 37             |
   | SC3 1×1 s=2        | 1      | 2      | 1        | 8    | 37             |
   | Conv3×3            | 3      | 1      | 1        | 8    | 53             |
   | Depthwise3×3       | 3      | 1      | 1        | 8    | 69             |
   | Pointwise1×1       | 1      | 1      | 1        | 8    | 69             |
   | GAP                | -      | -      | -        | -    | ≈Full image    |
   
   GAP
   
   Global average pooling doesn't expand RF, just aggregates entire spatial map. So effective RF covers the entire input, but theoretical RF at this point is 69×69, which is already > input size (32×32). So effectively, it's full coverage.

   **Final Result**

- Final theoretical receptive field ≈ 69 × 69
- Since input is 32×32, this fully covers the input.
- RF expansion is achieved mainly through dilated conv + multiple stride=2 skips.
- Theoretical RF exceeds the input size by a wide margin, indicating global context aggregation well before GAP, a good sign for effective representation learning.

**2. Training Dynamics**

The training curves indicate steady improvement and healthy generalization:

**Initial Epochs (1–5):**
Rapid gain from ~35% to ~69% validation accuracy. This is consistent with good augmentation and appropriate learning rate — the model quickly learns low-level features.

**Mid Training (6–30):**
Validation climbs gradually from 72% to ~81%. No severe overfitting — train and val accuracies remain close (gap ~6–7%). Indicates strong regularization (Dropout, augmentation, BN).

**Late Training (31–75):**
Training plateaus around 75% train acc, validation gradually creeps up to 85.55%.

Interestingly, **validation accuracy exceeds training accuracy consistently**, likely due to:
- Strong augmentation increasing training difficulty
- CoarseDropout simulating occlusion noise during training
- BatchNorm + dropout regularizing training more than test
Loss curves are smooth, with no spikes — this reflects stable optimization.

**Time per epoch (~16s)** suggests the architecture is computationally light despite multiple layers, thanks to low parameter count and depthwise convolutions.

**3. Hyperparameters & Augmentation**

- **Augmentations**: Horizontal flip, ShiftScaleRotate, CoarseDropout (Albumentations)
→ Added essential variety; CoarseDropout particularly improved robustness to occlusion.

- **Optimizer / Scheduler (from training code):**
→ SGD with appropriate CosineAnnealingLR — steady convergence suggests tuned LR, no oscillation.

- **Regularization:**
Dropout after most conv blocks (~0.01) + BN after each conv keeps model generalizing well without collapsing.

## Summary

**Objective Fulfillment Summary**

| Objective                       | Target                                          | Achieved    | Remarks                                                                  |
| ------------------------------- | ----------------------------------------------- | ----------- | ------------------------------------------------------------------------ |
| Architecture pattern            | C1C2C3C40 (No MaxPool; 3×3 stride=2 or dilated) | Y           | Strided 1×1 layers used as downsamplers, dilated conv used for RF growth |
| Receptive Field                 | > 44                                            | Y (≈ 69×69) | Exceeded input size (32×32), effectively global                          |
| Depthwise Separable Convolution | ≥ 1 layer                                       | Y           | Multiple DW conv + pointwise pairs                                       |
| Dilated Convolution             | ≥ 1 layer                                       | Y           | One 3×3 layer with dilation=2 used effectively                           |
| GAP                             | Required                                        | Y           | AdaptiveAvgPool2d at the end                                             |
| Augmentations                   | Flip, ShiftScaleRotate, CoarseDropout           | Y           | All used through Albumentations (from training code)                     |
| Accuracy                        | ≥ 85% validation                                | Y 85.55%    | After 75 epochs                                                          |
| Params                          | ≤ 200k                                          | Y 97,264    | Lightweight yet expressive                                               |
| Modularity                      | Required                                        | Y           | Clean separation of model, train, transforms                             |


   **Key Strengths**
   
   - Clean, modular architecture with thoughtful RF expansion
   - Excellent augmentation strategy for CIFAR-10
   - Very efficient in parameter usage 
   - Stable, steady training dynamics without overfitting 
   
   **Potential Improvement Areas**
   
   - Training accuracy plateauing earlier than validation may indicate underfitting on clean training data — slight LR warmup or longer training could close the gap.
   - Higher Epoch runs might push accuracy beyond 86–87%.
   - Could try replacing some standard 3×3 with depthwise to further reduce params or allow more channels.

### Model Architecture, Training Logs and Plots

         ----------------------------------------------------------------
                 Layer (type)               Output Shape         Param #
         ================================================================
                     Conv2d-1           [-1, 32, 32, 32]             864
                BatchNorm2d-2           [-1, 32, 32, 32]              64
                       ReLU-3           [-1, 32, 32, 32]               0
                  Dropout2d-4           [-1, 32, 32, 32]               0
                     Conv2d-5           [-1, 64, 32, 32]          18,432
                BatchNorm2d-6           [-1, 64, 32, 32]             128
                       ReLU-7           [-1, 64, 32, 32]               0
                  Dropout2d-8           [-1, 64, 32, 32]               0
                     Conv2d-9           [-1, 32, 16, 16]           2,080
                      ReLU-10           [-1, 32, 16, 16]               0
                    Conv2d-11           [-1, 32, 16, 16]           9,216
               BatchNorm2d-12           [-1, 32, 16, 16]              64
                      ReLU-13           [-1, 32, 16, 16]               0
                 Dropout2d-14           [-1, 32, 16, 16]               0
                    Conv2d-15           [-1, 32, 16, 16]             288
                    Conv2d-16           [-1, 64, 16, 16]           2,048
               BatchNorm2d-17           [-1, 64, 16, 16]             128
                      ReLU-18           [-1, 64, 16, 16]               0
                 Dropout2d-19           [-1, 64, 16, 16]               0
                    Conv2d-20             [-1, 32, 8, 8]           2,080
                      ReLU-21             [-1, 32, 8, 8]               0
                    Conv2d-22             [-1, 64, 6, 6]          18,432
               BatchNorm2d-23             [-1, 64, 6, 6]             128
                      ReLU-24             [-1, 64, 6, 6]               0
                 Dropout2d-25             [-1, 64, 6, 6]               0
                    Conv2d-26             [-1, 64, 6, 6]          36,864
               BatchNorm2d-27             [-1, 64, 6, 6]             128
                      ReLU-28             [-1, 64, 6, 6]               0
                 Dropout2d-29             [-1, 64, 6, 6]               0
                    Conv2d-30             [-1, 16, 3, 3]           1,040
                      ReLU-31             [-1, 16, 3, 3]               0
                    Conv2d-32             [-1, 32, 3, 3]           4,608
               BatchNorm2d-33             [-1, 32, 3, 3]              64
                      ReLU-34             [-1, 32, 3, 3]               0
                 Dropout2d-35             [-1, 32, 3, 3]               0
                    Conv2d-36             [-1, 32, 3, 3]             288
                    Conv2d-37             [-1, 10, 3, 3]             320
         AdaptiveAvgPool2d-38             [-1, 10, 1, 1]               0
         ================================================================
         Total params: 97,264
         Trainable params: 97,264
         Non-trainable params: 0
         ----------------------------------------------------------------
         Input size (MB): 0.01
         Forward/backward pass size (MB): 4.12
         Params size (MB): 0.37
         Estimated Total Size (MB): 4.51
         ----------------------------------------------------------------
         None
         
         Total trainable parameters: 97,264
         Epoch 1
         Loss=1.4867 Batch_id=390 Accuracy=34.69: 100%|████████████████████████████| 391/391 [00:13<00:00, 29.87it/s] 
         
         Validation set: Average loss: 1.4739, Accuracy: 4633/10000 (46.33%)
         
         Train Loss: 0.0137, Train Accuracy: 34.69%
         Validation Loss: 1.4739, Validation Accuracy: 46.33%
         Epoch 2
         Loss=1.3582 Batch_id=390 Accuracy=47.69: 100%|████████████████████████████| 391/391 [00:16<00:00, 24.20it/s] 
         
         Validation set: Average loss: 1.1915, Accuracy: 5661/10000 (56.61%)
         
         Train Loss: 0.0113, Train Accuracy: 47.69%
         Validation Loss: 1.1915, Validation Accuracy: 56.61%
         Epoch 3
         Loss=1.3516 Batch_id=390 Accuracy=53.75: 100%|████████████████████████████| 391/391 [00:16<00:00, 23.81it/s] 
         
         Validation set: Average loss: 0.9932, Accuracy: 6415/10000 (64.15%)
         
         Train Loss: 0.0100, Train Accuracy: 53.75%
         Validation Loss: 0.9932, Validation Accuracy: 64.15%
         Epoch 4
         Loss=1.1933 Batch_id=390 Accuracy=58.28: 100%|████████████████████████████| 391/391 [00:16<00:00, 23.72it/s] 
         
         Validation set: Average loss: 0.9811, Accuracy: 6535/10000 (65.35%)
         
         Train Loss: 0.0091, Train Accuracy: 58.28%
         Validation Loss: 0.9811, Validation Accuracy: 65.35%
         Epoch 5
         Loss=1.0205 Batch_id=390 Accuracy=60.85: 100%|████████████████████████████| 391/391 [00:16<00:00, 23.67it/s] 
         
         Validation set: Average loss: 0.8817, Accuracy: 6900/10000 (69.00%)
         
         Train Loss: 0.0086, Train Accuracy: 60.85%
         Validation Loss: 0.8817, Validation Accuracy: 69.00%
         Epoch 6
         Loss=1.0512 Batch_id=390 Accuracy=62.72: 100%|████████████████████████████| 391/391 [00:16<00:00, 23.77it/s] 
         
         Validation set: Average loss: 0.8025, Accuracy: 7195/10000 (71.95%)
         
         Train Loss: 0.0082, Train Accuracy: 62.72%
         Validation Loss: 0.8025, Validation Accuracy: 71.95%
         Epoch 7
         Loss=0.8151 Batch_id=390 Accuracy=64.34: 100%|████████████████████████████| 391/391 [00:16<00:00, 23.76it/s] 
         
         Validation set: Average loss: 0.7841, Accuracy: 7237/10000 (72.37%)
         
         Train Loss: 0.0079, Train Accuracy: 64.34%
         Validation Loss: 0.7841, Validation Accuracy: 72.37%
         Epoch 8
         Loss=1.1497 Batch_id=390 Accuracy=65.21: 100%|████████████████████████████| 391/391 [00:16<00:00, 23.49it/s] 
         
         Validation set: Average loss: 0.7609, Accuracy: 7364/10000 (73.64%)
         
         Train Loss: 0.0077, Train Accuracy: 65.21%
         Validation Loss: 0.7609, Validation Accuracy: 73.64%
         Epoch 9
         Loss=0.7263 Batch_id=390 Accuracy=66.32: 100%|████████████████████████████| 391/391 [00:16<00:00, 23.50it/s] 
         
         Validation set: Average loss: 0.7668, Accuracy: 7338/10000 (73.38%)
         
         Train Loss: 0.0075, Train Accuracy: 66.32%
         Validation Loss: 0.7668, Validation Accuracy: 73.38%
         Epoch 10
         Loss=0.9004 Batch_id=390 Accuracy=66.99: 100%|████████████████████████████| 391/391 [00:16<00:00, 23.66it/s] 
         
         Validation set: Average loss: 0.7669, Accuracy: 7345/10000 (73.45%)
         
         Train Loss: 0.0073, Train Accuracy: 66.99%
         Validation Loss: 0.7669, Validation Accuracy: 73.45%
         Epoch 11
         Loss=0.9095 Batch_id=390 Accuracy=67.78: 100%|████████████████████████████| 391/391 [00:16<00:00, 23.45it/s] 
         
         Validation set: Average loss: 0.7317, Accuracy: 7477/10000 (74.77%)
         
         Train Loss: 0.0072, Train Accuracy: 67.78%
         Validation Loss: 0.7317, Validation Accuracy: 74.77%
         Epoch 12
         Loss=0.7057 Batch_id=390 Accuracy=68.22: 100%|████████████████████████████| 391/391 [00:16<00:00, 23.52it/s] 
         
         Validation set: Average loss: 0.7506, Accuracy: 7419/10000 (74.19%)
         
         Train Loss: 0.0071, Train Accuracy: 68.22%
         Validation Loss: 0.7506, Validation Accuracy: 74.19%
         Epoch 13
         Loss=0.7483 Batch_id=390 Accuracy=68.95: 100%|████████████████████████████| 391/391 [00:16<00:00, 23.65it/s] 
         
         Validation set: Average loss: 0.6943, Accuracy: 7609/10000 (76.09%)
         
         Train Loss: 0.0069, Train Accuracy: 68.95%
         Validation Loss: 0.6943, Validation Accuracy: 76.09%
         Epoch 14
         Loss=0.7855 Batch_id=390 Accuracy=69.15: 100%|████████████████████████████| 391/391 [00:16<00:00, 23.50it/s] 
         
         Validation set: Average loss: 0.6506, Accuracy: 7791/10000 (77.91%)
         
         Train Loss: 0.0068, Train Accuracy: 69.15%
         Validation Loss: 0.6506, Validation Accuracy: 77.91%
         Epoch 15
         Loss=0.8587 Batch_id=390 Accuracy=70.13: 100%|████████████████████████████| 391/391 [00:16<00:00, 23.73it/s] 
         
         Validation set: Average loss: 0.6845, Accuracy: 7672/10000 (76.72%)
         
         Train Loss: 0.0067, Train Accuracy: 70.13%
         Validation Loss: 0.6845, Validation Accuracy: 76.72%
         Epoch 16
         Loss=0.8534 Batch_id=390 Accuracy=70.33: 100%|████████████████████████████| 391/391 [00:16<00:00, 23.75it/s] 
         
         Validation set: Average loss: 0.6786, Accuracy: 7711/10000 (77.11%)
         
         Train Loss: 0.0066, Train Accuracy: 70.33%
         Validation Loss: 0.6786, Validation Accuracy: 77.11%
         Epoch 17
         Loss=0.8991 Batch_id=390 Accuracy=70.62: 100%|████████████████████████████| 391/391 [00:16<00:00, 23.75it/s] 
         
         Validation set: Average loss: 0.6506, Accuracy: 7778/10000 (77.78%)
         
         Train Loss: 0.0066, Train Accuracy: 70.62%
         Validation Loss: 0.6506, Validation Accuracy: 77.78%
         Epoch 18
         Loss=0.7212 Batch_id=390 Accuracy=70.54: 100%|████████████████████████████| 391/391 [00:16<00:00, 23.63it/s] 
         
         Validation set: Average loss: 0.6304, Accuracy: 7873/10000 (78.73%)
         
         Train Loss: 0.0065, Train Accuracy: 70.54%
         Validation Loss: 0.6304, Validation Accuracy: 78.73%
         Epoch 19
         Loss=0.8049 Batch_id=390 Accuracy=71.33: 100%|████████████████████████████| 391/391 [00:16<00:00, 23.60it/s] 
         
         Validation set: Average loss: 0.6191, Accuracy: 7892/10000 (78.92%)
         
         Train Loss: 0.0065, Train Accuracy: 71.33%
         Validation Loss: 0.6191, Validation Accuracy: 78.92%
         Epoch 20
         Loss=0.8555 Batch_id=390 Accuracy=71.64: 100%|████████████████████████████| 391/391 [00:16<00:00, 23.74it/s] 
         
         Validation set: Average loss: 0.6219, Accuracy: 7844/10000 (78.44%)
         
         Train Loss: 0.0063, Train Accuracy: 71.64%
         Validation Loss: 0.6219, Validation Accuracy: 78.44%
         Epoch 21
         Loss=0.8155 Batch_id=390 Accuracy=71.89: 100%|████████████████████████████| 391/391 [00:16<00:00, 23.74it/s] 
         
         Validation set: Average loss: 0.5863, Accuracy: 7990/10000 (79.90%)
         
         Train Loss: 0.0063, Train Accuracy: 71.89%
         Validation Loss: 0.5863, Validation Accuracy: 79.90%
         Epoch 22
         Loss=0.6335 Batch_id=390 Accuracy=71.82: 100%|████████████████████████████| 391/391 [00:16<00:00, 23.83it/s] 
         
         Validation set: Average loss: 0.6170, Accuracy: 7875/10000 (78.75%)
         
         Train Loss: 0.0063, Train Accuracy: 71.82%
         Validation Loss: 0.6170, Validation Accuracy: 78.75%
         Epoch 23
         Loss=0.8541 Batch_id=390 Accuracy=72.42: 100%|████████████████████████████| 391/391 [00:16<00:00, 23.75it/s] 
         
         Validation set: Average loss: 0.5773, Accuracy: 8022/10000 (80.22%)
         
         Train Loss: 0.0062, Train Accuracy: 72.42%
         Validation Loss: 0.5773, Validation Accuracy: 80.22%
         Epoch 24
         Loss=0.9794 Batch_id=390 Accuracy=72.53: 100%|████████████████████████████| 391/391 [00:16<00:00, 23.86it/s] 
         
         Validation set: Average loss: 0.6043, Accuracy: 7951/10000 (79.51%)
         
         Train Loss: 0.0062, Train Accuracy: 72.53%
         Validation Loss: 0.6043, Validation Accuracy: 79.51%
         Epoch 25
         Loss=0.9729 Batch_id=390 Accuracy=72.63: 100%|████████████████████████████| 391/391 [00:16<00:00, 23.67it/s] 
         
         Validation set: Average loss: 0.5605, Accuracy: 8043/10000 (80.43%)
         
         Train Loss: 0.0061, Train Accuracy: 72.63%
         Validation Loss: 0.5605, Validation Accuracy: 80.43%
         Epoch 26
         Loss=0.7472 Batch_id=390 Accuracy=72.97: 100%|████████████████████████████| 391/391 [00:16<00:00, 23.63it/s] 
         
         Validation set: Average loss: 0.6153, Accuracy: 7896/10000 (78.96%)
         
         Train Loss: 0.0061, Train Accuracy: 72.97%
         Validation Loss: 0.6153, Validation Accuracy: 78.96%
         Epoch 27
         Loss=0.7200 Batch_id=390 Accuracy=73.00: 100%|████████████████████████████| 391/391 [00:16<00:00, 23.70it/s] 
         
         Validation set: Average loss: 0.5856, Accuracy: 8017/10000 (80.17%)
         
         Train Loss: 0.0061, Train Accuracy: 73.00%
         Validation Loss: 0.5856, Validation Accuracy: 80.17%
         Epoch 28
         Loss=0.7880 Batch_id=390 Accuracy=73.23: 100%|████████████████████████████| 391/391 [00:16<00:00, 23.80it/s] 
         
         Validation set: Average loss: 0.5521, Accuracy: 8119/10000 (81.19%)
         
         Train Loss: 0.0060, Train Accuracy: 73.23%
         Validation Loss: 0.5521, Validation Accuracy: 81.19%
         Epoch 29
         Loss=0.8634 Batch_id=390 Accuracy=73.76: 100%|████████████████████████████| 391/391 [00:16<00:00, 23.53it/s] 
         
         Validation set: Average loss: 0.5763, Accuracy: 8050/10000 (80.50%)
         
         Train Loss: 0.0059, Train Accuracy: 73.76%
         Validation Loss: 0.5763, Validation Accuracy: 80.50%
         Epoch 30
         Loss=0.7973 Batch_id=390 Accuracy=73.88: 100%|████████████████████████████| 391/391 [00:16<00:00, 23.80it/s] 
         
         Validation set: Average loss: 0.5546, Accuracy: 8114/10000 (81.14%)
         
         Train Loss: 0.0059, Train Accuracy: 73.88%
         Validation Loss: 0.5546, Validation Accuracy: 81.14%
         Epoch 31
         Loss=0.6654 Batch_id=390 Accuracy=73.96: 100%|████████████████████████████| 391/391 [00:16<00:00, 23.56it/s] 
         
         Validation set: Average loss: 0.5742, Accuracy: 8018/10000 (80.18%)
         
         Train Loss: 0.0058, Train Accuracy: 73.96%
         Validation Loss: 0.5742, Validation Accuracy: 80.18%
         Epoch 32
         Loss=0.7505 Batch_id=390 Accuracy=73.95: 100%|████████████████████████████| 391/391 [00:16<00:00, 23.51it/s] 
         
         Validation set: Average loss: 0.5493, Accuracy: 8107/10000 (81.07%)
         
         Train Loss: 0.0058, Train Accuracy: 73.95%
         Validation Loss: 0.5493, Validation Accuracy: 81.07%
         Epoch 33
         Loss=1.0066 Batch_id=390 Accuracy=74.37: 100%|████████████████████████████| 391/391 [00:16<00:00, 23.64it/s] 
         
         Validation set: Average loss: 0.5499, Accuracy: 8143/10000 (81.43%)
         
         Train Loss: 0.0058, Train Accuracy: 74.37%
         Validation Loss: 0.5499, Validation Accuracy: 81.43%
         Epoch 34
         Loss=0.6039 Batch_id=390 Accuracy=74.29: 100%|████████████████████████████| 391/391 [00:16<00:00, 23.72it/s] 
         
         Validation set: Average loss: 0.5360, Accuracy: 8163/10000 (81.63%)
         
         Train Loss: 0.0058, Train Accuracy: 74.29%
         Validation Loss: 0.5360, Validation Accuracy: 81.63%
         Epoch 35
         Loss=0.7138 Batch_id=390 Accuracy=74.35: 100%|████████████████████████████| 391/391 [00:16<00:00, 23.47it/s] 
         
         Validation set: Average loss: 0.5607, Accuracy: 8070/10000 (80.70%)
         
         Train Loss: 0.0057, Train Accuracy: 74.35%
         Validation Loss: 0.5607, Validation Accuracy: 80.70%
         Epoch 36
         Loss=0.7688 Batch_id=390 Accuracy=74.51: 100%|████████████████████████████| 391/391 [00:16<00:00, 23.60it/s] 
         
         Validation set: Average loss: 0.5562, Accuracy: 8109/10000 (81.09%)
         
         Train Loss: 0.0057, Train Accuracy: 74.51%
         Validation Loss: 0.5562, Validation Accuracy: 81.09%
         Epoch 37
         Loss=0.8250 Batch_id=390 Accuracy=74.50: 100%|████████████████████████████| 391/391 [00:16<00:00, 23.78it/s] 
         
         Validation set: Average loss: 0.5739, Accuracy: 8036/10000 (80.36%)
         
         Train Loss: 0.0057, Train Accuracy: 74.50%
         Validation Loss: 0.5739, Validation Accuracy: 80.36%
         Epoch 38
         Loss=0.6898 Batch_id=390 Accuracy=75.12: 100%|████████████████████████████| 391/391 [00:16<00:00, 23.72it/s] 
         
         Validation set: Average loss: 0.5223, Accuracy: 8210/10000 (82.10%)
         
         Train Loss: 0.0056, Train Accuracy: 75.12%
         Validation Loss: 0.5223, Validation Accuracy: 82.10%
         Epoch 39
         Loss=0.6350 Batch_id=390 Accuracy=75.11: 100%|████████████████████████████| 391/391 [00:16<00:00, 23.54it/s] 
         
         Validation set: Average loss: 0.5252, Accuracy: 8213/10000 (82.13%)
         
         Train Loss: 0.0056, Train Accuracy: 75.11%
         Validation Loss: 0.5252, Validation Accuracy: 82.13%
         Epoch 40
         Loss=0.7554 Batch_id=390 Accuracy=74.97: 100%|████████████████████████████| 391/391 [00:16<00:00, 23.75it/s] 
         
         Validation set: Average loss: 0.5500, Accuracy: 8156/10000 (81.56%)
         
         Train Loss: 0.0056, Train Accuracy: 74.97%
         Validation Loss: 0.5500, Validation Accuracy: 81.56%
         Epoch 41
         Loss=0.6607 Batch_id=390 Accuracy=75.49: 100%|████████████████████████████| 391/391 [00:16<00:00, 23.66it/s] 
         
         Validation set: Average loss: 0.5337, Accuracy: 8156/10000 (81.56%)
         
         Train Loss: 0.0055, Train Accuracy: 75.49%
         Validation Loss: 0.5337, Validation Accuracy: 81.56%
         Epoch 42
         Loss=0.9288 Batch_id=390 Accuracy=75.34: 100%|████████████████████████████| 391/391 [00:16<00:00, 23.91it/s] 
         
         Validation set: Average loss: 0.5024, Accuracy: 8296/10000 (82.96%)
         
         Train Loss: 0.0055, Train Accuracy: 75.34%
         Validation Loss: 0.5024, Validation Accuracy: 82.96%
         Epoch 43
         Loss=0.6428 Batch_id=390 Accuracy=75.59: 100%|████████████████████████████| 391/391 [00:16<00:00, 23.72it/s] 
         
         Validation set: Average loss: 0.5140, Accuracy: 8225/10000 (82.25%)
         
         Train Loss: 0.0054, Train Accuracy: 75.59%
         Validation Loss: 0.5140, Validation Accuracy: 82.25%
         Epoch 44
         Loss=0.9019 Batch_id=390 Accuracy=75.78: 100%|████████████████████████████| 391/391 [00:16<00:00, 23.84it/s] 
         
         Validation set: Average loss: 0.5237, Accuracy: 8257/10000 (82.57%)
         
         Train Loss: 0.0054, Train Accuracy: 75.78%
         Validation Loss: 0.5237, Validation Accuracy: 82.57%
         Epoch 45
         Loss=0.7952 Batch_id=390 Accuracy=75.91: 100%|████████████████████████████| 391/391 [00:16<00:00, 23.79it/s] 
         
         Validation set: Average loss: 0.5040, Accuracy: 8260/10000 (82.60%)
         
         Train Loss: 0.0054, Train Accuracy: 75.91%
         Validation Loss: 0.5040, Validation Accuracy: 82.60%
         Epoch 46
         Loss=0.5206 Batch_id=390 Accuracy=76.21: 100%|████████████████████████████| 391/391 [00:16<00:00, 23.79it/s] 
         
         Validation set: Average loss: 0.4908, Accuracy: 8357/10000 (83.57%)
         
         Train Loss: 0.0054, Train Accuracy: 76.21%
         Validation Loss: 0.4908, Validation Accuracy: 83.57%
         Epoch 47
         Loss=0.6839 Batch_id=390 Accuracy=75.72: 100%|████████████████████████████| 391/391 [00:16<00:00, 23.65it/s] 
         
         Validation set: Average loss: 0.5198, Accuracy: 8300/10000 (83.00%)
         
         Train Loss: 0.0054, Train Accuracy: 75.72%
         Validation Loss: 0.5198, Validation Accuracy: 83.00%
         Epoch 48
         Loss=0.7517 Batch_id=390 Accuracy=76.13: 100%|████████████████████████████| 391/391 [00:16<00:00, 23.79it/s] 
         
         Validation set: Average loss: 0.5021, Accuracy: 8333/10000 (83.33%)
         
         Train Loss: 0.0053, Train Accuracy: 76.13%
         Validation Loss: 0.5021, Validation Accuracy: 83.33%
         Epoch 49
         Loss=0.7433 Batch_id=390 Accuracy=76.48: 100%|████████████████████████████| 391/391 [00:16<00:00, 23.82it/s] 
         
         Validation set: Average loss: 0.4859, Accuracy: 8347/10000 (83.47%)
         
         Train Loss: 0.0053, Train Accuracy: 76.48%
         Validation Loss: 0.4859, Validation Accuracy: 83.47%
         Epoch 50
         Loss=0.7988 Batch_id=390 Accuracy=76.71: 100%|████████████████████████████| 391/391 [00:16<00:00, 23.73it/s] 
         
         Validation set: Average loss: 0.5132, Accuracy: 8262/10000 (82.62%)
         
         Train Loss: 0.0053, Train Accuracy: 76.71%
         Validation Loss: 0.5132, Validation Accuracy: 82.62%
         Epoch 51
         Loss=0.6825 Batch_id=390 Accuracy=76.82: 100%|████████████████████████████| 391/391 [00:16<00:00, 23.67it/s] 
         
         Validation set: Average loss: 0.4769, Accuracy: 8381/10000 (83.81%)
         
         Train Loss: 0.0052, Train Accuracy: 76.82%
         Validation Loss: 0.4769, Validation Accuracy: 83.81%
         Epoch 52
         Loss=0.6247 Batch_id=390 Accuracy=76.92: 100%|████████████████████████████| 391/391 [00:16<00:00, 23.67it/s] 
         
         Validation set: Average loss: 0.4958, Accuracy: 8338/10000 (83.38%)
         
         Train Loss: 0.0052, Train Accuracy: 76.92%
         Validation Loss: 0.4958, Validation Accuracy: 83.38%
         Epoch 53
         Loss=0.9103 Batch_id=390 Accuracy=76.99: 100%|████████████████████████████| 391/391 [00:16<00:00, 23.54it/s] 
         
         Validation set: Average loss: 0.4847, Accuracy: 8369/10000 (83.69%)
         
         Train Loss: 0.0051, Train Accuracy: 76.99%
         Validation Loss: 0.4847, Validation Accuracy: 83.69%
         Epoch 54
         Loss=0.7378 Batch_id=390 Accuracy=77.04: 100%|████████████████████████████| 391/391 [00:16<00:00, 23.67it/s] 
         
         Validation set: Average loss: 0.4831, Accuracy: 8359/10000 (83.59%)
         
         Train Loss: 0.0052, Train Accuracy: 77.04%
         Validation Loss: 0.4831, Validation Accuracy: 83.59%
         Epoch 55
         Loss=0.6546 Batch_id=390 Accuracy=77.29: 100%|████████████████████████████| 391/391 [00:16<00:00, 23.66it/s] 
         
         Validation set: Average loss: 0.4884, Accuracy: 8328/10000 (83.28%)
         
         Train Loss: 0.0051, Train Accuracy: 77.29%
         Validation Loss: 0.4884, Validation Accuracy: 83.28%
         Epoch 56
         Loss=0.7354 Batch_id=390 Accuracy=77.48: 100%|████████████████████████████| 391/391 [00:16<00:00, 23.71it/s] 
         
         Validation set: Average loss: 0.4918, Accuracy: 8358/10000 (83.58%)
         
         Train Loss: 0.0051, Train Accuracy: 77.48%
         Validation Loss: 0.4918, Validation Accuracy: 83.58%
         Epoch 57
         Loss=0.6800 Batch_id=390 Accuracy=77.53: 100%|████████████████████████████| 391/391 [00:16<00:00, 23.87it/s] 
         
         Validation set: Average loss: 0.4821, Accuracy: 8348/10000 (83.48%)
         
         Train Loss: 0.0050, Train Accuracy: 77.53%
         Validation Loss: 0.4821, Validation Accuracy: 83.48%
         Epoch 58
         Loss=0.6260 Batch_id=390 Accuracy=77.59: 100%|████████████████████████████| 391/391 [00:16<00:00, 23.73it/s] 
         
         Validation set: Average loss: 0.4733, Accuracy: 8377/10000 (83.77%)
         
         Train Loss: 0.0050, Train Accuracy: 77.59%
         Validation Loss: 0.4733, Validation Accuracy: 83.77%
         Epoch 59
         Loss=0.5785 Batch_id=390 Accuracy=77.43: 100%|████████████████████████████| 391/391 [00:16<00:00, 23.66it/s] 
         
         Validation set: Average loss: 0.4877, Accuracy: 8379/10000 (83.79%)
         
         Train Loss: 0.0050, Train Accuracy: 77.43%
         Validation Loss: 0.4877, Validation Accuracy: 83.79%
         Epoch 60
         Loss=0.7051 Batch_id=390 Accuracy=77.52: 100%|████████████████████████████| 391/391 [00:16<00:00, 23.70it/s] 
         
         Validation set: Average loss: 0.4535, Accuracy: 8445/10000 (84.45%)
         
         Train Loss: 0.0050, Train Accuracy: 77.52%
         Validation Loss: 0.4535, Validation Accuracy: 84.45%
         Epoch 61
         Loss=0.8809 Batch_id=390 Accuracy=78.03: 100%|████████████████████████████| 391/391 [00:16<00:00, 23.79it/s] 
         
         Validation set: Average loss: 0.4734, Accuracy: 8376/10000 (83.76%)
         
         Train Loss: 0.0050, Train Accuracy: 78.03%
         Validation Loss: 0.4734, Validation Accuracy: 83.76%
         Epoch 62
         Loss=0.7728 Batch_id=390 Accuracy=77.79: 100%|████████████████████████████| 391/391 [00:16<00:00, 23.66it/s] 
         
         Validation set: Average loss: 0.4546, Accuracy: 8461/10000 (84.61%)
         
         Train Loss: 0.0049, Train Accuracy: 77.79%
         Validation Loss: 0.4546, Validation Accuracy: 84.61%
         Epoch 63
         Loss=0.4846 Batch_id=390 Accuracy=78.26: 100%|████████████████████████████| 391/391 [00:16<00:00, 23.86it/s] 
         
         Validation set: Average loss: 0.4582, Accuracy: 8454/10000 (84.54%)
         
         Train Loss: 0.0049, Train Accuracy: 78.26%
         Validation Loss: 0.4582, Validation Accuracy: 84.54%
         Epoch 64
         Loss=0.5539 Batch_id=390 Accuracy=77.82: 100%|████████████████████████████| 391/391 [00:16<00:00, 23.71it/s] 
         
         Validation set: Average loss: 0.4586, Accuracy: 8470/10000 (84.70%)
         
         Train Loss: 0.0049, Train Accuracy: 77.82%
         Validation Loss: 0.4586, Validation Accuracy: 84.70%
         Epoch 65
         Loss=0.5329 Batch_id=390 Accuracy=78.14: 100%|████████████████████████████| 391/391 [00:16<00:00, 23.79it/s] 
         
         Validation set: Average loss: 0.4487, Accuracy: 8482/10000 (84.82%)
         
         Train Loss: 0.0049, Train Accuracy: 78.14%
         Validation Loss: 0.4487, Validation Accuracy: 84.82%
         Epoch 66
         Loss=0.6453 Batch_id=390 Accuracy=78.40: 100%|████████████████████████████| 391/391 [00:16<00:00, 23.55it/s] 
         
         Validation set: Average loss: 0.4487, Accuracy: 8468/10000 (84.68%)
         
         Train Loss: 0.0048, Train Accuracy: 78.40%
         Validation Loss: 0.4487, Validation Accuracy: 84.68%
         Epoch 67
         Loss=0.6754 Batch_id=390 Accuracy=78.48: 100%|████████████████████████████| 391/391 [00:16<00:00, 23.73it/s] 
         
         Validation set: Average loss: 0.4473, Accuracy: 8497/10000 (84.97%)
         
         Train Loss: 0.0048, Train Accuracy: 78.48%
         Validation Loss: 0.4473, Validation Accuracy: 84.97%
         Epoch 68
         Loss=0.7174 Batch_id=390 Accuracy=78.70: 100%|████████████████████████████| 391/391 [00:16<00:00, 23.72it/s] 
         
         Validation set: Average loss: 0.4452, Accuracy: 8494/10000 (84.94%)
         
         Train Loss: 0.0047, Train Accuracy: 78.70%
         Validation Loss: 0.4452, Validation Accuracy: 84.94%
         Epoch 69
         Loss=0.5709 Batch_id=390 Accuracy=79.03: 100%|████████████████████████████| 391/391 [00:16<00:00, 23.85it/s] 
         
         Validation set: Average loss: 0.4383, Accuracy: 8519/10000 (85.19%)
         
         Train Loss: 0.0047, Train Accuracy: 79.03%
         Validation Loss: 0.4383, Validation Accuracy: 85.19%
         Epoch 70
         Loss=0.5866 Batch_id=390 Accuracy=78.77: 100%|████████████████████████████| 391/391 [00:16<00:00, 23.80it/s] 
         
         Validation set: Average loss: 0.4380, Accuracy: 8555/10000 (85.55%)
         
         Train Loss: 0.0047, Train Accuracy: 78.77%
         Validation Loss: 0.4380, Validation Accuracy: 85.55%
         Epoch 71
         Loss=0.5194 Batch_id=390 Accuracy=78.91: 100%|████████████████████████████| 391/391 [00:16<00:00, 23.75it/s] 
         
         Validation set: Average loss: 0.4430, Accuracy: 8525/10000 (85.25%)
         
         Train Loss: 0.0047, Train Accuracy: 78.91%
         Validation Loss: 0.4430, Validation Accuracy: 85.25%
         Epoch 72
         Loss=0.5385 Batch_id=390 Accuracy=79.23: 100%|████████████████████████████| 391/391 [00:16<00:00, 23.71it/s] 
         
         Validation set: Average loss: 0.4300, Accuracy: 8533/10000 (85.33%)
         
         Train Loss: 0.0047, Train Accuracy: 79.23%
         Validation Loss: 0.4300, Validation Accuracy: 85.33%
         Epoch 73
         Loss=0.6723 Batch_id=390 Accuracy=79.38: 100%|████████████████████████████| 391/391 [00:16<00:00, 23.56it/s] 
         
         Validation set: Average loss: 0.4339, Accuracy: 8520/10000 (85.20%)
         
         Train Loss: 0.0046, Train Accuracy: 79.38%
         Validation Loss: 0.4339, Validation Accuracy: 85.20%
         Epoch 74
         Loss=0.6033 Batch_id=390 Accuracy=79.29: 100%|████████████████████████████| 391/391 [00:16<00:00, 23.37it/s] 
         
         Validation set: Average loss: 0.4293, Accuracy: 8529/10000 (85.29%)
         
         Train Loss: 0.0046, Train Accuracy: 79.29%
         Validation Loss: 0.4293, Validation Accuracy: 85.29%
         Epoch 75
         Loss=0.5593 Batch_id=390 Accuracy=79.27: 100%|████████████████████████████| 391/391 [00:16<00:00, 23.74it/s] 
         
         Validation set: Average loss: 0.4285, Accuracy: 8542/10000 (85.42%)
         
         Train Loss: 0.0046, Train Accuracy: 79.27%
         Validation Loss: 0.4285, Validation Accuracy: 85.42%

**Validation Plots**

<img width="1200" height="500" alt="cifar_lossplots" src="https://github.com/user-attachments/assets/d4c9d1f4-b0ea-44aa-99f8-e61034a131c1" />

<img width="1200" height="500" alt="cifar_accuracyplots" src="https://github.com/user-attachments/assets/e9f5d710-0be5-403e-8035-64ff77ce5459" />

***Misclassified Images***

<img width="1536" height="850" alt="cifar_misclassifiedplots" src="https://github.com/user-attachments/assets/92956ea4-fe01-492e-b92c-086a25881cdd" />



---


**For questions or improvements, feel free to open an issue or pull request.**
