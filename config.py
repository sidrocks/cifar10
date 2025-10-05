import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 75
BATCH_SIZE = 128
MEAN = (0.49139968, 0.48215841, 0.44653091)
STD = (0.24703223, 0.24348513, 0.26158784)
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']