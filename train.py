import torch
from torchvision import datasets
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchsummary import summary

from config import DEVICE, EPOCHS, BATCH_SIZE, MEAN, STD
from model import Net
from transforms import Cifar10Albumentations
from engine import train, test, get_misclassified, plot_misclassified, count_parameters

def main():
    albumentations_transform = Cifar10Albumentations(MEAN, STD)
    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=lambda img: albumentations_transform(img, train=True))
    test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=lambda img: albumentations_transform(img, train=False))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model = Net().to(DEVICE)
    optimizer = SGD(model.parameters(), lr=0.07, momentum=0.9, weight_decay=7e-5, nesterov=True)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=0.0005)

    print(summary(model, input_size=(3, 32, 32)))
    print(f"\nTotal trainable parameters: {count_parameters(model):,}")

    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch}")
        train_loss, train_accuracy = train(model, DEVICE, train_loader, optimizer, scheduler)
        test_loss, test_accuracy = test(model, DEVICE, test_loader)
        scheduler.step()
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
        print(f"Validation Loss: {test_loss:.4f}, Validation Accuracy: {test_accuracy:.2f}%")

    misclassified = get_misclassified(model, test_loader, DEVICE)
    plot_misclassified(misclassified, n=25)

if __name__ == '__main__':
    main()