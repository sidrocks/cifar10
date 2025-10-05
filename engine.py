import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from config import CLASSES

def train(model, device, train_loader, optimizer, scheduler):
    model.train()
    pbar = tqdm(train_loader)
    train_loss = 0
    correct = 0
    processed = 0
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
        pbar.set_description(desc= f'Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    train_loss /= len(train_loader.dataset)
    train_accuracy = 100. * correct / len(train_loader.dataset)
    return train_loss, train_accuracy

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, test_accuracy

def get_misclassified(model, loader, device):
    model.eval()
    misclassified = []
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            wrong_idx = pred.squeeze().ne(target).nonzero(as_tuple=True)[0]
            for idx in wrong_idx:
                misclassified.append((data[idx].cpu(), pred[idx].item(), target[idx].item()))
    return misclassified

def plot_misclassified(misclassified, n=25):
    plt.figure(figsize=(10,10))
    for i, (img, pred, true) in enumerate(misclassified[:n]):
        plt.subplot(5,5,i+1)
        plt.imshow(img.permute(1, 2, 0), cmap="gray")
        plt.title(f"P:{CLASSES[pred]}, T:{CLASSES[true]}")
        plt.axis("off")
    plt.show()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)