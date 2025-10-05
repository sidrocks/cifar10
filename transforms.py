import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from config import MEAN, STD

def _coarse_dropout_fill_value_from_mean(mean_rgb):
    return tuple(int(m * 255.0) for m in mean_rgb)

class Cifar10Albumentations:
    def __init__(self, mean, std):
        fill_value = _coarse_dropout_fill_value_from_mean(mean)
        self.train_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Affine(translate_percent=0.0625, scale=(0.9, 1.1), rotate=(-15, 15), p=0.5),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2, p=0.25),
            A.ToGray(p=0.15),
            A.CoarseDropout(num_holes_range=(1, 1), hole_height_range=(16, 16), hole_width_range=(16, 16), p=0.5, fill=fill_value),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
        self.test_transform = A.Compose([
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])

    def __call__(self, img, train=True):
        img = np.array(img)
        if train:
            return self.train_transform(image=img)['image']
        else:
            return self.test_transform(image=img)['image']

class CutMix:
    def __init__(self, alpha=1.0, p=0.5):
        self.alpha = alpha
        self.p = p

    def _rand_bbox(self, size, lam):
        W, H = size[2], size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        x1 = np.clip(cx - cut_w // 2, 0, W)
        y1 = np.clip(cy - cut_h // 2, 0, H)
        x2 = np.clip(cx + cut_w // 2, 0, W)
        y2 = np.clip(cy + cut_h // 2, 0, H)
        return x1, y1, x2, y2

    def __call__(self, x, y):
        if np.random.rand() > self.p or self.alpha <= 0:
            return x, y, y, 1.0
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size, device=x.device)
        x1, y1, x2, y2 = self._rand_bbox(x.size(), lam)
        x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
        lam = 1 - ((x2 - x1) * (y2 - y1) / (x.size(-1) * x.size(-2)))
        y_a, y_b = y, y[index]
        return x, y_a, y_b, lam