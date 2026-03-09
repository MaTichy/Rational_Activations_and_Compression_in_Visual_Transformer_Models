"""Dataset loading utilities for training and evaluation."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as T


def get_cifar10(batch_size: int = 64, num_workers: int = 4, data_dir: str = "./data") -> tuple[DataLoader, DataLoader, DataLoader, int]:
    """Load CIFAR-10 with standard normalization.

    Returns:
        Tuple of (train_loader, val_loader, test_loader, num_classes).
    """
    normalize = T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])

    train_transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize,
    ])
    test_transform = T.Compose([T.ToTensor(), normalize])

    train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    test_set = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)

    # Split train into train/val (90/10)
    train_len = int(0.9 * len(train_set))
    val_len = len(train_set) - train_len
    train_subset, val_subset = random_split(train_set, [train_len, val_len])

    pin = torch.cuda.is_available()
    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin)
    train_loader = DataLoader(train_subset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_subset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader, 10


def get_imagenette(batch_size: int = 64, num_workers: int = 4, data_dir: str = "./data/imagenette") -> tuple[DataLoader, DataLoader, DataLoader, int]:
    """Load Imagenette (10-class subset of ImageNet) at 160px resolution.

    Returns:
        Tuple of (train_loader, val_loader, test_loader, num_classes).
    """
    img_size = 160
    normalize = T.Normalize(mean=[0.460, 0.455, 0.427], std=[0.229, 0.223, 0.231])

    train_transform = T.Compose([
        T.RandomResizedCrop(img_size),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize,
    ])
    test_transform = T.Compose([T.Resize((img_size, img_size)), T.ToTensor(), normalize])

    train_set = torchvision.datasets.ImageFolder(f"{data_dir}/train", transform=train_transform)
    test_set = torchvision.datasets.ImageFolder(f"{data_dir}/val", transform=test_transform)

    train_len = int(0.9 * len(train_set))
    val_len = len(train_set) - train_len
    train_subset, val_subset = random_split(train_set, [train_len, val_len])

    pin = torch.cuda.is_available()
    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin)
    train_loader = DataLoader(train_subset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_subset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader, 10
