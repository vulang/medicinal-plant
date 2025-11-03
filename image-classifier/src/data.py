import os
from typing import Tuple
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def build_transforms(img_size: int = 224, is_train: bool = True):
    if is_train:
        tfm = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
            transforms.ToTensor()
        ])
    else:
        tfm = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])
    return tfm

def build_dataloaders(train_dir: str, val_dir: str, img_size: int, batch_size: int, num_workers: int):
    train_ds = datasets.ImageFolder(train_dir, transform=build_transforms(img_size, is_train=True))
    val_ds = datasets.ImageFolder(val_dir, transform=build_transforms(img_size, is_train=False))

    class_names = train_ds.classes
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, class_names

def build_testloader(test_dir: str, img_size: int, batch_size: int, num_workers: int, class_names):
    test_ds = datasets.ImageFolder(test_dir, transform=build_transforms(img_size, is_train=False))
    # Ensure class order matches training
    assert test_ds.classes == class_names, "Test classes differ from training classes."
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return test_loader
