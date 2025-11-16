from typing import Optional, Dict, Any, List, Tuple
from PIL import Image, UnidentifiedImageError
import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from torchvision import datasets, transforms
from torchvision.datasets.folder import IMG_EXTENSIONS
from torchvision.transforms.functional import InterpolationMode

try:
    from timm.data import create_transform
except ImportError:
    create_transform = None

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_INTERP_MAP = {
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
    "nearest": InterpolationMode.NEAREST,
}

def _get_interp(mode: str):
    return _INTERP_MAP.get(mode, InterpolationMode.BILINEAR)


def _is_valid_image(path: str) -> bool:
    if not path.lower().endswith(IMG_EXTENSIONS):
        return False
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except (UnidentifiedImageError, OSError):
        print(f"[WARN] Skipping unreadable image: {path}")
        return False

def build_transforms(
    img_size: int = 224,
    is_train: bool = True,
    data_cfg: Optional[Dict[str, Any]] = None,
    use_timm_augment: bool = False
):
    """
    Build preprocessing pipeline. For training we default to the gentler
    augmentations that worked for the original scaffold, but you can enable
    the timm recipe for models such as ConvNeXt V2 by turning on
    `use_timm_augment` in the config.
    """
    mean = data_cfg.get("mean", IMAGENET_MEAN) if data_cfg else IMAGENET_MEAN
    std = data_cfg.get("std", IMAGENET_STD) if data_cfg else IMAGENET_STD
    interpolation = _get_interp((data_cfg or {}).get("interpolation", "bilinear"))

    if is_train and use_timm_augment and data_cfg and create_transform is not None:
        cfg = data_cfg.copy()
        in_ch, _, _ = cfg.get("input_size", (3, img_size, img_size))
        cfg["input_size"] = (in_ch, img_size, img_size)
        return create_transform(**cfg, is_training=True)

    if is_train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=interpolation),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    resize_size = img_size
    crop_pct = (data_cfg or {}).get("crop_pct", 1.0)
    if crop_pct < 1.0:
        resize_size = int(img_size / crop_pct)

    if crop_pct < 1.0:
        eval_ops = [
            transforms.Resize(resize_size, interpolation=interpolation),
            transforms.CenterCrop(img_size),
        ]
    else:
        # When no crop_pct is provided ensure deterministic resize to fixed edge
        eval_ops = [
            transforms.Resize((img_size, img_size), interpolation=interpolation),
        ]
    eval_ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return transforms.Compose(eval_ops)

def _clone_collate(batch: List[Tuple[torch.Tensor, int]]):
    """
    Clone tensors while forcing them onto fresh storage so that
    PyTorch's shared-memory resize step in worker processes does
    not trip over non-resizable buffers produced by PIL->Tensor.
    """
    images, targets = zip(*batch)
    stacked = torch.stack([img.detach().clone() for img in images], dim=0)
    return stacked, default_collate(targets)


def build_dataloaders(
    train_dir: str,
    val_dir: str,
    img_size: int,
    batch_size: int,
    num_workers: int,
    data_cfg: Optional[Dict[str, Any]] = None,
    use_timm_augment: bool = False
):
    collate = _clone_collate
    # Some classes currently have empty folders; allow_empty avoids hard failures while keeping class ordering consistent.
    train_ds = datasets.ImageFolder(
        train_dir,
        transform=build_transforms(img_size, is_train=True, data_cfg=data_cfg, use_timm_augment=use_timm_augment),
        is_valid_file=_is_valid_image,
        allow_empty=True
    )
    val_ds = datasets.ImageFolder(
        val_dir,
        transform=build_transforms(img_size, is_train=False, data_cfg=data_cfg, use_timm_augment=use_timm_augment),
        is_valid_file=_is_valid_image,
        allow_empty=True
    )

    class_names = train_ds.classes
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate
    )
    return train_loader, val_loader, class_names

def build_testloader(
    test_dir: str,
    img_size: int,
    batch_size: int,
    num_workers: int,
    class_names,
    data_cfg: Optional[Dict[str, Any]] = None,
    use_timm_augment: bool = False
):
    test_ds = datasets.ImageFolder(
        test_dir,
        transform=build_transforms(img_size, is_train=False, data_cfg=data_cfg, use_timm_augment=use_timm_augment),
        is_valid_file=_is_valid_image,
        allow_empty=True
    )
    # Ensure class order matches training
    assert test_ds.classes == class_names, "Test classes differ from training classes."
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_clone_collate
    )
    return test_loader
