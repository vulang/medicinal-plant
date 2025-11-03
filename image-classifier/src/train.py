import os
import yaml
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .data import build_dataloaders
from .model import build_model
from .utils import set_seed, resolve_device

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="Train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="Val", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total

def main(cfg_path: str = "config.yaml"):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))
    device = resolve_device(cfg.get("device", "auto"))
    os.makedirs(cfg["save_dir"], exist_ok=True)
    os.makedirs(cfg["outputs_dir"], exist_ok=True)

    train_loader, val_loader, class_names = build_dataloaders(
        cfg["train_dir"], cfg["val_dir"], cfg["img_size"], cfg["batch_size"], cfg["num_workers"]
    )
    num_classes = len(class_names)

    model = build_model(cfg["model_name"], num_classes, pretrained=cfg.get("pretrained", True)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    best_val_acc = 0.0
    best_ckpt = os.path.join(cfg["save_dir"], f"{cfg['model_name']}_best.pt")

    for epoch in range(1, cfg["epochs"] + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch}/{cfg['epochs']} | "
              f"Train Loss {train_loss:.4f} Acc {train_acc:.4f} | "
              f"Val Loss {val_loss:.4f} Acc {val_acc:.4f}")

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "classes": class_names,
                "config": cfg
            }, best_ckpt)

    # Save last checkpoint
    last_ckpt = os.path.join(cfg["save_dir"], f"{cfg['model_name']}_last.pt")
    torch.save({"model_state": model.state_dict(), "classes": class_names, "config": cfg}, last_ckpt)
    print(f"Best val acc: {best_val_acc:.4f}. Saved best -> {best_ckpt}")
    print(f"Saved last -> {last_ckpt}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
