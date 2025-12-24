import os
import copy
import yaml
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch import amp
from sklearn.metrics import f1_score
import mlflow
import mlflow.pytorch
from mlflow.models import infer_signature

try:
    from timm.data import Mixup
    from timm.loss import SoftTargetCrossEntropy
except ImportError:
    Mixup = None
    SoftTargetCrossEntropy = None

from .data import build_dataloaders
from .model import build_model
from .utils import set_seed, resolve_device


def flatten_config(config, parent_key="", sep="."):
    """Flatten nested config dictionaries so they can be logged to MLflow."""
    items = {}
    for key, value in config.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            items.update(flatten_config(value, new_key, sep=sep))
        else:
            if isinstance(value, (list, tuple)):
                items[new_key] = ",".join(map(str, value))
            else:
                items[new_key] = value
    return items


def freeze_layers(model, freeze: bool = True):
    """Freeze/unfreeze backbone layers for two-phase training."""
    if hasattr(model, "features"):
        modules = model.features
    elif hasattr(model, "backbone"):
        modules = model.backbone
    else:
        # Fallback: treat everything except final layer as backbone
        children = list(model.children())
        modules = children[:-1]

    for module in modules if isinstance(modules, (list, tuple)) else [modules]:
        for param in module.parameters():
            param.requires_grad = not freeze


def compute_topk_accuracy(outputs, labels, k: int = 5):
    """Compute top-k accuracy."""
    with torch.no_grad():
        batch_size = labels.size(0)
        k = min(k, outputs.size(1))
        _, topk_preds = outputs.topk(k, dim=1, largest=True, sorted=True)
        correct = topk_preds.eq(labels.view(-1, 1).expand_as(topk_preds))
        return correct.any(dim=1).float().sum().item() / batch_size


class FocalLoss(nn.Module):
    """Standard focal loss for hard labels."""

    def __init__(self, gamma: float = 2.0, alpha: float | None = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(inputs, dim=1)
        probs = log_probs.exp()
        targets = targets.long()

        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        alpha_factor = 1.0
        if self.alpha is not None:
            alpha_factor = self.alpha

        loss = -alpha_factor * ((1 - pt) ** self.gamma) * log_pt

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    use_amp=True,
    max_grad_norm=1.0,
    mixup_fn=None
):
    """Train for one epoch with optional mixed precision, mixup/cutmix, and grad clipping."""
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    top5_correct = 0.0
    scaler = amp.GradScaler(device.type) if use_amp and device.type == "cuda" else None

    for images, labels in tqdm(loader, desc="Train", leave=False):
        batch_images, batch_labels = images.to(device), labels.to(device)

        if mixup_fn is not None and batch_images.size(0) % 2 != 0:
            # timm Mixup requires an even batch size; trim one sample if needed.
            batch_images = batch_images[:-1]
            batch_labels = batch_labels[:-1]
            if batch_images.numel() == 0:
                continue

        inputs, targets = (mixup_fn(batch_images, batch_labels) if mixup_fn is not None
                           else (batch_images, batch_labels))
        optimizer.zero_grad()

        if scaler is not None:
            with amp.autocast(device_type=device.type):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            if max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        batch_size = batch_labels.size(0)
        running_loss += loss.item() * batch_size
        preds = outputs.argmax(dim=1)
        correct += (preds == batch_labels).sum().item()
        total += batch_size
        top5_correct += compute_topk_accuracy(outputs, batch_labels, k=5) * batch_size

    return running_loss / total, correct / total, top5_correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate model."""
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    top5_correct = 0.0
    all_preds, all_labels = [], []

    for images, labels in tqdm(loader, desc="Val", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        top5_correct += compute_topk_accuracy(outputs, labels, k=5) * images.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    val_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return running_loss / total, correct / total, top5_correct / total, val_f1


def main(cfg_path: str = "config.yaml"):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    resume_from = cfg.get("resume_from")
    if resume_from:
        resume_from = os.path.expanduser(resume_from)

    set_seed(cfg.get("seed", 42))
    device = resolve_device(cfg.get("device", "auto"))
    os.makedirs(cfg["save_dir"], exist_ok=True)
    os.makedirs(cfg["outputs_dir"], exist_ok=True)

    mlflow_cfg = cfg.get("mlflow") or {}
    mlflow_enabled = mlflow_cfg.get("enabled", False)
    artifact_subdir = mlflow_cfg.get("artifact_subdir") or "checkpoints"
    register_model = mlflow_cfg.get("register_model", False)
    registered_model_name = mlflow_cfg.get("registered_model_name")
    model_artifact_subdir = mlflow_cfg.get("model_artifact_subdir") or "model"
    mlflow_run = None
    if mlflow_enabled:
        tracking_uri = mlflow_cfg.get("tracking_uri")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        experiment_name = mlflow_cfg.get("experiment_name")
        if experiment_name:
            mlflow.set_experiment(experiment_name)
        run_name = mlflow_cfg.get("run_name")
        mlflow_run = mlflow.start_run(run_name=run_name)
        tags = mlflow_cfg.get("tags")
        if tags and isinstance(tags, dict):
            mlflow.set_tags(tags)

    class_dirs = sorted(entry.name for entry in os.scandir(cfg["train_dir"]) if entry.is_dir())
    if not class_dirs:
        raise RuntimeError(f"No class folders found under {cfg['train_dir']}.")
    num_classes = len(class_dirs)
    print(f"Found {num_classes} classes.")

    model = build_model(cfg["model_name"], num_classes, pretrained=cfg.get("pretrained", True)).to(device)
    data_cfg = getattr(model, "data_config", None)

    mixup_alpha = cfg.get("mixup_alpha", 0.0)
    cutmix_alpha = cfg.get("cutmix_alpha", 0.0)
    mixup_prob = cfg.get("mixup_prob", 0.0)
    mixup_switch_prob = cfg.get("mixup_switch_prob", 0.5)
    mixup_mode = cfg.get("mixup_mode", "batch")
    mixup_active = (mixup_alpha > 0 or cutmix_alpha > 0) and mixup_prob > 0
    if mixup_active and cfg["batch_size"] % 2 != 0:
        print(f"[WARN] Mixup/CutMix enabled but batch_size ({cfg['batch_size']}) is odd. "
              "Dropping the last sample of odd batches to keep pairs.")

    train_loader, val_loader, class_names = build_dataloaders(
        cfg["train_dir"],
        cfg["val_dir"],
        cfg["img_size"],
        cfg["batch_size"],
        cfg["num_workers"],
        data_cfg=data_cfg,
        use_timm_augment=cfg.get("use_timm_augment", False),
        drop_last_train=mixup_active,
        drop_last_val=False,
        use_weighted_sampler=cfg.get("use_weighted_sampler", False)
    )
    if class_names != class_dirs:
        print("[WARN] Class order from dataloader differed from directory listing; using loader order.")

    if mlflow_enabled:
        params_to_log = flatten_config({k: v for k, v in cfg.items() if k != "mlflow"})
        for key, value in params_to_log.items():
            if value is None:
                continue
            mlflow.log_param(key, value)
        mlflow.log_param("num_classes", num_classes)
        mlflow.log_param("classes", ",".join(class_names))
        mlflow.log_dict(cfg, os.path.join("configs", "config_used.yaml"))

    mixup_fn = None
    if mixup_active:
        if Mixup is None or SoftTargetCrossEntropy is None:
            raise ImportError("timm is required for mixup/cutmix training. Install it with `pip install timm`.")
        mixup_fn = Mixup(
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            prob=mixup_prob,
            switch_prob=mixup_switch_prob,
            mode=mixup_mode,
            label_smoothing=cfg.get("label_smoothing", 0.0),
            num_classes=num_classes
        )
        train_criterion = SoftTargetCrossEntropy()
    else:
        loss_type = cfg.get("loss_type", "cross_entropy")
        if loss_type == "focal":
            train_criterion = FocalLoss(
                gamma=cfg.get("focal_gamma", 2.0),
                alpha=cfg.get("focal_alpha"),
                reduction="mean",
            )
        else:
            train_criterion = nn.CrossEntropyLoss(label_smoothing=cfg.get("label_smoothing", 0.0))
    val_criterion = nn.CrossEntropyLoss(label_smoothing=cfg.get("label_smoothing", 0.0))
    optimizer = AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg.get("weight_decay", 0.01)
    )

    warmup_epochs = cfg.get("warmup_epochs", 5)
    total_epochs = cfg.get("epochs", 10)
    if warmup_epochs > 0:
        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
        main_scheduler = CosineAnnealingLR(optimizer, T_max=max(total_epochs - warmup_epochs, 1))
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_epochs]
        )
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=max(total_epochs, 1))

    finetune_lr = cfg.get("finetune_lr")
    use_amp = cfg.get("use_amp", True) and device.type == "cuda"
    max_grad_norm = cfg.get("max_grad_norm", 1.0)
    warmup_phase1_epochs = cfg.get("warmup_epochs_phase1", 0)
    patience = cfg.get("patience", 10)
    best_ckpt_interval = max(1, int(cfg.get("best_checkpoint_interval", 10)))

    best_val_acc = 0.0
    best_val_top5 = 0.0
    best_val_f1 = 0.0
    patience_counter = 0
    best_ckpt = os.path.join(cfg["save_dir"], f"{cfg['model_name']}_best.pt")

    resume_checkpoint = None
    resume_epoch = 0
    completed_phase1 = 0
    completed_phase2 = 0
    if resume_from:
        if not os.path.isfile(resume_from):
            raise FileNotFoundError(f"Resume checkpoint not found at {resume_from}")
        print(f"[INFO] Loading checkpoint from {resume_from}")
        resume_checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(resume_checkpoint["model_state"])
        if "optimizer_state" in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
        scheduler_loaded = False
        if "scheduler_state" in resume_checkpoint:
            try:
                scheduler.load_state_dict(resume_checkpoint["scheduler_state"])
                scheduler_loaded = True
            except Exception as e:
                if finetune_lr:
                    try:
                        scheduler = CosineAnnealingLR(optimizer, T_max=max(total_epochs, 1))
                        scheduler.load_state_dict(resume_checkpoint["scheduler_state"])
                        scheduler_loaded = True
                        print("[INFO] Recreated cosine scheduler for fine-tuning resume.")
                    except Exception as inner_exc:
                        print(f"[WARN] Could not load scheduler state even after rebuild: {inner_exc}")
                if not scheduler_loaded:
                    print(f"[WARN] Could not load scheduler state from checkpoint: {e}")
        resume_epoch = int(resume_checkpoint.get("epoch", 0))
        best_val_acc = resume_checkpoint.get("best_val_acc", 0.0)
        best_val_top5 = resume_checkpoint.get("best_val_top5", 0.0)
        best_val_f1 = resume_checkpoint.get("best_val_f1", 0.0)
        patience_counter = resume_checkpoint.get("patience_counter", 0)
        ckpt_classes = resume_checkpoint.get("classes")
        if ckpt_classes and ckpt_classes != class_names:
            print("[WARN] Class order in checkpoint differs from current dataloader order.")
        completed_phase1 = min(resume_epoch, warmup_phase1_epochs)
        completed_phase2 = max(resume_epoch - warmup_phase1_epochs, 0)

    print("=" * 70)
    print("Training Configuration")
    print(f"  Model: {cfg['model_name']}")
    print(f"  Classes: {num_classes}")
    print(f"  Device: {device}")
    print(f"  Mixed Precision: {use_amp}")
    print(f"  Phase1 (head) epochs: {warmup_phase1_epochs}")
    print(f"  Total fine-tune epochs: {total_epochs}")
    print(f"  Initial LR: {cfg['learning_rate']}")
    print(f"  Weight Decay: {cfg.get('weight_decay', 0.01)}")
    print(f"  Label Smoothing: {cfg.get('label_smoothing', 0.0)}")
    print(f"  Early stop patience: {patience}")
    print(f"  Best checkpoint interval: {best_ckpt_interval} epochs")
    if resume_from:
        print(f"  Resume from: {resume_from} (epoch {resume_epoch})")
    if mixup_active:
        print(f"  Mixup/CutMix: mixup_alpha={mixup_alpha}, cutmix_alpha={cutmix_alpha}, prob={mixup_prob}, "
              f"switch_prob={mixup_switch_prob}, mode={mixup_mode}")
    print("=" * 70)

    total_planned_epochs = warmup_phase1_epochs + total_epochs
    if resume_epoch >= total_planned_epochs:
        print(f"[INFO] Checkpoint epoch {resume_epoch} meets or exceeds configured total ({total_planned_epochs}). Nothing to train.")
        return

    global_epoch = resume_epoch

    def log_epoch_metrics(step, metrics):
        if mlflow_enabled:
            mlflow.log_metrics(metrics, step=step)

    best_state = None
    best_state_pending = False

    def stage_best_checkpoint(epoch, val_metrics):
        """Capture the current best state so it can be saved on interval."""
        nonlocal best_state, best_state_pending
        best_state = {
            "epoch": epoch,
            "model_state": copy.deepcopy(model.state_dict()),
            "optimizer_state": copy.deepcopy(optimizer.state_dict()),
            "best_val_acc": val_metrics["val_acc"],
            "best_val_top5": val_metrics["val_top5"],
            "best_val_f1": val_metrics["val_f1"],
            "patience_counter": 0,
            "scheduler_state": copy.deepcopy(scheduler.state_dict()),
            "classes": class_names,
            "config": cfg
        }
        best_state_pending = True

    def maybe_save_best(force: bool = False):
        """Persist staged best checkpoint based on interval or when forced."""
        nonlocal best_state_pending
        if best_state is None:
            return
        if not force and not best_state_pending:
            return
        if not force and (global_epoch % best_ckpt_interval != 0):
            return
        torch.save(best_state, best_ckpt)
        best_state_pending = False
        print(f"  → Saved new best model (Val Acc {best_state['best_val_acc']:.4f}, "
              f"Top5 {best_state['best_val_top5']:.4f}, F1 {best_state['best_val_f1']:.4f})")
        if mlflow_enabled:
            mlflow.log_artifact(best_ckpt, artifact_path=artifact_subdir)

    # Phase 1: train classifier head only
    if warmup_phase1_epochs > completed_phase1:
        print("\n" + "*" * 70)
        print(f"PHASE 1: Training classifier head for {warmup_phase1_epochs} epochs (starting at {completed_phase1 + 1})")
        print("*" * 70)
        freeze_layers(model, freeze=True)

        for epoch in range(completed_phase1 + 1, warmup_phase1_epochs + 1):
            train_loss, train_acc, train_top5 = train_one_epoch(
                model, train_loader, train_criterion, optimizer, device, use_amp, max_grad_norm, mixup_fn
            )
            val_loss, val_acc, val_top5, val_f1 = validate(model, val_loader, val_criterion, device)
            scheduler.step()

            lr = optimizer.param_groups[0]["lr"]
            print(f"[Phase1] Epoch {epoch}/{warmup_phase1_epochs} | LR {lr:.2e} | "
                  f"Train Loss {train_loss:.4f} Acc {train_acc:.4f} Top5 {train_top5:.4f} | "
                  f"Val Loss {val_loss:.4f} Acc {val_acc:.4f} Top5 {val_top5:.4f} F1 {val_f1:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_top5 = val_top5
                best_val_f1 = val_f1

            log_epoch_metrics(global_epoch + 1, {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "train_top5": train_top5,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_top5": val_top5,
                "val_f1": val_f1,
                "learning_rate": lr,
            })
            global_epoch += 1

    elif warmup_phase1_epochs > 0:
        print("\n" + "*" * 70)
        print(f"PHASE 1: Skipping (already completed {completed_phase1} / {warmup_phase1_epochs} epochs in checkpoint)")
        print("*" * 70)

    # Phase 2: fine-tune entire network
    print("\n" + "*" * 70)
    print("PHASE 2: Fine-tuning all layers")
    print(f"Starting at epoch {completed_phase2 + 1}/{total_epochs}")
    print("*" * 70)
    freeze_layers(model, freeze=False)

    apply_finetune_lr = finetune_lr and (resume_checkpoint is None or resume_epoch < warmup_phase1_epochs)
    if apply_finetune_lr:
        for param_group in optimizer.param_groups:
            param_group["lr"] = finetune_lr
        scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs)
        print(f"Reduced LR to {finetune_lr:.2e} for fine-tuning.")
    elif finetune_lr and resume_checkpoint is not None:
        print(f"Keeping optimizer LR from checkpoint (fine-tune LR already applied in {resume_from}).")

    for epoch in range(completed_phase2 + 1, total_epochs + 1):
        train_loss, train_acc, train_top5 = train_one_epoch(
            model, train_loader, train_criterion, optimizer, device, use_amp, max_grad_norm, mixup_fn
        )
        val_loss, val_acc, val_top5, val_f1 = validate(model, val_loader, val_criterion, device)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        print(f"[Phase2] Epoch {epoch}/{total_epochs} | LR {lr:.2e} | "
              f"Train Loss {train_loss:.4f} Acc {train_acc:.4f} Top5 {train_top5:.4f} | "
              f"Val Loss {val_loss:.4f} Acc {val_acc:.4f} Top5 {val_top5:.4f} F1 {val_f1:.4f}")

        log_epoch_metrics(global_epoch + 1, {
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_top5": train_top5,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_top5": val_top5,
            "val_f1": val_f1,
            "learning_rate": lr,
        })
        global_epoch += 1
        current_epoch_total = global_epoch

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_top5 = val_top5
            best_val_f1 = val_f1
            patience_counter = 0
            stage_best_checkpoint(current_epoch_total, {
                "val_acc": best_val_acc,
                "val_top5": best_val_top5,
                "val_f1": best_val_f1,
            })
            maybe_save_best()
            if mlflow_enabled:
                mlflow.log_metrics({
                    "best_val_acc": best_val_acc,
                    "best_val_top5": best_val_top5,
                    "best_val_f1": best_val_f1,
                }, step=global_epoch)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                maybe_save_best(force=True)
                print("=" * 70)
                print(f"Early stopping triggered after {global_epoch} total epochs.")
                print("=" * 70)
                break
        maybe_save_best()

    maybe_save_best(force=True)

    last_ckpt = os.path.join(cfg["save_dir"], f"{cfg['model_name']}_last.pt")
    torch.save({
        "epoch": global_epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "val_acc": val_acc,
        "val_top5": val_top5,
        "val_f1": val_f1,
        "best_val_acc": best_val_acc,
        "best_val_top5": best_val_top5,
        "best_val_f1": best_val_f1,
        "patience_counter": patience_counter,
        "scheduler_state": scheduler.state_dict(),
        "classes": class_names,
        "config": cfg
    }, last_ckpt)
    if mlflow_enabled:
        mlflow.log_artifact(last_ckpt, artifact_path=artifact_subdir)

    print("=" * 70)
    print("Training complete")
    print(f"  Best Val Accuracy: {best_val_acc:.4f}")
    print(f"  Best Val Top-5 Accuracy: {best_val_top5:.4f}")
    print(f"  Best Val F1 Score: {best_val_f1:.4f}")
    print(f"  Best checkpoint: {best_ckpt}")
    print(f"  Last checkpoint: {last_ckpt}")
    print("=" * 70)

    if mlflow_enabled:
        if register_model and registered_model_name:
            try:
                ckpt_to_register = best_ckpt if os.path.exists(best_ckpt) else last_ckpt
                reg_ckpt = torch.load(ckpt_to_register, map_location="cpu")
                reg_model = build_model(cfg["model_name"], len(reg_ckpt["classes"]), pretrained=False)
                reg_model.load_state_dict(reg_ckpt["model_state"])
                reg_model.eval()

                # Build a minimal input_example and signature so MLflow can infer schema.
                input_example = None
                signature = None
                try:
                    data_cfg = getattr(reg_model, "data_config", {}) or {}
                    input_size = data_cfg.get("input_size") if isinstance(data_cfg, dict) else None
                    if not input_size:
                        img_size = cfg.get("img_size")
                        if isinstance(img_size, int):
                            input_size = (3, img_size, img_size)
                    if not input_size:
                        input_size = (3, 224, 224)

                    example_input = torch.zeros((1, *input_size), dtype=torch.float32)
                    with torch.no_grad():
                        example_output = reg_model(example_input)
                    input_example = example_input.numpy()
                    signature = infer_signature(input_example, example_output.detach().numpy())
                except Exception as signature_exc:
                    print(f"[WARN] Could not build MLflow input_example/signature: {signature_exc}")

                # Log the model artifact first
                mlflow.pytorch.log_model(
                    reg_model,
                    artifact_path=model_artifact_subdir,
                    signature=signature,
                    input_example=input_example,
                )
                artifact_uri = mlflow.get_artifact_uri(model_artifact_subdir)

                # Try to register the logged artifact
                try:
                    mlflow.register_model(model_uri=artifact_uri, name=registered_model_name)
                    print(f"Registered model to MLflow Model Registry as '{registered_model_name}'.")
                except Exception as reg_exc:
                    msg = str(reg_exc)
                    if "logged-models" in msg or "404" in msg:
                        print("[WARN] MLflow server does not expose the Model Registry endpoints (logged-models). "
                              "Model artifacts were logged but not registered. "
                              "Upgrade/enable the MLflow Model Registry or set mlflow.register_model=false.")
                    else:
                        print(f"[WARN] Failed to register model to MLflow: {reg_exc}")
            except Exception as e:
                print(f"[WARN] Failed to log/register model to MLflow: {e}")

        mlflow.log_metrics({
            "final_val_acc": val_acc,
            "final_val_top5": val_top5,
            "final_val_f1": val_f1,
            "best_val_acc": best_val_acc,
            "best_val_top5": best_val_top5,
            "best_val_f1": best_val_f1,
        }, step=global_epoch)

    if mlflow_run is not None:
        mlflow.end_run()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train medicinal plant classifier")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
