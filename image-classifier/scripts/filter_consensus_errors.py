"""Move consensus high-confidence label errors into a quarantine folder.

Both ConvNeXt and Swin must:
- Predict the same label (top-1)
- Assign at least `--min-confidence` probability to that label
- Disagree with the folder name (assumed ground-truth label)

Matched images are moved to `suspect_images` (or a custom path) so they can be
reviewed manually instead of being deleted outright.
"""

import argparse
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
import yaml
from PIL import Image, UnidentifiedImageError

# Ensure local src/ is importable when running as a script
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data import IMG_EXTENSIONS, build_transforms
from src.model import build_model
from src.utils import resolve_device


@dataclass
class ModelBundle:
    name: str
    model: torch.nn.Module
    transform: any  # torchvision transform
    classes: List[str]
    device: torch.device


def _load_model_bundle(model_cfg: dict, global_cfg: dict, device: torch.device) -> Tuple[ModelBundle, List[str]]:
    """Load a single model checkpoint and its transform."""
    model_name = model_cfg["model_name"]
    ckpt_path = model_cfg.get("checkpoint") or os.path.join(global_cfg["save_dir"], f"{model_name}_best.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    classes = ckpt["classes"]

    model = build_model(model_name, len(classes), pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    data_cfg = getattr(model, "data_config", None)

    transform = build_transforms(
        global_cfg["img_size"],
        is_train=False,
        data_cfg=data_cfg,
        use_timm_augment=global_cfg.get("use_timm_augment", False),
    )

    return ModelBundle(name=model_name, model=model, transform=transform, classes=classes, device=device), classes


def load_models(cfg: dict, device: torch.device) -> Tuple[List[ModelBundle], List[str]]:
    ensemble_cfg = cfg.get("ensemble_models") or []
    if len(ensemble_cfg) < 2:
        raise ValueError("Ensemble config must list at least two models for consensus filtering.")

    bundles: List[ModelBundle] = []
    class_names: List[str] | None = None
    for model_cfg in ensemble_cfg:
        bundle, classes = _load_model_bundle(model_cfg, cfg, device)
        if class_names is None:
            class_names = classes
        elif classes != class_names:
            raise ValueError("Class ordering mismatch across ensemble checkpoints.")
        bundles.append(bundle)

    return bundles, class_names or []


def predict_one(img: Image.Image, bundle: ModelBundle) -> Tuple[str, float]:
    x = bundle.transform(img).unsqueeze(0).to(bundle.device)
    with torch.inference_mode():
        probs = bundle.model(x).softmax(dim=1)
    top_prob, top_idx = probs.squeeze(0).max(dim=0)
    return bundle.classes[int(top_idx.item())], float(top_prob.item())


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def unique_destination(dest_dir: str, filename: str) -> str:
    base, ext = os.path.splitext(filename)
    candidate = os.path.join(dest_dir, filename)
    counter = 1
    while os.path.exists(candidate):
        candidate = os.path.join(dest_dir, f"{base}_{counter}{ext}")
        counter += 1
    return candidate


def move_consensus_errors(
    source_dir: str,
    dest_dir: str,
    bundles: List[ModelBundle],
    min_confidence: float,
    dry_run: bool,
):
    ensure_dir(dest_dir)

    moved = []
    skipped_unreadable = 0
    total_images = 0

    for root, _, files in os.walk(source_dir):
        true_label = os.path.basename(root)
        for fname in files:
            if not fname.lower().endswith(tuple(IMG_EXTENSIONS)):
                continue
            total_images += 1
            path = os.path.join(root, fname)
            try:
                img = Image.open(path).convert("RGB")
            except (UnidentifiedImageError, OSError):
                skipped_unreadable += 1
                continue

            preds = []
            for bundle in bundles:
                pred_label, prob = predict_one(img, bundle)
                preds.append((bundle.name, pred_label, prob))

            # Consensus: all predicted labels identical and differ from folder name.
            pred_labels = {p[1] for p in preds}
            if len(pred_labels) != 1:
                continue
            consensus_label = pred_labels.pop()
            if consensus_label == true_label:
                continue

            # High confidence: every model must exceed the threshold.
            if any(prob < min_confidence for _, _, prob in preds):
                continue

            # Prepare destination path grouped by true/pred labels for fast review.
            label_dir = f"{true_label}__pred-{consensus_label}"
            dest_subdir = os.path.join(dest_dir, label_dir)
            ensure_dir(dest_subdir)
            target_path = unique_destination(dest_subdir, fname)

            if not dry_run:
                shutil.move(path, target_path)

            moved.append(
                {
                    "source": path,
                    "dest": target_path,
                    "true_label": true_label,
                    "pred_label": consensus_label,
                    "details": {name: prob for name, _, prob in preds},
                }
            )

    return moved, total_images, skipped_unreadable


def parse_args():
    parser = argparse.ArgumentParser(description="Quarantine consensus label errors using Swin + ConvNeXt.")
    parser.add_argument("--config", type=str, default="config_ensemble.yaml", help="YAML config with ensemble_models list.")
    parser.add_argument("--source", type=str, default="source", help="Dataset root to scan (class subfolders expected).")
    parser.add_argument("--destination", type=str, default="suspect_images", help="Where to move consensus errors.")
    parser.add_argument("--min-confidence", type=float, default=0.95, help="Per-model minimum softmax probability for the predicted label.")
    parser.add_argument("--dry-run", action="store_true", help="Report matches without moving files.")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = resolve_device(cfg.get("device", "auto"))
    bundles, class_names = load_models(cfg, device)
    print(f"Loaded {len(bundles)} models; {len(class_names)} classes.")
    print(f"Scanning {args.source} -> suspect folder: {args.destination}")
    print(f"Consensus rule: agree + prob >= {args.min_confidence} each (dry_run={args.dry_run})")

    moved, total, unreadable = move_consensus_errors(
        source_dir=args.source,
        dest_dir=args.destination,
        bundles=bundles,
        min_confidence=args.min_confidence,
        dry_run=args.dry_run,
    )

    print(f"Checked {total} images. Unreadable: {unreadable}.")
    print(f"Consensus suspects: {len(moved)}")
    if moved:
        print("Examples:")
        for item in moved[:10]:
            detail_str = ", ".join(f"{k}={v:.3f}" for k, v in item["details"].items())
            print(f"- {item['source']} -> {item['dest']} (true={item['true_label']}, pred={item['pred_label']}, {detail_str})")
    else:
        print("No consensus errors found with current threshold.")


if __name__ == "__main__":
    main()
