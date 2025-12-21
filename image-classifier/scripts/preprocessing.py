#!/usr/bin/env python3
"""
Filter crawler downloads to keep only images that likely contain plants.

Uses OpenCLIP zero-shot classification ("plant" vs "not a plant") and copies
accepted files into a destination folder, preserving class subdirectories.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path
from typing import Iterable, Tuple

import torch
import numpy as np
import cv2
from PIL import Image

try:
    import open_clip
except ImportError as exc:  # pragma: no cover - convenience message
    raise ImportError(
        "open_clip_torch is required. Install with `pip install open_clip_torch`."
    ) from exc


PLANT_PROMPTS = [
    "a photo of a medicinal plant",
    "a photo of an herb plant",
    "a close-up photo of plant leaves",
    "a photo of a plant in nature",
    "a photo of a bear", # class 148 - Ursus arctos Linné"
]
NON_PLANT_PROMPTS = [
    "a photo of an animal",
    "a photo of a person",
    "a photo of an object that is not a plant",
    "a photo of a building",
    "a photo of a hand",
]
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy images that look like plants into a filtered folder."
    )
    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--source",
        type=Path,
        default=repo_root / "crawler" / "photos" / "plants",
        help="Source folder containing class subdirectories (default: ../crawler/photos/plants).",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=repo_root / "image-classifier" / "data" / "filtered",
        help="Destination root to store filtered images (default: ./image-classifier/data/filtered).",
    )
    parser.add_argument(
        "--removed-destination",
        type=Path,
        default=repo_root / "image-classifier" / "data" / "removed-by-preprocessed",
        help="Root folder to store removed images, organized by reason/class (default: ./image-classifier/data/removed-by-preprocessed).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.65,
        help="Minimum plant probability to keep an image (default: 0.65).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for model inference (default: 64).",
    )
    parser.add_argument(
        "--min-laplacian-var",
        type=float,
        default=30.0,
        help="Minimum variance of Laplacian (blur threshold). Lower rejects blurrier images.",
    )
    parser.add_argument(
        "--min-contrast",
        type=float,
        default=0.03,
        help="Minimum grayscale contrast (p99-p1 normalized) to keep an image (default: 0.03).",
    )
    parser.add_argument(
        "--aesthetic-threshold",
        type=float,
        default=0.20,
        help="Minimum aesthetic probability (CLIP high-quality vs low-quality) to keep an image (default: 0.20).",
    )
    parser.add_argument(
        "--min-width",
        type=int,
        default=224,
        help="Minimum image width to keep (default: 224).",
    )
    parser.add_argument(
        "--min-height",
        type=int,
        default=224,
        help="Minimum image height to keep (default: 224).",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=224 * 224,
        help="Minimum image area (width*height) to keep; set to 0 to disable (default: 224*224).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: auto|cuda|cpu (default: auto).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ViT-B-32",
        help="OpenCLIP model name (default: ViT-B-32).",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="laion2b_s34b_b79k",
        help="OpenCLIP pretrained tag (default: laion2b_s34b_b79k).",
    )
    parser.add_argument(
        "--skip-clip-filter",
        action="store_true",
        help="Skip OpenCLIP filtering and only apply dedup/size/quality checks.",
    )
    return parser.parse_args()


def get_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_model(model_name: str, pretrained: str, device: torch.device):
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.to(device)
    model.eval()
    with torch.no_grad():
        plant_tokens = tokenizer(PLANT_PROMPTS).to(device)
        non_plant_tokens = tokenizer(NON_PLANT_PROMPTS).to(device)
        plant_feat = model.encode_text(plant_tokens).float()
        non_plant_feat = model.encode_text(non_plant_tokens).float()
        plant_feat = (plant_feat / plant_feat.norm(dim=-1, keepdim=True)).mean(dim=0)
        non_plant_feat = (non_plant_feat / non_plant_feat.norm(dim=-1, keepdim=True)).mean(dim=0)
        text_features = torch.stack(
            [
                plant_feat / plant_feat.norm(),
                non_plant_feat / non_plant_feat.norm(),
            ],
            dim=0,
        )
        # Aesthetic prompts (high vs low quality)
        aesthetic_tokens = tokenizer(
            ["a high quality, well-lit photo", "a low quality, blurry photo"]
        ).to(device)
        aesthetic_feat = model.encode_text(aesthetic_tokens).float()
        aesthetic_feat = aesthetic_feat / aesthetic_feat.norm(dim=-1, keepdim=True)
    return model, preprocess, text_features, aesthetic_feat


def iter_image_files(class_dir: Path) -> Iterable[Path]:
    for path in sorted(class_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in VALID_EXTS:
            yield path


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def deduplicate(
    files: list[Path], removed_root: Path, class_name: str
) -> tuple[list[Path], list[Path]]:
    """Return unique files and copy duplicates into removed folder based on SHA-256 of file content."""
    seen = {}
    unique = []
    duplicates = []
    dup_dir = removed_root / "duplicates" / class_name
    for path in files:
        digest = file_sha256(path)
        if digest in seen:
            duplicates.append(path)
            copy_with_unique_name(path, dup_dir)
            continue
        seen[digest] = path
        unique.append(path)
    return unique, duplicates


def filter_small_images(
    files: list[Path],
    min_width: int,
    min_height: int,
    min_area: int,
    removed_root: Path,
    class_name: str,
) -> tuple[list[Path], list[Path]]:
    """Return files that meet size constraints and copy rejected ones."""
    kept = []
    rejected = []
    small_dir = removed_root / "small" / class_name
    for path in files:
        try:
            with Image.open(path) as img:
                width, height = img.size
        except Exception:
            rejected.append(path)
            copy_with_unique_name(path, small_dir)
            continue
        area = width * height
        if width >= min_width and height >= min_height and (min_area == 0 or area >= min_area):
            kept.append(path)
        else:
            rejected.append(path)
            copy_with_unique_name(path, small_dir)
    return kept, rejected


def quality_checks(path: Path, min_lap_var: float, min_contrast: float) -> Tuple[bool, float, float]:
    """Return (pass, lap_var, contrast)."""
    data = path.read_bytes()
    img_array = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False, 0.0, 0.0
    lap_var = float(cv2.Laplacian(img, cv2.CV_64F).var())
    p1, p99 = np.percentile(img, [1, 99])
    contrast = float((p99 - p1) / 255.0)
    ok = lap_var >= min_lap_var and contrast >= min_contrast
    return ok, lap_var, contrast


def score_batch(
    model,
    preprocess,
    text_features: torch.Tensor,
    aesthetic_features: torch.Tensor,
    batch: list[Path],
    device: torch.device,
) -> list[float]:
    images = []
    for path in batch:
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            images.append(None)
            continue
        images.append(preprocess(img))
    valid_indices = [i for i, img in enumerate(images) if img is not None]
    if not valid_indices:
        return [0.0] * len(batch)
    batch_tensor = torch.stack([images[i] for i in valid_indices]).to(device)
    with torch.no_grad():
        image_features = model.encode_image(batch_tensor).float()
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = (model.logit_scale.exp() * image_features @ text_features.T)
        probs = logits.softmax(dim=1)[:, 0].cpu().tolist()
        aesthetic_logits = (model.logit_scale.exp() * image_features @ aesthetic_features.T)
        aesthetic_probs = aesthetic_logits.softmax(dim=1)[:, 0].cpu().tolist()
    scores = [0.0] * len(batch)
    aesthetic_scores = [0.0] * len(batch)
    for idx, prob in zip(valid_indices, probs):
        scores[idx] = prob
    for idx, prob in zip(valid_indices, aesthetic_probs):
        aesthetic_scores[idx] = prob
    return list(zip(scores, aesthetic_scores))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_with_unique_name(src: Path, dst_dir: Path) -> Path:
    """Copy src into dst_dir, avoiding name collisions."""
    ensure_dir(dst_dir)
    dst = dst_dir / src.name
    if dst.exists():
        stem = dst.stem
        suffix = dst.suffix
        counter = 1
        while True:
            candidate = dst_dir / f"{stem}_{counter}{suffix}"
            if not candidate.exists():
                dst = candidate
                break
            counter += 1
    shutil.copy2(src, dst)
    return dst


def copy_file(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)


def filter_class(
    class_dir: Path,
    dst_root: Path,
    model,
    preprocess,
    text_features: torch.Tensor | None,
    aesthetic_features: torch.Tensor | None,
    device: torch.device,
    batch_size: int,
    threshold: float,
    aesthetic_threshold: float,
    min_width: int,
    min_height: int,
    min_area: int,
    min_lap_var: float,
    min_contrast: float,
    clip_enabled: bool,
    removed_root: Path,
) -> tuple[int, int, int, int, int, int]:
    files = list(iter_image_files(class_dir))
    unique_files, duplicates = deduplicate(files, removed_root, class_dir.name)
    sized_files, too_small = filter_small_images(
        unique_files, min_width, min_height, min_area, removed_root, class_dir.name
    )
    kept = 0
    blurry_or_low_contrast = 0
    openclip_rejected = 0
    for i in range(0, len(sized_files), batch_size):
        batch = sized_files[i : i + batch_size]
        if clip_enabled:
            quality_pass_paths: list[Path] = []
            for path in batch:
                ok, _, _ = quality_checks(path, min_lap_var, min_contrast)
                if not ok:
                    blurry_or_low_contrast += 1
                    copy_with_unique_name(path, removed_root / "blurry" / class_dir.name)
                    continue
                quality_pass_paths.append(path)

            if quality_pass_paths:
                scored = score_batch(
                    model, preprocess, text_features, aesthetic_features, quality_pass_paths, device
                )
                for path, (plant_prob, aesth_prob) in zip(quality_pass_paths, scored):
                    if plant_prob >= threshold and aesth_prob >= aesthetic_threshold:
                        dst = dst_root / class_dir.name / path.name
                        copy_file(path, dst)
                        kept += 1
                    else:
                        copy_with_unique_name(path, removed_root / "open-clip" / class_dir.name)
                        openclip_rejected += 1
        else:
            for path in batch:
                ok, _, _ = quality_checks(path, min_lap_var, min_contrast)
                if not ok:
                    blurry_or_low_contrast += 1
                    copy_with_unique_name(path, removed_root / "blurry" / class_dir.name)
                    continue
                dst = dst_root / class_dir.name / path.name
                copy_file(path, dst)
                kept += 1
    return kept, len(files), len(duplicates), len(too_small), blurry_or_low_contrast, openclip_rejected


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    if not args.source.exists():
        raise FileNotFoundError(f"Source directory not found: {args.source}")
    ensure_dir(args.destination)
    ensure_dir(args.removed_destination)

    clip_enabled = not args.skip_clip_filter
    if clip_enabled:
        print(f"Loading OpenCLIP {args.model} ({args.pretrained}) on {device}...")
        model, preprocess, text_features, aesthetic_features = load_model(args.model, args.pretrained, device)
    else:
        print("Skipping OpenCLIP filtering (dedup + size + quality only).")
        model = preprocess = text_features = aesthetic_features = None

    class_dirs = [p for p in sorted(args.source.iterdir()) if p.is_dir()]
    if not class_dirs:
        raise RuntimeError(f"No class folders found under {args.source}")

    total_kept = 0
    total_seen = 0
    total_dupes = 0
    total_small = 0
    total_blurry = 0
    total_openclip = 0
    for class_dir in class_dirs:
        kept, seen, dupes, small, blurry, openclip_rejects = filter_class(
            class_dir,
            args.destination,
            model,
            preprocess,
            text_features,
            aesthetic_features,
            device,
            args.batch_size,
            args.threshold,
            args.aesthetic_threshold,
            args.min_width,
            args.min_height,
            args.min_area,
            args.min_laplacian_var,
            args.min_contrast,
            clip_enabled,
            args.removed_destination,
        )
        total_kept += kept
        total_seen += seen
        total_dupes += dupes
        total_small += small
        total_blurry += blurry
        total_openclip += openclip_rejects
        print(
            f"{class_dir.name}: kept {kept}/{seen} "
            f"(dupes {dupes}, too small {small}, blurry/low-contrast {blurry}, open-clip {openclip_rejects}, "
            f"{(kept/seen*100 if seen else 0):.1f}% passed)"
        )

    print(
        f"Done. Kept {total_kept}/{total_seen} images "
        f"({(total_kept/total_seen*100 if total_seen else 0):.1f}%) "
        f"into '{args.destination}' "
        f"(removed dupes {total_dupes}, too small {total_small}, blurry/low-contrast {total_blurry}, open-clip {total_openclip})."
    )


if __name__ == "__main__":
    main()
