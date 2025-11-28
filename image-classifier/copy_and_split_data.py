#!/usr/bin/env python3
"""
Utility for copying the raw crawler output into the classifier's data directory.

It expects crawler photos to be stored as class-specific directories under
``crawler/photos/plants`` (relative to the repo root). The script copies every
file into ``image-classifier/data`` and splits each class into ``train``,
``val``, and ``test`` subsets.
"""

from __future__ import annotations

import argparse
import math
import random
import shutil
from pathlib import Path
from typing import Iterable

from PIL import Image
import yaml

SplitNames = ("train", "val", "test")
#classes_to_ignore = [9, 10, 11, 13, 48, 56, 57, 110, 117, 139, 141, 142, 173, 177, 179, 180]
classes_to_ignore = []
IGNORED_CLASS_NAMES = {str(class_id) for class_id in classes_to_ignore}
MAX_FILES_PER_CLASS = 2000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy crawler data into train/val/test folders for modeling."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parent / "source",
        help="Folder containing one sub-folder per class (default: ./source).",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
        help="Destination root for the split dataset (default: ./data).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of each class to place in the validation split (default: 0.1).",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Fraction of each class to place in the test split (default: 0.1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for shuffling files before splitting (default: 7).",
    )
    parser.add_argument(
        "--ignored-log",
        type=Path,
        help="Log file for classes that are skipped (default: <destination>/ignored_classes.log).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "config.yaml",
        help="Path to the training config file containing img_size (default: ./config.yaml).",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Remove the destination directory before copying to ensure a clean split.",
    )
    return parser.parse_args()


def validate_inputs(source: Path, val_ratio: float, test_ratio: float) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Source directory '{source}' does not exist.")
    if not source.is_dir():
        raise NotADirectoryError(f"Source path '{source}' is not a directory.")
    if not 0 <= val_ratio < 1 or not 0 <= test_ratio < 1:
        raise ValueError("Validation and test ratios must both be between 0 and 1.")
    if val_ratio + test_ratio >= 1:
        raise ValueError("Validation and test ratios must sum to less than 1.")


def list_classes(source: Path) -> list[Path]:
    class_dirs = [entry for entry in sorted(source.iterdir()) if entry.is_dir()]
    if not class_dirs:
        raise ValueError(f"No class directories found inside '{source}'.")
    return class_dirs


def safe_makedirs(destination: Path, clear: bool, source: Path | None = None) -> None:
    """Create destination split folders, optionally clearing previous content.

    If the source folder lives inside the destination (e.g., source=./data/filtered,
    destination=./data), we only clear the split subfolders to avoid deleting the
    source.
    """
    should_preserve_source = source is not None and source.exists() and source.is_relative_to(destination)
    if destination.exists() and clear:
        if should_preserve_source:
            for split_name in SplitNames:
                shutil.rmtree(destination / split_name, ignore_errors=True)
        else:
            shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for split_name in SplitNames:
        (destination / split_name).mkdir(parents=True, exist_ok=True)


def split_counts(total: int, val_ratio: float, test_ratio: float) -> tuple[int, int, int]:
    if total == 0:
        return (0, 0, 0)
    val_count = math.floor(total * val_ratio)
    test_count = math.floor(total * test_ratio)
    if val_count + test_count >= total:
        # Keep at least one sample in the training split when possible.
        overflow = val_count + test_count - (total - 1)
        if overflow > 0:
            # Reclaim overflow from the larger split first.
            if val_count >= test_count:
                take = min(overflow, val_count)
                val_count -= take
                overflow -= take
            if overflow > 0:
                take = min(overflow, test_count)
                test_count -= take
    train_count = total - val_count - test_count
    return (train_count, val_count, test_count)


def copy_files(
    files: Iterable[Path],
    split_sizes: tuple[int, int, int],
    class_name: str,
    destination: Path,
) -> None:
    train_size, val_size, test_size = split_sizes
    files = list(files)
    train_files = files[:train_size]
    val_files = files[train_size : train_size + val_size]
    test_files = files[train_size + val_size :]

    for subset, subset_files in zip(SplitNames, (train_files, val_files, test_files)):
        subset_dir = destination / subset / class_name
        subset_dir.mkdir(parents=True, exist_ok=True)
        for file_path in subset_files:
            target_path = subset_dir / file_path.name
            shutil.copy2(file_path, target_path)


def load_img_size(config_path: Path) -> int:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file '{config_path}' does not exist.")
    with config_path.open("r", encoding="utf-8") as config_file:
        cfg = yaml.safe_load(config_file) or {}
    if "img_size" not in cfg:
        raise KeyError(f"'img_size' not found in config file '{config_path}'.")
    try:
        img_size = int(cfg["img_size"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"'img_size' in '{config_path}' must be an integer.") from exc
    if img_size <= 0:
        raise ValueError(f"'img_size' in '{config_path}' must be positive.")
    return img_size


def filter_small_images(files: Iterable[Path], min_size: int) -> tuple[list[Path], int]:
    kept_files: list[Path] = []
    skipped_count = 0
    for file_path in files:
        try:
            with Image.open(file_path) as img:
                width, height = img.size
        except Exception:
            skipped_count += 1
            continue
        if width < min_size or height < min_size:
            skipped_count += 1
            continue
        kept_files.append(file_path)
    return kept_files, skipped_count


def main() -> None:
    args = parse_args()
    img_size = load_img_size(args.config)
    validate_inputs(args.source, args.val_ratio, args.test_ratio)
    class_dirs = list_classes(args.source)

    safe_makedirs(args.destination, args.clear, args.source)
    random.seed(args.seed)

    total_files_copied = 0
    ignored_classes: list[tuple[str, int]] = []
    ignored_by_list = 0
    ignored_log_path = args.ignored_log or args.destination / "ignored_classes.log"

    for class_dir in class_dirs:
        files = [path for path in class_dir.iterdir() if path.is_file()]
        if class_dir.name in IGNORED_CLASS_NAMES:
            ignored_classes.append((class_dir.name, len(files)))
            ignored_by_list += 1
            print(f"Skipping {class_dir.name}: listed in classes_to_ignore.")
            continue
        filtered_files, skipped_small = filter_small_images(files, img_size)
        if skipped_small:
            print(f"{class_dir.name}: filtered out {skipped_small} files smaller than {img_size}px.")
        random.shuffle(filtered_files)
        filtered_count = len(filtered_files)
        if filtered_count > MAX_FILES_PER_CLASS:
            filtered_files = filtered_files[:MAX_FILES_PER_CLASS]
        splits = split_counts(len(filtered_files), args.val_ratio, args.test_ratio)
        copy_files(filtered_files, splits, class_dir.name, args.destination)
        total_files_copied += len(filtered_files)
        notes = []
        if skipped_small:
            notes.append(f"filtered_out={skipped_small} <{img_size}px")
        if filtered_count > len(filtered_files):
            notes.append(f"capped from {filtered_count}")
        note_text = f" ({'; '.join(notes)})" if notes else ""
        print(
            f"{class_dir.name}: {len(filtered_files)} files{note_text} -> "
            f"train={splits[0]}, val={splits[1]}, test={splits[2]}"
        )

    print(f"Finished copying {total_files_copied} files into '{args.destination}'.")
    if ignored_classes:
        ignored_log_path.parent.mkdir(parents=True, exist_ok=True)
        with ignored_log_path.open("w", encoding="utf-8") as log_file:
            log_file.write("class_name,file_count\n")
            for class_name, count in ignored_classes:
                log_file.write(f"{class_name},{count}\n")
        reasons = []
        if ignored_by_list:
            reasons.append(f"{ignored_by_list} in classes_to_ignore")
        reason_summary = "; ".join(reasons) if reasons else "ignored"
        print(f"Ignored {len(ignored_classes)} classes ({reason_summary}). See '{ignored_log_path}' for details.")
    else:
        print("No classes were ignored.")


if __name__ == "__main__":
    main()
