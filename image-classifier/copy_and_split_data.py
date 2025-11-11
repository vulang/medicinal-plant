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

SplitNames = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy crawler data into train/val/test folders for modeling."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "crawler" / "photos" / "plants",
        help="Folder containing one sub-folder per class (default: ../crawler/photos/plants).",
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


def safe_makedirs(destination: Path, clear: bool) -> None:
    if destination.exists() and clear:
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


def main() -> None:
    args = parse_args()
    validate_inputs(args.source, args.val_ratio, args.test_ratio)
    class_dirs = list_classes(args.source)

    safe_makedirs(args.destination, args.clear)
    random.seed(args.seed)

    total_files_copied = 0
    for class_dir in class_dirs:
        files = [path for path in class_dir.iterdir() if path.is_file()]
        random.shuffle(files)
        splits = split_counts(len(files), args.val_ratio, args.test_ratio)
        copy_files(files, splits, class_dir.name, args.destination)
        total_files_copied += len(files)
        print(
            f"{class_dir.name}: {len(files)} files -> "
            f"train={splits[0]}, val={splits[1]}, test={splits[2]}"
        )

    print(f"Finished copying {total_files_copied} files into '{args.destination}'.")


if __name__ == "__main__":
    main()
