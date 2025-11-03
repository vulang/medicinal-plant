#!/usr/bin/env python3
"""Utility to split a combined dataset into train/val folder structures."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a dataset arranged in per-class folders into train/val subsets."
    )
    parser.add_argument(
        "--combined-dir",
        type=Path,
        default=Path("data/combined"),
        help="Path to the combined dataset directory (default: data/combined).",
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=Path("data/train"),
        help="Destination directory for the training split (default: data/train).",
    )
    parser.add_argument(
        "--val-dir",
        type=Path,
        default=Path("data/val"),
        help="Destination directory for the validation split (default: data/val).",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Proportion of samples per class to allocate to the training split (default: 0.8).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for reproducible shuffling (default: 42).",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of moving them (default: move).",
    )
    return parser.parse_args()


def ensure_destination_dirs(train_dir: Path, val_dir: Path) -> None:
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)


def collect_class_files(class_dir: Path) -> List[Path]:
    return [
        path
        for path in class_dir.iterdir()
        if path.is_file() and not path.name.startswith(".")
    ]


def split_indices(num_items: int, train_ratio: float) -> Tuple[List[int], List[int]]:
    if num_items == 0:
        return [], []

    n_train = int(num_items * train_ratio)
    if n_train == num_items and num_items > 1:
        n_train -= 1
    if n_train == 0 and num_items > 1:
        n_train = 1
    train_indices = list(range(n_train))
    val_indices = list(range(n_train, num_items))
    return train_indices, val_indices


def transfer_files(
    files: Iterable[Path], destination_dir: Path, use_copy: bool
) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)
    for file_path in files:
        destination = destination_dir / file_path.name
        if destination.exists():
            continue
        if use_copy:
            shutil.copy2(file_path, destination)
        else:
            shutil.move(str(file_path), destination)


def split_dataset(
    combined_dir: Path,
    train_dir: Path,
    val_dir: Path,
    train_ratio: float,
    seed: int,
    use_copy: bool,
) -> None:
    if train_ratio <= 0 or train_ratio >= 1:
        raise ValueError("train_ratio must be between 0 and 1 (exclusive).")

    ensure_destination_dirs(train_dir, val_dir)

    rng = random.Random(seed)
    class_dirs = sorted(
        (path for path in combined_dir.iterdir() if path.is_dir()),
        key=lambda p: p.name,
    )

    for class_dir in class_dirs:
        files = collect_class_files(class_dir)
        if not files:
            continue

        rng.shuffle(files)
        train_idx, val_idx = split_indices(len(files), train_ratio)
        train_files = [files[i] for i in train_idx]
        val_files = [files[i] for i in val_idx]

        transfer_files(train_files, train_dir / class_dir.name, use_copy)
        transfer_files(val_files, val_dir / class_dir.name, use_copy)


def main() -> None:
    args = parse_args()
    split_dataset(
        combined_dir=args.combined_dir,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        train_ratio=args.train_ratio,
        seed=args.seed,
        use_copy=args.copy,
    )


if __name__ == "__main__":
    main()
