#!/usr/bin/env python3
"""Find duplicated photo names under the dataset source directory."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List


# Common image extensions; expand with --all-files to include everything.
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


def iter_targets(root: Path, include_all: bool) -> Iterable[Path]:
    """Yield files under root that should be checked."""
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if include_all or path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def find_duplicates(
    files: Iterable[Path], case_insensitive: bool
) -> Dict[str, List[Path]]:
    """Collect files keyed by their name and return only duplicates."""
    by_name: Dict[str, List[Path]] = {}
    for path in files:
        key = path.name.lower() if case_insensitive else path.name
        by_name.setdefault(key, []).append(path)
    return {name: paths for name, paths in by_name.items() if len(paths) > 1}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check for duplicated photo names inside a directory tree."
    )
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "source",
        help="root folder to scan (default: image-classifier/source relative to this script)",
    )
    parser.add_argument(
        "--all-files",
        action="store_true",
        help="scan all files, not just common image extensions",
    )
    parser.add_argument(
        "--case-insensitive",
        action="store_true",
        help="treat file names in a case-insensitive manner",
    )
    parser.add_argument(
        "--fail-on-duplicates",
        action="store_true",
        help="return exit code 1 if duplicates are found",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="optional path to write duplicates as CSV (columns: name, path)",
    )

    args = parser.parse_args()
    root = args.root

    if not root.exists():
        parser.error(f"root path does not exist: {root}")

    files = list(iter_targets(root, args.all_files))
    duplicates = find_duplicates(files, args.case_insensitive)

    print(f"Scanned {len(files)} file(s) under {root}")
    print(f"Found {len(duplicates)} duplicate name(s).")

    if duplicates:
        print("\nDuplicates:")
        for name, paths in sorted(duplicates.items()):
            print(f"{name} ({len(paths)}):")
            for path in sorted(paths):
                try:
                    relative = path.relative_to(root)
                except ValueError:
                    relative = path
                print(f"  {relative}")

    if args.output_csv:
        output_path = args.output_csv
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            writer.writerow(["name", "path"])
            for name, paths in sorted(duplicates.items()):
                for path in sorted(paths):
                    try:
                        relative = path.relative_to(root)
                    except ValueError:
                        relative = path
                    writer.writerow([name, str(relative)])
        print(f"Wrote duplicates to {output_path}")

    return 1 if duplicates and args.fail_on_duplicates else 0


if __name__ == "__main__":
    raise SystemExit(main())
