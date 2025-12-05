"""
Train multiple backbones sequentially and emit an ensemble-ready config.

Usage:
    python -m src.train_ensemble --configs config.yaml config_vit.yaml config_swin_b.yaml \\
        --weights 0.34 0.33 0.33 --output-config config_ensemble.yaml
"""

import argparse
import os
import sys
import yaml
import mlflow

from .train import main as train_single


def _load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _resolve_checkpoint(cfg: dict) -> str:
    save_dir = os.path.expanduser(cfg.get("save_dir", "models"))
    model_name = cfg["model_name"]
    best_ckpt = os.path.join(save_dir, f"{model_name}_best.pt")
    last_ckpt = os.path.join(save_dir, f"{model_name}_last.pt")
    if os.path.exists(best_ckpt):
        return best_ckpt
    if os.path.exists(last_ckpt):
        return last_ckpt
    raise FileNotFoundError(f"No checkpoint found for {model_name} (looked for {best_ckpt} and {last_ckpt})")


def _warn_if_dataset_mismatch(base_cfg: dict, cfg: dict, cfg_path: str):
    keys_to_check = ["train_dir", "val_dir", "test_dir", "img_size"]
    for key in keys_to_check:
        base_val = base_cfg.get(key)
        new_val = cfg.get(key)
        if base_val != new_val:
            print(f"[WARN] {key} mismatch between base config and {cfg_path} ({base_val} vs {new_val}). "
                  "Ensure splits and preprocessing stay aligned for the ensemble.")


def _build_ensemble_config(base_cfg: dict, models: list[dict]) -> dict:
    return {
        "seed": base_cfg.get("seed", 42),
        "data_dir": base_cfg.get("data_dir", "data"),
        "train_dir": base_cfg.get("train_dir", "data/train"),
        "val_dir": base_cfg.get("val_dir", "data/val"),
        "test_dir": base_cfg.get("test_dir", "data/test"),
        "img_size": base_cfg.get("img_size", 224),
        "batch_size": base_cfg.get("batch_size", 16),
        "num_workers": base_cfg.get("num_workers", 4),
        "model_name": base_cfg.get("model_name"),
        "pretrained": base_cfg.get("pretrained", True),
        "use_timm_augment": base_cfg.get("use_timm_augment", False),
        "device": base_cfg.get("device", "auto"),
        "save_dir": base_cfg.get("save_dir", "models"),
        "outputs_dir": base_cfg.get("outputs_dir", "outputs"),
        "ensemble_models": models,
        "mlflow": base_cfg.get("mlflow"),
    }


def _train_and_collect(config_path: str, weight: float) -> tuple[dict, dict]:
    print("=" * 70)
    print(f"Training config -> {config_path}")
    print("=" * 70)
    train_single(config_path)
    # Ensure MLflow run is closed before starting the next config.
    mlflow.end_run()
    cfg = _load_config(config_path)
    ckpt = _resolve_checkpoint(cfg)
    model_entry = {
        "model_name": cfg["model_name"],
        "checkpoint": ckpt,
        "weight": float(weight),
    }
    return model_entry, cfg


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Train multiple configs and build an ensemble config.")
    parser.add_argument(
        "--configs",
        nargs="+",
        required=True,
        help="List of training config YAMLs to run sequentially (e.g., ConvNeXt + ViT + SwinB).",
    )
    parser.add_argument(
        "--weights",
        nargs="*",
        type=float,
        help="Optional weights for the ensemble_models list (must match length of --configs). Defaults to 1.0 for each.",
    )
    parser.add_argument(
        "--output-config",
        default="config_ensemble.yaml",
        help="Path to write the ensemble-ready YAML (defaults to config_ensemble.yaml).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing output config instead of failing.",
    )
    args = parser.parse_args(argv)

    if args.weights and len(args.weights) != len(args.configs):
        raise ValueError("Length of --weights must match length of --configs.")
    if os.path.exists(args.output_config) and not args.force:
        raise FileExistsError(f"{args.output_config} already exists. Use --force to overwrite it.")

    ensemble_models: list[dict] = []
    base_cfg: dict | None = None

    for idx, cfg_path in enumerate(args.configs):
        weight = args.weights[idx] if args.weights else 1.0
        model_entry, cfg = _train_and_collect(cfg_path, weight)
        ensemble_models.append(model_entry)
        if base_cfg is None:
            base_cfg = cfg
        else:
            _warn_if_dataset_mismatch(base_cfg, cfg, cfg_path)

    if base_cfg is None:
        raise RuntimeError("No configs were processed; nothing to do.")

    ensemble_cfg = _build_ensemble_config(base_cfg, ensemble_models)

    with open(args.output_config, "w") as f:
        yaml.safe_dump(ensemble_cfg, f, sort_keys=False)

    print("-" * 70)
    print(f"Wrote ensemble config -> {args.output_config}")
    print("Ensemble members:")
    for member in ensemble_models:
        print(f"  - {member['model_name']} @ {member['checkpoint']} (weight={member['weight']})")


if __name__ == "__main__":
    main(sys.argv[1:])
