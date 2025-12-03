import argparse
import yaml
import torch
from PIL import Image
import os
from src.data import build_transforms
from src.model import build_model
from src.utils import resolve_device

def load_checkpoint(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    return ckpt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = resolve_device(cfg.get("device", "auto"))

    ensemble_cfg = cfg.get("ensemble_models") or []
    models_with_weights = []
    data_cfg = None

    if ensemble_cfg:
        class_names = None
        for model_cfg in ensemble_cfg:
            model_name = model_cfg["model_name"]
            ckpt_path = model_cfg.get("checkpoint") or os.path.join(cfg["save_dir"], f"{model_name}_best.pt")
            weight = float(model_cfg.get("weight", 1.0))
            ckpt = load_checkpoint(ckpt_path, device)
            classes = ckpt["classes"]
            if class_names is None:
                class_names = classes
            elif classes != class_names:
                raise ValueError("Class ordering mismatch across ensemble checkpoints.")
            model = build_model(model_name, len(classes), pretrained=False).to(device)
            model.load_state_dict(ckpt["model_state"])
            model.eval()
            if data_cfg is None:
                data_cfg = getattr(model, "data_config", None)
            models_with_weights.append((model, weight))
    else:
        ckpt_path = os.path.join(cfg["save_dir"], f"{cfg['model_name']}_best.pt")
        ckpt = load_checkpoint(ckpt_path, device)
        class_names = ckpt["classes"]
        model = build_model(cfg["model_name"], len(class_names), pretrained=False).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        data_cfg = getattr(model, "data_config", None)
        models_with_weights = [(model, 1.0)]

    tfm = build_transforms(
        cfg["img_size"],
        is_train=False,
        data_cfg=data_cfg,
        use_timm_augment=cfg.get("use_timm_augment", False)
    )
    img = Image.open(args.image).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)
    with torch.no_grad():
        prob_sum = None
        weight_sum = sum(w for _, w in models_with_weights)
        for model, weight in models_with_weights:
            outputs = model(x).softmax(dim=1) * float(weight)
            prob_sum = outputs if prob_sum is None else prob_sum + outputs
        probs = (prob_sum / max(weight_sum, 1e-6)).cpu().squeeze().tolist()
    top_idx = int(torch.tensor(probs).argmax().item())
    print(f"Predicted: {class_names[top_idx]} (p={probs[top_idx]:.4f})")

if __name__ == "__main__":
    main()
