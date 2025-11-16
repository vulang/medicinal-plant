import os
import yaml
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from .data import build_testloader
from .model import build_model
from .utils import resolve_device, plot_confusion_matrix, save_classification_report

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for images, labels in tqdm(loader, desc="Test", leave=False):
        images = images.to(device)
        outputs = model(images).softmax(dim=1)
        preds = outputs.argmax(dim=1).cpu().tolist()
        y_pred.extend(preds)
        y_true.extend(labels.tolist())
    return y_true, y_pred

def main(cfg_path: str = "config.yaml"):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = resolve_device(cfg.get("device", "auto"))
    best_ckpt = os.path.join(cfg["save_dir"], f"{cfg['model_name']}_best.pt")
    ckpt = torch.load(best_ckpt, map_location="cpu")
    class_names = ckpt["classes"]
    num_classes = len(class_names)

    model = build_model(cfg["model_name"], num_classes, pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])

    data_cfg = getattr(model, "data_config", None)
    test_loader = build_testloader(
        cfg["test_dir"],
        cfg["img_size"],
        cfg["batch_size"],
        cfg["num_workers"],
        class_names,
        data_cfg=data_cfg,
        use_timm_augment=cfg.get("use_timm_augment", False)
    )

    y_true, y_pred = evaluate(model, test_loader, device)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test Macro F1: {f1:.4f}")

    os.makedirs(cfg["outputs_dir"], exist_ok=True)
    cm_path = os.path.join(cfg["outputs_dir"], "confusion_matrix.png")
    rpt_path = os.path.join(cfg["outputs_dir"], "classification_report.txt")
    plot_confusion_matrix(y_true, y_pred, class_names, cm_path)
    save_classification_report(y_true, y_pred, class_names, rpt_path)
    print(f"Saved confusion matrix -> {cm_path}")
    print(f"Saved classification report -> {rpt_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
