import os
import yaml
import torch
import mlflow
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from .data import build_testloader
from .model import build_model
from .utils import resolve_device, plot_confusion_matrix, save_classification_report


def _load_model_from_checkpoint(model_name: str, ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    class_names = ckpt["classes"]
    model = build_model(model_name, len(class_names), pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    data_cfg = getattr(model, "data_config", None)
    return model, class_names, data_cfg

@torch.no_grad()
def evaluate(models_with_weights, loader, device):
    """
    Run evaluation for a single model or an ensemble. `models_with_weights` is
    a list of (model, weight) tuples. We average the softmax probabilities
    using the provided weights before taking the argmax.
    """
    y_true, y_pred = [], []
    weight_sum = sum(w for _, w in models_with_weights)
    for images, labels in tqdm(loader, desc="Test", leave=False):
        images = images.to(device)
        probs_sum = None
        for model, weight in models_with_weights:
            outputs = model(images).softmax(dim=1)
            weighted = outputs * float(weight)
            probs_sum = weighted if probs_sum is None else probs_sum + weighted
        probs_avg = probs_sum / max(weight_sum, 1e-6)
        preds = probs_avg.argmax(dim=1).cpu().tolist()
        y_pred.extend(preds)
        y_true.extend(labels.tolist())
    return y_true, y_pred

def main(cfg_path: str = "config.yaml"):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    mlflow_cfg = cfg.get("mlflow") or {}
    mlflow_enabled = mlflow_cfg.get("enabled", False)
    artifact_subdir = mlflow_cfg.get("eval_artifact_subdir") or "eval"
    mlflow_run = None
    try:
        if mlflow_enabled:
            tracking_uri = mlflow_cfg.get("tracking_uri")
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            experiment_name = mlflow_cfg.get("experiment_name")
            if experiment_name:
                mlflow.set_experiment(experiment_name)
            run_name = mlflow_cfg.get("run_name") or "eval"
            mlflow_run = mlflow.start_run(run_name=run_name)
            tags = mlflow_cfg.get("tags")
            if tags and isinstance(tags, dict):
                mlflow.set_tags(tags)

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
                print(f"Loading ensemble checkpoint -> {ckpt_path} (model={model_name}, weight={weight})")
                model, classes, model_data_cfg = _load_model_from_checkpoint(model_name, ckpt_path, device)
                if class_names is None:
                    class_names = classes
                elif classes != class_names:
                    raise ValueError("Class ordering mismatch across ensemble checkpoints.")
                if data_cfg is None:
                    data_cfg = model_data_cfg
                models_with_weights.append((model, weight))
        else:
            best_ckpt = os.path.join(cfg["save_dir"], f"{cfg['model_name']}_best.pt")
            last_ckpt = os.path.join(cfg["save_dir"], f"{cfg['model_name']}_last.pt")
            if os.path.exists(best_ckpt):
                ckpt_path = best_ckpt
            elif os.path.exists(last_ckpt):
                ckpt_path = last_ckpt
            else:
                raise FileNotFoundError(f"Checkpoint not found. Tried {best_ckpt} and {last_ckpt}")
            print(f"Loading checkpoint -> {ckpt_path}")

            model, class_names, data_cfg = _load_model_from_checkpoint(cfg["model_name"], ckpt_path, device)
            models_with_weights = [(model, 1.0)]

        num_classes = len(class_names)
        test_loader = build_testloader(
            cfg["test_dir"],
            cfg["img_size"],
            cfg["batch_size"],
            cfg["num_workers"],
            class_names,
            data_cfg=data_cfg,
            use_timm_augment=cfg.get("use_timm_augment", False)
        )

        y_true, y_pred = evaluate(models_with_weights, test_loader, device)
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

        if mlflow_enabled:
            if ensemble_cfg:
                checkpoints = ";".join(
                    f"{m['model_name']}@{os.path.basename(m.get('checkpoint') or os.path.join(cfg['save_dir'], m['model_name'] + '_best.pt'))}"
                    for m in ensemble_cfg
                )
                mlflow.log_param("eval_ensemble_checkpoints", checkpoints)
            else:
                mlflow.log_param("eval_checkpoint", os.path.basename(ckpt_path))
            mlflow.log_metrics({
                "test_acc": acc,
                "test_macro_f1": f1,
            })
            mlflow.log_artifact(cm_path, artifact_path=artifact_subdir)
            mlflow.log_artifact(rpt_path, artifact_path=artifact_subdir)
    finally:
        if mlflow_run is not None:
            mlflow.end_run()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
