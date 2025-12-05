# Medicinal Plant Image Classifier

End-to-end project scaffold to train, evaluate, and demo a **medicinal plant species classifier** from images.

## 🔧 Features
- **Folder-based dataset** (ImageNet-style): `data/train/<class>/`, `data/val/<class>/`, `data/test/<class>/`
- **Transfer Learning** with torchvision/timm (ResNet18 by default; also ResNet50, EfficientNet-B0, MobileNetV3-Small, ConvNeXt Tiny/Small/Base/Large, ConvNeXt V2 Base `fcmae_ft_in22k_in1k`, **Swin-T / Swin-B** from torchvision, and ViT/DeiT/Swin variants via timm)
- **Two-phase training** (head warmup + full fine-tune) with mixed-precision, LR warmup, cosine decay, label smoothing, grad clipping, and top-5 metrics.
- **Config-driven** (`config.yaml`): image size, batch size, epochs, learning rate, model name
- **Metrics & Confusion Matrix** saved to `outputs/`
- **Streamlit app** for quick demo (`app/streamlit_app.py`)
- **MLflow-ready**: log params/metrics/artifacts locally or to a remote tracking server; optional Model Registry push
  - **Run an MLflow tracking server** backed by a database (example: Postgres + local artifacts):
    ```
    mlflow server \
      --backend-store-uri postgresql+psycopg2://user:password@localhost:5432/mlflowdb \
      --artifacts-destination /abs/path/to/mlflow-artifacts \
      --host 0.0.0.0 --port 5000
    ```
    For S3/MinIO artifacts, set `MLFLOW_S3_ENDPOINT_URL` (if needed) and use `--artifacts-destination s3://my-bucket/path`. Point `mlflow.tracking_uri` in `config.yaml` to `http://<server>:5000` (already set to `http://localhost:5000`).

## 📂 Project Structure
```
medicinal-plant-image-classifier/
├── app/
│   └── streamlit_app.py          # Web demo: upload image -> predict species
├── data/
│   ├── train/                    # e.g., train/Panax_ginseng/*.jpg
│   ├── val/
│   └── test/
├── models/                       # Saved checkpoints (.pt)
├── outputs/                      # Metrics, confusion matrix, logs
├── src/
│   ├── data.py                   # Datasets & augmentations
│   ├── model.py                  # Build/Load torchvision models
│   ├── train.py                  # Training loop + validation
│   ├── eval.py                   # Test set evaluation + reports
│   └── utils.py                  # Helpers (seed, metrics, plots)
├── scripts/
│   └── predict_one.py            # CLI: predict one image
├── config.yaml                   # Training configuration
├── requirements.txt
└── README.md
```

## 🚀 Quickstart

1) **Create your dataset** in this layout (or auto-populate it with the copy script below):
```
data/
  train/
    Panax_ginseng/ *.jpg
    Zingiber_officinale/ *.jpg
    Glycyrrhiza_uralensis/ *.jpg
  val/
    Panax_ginseng/ *.jpg
    Zingiber_officinale/ *.jpg
    Glycyrrhiza_uralensis/ *.jpg
  test/
    ...
```

> (Optional) First filter crawler downloads to keep plant-only images (OpenCLIP zero-shot with quality/aesthetic filters):
> ```
> python3 scripts/filter_non_plants.py --source ../crawler/photos/plants --destination source
> ```
> To split the (filtered) photos into train/val/test, run:
> ```
> python3 copy_and_split_data.py --clear
> ```
> Use `--val-ratio`/`--test-ratio` to adjust split sizes and `--source`/`--destination` if your paths differ from the defaults.

2) **Create a virtual environment**
```
python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
```

3) **Install dependencies**
```
pip install -r requirements.txt
```

4) **Train**
```
python -m src.train --config config.yaml
```

5) **Evaluate on test**
```
python -m src.eval --config config.yaml
```

6) **Demo app**
```
streamlit run app/streamlit_app.py
```

### Train an ensemble (ConvNeXt + ViT/SwinB)
Train ConvNeXt V2 Base, ViT-B/16 (in22k), and Swin-B back-to-back and emit an ensemble config in one shot:
```
python -m src.train_ensemble --configs config.yaml config_vit.yaml config_swin_b.yaml \
  --weights 0.34 0.33 0.33 --output-config config_ensemble.yaml --force
```
Then evaluate with the generated `config_ensemble.yaml`:
```
python -m src.eval --config config_ensemble.yaml
```

## 📈 MLflow Tracking (optional)
- Enable logging by setting `mlflow.enabled: true` in `config.yaml` (default). Configs now default to `mlflow.tracking_uri: http://localhost:5000`; change this to your server URL if different.
- Start a local UI (if using the default local store): `mlflow ui --backend-store-uri mlruns --port 5000` and open `http://127.0.0.1:5000`.
- Training logs params, per-epoch metrics, and uploads checkpoints under `mlflow.artifact_subdir` (default `checkpoints`). Evaluation logs test metrics and confusion-matrix/report artifacts under `mlflow.eval_artifact_subdir` (default `eval`).
- To push the best checkpoint to the MLflow Model Registry, set `mlflow.register_model: true` and choose `mlflow.registered_model_name`/`mlflow.model_artifact_subdir`.
- Optional keys: `mlflow.experiment_name`, `mlflow.run_name`, and `mlflow.tags` let you organize runs; leave them unset to use MLflow defaults.

### Docker Compose (SQLite-backed MLflow)
`docker-compose.yaml` snippet to spin up an MLflow tracking server with SQLite (artifacts on the host):
```yaml
version: "3.9"
services:
  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.9.2
    ports:
      - "5000:5000"
    volumes:
      - ./mlruns:/mlruns                # SQLite DB and default artifacts
      - ./mlartifacts:/mlartifacts      # Artifact root (optional separate path)
    command: >
      mlflow server
        --backend-store-uri sqlite:///mlruns/mlflow.db
        --artifacts-destination /mlartifacts
        --host 0.0.0.0
        --port 5000
```
Start it with `docker compose up -d mlflow`; `config.yaml` already points `mlflow.tracking_uri` to `http://localhost:5000`.

## 🧪 Notes
- This scaffold uses **torchvision.models** with pretrained weights and replaces the classifier head.
- Extra backbones such as ConvNeXt V2 use **timm** under the hood; install via `pip install -r requirements.txt`.
- Vision Transformer backbones (e.g., `vit_base_patch16_224`, `deit_base_patch16_224`, `swin_base_patch4_window7_224`) are available through `model_name`; set `img_size` to match the chosen variant (224 by default) and consider `use_timm_augment: true` for timm defaults. Torchvision `swin_t` / `swin_b` are also supported directly.
- Data transforms default to the lightweight Resize/ColorJitter pipeline; set `use_timm_augment: true` in `config.yaml` if you want the more aggressive timm RandomResizedCrop recipe that matches ConvNeXt V2 pretraining.
- Training behaviour (AMP, label smoothing or focal loss, warmup epochs, patience, fine-tune LR, grad clipping) is controlled entirely from `config.yaml`.
- It **auto-infers class names** from the folder structure.
- Confusion matrix and classification report are written to `outputs/`.
- For Apple Silicon, consider installing PyTorch with the **MPS** (Metal) backend: https://pytorch.org/get-started/locally/
