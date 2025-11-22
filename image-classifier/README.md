# Medicinal Plant Image Classifier

End-to-end project scaffold to train, evaluate, and demo a **medicinal plant species classifier** from images.

## 🔧 Features
- **Folder-based dataset** (ImageNet-style): `data/train/<class>/`, `data/val/<class>/`, `data/test/<class>/`
- **Transfer Learning** with torchvision/timm (ResNet18 by default; also ResNet50, EfficientNet-B0, MobileNetV3-Small, ConvNeXt Tiny/Small/Base/Large, ConvNeXt V2 Base `fcmae_ft_in22k_in1k`, and ViT/DeiT/Swin variants via timm)
- **Two-phase training** (head warmup + full fine-tune) with mixed-precision, LR warmup, cosine decay, label smoothing, grad clipping, and top-5 metrics.
- **Config-driven** (`config.yaml`): image size, batch size, epochs, learning rate, model name
- **Metrics & Confusion Matrix** saved to `outputs/`
- **Streamlit app** for quick demo (`app/streamlit_app.py`)

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

> To copy photos from `crawler/photos/plants/` and split them automatically, run:
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

## 🧪 Notes
- This scaffold uses **torchvision.models** with pretrained weights and replaces the classifier head.
- Extra backbones such as ConvNeXt V2 use **timm** under the hood; install via `pip install -r requirements.txt`.
- Vision Transformer backbones (e.g., `vit_base_patch16_224`, `deit_base_patch16_224`, `swin_base_patch4_window7_224`) are available through `model_name`; set `img_size` to match the chosen variant (224 by default) and consider `use_timm_augment: true` for timm defaults.
- Data transforms default to the lightweight Resize/ColorJitter pipeline; set `use_timm_augment: true` in `config.yaml` if you want the more aggressive timm RandomResizedCrop recipe that matches ConvNeXt V2 pretraining.
- Training behaviour (AMP, label smoothing, warmup epochs, patience, fine-tune LR, grad clipping) is controlled entirely from `config.yaml`.
- It **auto-infers class names** from the folder structure.
- Confusion matrix and classification report are written to `outputs/`.
- For Apple Silicon, consider installing PyTorch with the **MPS** (Metal) backend: https://pytorch.org/get-started/locally/
