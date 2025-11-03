# Medicinal Plant Image Classifier

End-to-end project scaffold to train, evaluate, and demo a **medicinal plant species classifier** from images.

## 🔧 Features
- **Folder-based dataset** (ImageNet-style): `data/train/<class>/`, `data/val/<class>/`, `data/test/<class>/`
- **Transfer Learning** with torchvision (ResNet18 by default)
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

1) **Create your dataset** in this layout:
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
- It **auto-infers class names** from the folder structure.
- Confusion matrix and classification report are written to `outputs/`.
- For Apple Silicon, consider installing PyTorch with the **MPS** (Metal) backend: https://pytorch.org/get-started/locally/
