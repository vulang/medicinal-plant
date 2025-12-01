import csv
import io
import os
import logging
import sys
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image, UnidentifiedImageError
import torch
import torch.nn.functional as F
from torchvision import models, transforms
import yaml

try:
    import timm
except ImportError:
    timm = None


logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "image-classifier"))
DEFAULT_CONFIG_PATH = REPO_ROOT / "image-classifier" / "config.yaml"
DEFAULT_MODELS_DIR = REPO_ROOT / "image-classifier" / "models"
DEFAULT_PLANT_META_PATH = REPO_ROOT / "crawler" / "data" / "plant.csv"
MIN_CUDA_COMPUTE_CAPABILITY = (5, 0)
from merge_config import LOW_PERFORMANCE_CLASSES


class ClassConfidence(BaseModel):
    plant_id: str
    plant_name: str
    confidence: float


class MergedClass(BaseModel):
    plant_id: str
    plant_name: str


class FamilyMergeInfo(BaseModel):
    family_name: str
    classes: List[MergedClass]


class PredictionResponse(BaseModel):
    plant_id: str
    plant_name: str
    confidence: float
    class_confidences: List[ClassConfidence]
    family_merge: FamilyMergeInfo | None = None


def _cuda_device_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        capability = torch.cuda.get_device_capability()
    except (AssertionError, RuntimeError) as exc:
        logger.warning("Unable to determine CUDA device capability; falling back to CPU.", exc_info=exc)
        return False
    if capability < MIN_CUDA_COMPUTE_CAPABILITY:
        logger.warning(
            "CUDA device capability sm_%d%d is not supported by this PyTorch build; using CPU instead.",
            capability[0],
            capability[1],
        )
        return False
    return True


def _resolve_device(preferred: str = "auto") -> torch.device:
    preferred = preferred.lower()
    if preferred == "cpu":
        return torch.device("cpu")
    if preferred == "cuda":
        if _cuda_device_supported():
            return torch.device("cuda")
        logger.warning("MODEL_DEVICE explicitly set to CUDA but falling back to CPU.")
        return torch.device("cpu")
    if preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if _cuda_device_supported():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_model(model_name: str, num_classes: int, pretrained: bool = False) -> torch.nn.Module:
    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, num_classes)
        return model
    if model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, num_classes)
        return model
    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(in_features, num_classes)
        return model
    if model_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
        in_features = model.classifier[3].in_features
        model.classifier[3] = torch.nn.Linear(in_features, num_classes)
        return model
    # Fallback to timm for additional architectures used during training.
    if timm is None:
        raise ValueError(
            f"Unsupported model_name: {model_name}. Install timm to enable additional architectures."
        )
    try:
        return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    except Exception as exc:
        raise ValueError(f"Unsupported model_name: {model_name}") from exc


def _build_inference_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])


def _load_plant_metadata(csv_path: Path) -> Dict[str, str]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Plant metadata CSV not found at {csv_path}")
    mapping: Dict[str, str] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            plant_id = (row.get("ID") or "").strip()
            latin_name = (row.get("Plant latin name") or "").strip()
            if not plant_id:
                continue
            mapping[plant_id] = latin_name or plant_id
    return mapping


def _load_family_lookup(csv_path: Path) -> Dict[str, str]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Plant metadata CSV not found at {csv_path}")
    mapping: Dict[str, str] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            plant_id = (row.get("ID") or "").strip()
            family_name = (row.get("Family name") or "").strip()
            if not plant_id or not family_name:
                continue
            mapping[plant_id] = family_name
    return mapping


def _build_family_merge_lookup(
    class_names: List[str],
    family_lookup: Dict[str, str],
    plant_lookup: Dict[str, str],
) -> Dict[str, FamilyMergeInfo]:
    """Return a map of class_id -> merged family info for merged families."""
    from collections import defaultdict

    family_groups: Dict[str, List[str]] = defaultdict(list)
    for class_id in class_names:
        if class_id not in LOW_PERFORMANCE_CLASSES:
            continue
        family_name = family_lookup.get(class_id)
        if not family_name:
            continue
        family_groups[family_name].append(class_id)

    merge_lookup: Dict[str, FamilyMergeInfo] = {}
    for family_name, ids in family_groups.items():
        if len(ids) < 2:
            continue
        try:
            sorted_ids = sorted(ids, key=lambda x: int(x))
        except ValueError:
            sorted_ids = sorted(ids)
        class_entries = [
            MergedClass(plant_id=class_id, plant_name=plant_lookup.get(class_id, class_id))
            for class_id in sorted_ids
        ]
        info = FamilyMergeInfo(family_name=family_name, classes=class_entries)
        for class_id in sorted_ids:
            merge_lookup[class_id] = info
    return merge_lookup


class PlantClassifierService:
    def __init__(self):
        config_path = Path(os.getenv("MODEL_CONFIG_PATH", DEFAULT_CONFIG_PATH))
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at {config_path}")
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        checkpoint_path_env = os.getenv("MODEL_CHECKPOINT_PATH")
        self.checkpoint_path = Path(checkpoint_path_env) if checkpoint_path_env else (
            DEFAULT_MODELS_DIR / f"{self.cfg['model_name']}_best.pt"
        )
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {self.checkpoint_path}")

        self.device = _resolve_device(os.getenv("MODEL_DEVICE", self.cfg.get("device", "auto")))
        self.transform = _build_inference_transform(self.cfg["img_size"])

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.class_names: List[str] = checkpoint["classes"]
        self.model = _build_model(
            self.cfg["model_name"],
            num_classes=len(self.class_names),
            pretrained=False,
        )
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(self.device)
        self.model.eval()
        metadata_path = Path(os.getenv("PLANT_METADATA_PATH", DEFAULT_PLANT_META_PATH))
        self.plant_lookup = _load_plant_metadata(metadata_path)
        self.family_lookup = _load_family_lookup(metadata_path)
        self.family_merge_lookup = _build_family_merge_lookup(
            self.class_names, self.family_lookup, self.plant_lookup
        )
        logger.info("Loaded model '%s' on %s", self.cfg["model_name"], self.device)

    def _fallback_to_cpu(self) -> None:
        if self.device.type == "cpu":
            return
        logger.warning("Switching inference to CPU due to CUDA execution failure.")
        self.device = torch.device("cpu")
        self.model.to(self.device)

    def predict(self, image_bytes: bytes) -> PredictionResponse:
        try:
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}") from exc

        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            try:
                logits = self.model(tensor)
            except RuntimeError as exc:
                message = str(exc).lower()
                if self.device.type == "cuda" and ("no kernel image" in message or "cuda error" in message):
                    self._fallback_to_cpu()
                    tensor = tensor.to(self.device)
                    logits = self.model(tensor)
                else:
                    raise
            probs = F.softmax(logits, dim=1).squeeze(0)

        top_prob, top_idx = torch.max(probs, dim=0)
        confidences: List[ClassConfidence] = []
        for i, plant_id in enumerate(self.class_names):
            plant_name = self.plant_lookup.get(plant_id, plant_id)
            confidences.append(ClassConfidence(
                plant_id=plant_id,
                plant_name=plant_name,
                confidence=float(probs[i])
            ))

        plant_id = self.class_names[top_idx.item()]
        plant_name = self.plant_lookup.get(plant_id, plant_id)
        family_merge = self.family_merge_lookup.get(plant_id)
        return PredictionResponse(
            plant_id=plant_id,
            plant_name=plant_name,
            confidence=float(top_prob),
            class_confidences=confidences,
            family_merge=family_merge,
        )


classifier_service = PlantClassifierService()
app = FastAPI(title="Medicinal Plant Chat API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_name": classifier_service.cfg["model_name"],
        "checkpoint": str(classifier_service.checkpoint_path),
        "device": str(classifier_service.device),
        "num_classes": len(classifier_service.class_names),
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file received.")
    return classifier_service.predict(image_bytes)


@app.get("/")
def root():
    return {
        "message": "Medicinal plant classifier API",
        "endpoints": ["/health", "/predict"],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=bool(int(os.getenv("RELOAD", "0"))),
    )
