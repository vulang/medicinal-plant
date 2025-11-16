## Medicinal Plant Chat API

FastAPI service that wraps the trained plant classifier so the Angular chatbot can request predictions.

### Configuration

Environment variables (all optional):

| Name | Description | Default |
| --- | --- | --- |
| `MODEL_CONFIG_PATH` | Path to the YAML config produced for training. | `../../image-classifier/config.yaml` |
| `MODEL_CHECKPOINT_PATH` | Specific checkpoint to load. | `../../image-classifier/models/<model_name>_best.pt` |
| `MODEL_DEVICE` | `auto`, `cpu`, `cuda`, or `mps`. | Value from config or auto-detected |
| `ALLOWED_ORIGINS` | Comma-separated list for CORS. | `*` |
| `PLANT_METADATA_PATH` | CSV containing plant metadata (ID → Latin name). | `../../crawler/data/plant.csv` |

### Setup

```bash
cd application/chat-api
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Running locally

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

`POST /predict` accepts a multipart `file` field with an image and returns the predicted plant and class probabilities. Use `GET /health` for readiness probes.
