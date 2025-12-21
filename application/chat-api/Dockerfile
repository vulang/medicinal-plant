FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system dependencies required by PyTorch/Pillow wheels and ngrok.
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential curl ca-certificates && \
    curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null && \
    echo "deb https://ngrok-agent.s3.amazonaws.com $(. /etc/os-release && echo ${VERSION_CODENAME:-stable}) main" \
      | tee /etc/apt/sources.list.d/ngrok.list >/dev/null && \
    apt-get update && \
    apt-get install -y --no-install-recommends ngrok && \
    rm -rf /var/lib/apt/lists/*

COPY application/chat-api/requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && pip install -r /tmp/requirements.txt

# Copy the FastAPI application as well as the trained model artifacts.
COPY application/chat-api /app/application/chat-api
# Only ship the two trained checkpoints needed for inference to keep the image slim.
COPY image-classifier/merge_config.py /app/image-classifier/merge_config.py
COPY image-classifier/models/convnextv2_base.fcmae_ft_in22k_in1k_best.pt /app/image-classifier/models/
COPY image-classifier/models/swin_base_patch4_window7_224.ms_in22k_best.pt /app/image-classifier/models/
# Plant metadata used for ID-to-name mapping.
COPY crawler/data/plants.csv /app/crawler/data/plants.csv

COPY application/chat-api/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

WORKDIR /app/application/chat-api

EXPOSE 8000

CMD ["/app/entrypoint.sh"]
