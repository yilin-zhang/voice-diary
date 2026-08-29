FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=5000 \
    PORT_HEALTH=5000 \
    HF_HOME=/runpod-volume/huggingface-cache \
    HF_HUB_OFFLINE=1 \
    VOICE_DIARY_TEMP_DIR=/dev/shm/voice-diary

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m pip install --no-cache-dir --ignore-installed \
      "blinker>=1.9" \
      "cryptography>=46" \
    && python -m pip install --no-cache-dir ".[gpu]" \
    && python -m pip install --no-cache-dir --force-reinstall \
      "transformers @ git+https://github.com/huggingface/transformers.git@805a9e939fa8c1bff8d8ffdf041c051b71a914aa" \
    && python -c "from qwen_asr import Qwen3ASRModel; from transformers import AutoModelForMultimodalLM, AutoProcessor"

EXPOSE 5000

CMD ["uvicorn", "voice_diary.app:app", "--host", "0.0.0.0", "--port", "5000", "--no-access-log"]
