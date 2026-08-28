FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=5000 \
    PORT_HEALTH=5000 \
    HF_HOME=/models/huggingface \
    VOICE_DIARY_TEMP_DIR=/dev/shm/voice-diary

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m pip install --no-cache-dir ".[gpu]"

EXPOSE 5000

CMD ["uvicorn", "voice_diary.app:app", "--host", "0.0.0.0", "--port", "5000", "--no-access-log"]
