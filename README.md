# Voice Diary

Privacy-first transcription and diary editing for personal voice memos. The Mac is the
source of truth: audio and generated files remain local. A Runpod load-balanced worker
temporarily receives audio, runs Qwen3-ASR and Qwen3, returns the result, and unlinks the
request file.

## Data flow

```text
local memo -> local ffmpeg chunks -> HTTPS multipart -> ephemeral Runpod worker
           <- transcript / cleaned text / diary <- direct HTTP response
```

The worker has no database, S3 integration, or network volume. Uploads use random names in
`/dev/shm` and are removed in a `finally` block. Application logs contain request IDs,
paths, status codes, elapsed time, and exception types only.

This is application-level zero persistence, not a claim that the cloud provider retains no
operational metadata. Runpod endpoint logs must never receive audio, transcripts, prompts,
model output, original filenames, headers, or exception messages.

## Project layout

- `src/voice_diary/app.py` — load-balanced FastAPI service.
- `src/voice_diary/models.py` — fake and Qwen model backends.
- `src/voice_diary/privacy.py` — bounded random-name temporary uploads.
- `src/voice_diary/client.py` — local chunk/upload/save CLI.
- `src/voice_diary/prompts/diary_zh.txt` — fact-preserving Chinese diary prompt.

## Local API tests

Use Python 3.11–3.13:

```bash
uv sync --extra dev
VOICE_DIARY_FAKE_MODE=1 uv run uvicorn voice_diary.app:app \
  --host 127.0.0.1 --port 8000 --no-access-log
uv run pytest
uv run ruff check .
```

Fake mode validates HTTP, cleanup, authentication, and local-client behavior. It must never
be enabled on the production endpoint.

## Local client

The client always normalizes and chunks the memo locally with `ffmpeg`. It then calls
`/v1/transcribe` for each chunk and `/v1/rewrite` once for the combined transcript.

```bash
export VOICE_DIARY_ENDPOINT_URL="https://ENDPOINT_ID.api.runpod.ai"
export VOICE_DIARY_APP_TOKEN="..."       # optional second factor

uv run voice-diary path/to/memo.m4a \
  --date 2026-08-28 \
  --title "今天的记录"
```

The client uses `RUNPOD_API_KEY` when set; otherwise it reads the selected
`RUNPOD_PROFILE` (default: `default`) from `~/.runpod/config.toml`. The key is never
written to output files or logs.

Results are written under `.voice-diary-output/`:

- `raw_transcript.json`
- `cleaned_transcript.md`
- `diary.md`
- `metadata.json`

## Production image

The image uses Runpod's PyTorch base and installs Qwen3-ASR, Transformers, and ffmpeg. The
GitHub Actions workflow builds `linux/amd64` and publishes to GHCR. Before processing real
audio, confirm the GitHub repository and GHCR package are private.

Production defaults:

- ASR: `Qwen/Qwen3-ASR-1.7B`
- timestamps: `Qwen/Qwen3-ForcedAligner-0.6B`
- editor: `Qwen/Qwen3-14B-FP8` with thinking disabled
- upload ceiling: 28 MiB (below Runpod's 30 MB load-balancer limit)
- port and health port: `5000`

Runpod endpoint configuration should use a 24 GB Ada GPU pool, one request per
worker, `workersMin=0`, `workersMax=1`, `endpointType=LOAD_BALANCER`, no network volume,
`flashboot=OFF`, and a short idle timeout. Validate actual VRAM usage before trying a 48 GB
or larger editor model.

## Required verification before real data

1. Run unit tests and static checks.
2. Build the private image and deploy with fake mode enabled.
3. Call it with synthetic audio and inspect all endpoint/worker logs for content leakage.
4. Redeploy with fake mode disabled and run a non-sensitive ASR fixture.
5. Verify the response, temporary-file cleanup, GPU memory, cold start, and scale-to-zero.
6. Only then process a real voice memo.
