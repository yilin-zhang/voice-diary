from __future__ import annotations

import asyncio
import hmac
import json
import logging
import tempfile
import time
import traceback
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .audio import split_audio
from .config import Settings
from .models import FakeBackend, ModelManager, ModelsNotReady, QwenBackend
from .privacy import UploadTooLarge, private_upload
from .schemas import DiaryResult, Health, ProcessResult, RewriteRequest, Transcript

logger = logging.getLogger("voice_diary.api")


def _exception_location(exc: Exception) -> str:
    frames = traceback.extract_tb(exc.__traceback__)[-4:]
    return ">".join(
        f"{frame.filename.rsplit('/', 1)[-1]}:{frame.lineno}:{frame.name}" for frame in frames
    )


def _manager_for(settings: Settings) -> ModelManager:
    backend = FakeBackend() if settings.fake_mode else QwenBackend(settings)
    return ModelManager(backend)


def create_app(
    *,
    settings: Settings | None = None,
    manager: ModelManager | None = None,
    start_loader: bool = True,
) -> FastAPI:
    config = settings or Settings.from_env()
    models = manager or _manager_for(config)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        task: asyncio.Task[None] | None = None
        if start_loader:
            task = asyncio.create_task(asyncio.to_thread(models.load))
        yield
        if task and not task.done():
            task.cancel()

    api = FastAPI(
        title="Voice Diary",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    api.state.settings = config
    api.state.models = models

    @api.middleware("http")
    async def metadata_only_log(request, call_next):  # type: ignore[no-untyped-def]
        request_id = uuid.uuid4().hex
        started = time.monotonic()
        try:
            response = await call_next(request)
        except Exception as exc:
            logger.error(
                "request_failed request_id=%s path=%s error_type=%s error_location=%s",
                request_id,
                request.url.path,
                type(exc).__name__,
                _exception_location(exc),
            )
            return JSONResponse(
                status_code=500,
                content={"detail": "processing failed", "request_id": request_id},
                headers={"X-Request-ID": request_id, "Cache-Control": "no-store"},
            )
        elapsed_ms = round((time.monotonic() - started) * 1000)
        logger.info(
            "request_complete request_id=%s path=%s status=%s elapsed_ms=%s",
            request_id,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        response.headers["X-Request-ID"] = request_id
        response.headers["Cache-Control"] = "no-store"
        return response

    async def authorize(x_voice_diary_key: str | None = Header(default=None)) -> None:
        if config.app_token is None:
            return
        if x_voice_diary_key is None or not hmac.compare_digest(
            x_voice_diary_key, config.app_token
        ):
            raise HTTPException(status_code=401, detail="unauthorized")

    def require_models() -> None:
        try:
            models.ensure_ready()
        except ModelsNotReady as exc:
            raise HTTPException(status_code=503, detail="models are not ready") from exc

    async def run_transcription(upload: UploadFile, language: str | None) -> Transcript:
        require_models()
        try:
            async with private_upload(
                upload,
                directory=config.temp_dir,
                max_bytes=config.max_upload_bytes,
            ) as (audio_path, _size):
                return await asyncio.to_thread(models.transcribe, audio_path, language)
        except UploadTooLarge as exc:
            raise HTTPException(
                status_code=413,
                detail=f"audio exceeds {config.max_upload_bytes} bytes; split it locally",
            ) from exc

    async def run_rewrite(request: RewriteRequest) -> DiaryResult:
        require_models()
        return await asyncio.to_thread(
            models.rewrite,
            request.transcript,
            date=request.date,
            title_hint=request.title_hint,
        )

    def stream_error(exc: Exception) -> str:
        logger.error(
            "stream_failed error_type=%s error_location=%s",
            type(exc).__name__,
            _exception_location(exc),
        )
        return 'event: error\ndata: {"error":"processing failed"}\n\n'

    def stream_event(event: str, payload: dict[str, Any]) -> str:
        data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        return f"event: {event}\ndata: {data}\n\n"

    @api.get("/ping", response_model=Health)
    async def ping() -> Health | Response:
        if models.state == "ready":
            return Health(status="healthy")
        if models.state == "failed":
            return JSONResponse(status_code=503, content={"status": "failed"})
        return Response(status_code=204)

    @api.post(
        "/v1/transcribe",
        response_model=Transcript,
        dependencies=[Depends(authorize)],
    )
    async def transcribe(
        audio: Annotated[UploadFile, File()],
        language: Annotated[str | None, Form()] = "Chinese",
    ) -> Transcript:
        return await run_transcription(audio, language)

    @api.post(
        "/v1/rewrite",
        response_model=DiaryResult,
        dependencies=[Depends(authorize)],
    )
    async def rewrite(request: RewriteRequest) -> DiaryResult:
        return await run_rewrite(request)

    @api.post(
        "/v1/rewrite-stream",
        dependencies=[Depends(authorize)],
    )
    async def rewrite_stream(request: RewriteRequest) -> StreamingResponse:
        async def events():  # type: ignore[no-untyped-def]
            task = asyncio.create_task(run_rewrite(request))
            while True:
                try:
                    result = await asyncio.wait_for(asyncio.shield(task), timeout=10)
                except TimeoutError:
                    yield ": keep-alive\n\n"
                    continue
                except Exception as exc:
                    yield stream_error(exc)
                    return
                yield stream_event("diary", result.model_dump())
                return

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no", "Cache-Control": "no-store"},
        )

    @api.post(
        "/v1/process-stream",
        dependencies=[Depends(authorize)],
    )
    async def process_stream(
        audio: Annotated[UploadFile, File()],
        language: Annotated[str | None, Form()] = "Chinese",
        date: Annotated[str | None, Form()] = None,
        title_hint: Annotated[str | None, Form()] = None,
    ) -> StreamingResponse:
        async def run_pipeline(
            queue: asyncio.Queue[tuple[str, dict[str, Any]]],
        ) -> DiaryResult:
            require_models()
            try:
                async with private_upload(
                    audio,
                    directory=config.temp_dir,
                    max_bytes=config.max_upload_bytes,
                ) as (audio_path, _size):
                    with tempfile.TemporaryDirectory(
                        prefix="voice-diary-chunks-",
                        dir=config.temp_dir,
                    ) as raw_chunks:
                        chunks = await asyncio.to_thread(
                            split_audio,
                            audio_path,
                            Path(raw_chunks),
                            config.asr_chunk_seconds,
                        )
                        transcripts: list[dict[str, Any]] = []
                        for index, chunk in enumerate(chunks):
                            result = await asyncio.to_thread(models.transcribe, chunk, language)
                            item = result.model_dump()
                            item["chunk_index"] = index
                            item["offset_seconds"] = index * config.asr_chunk_seconds
                            transcripts.append(item)
            except UploadTooLarge as exc:
                raise HTTPException(
                    status_code=413,
                    detail=f"audio exceeds {config.max_upload_bytes} bytes",
                ) from exc

            full_transcript = "\n".join(item["text"].strip() for item in transcripts).strip()
            if not full_transcript:
                raise ValueError("ASR returned an empty transcript")
            await queue.put(
                (
                    "transcript",
                    {"text": full_transcript, "chunks": transcripts},
                )
            )
            return await asyncio.to_thread(
                models.rewrite,
                full_transcript,
                date=date,
                title_hint=title_hint,
            )

        async def events():  # type: ignore[no-untyped-def]
            queue: asyncio.Queue[tuple[str, dict[str, Any]]] = asyncio.Queue()
            task = asyncio.create_task(run_pipeline(queue))
            while not task.done():
                try:
                    event, payload = await asyncio.wait_for(queue.get(), timeout=10)
                except TimeoutError:
                    yield ": keep-alive\n\n"
                    continue
                yield stream_event(event, payload)

            while not queue.empty():
                event, payload = queue.get_nowait()
                yield stream_event(event, payload)

            try:
                result = task.result()
            except Exception as exc:
                yield stream_error(exc)
                return
            yield stream_event("diary", result.model_dump())

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no", "Cache-Control": "no-store"},
        )

    @api.post(
        "/v1/process",
        response_model=ProcessResult,
        dependencies=[Depends(authorize)],
    )
    async def process(
        audio: Annotated[UploadFile, File()],
        language: Annotated[str | None, Form()] = "Chinese",
        date: Annotated[str | None, Form()] = None,
        title_hint: Annotated[str | None, Form()] = None,
    ) -> ProcessResult:
        transcript = await run_transcription(audio, language)
        rewrite_result = await run_rewrite(
            RewriteRequest(transcript=transcript.text, date=date, title_hint=title_hint)
        )
        return ProcessResult(transcript=transcript, diary=rewrite_result)

    return api


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
logging.getLogger("uvicorn.access").disabled = True
app = create_app()
