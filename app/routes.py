"""HTTP endpoints and their JSON response schemas."""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import BinaryIO, Literal, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .jobs import QueueFull
from .transcriber import SUPPORTED_EXTENSIONS, AudioError, probe_audio

router = APIRouter()


class Segment(BaseModel):
    id: int
    start: float  # seconds from the start of the recording
    end: float
    text: str


class TranscriptionResult(BaseModel):
    job_id: UUID
    status: Literal["completed"] = "completed"
    language: str
    language_probability: float
    duration_seconds: float
    text: str
    segments: list[Segment]
    model: str
    device: str
    compute_type: str
    processing_seconds: float


class JobStatus(BaseModel):
    job_id: UUID
    status: Literal["queued", "processing", "completed", "failed"]
    filename: str
    duration_seconds: Optional[float]
    progress: float  # 0..1
    error: Optional[str]
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime]
    finished_at: Optional[datetime]
    result_url: str


def _error(status: int, code: str, message: str, retry_after: Optional[int] = None) -> HTTPException:
    headers = {"Retry-After": str(retry_after)} if retry_after else None
    return HTTPException(status, detail={"code": code, "message": message}, headers=headers)


def _status_body(job: dict) -> dict:
    fields = {k: job[k] for k in JobStatus.model_fields if k in job}
    return JobStatus(**fields, job_id=job["id"], result_url=f"/v1/transcriptions/{job['id']}/result").model_dump(mode="json")


def _result_body(job: dict) -> dict:
    return TranscriptionResult(job_id=job["id"], **job["result"]).model_dump(mode="json")


def _save_upload(src: BinaryIO, dest: Path, max_bytes: int) -> int:
    """Copy the upload in 1 MiB pieces, stopping as soon as it's over the limit."""
    size = 0
    with open(dest, "wb") as out:
        while piece := src.read(1 << 20):
            size += len(piece)
            if size > max_bytes:
                raise _error(413, "file_too_large", f"The file exceeds the {max_bytes >> 20} MB limit.")
            out.write(piece)
    return size


@router.post("/v1/transcriptions", status_code=202, response_model=JobStatus,
             responses={200: {"model": TranscriptionResult, "description": "Short recording, finished"}})
async def create_transcription(request: Request,
                               file: UploadFile = File(..., description="WAV, MP3, M4A or FLAC"),
                               language: Optional[str] = Form(None, description="e.g. 'en'; omit to auto-detect")):
    settings, runner, db = request.app.state.settings, request.app.state.runner, request.app.state.db

    ext = Path(file.filename or "").suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise _error(415, "unsupported_format", f"Unsupported file type '{ext}'. "
                                                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}.")
    if language and language not in runner.model.supported_languages:
        raise _error(422, "unsupported_language", f"Unknown language code '{language}'.")
    if not runner.has_capacity():
        raise _error(503, "server_busy", "Too many transcriptions in progress; try again later.", 30)

    # Save under a temporary name and rename only once it's validated, so a stored file is always complete.
    job_id = uuid4()
    final_path = settings.audio_dir / f"{job_id}{ext}"
    tmp_path = settings.audio_dir / f".upload-{job_id}{ext}"
    try:
        if await run_in_threadpool(_save_upload, file.file, tmp_path, settings.max_upload_mb << 20) == 0:
            raise _error(422, "empty_file", "The uploaded file is empty.")
        info = await run_in_threadpool(probe_audio, tmp_path)
        if (info.duration_seconds or 0) > settings.max_audio_duration_seconds:
            raise _error(422, "audio_too_long", f"The limit is {settings.max_audio_duration_seconds:.0f} seconds.")
        tmp_path.replace(final_path)
    except AudioError as e:
        raise _error(422, e.code, str(e)) from e
    finally:
        tmp_path.unlink(missing_ok=True)

    job = await run_in_threadpool(db.create_job, job_id, file.filename, str(final_path), info.duration_seconds, language)
    try:
        future = runner.submit(job_id, final_path, language)
    except QueueFull:
        await run_in_threadpool(db.fail_job, job_id, "server_busy: the queue was full")
        final_path.unlink(missing_ok=True)
        raise _error(503, "server_busy", "Too many transcriptions in progress; try again later.", 30) from None

    # Short recordings: await the result without blocking the event loop. shield() keeps the job
    # running if the wait times out or the client disconnects; the client then gets a job ID.
    if info.duration_seconds is not None and info.duration_seconds <= settings.sync_max_duration_seconds:
        try:
            await asyncio.wait_for(asyncio.shield(asyncio.wrap_future(future)), settings.sync_timeout_seconds)
        except asyncio.TimeoutError:
            pass
        job = await run_in_threadpool(db.get_job, job_id)
        if job["status"] == "completed":
            return JSONResponse(_result_body(job), status_code=200)
        if job["status"] == "failed":
            code, _, message = job["error"].partition(": ")
            raise _error(422, code, message)

    return JSONResponse(_status_body(job), status_code=202, headers={"Location": f"/v1/transcriptions/{job_id}"})


# Plain `def` endpoints run in FastAPI's thread pool, so database calls don't block the event loop.

@router.get("/v1/transcriptions/{job_id}", response_model=JobStatus)
def get_status(job_id: UUID, request: Request):
    job = request.app.state.db.get_job(job_id)
    if job is None:
        raise _error(404, "not_found", f"No transcription job {job_id}.")
    return _status_body(job)


@router.get("/v1/transcriptions/{job_id}/result", response_model=TranscriptionResult)
def get_result(job_id: UUID, request: Request):
    job = request.app.state.db.get_job(job_id)
    if job is None:
        raise _error(404, "not_found", f"No transcription job {job_id}.")
    if job["status"] == "failed":
        raise _error(409, "job_failed", job["error"])
    if job["status"] != "completed":
        raise _error(409, "not_ready", f"Job is {job['status']} ({job['progress']:.0%} done).", 5)
    return _result_body(job)


@router.get("/health")
def health(request: Request):
    state = request.app.state
    db_ok = state.db.ping()
    body = {"status": "ok" if db_ok else "degraded", "database": "ok" if db_ok else "unreachable",
            "model": state.settings.whisper_model, "device": state.runner.runtime.device,
            "compute_type": state.runner.runtime.compute_type}
    return JSONResponse(body, status_code=200 if db_ok else 503)
