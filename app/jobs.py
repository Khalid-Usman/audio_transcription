"""Runs transcriptions in a bounded thread pool, off the asyncio event loop.

Threads (not processes or Celery): the model is loaded once and shared, and
CTranslate2 releases the GIL while computing, so threads run truly in parallel.
Job state is in PostgreSQL, so unfinished jobs are re-queued after a restart.
"""

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Optional
from uuid import UUID

from faster_whisper import WhisperModel

from .config import Settings
from .db import Database
from .transcriber import AudioError, Runtime, transcribe

log = logging.getLogger(__name__)


class QueueFull(Exception):
    """MAX_PENDING_JOBS jobs are already waiting or running."""


class JobRunner:
    def __init__(self, settings: Settings, db: Database, model: WhisperModel, runtime: Runtime):
        self.settings, self.db, self.model, self.runtime = settings, db, model, runtime
        self._pool = ThreadPoolExecutor(settings.max_concurrent_transcriptions, thread_name_prefix="transcribe")
        self._pending = 0
        self._lock = threading.Lock()

    def has_capacity(self) -> bool:
        with self._lock:
            return self._pending < self.settings.max_pending_jobs

    def submit(self, job_id: UUID, audio_path: Path, language: Optional[str], force: bool = False) -> Future:
        with self._lock:
            if not force and self._pending >= self.settings.max_pending_jobs:
                raise QueueFull()
            self._pending += 1
        future = self._pool.submit(self._run, job_id, audio_path, language)
        future.add_done_callback(self._release_slot)
        return future

    def _release_slot(self, _: Future) -> None:
        with self._lock:
            self._pending -= 1

    def _run(self, job_id: UUID, audio_path: Path, language: Optional[str]) -> None:
        if self.db.start_job(job_id) is None:  # atomic queued -> processing; already taken otherwise
            return
        last_saved = 0.0

        def on_progress(fraction: float) -> None:  # write progress at most every 2 s
            nonlocal last_saved
            if time.monotonic() - last_saved >= 2:
                last_saved = time.monotonic()
                self.db.update_progress(job_id, fraction)

        try:
            result = transcribe(self.model, self.runtime, self.settings, audio_path, language, on_progress)
            self.db.complete_job(job_id, result)
            log.info("job %s done: %.1f s of audio in %.1f s", job_id, result["duration_seconds"],
                     result["processing_seconds"])
        except AudioError as e:
            self.db.fail_job(job_id, f"{e.code}: {e}")
        except Exception as e:  # noqa: BLE001 - a failing job must never kill the worker thread
            log.exception("job %s failed", job_id)
            self.db.fail_job(job_id, f"internal_error: {e}")

    def recover(self) -> None:
        """Re-queue jobs left unfinished by a previous run (crash or restart)."""
        for row in self.db.requeue_unfinished():
            if Path(row["audio_path"]).exists():
                self.submit(row["id"], Path(row["audio_path"]), row["language"], force=True)
            else:
                self.db.fail_job(row["id"], "audio_missing: the uploaded audio file no longer exists")

    def shutdown(self) -> None:
        # Finish running jobs; queued ones stay 'queued' in the database and recover() resumes them.
        self._pool.shutdown(wait=True, cancel_futures=True)
