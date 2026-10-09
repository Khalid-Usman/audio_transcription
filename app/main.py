"""FastAPI app. Run with:  uvicorn app.main:app --host 0.0.0.0 --port 8000

Use one uvicorn process (no --workers): concurrency comes from the thread pool,
each extra process would load another copy of the model, and start-up job
recovery assumes a single process owns the queue.
"""

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool

from .config import Settings, get_settings
from .db import Database
from .jobs import JobRunner
from .routes import router
from .transcriber import load_model


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logging.basicConfig(level=settings.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
        settings.audio_dir.mkdir(parents=True, exist_ok=True)
        for leftover in settings.audio_dir.glob(".upload-*"):  # partial uploads from a crash
            leftover.unlink(missing_ok=True)

        db = Database(settings.database_url, pool_size=settings.max_concurrent_transcriptions + 5)
        await run_in_threadpool(db.open)
        try:
            model, runtime = await run_in_threadpool(load_model, settings)  # once per process
        except Exception:
            db.close()
            raise
        app.state.settings, app.state.db = settings, db
        app.state.runner = JobRunner(settings, db, model, runtime)
        app.state.runner.recover()
        yield
        await run_in_threadpool(app.state.runner.shutdown)
        db.close()

    app = FastAPI(title="Audio Transcription Service", version="1.0.0", lifespan=lifespan)
    app.include_router(router)
    return app


app = create_app()
