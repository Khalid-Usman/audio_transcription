"""PostgreSQL storage for job state and results (one table, plain SQL).

Each status change is a single UPDATE guarded by the expected current status,
so a job can't be processed twice or marked completed after it failed.
"""

from typing import Any, Optional
from uuid import UUID

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool, PoolTimeout

SCHEMA = """
CREATE TABLE IF NOT EXISTS transcription_jobs (
    id               UUID PRIMARY KEY,
    status           TEXT NOT NULL CHECK (status IN ('queued', 'processing', 'completed', 'failed')),
    filename         TEXT NOT NULL,
    audio_path       TEXT NOT NULL,
    language         TEXT,            -- requested by the client; NULL = auto-detect
    duration_seconds DOUBLE PRECISION,
    progress         DOUBLE PRECISION NOT NULL DEFAULT 0,
    error            TEXT,
    result           JSONB,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at       TIMESTAMPTZ,
    finished_at      TIMESTAMPTZ
);
"""

Row = Optional[dict[str, Any]]


class Database:
    def __init__(self, url: str, pool_size: int):
        self.pool = ConnectionPool(url, min_size=1, max_size=pool_size, open=False,
                                   kwargs={"row_factory": dict_row})
        self._url = url

    def open(self) -> None:
        try:
            self.pool.open(wait=True, timeout=10)
        except (PoolTimeout, psycopg.OperationalError) as e:
            self.pool.close()
            info = psycopg.conninfo.conninfo_to_dict(self._url)  # shown without the password
            raise RuntimeError(f"Cannot connect to PostgreSQL (host={info.get('host')} port={info.get('port')} "
                               f"dbname={info.get('dbname')}). Check DATABASE_URL and that the server is running.") from e
        self._run(SCHEMA)

    def close(self) -> None:
        self.pool.close()

    def _run(self, sql: str, params: tuple = ()) -> Row:
        with self.pool.connection() as conn:
            cur = conn.execute(sql, params)
            return cur.fetchone() if cur.description else None

    def create_job(self, job_id: UUID, filename: str, audio_path: str,
                   duration: Optional[float], language: Optional[str]) -> Row:
        return self._run("INSERT INTO transcription_jobs (id, status, filename, audio_path, duration_seconds, language) "
                         "VALUES (%s, 'queued', %s, %s, %s, %s) RETURNING *",
                         (job_id, filename, audio_path, duration, language))

    def get_job(self, job_id: UUID) -> Row:
        return self._run("SELECT * FROM transcription_jobs WHERE id = %s", (job_id,))

    def start_job(self, job_id: UUID) -> Row:
        """queued -> processing; returns None if another worker already took it."""
        return self._run("UPDATE transcription_jobs SET status = 'processing', started_at = now(), updated_at = now() "
                         "WHERE id = %s AND status = 'queued' RETURNING *", (job_id,))

    def update_progress(self, job_id: UUID, progress: float) -> None:
        self._run("UPDATE transcription_jobs SET progress = %s, updated_at = now() "
                  "WHERE id = %s AND status = 'processing'", (round(progress, 4), job_id))

    def complete_job(self, job_id: UUID, result: dict) -> None:
        self._run("UPDATE transcription_jobs SET status = 'completed', result = %s, progress = 1, "
                  "duration_seconds = %s, finished_at = now(), updated_at = now() "
                  "WHERE id = %s AND status = 'processing'", (Jsonb(result), result["duration_seconds"], job_id))

    def fail_job(self, job_id: UUID, error: str) -> None:
        self._run("UPDATE transcription_jobs SET status = 'failed', error = %s, finished_at = now(), updated_at = now() "
                  "WHERE id = %s AND status IN ('queued', 'processing')", (error[:2000], job_id))

    def requeue_unfinished(self) -> list[dict[str, Any]]:
        """At start-up: put jobs interrupted by a crash back in the queue and return all queued jobs."""
        with self.pool.connection() as conn:
            conn.execute("UPDATE transcription_jobs SET status = 'queued', progress = 0, started_at = NULL "
                         "WHERE status = 'processing'")
            return conn.execute("SELECT id, audio_path, language FROM transcription_jobs "
                                "WHERE status = 'queued' ORDER BY created_at").fetchall()

    def ping(self) -> bool:
        try:
            self._run("SELECT 1")
            return True
        except Exception:  # noqa: BLE001 - the health check reports, it never raises
            return False
