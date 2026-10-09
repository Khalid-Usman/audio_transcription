"""Application settings, read from environment variables (and an optional .env file)."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Database: job state and transcription results ---
    database_url: str = "postgresql://postgres:postgres@localhost:5432/transcriptions"

    # --- Whisper model ---
    # A size name (tiny, base, small, medium, large-v3, turbo, ...) downloaded on first
    # use, or a path to a local CTranslate2 model directory.
    whisper_model: str = "small"
    whisper_device: Literal["auto", "cpu", "cuda"] = "auto"
    # "auto" picks float16 on CUDA and int8 on CPU. Other values: int8, int8_float16,
    # int8_float32, int16, float16, bfloat16, float32 (must be supported by the device).
    whisper_compute_type: str = "auto"
    whisper_cpu_threads: int = 0          # 0 = let CTranslate2 decide
    whisper_beam_size: int = 5
    whisper_vad_filter: bool = True       # skip silence; also keeps long-audio timestamps accurate
    model_download_dir: Path = Path("models")

    # --- Concurrency ---
    max_concurrent_transcriptions: int = 1  # transcriptions running at the same time
    max_pending_jobs: int = 20              # queued + running; more -> HTTP 503

    # --- Requests ---
    sync_max_duration_seconds: float = 30.0   # recordings up to this length get the result directly
    sync_timeout_seconds: float = 120.0       # ...unless it takes longer than this (then 202 + job id)
    max_upload_mb: int = 500
    max_audio_duration_seconds: float = 3 * 3600.0

    # --- Storage ---
    data_dir: Path = Path("data")             # uploaded audio is kept in data_dir/audio

    log_level: str = "INFO"

    @property
    def audio_dir(self) -> Path:
        return self.data_dir / "audio"


@lru_cache
def get_settings() -> Settings:
    return Settings()
