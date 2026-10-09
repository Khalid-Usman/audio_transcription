"""Device selection, model loading, audio validation and transcription.

Audio is decoded by PyAV (bundled with faster-whisper; its wheels include FFmpeg),
so no system ffmpeg is needed. Long recordings: faster-whisper walks the file in
30-second windows, and with the Silero VAD filter it transcribes only speech and
maps timestamps back to the original timeline, so no manual chunking (which
would split words) is needed. Decoded audio takes ~230 MB of RAM per hour.
"""

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import av
import ctranslate2
import numpy as np
from faster_whisper import WhisperModel, decode_audio

from .config import Settings

SAMPLE_RATE = 16_000
SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".webm", ".aac"}


class AudioError(ValueError):
    """The file can't be transcribed; ``code`` and the message are returned to the client."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class AudioInfo:
    duration_seconds: Optional[float]  # None if the container doesn't record it


@dataclass(frozen=True)
class Runtime:
    device: str        # "cpu" or "cuda"
    compute_type: str  # e.g. "int8", "float16"


def probe_audio(path: Path) -> AudioInfo:
    """Check by content (not extension) that the file is decodable audio."""
    try:
        with av.open(str(path)) as container:
            if not container.streams.audio:
                raise AudioError("no_audio_stream", "The file contains no audio stream.")
            stream = container.streams.audio[0]
            next(container.decode(stream), None)  # decode one frame to catch corrupt data
            if container.duration:
                return AudioInfo(container.duration / av.time_base)
            return AudioInfo(float(stream.duration * stream.time_base) if stream.duration else None)
    except AudioError:
        raise
    except Exception as e:  # PyAV raises many different FFmpeg errors for unreadable input
        raise AudioError("invalid_audio", "The file is not a readable audio file.") from e


def resolve_runtime(device: str, compute_type: str) -> Runtime:
    """Turn WHISPER_DEVICE / WHISPER_COMPUTE_TYPE into a supported pair, or fail with a clear message."""
    try:
        gpus = ctranslate2.get_cuda_device_count()
    except Exception:  # noqa: BLE001 - no CUDA runtime at all
        gpus = 0
    if device == "auto":
        device = "cuda" if gpus else "cpu"
    elif device == "cuda" and not gpus:
        if sys.platform == "darwin":
            raise RuntimeError("WHISPER_DEVICE=cuda is not available on macOS: faster-whisper has no Apple GPU "
                               "(Metal) backend, so Apple Silicon runs on the CPU. Use WHISPER_DEVICE=cpu or auto.")
        raise RuntimeError("WHISPER_DEVICE=cuda but no CUDA GPU is visible. Check the NVIDIA driver (nvidia-smi); "
                           "CUDA builds of CTranslate2 exist only for Linux x86_64 and Windows x64. "
                           "Use WHISPER_DEVICE=cpu or auto to run on the CPU.")

    supported = ctranslate2.get_supported_compute_types(device)
    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" and "float16" in supported else \
            "int8" if "int8" in supported else "float32"
    elif compute_type not in supported:
        raise RuntimeError(f"WHISPER_COMPUTE_TYPE={compute_type} is not supported on {device} here. "
                           f"Supported: {', '.join(sorted(supported))}.")
    return Runtime(device, compute_type)


def load_model(settings: Settings) -> tuple[WhisperModel, Runtime]:
    """Load once at start-up, with a 1-second warm-up so problems such as missing CUDA
    libraries or a bad model name surface now instead of on the first request."""
    runtime = resolve_runtime(settings.whisper_device, settings.whisper_compute_type)
    try:
        model = WhisperModel(settings.whisper_model, device=runtime.device, compute_type=runtime.compute_type,
                             cpu_threads=settings.whisper_cpu_threads,
                             num_workers=settings.max_concurrent_transcriptions,  # parallel calls on one model
                             download_root=str(settings.model_download_dir))
        list(model.transcribe(np.zeros(SAMPLE_RATE, np.float32), language="en", beam_size=1)[0])
    except Exception as e:
        hint = ""
        if not Path(settings.whisper_model).exists():
            hint += " Model names are downloaded from Hugging Face on first use: check network access, " \
                    "or set WHISPER_MODEL to a local model directory."
        if runtime.device == "cuda":
            hint += " NVIDIA GPUs also need the CUDA 12 cuBLAS and cuDNN 9 libraries (see README), " \
                    "or set WHISPER_DEVICE=cpu."
        raise RuntimeError(f"Could not load Whisper model '{settings.whisper_model}' "
                           f"on {runtime.device}/{runtime.compute_type}: {e}.{hint}") from e
    return model, runtime


def transcribe(model: WhisperModel, runtime: Runtime, settings: Settings, audio_path: Path,
               language: Optional[str] = None, on_progress: Optional[Callable[[float], None]] = None) -> dict:
    """Transcribe one file. CPU/GPU-heavy and blocking: call it from a worker thread."""
    started = time.perf_counter()
    try:
        audio = decode_audio(str(audio_path), sampling_rate=SAMPLE_RATE)  # mono float32
    except Exception as e:
        raise AudioError("invalid_audio", "The audio could not be decoded.") from e
    duration = len(audio) / SAMPLE_RATE
    if duration > settings.max_audio_duration_seconds:  # for files whose header had no duration
        raise AudioError("audio_too_long", f"The limit is {settings.max_audio_duration_seconds:.0f} seconds.")

    segments_iter, info = model.transcribe(
        audio, language=language, beam_size=settings.whisper_beam_size,
        vad_filter=settings.whisper_vad_filter, vad_parameters={"min_silence_duration_ms": 500},
        condition_on_previous_text=False,  # stops one bad segment from repeating through long audio
    )
    segments = []
    for seg in segments_iter:  # decoding happens lazily, one segment at a time
        start = min(max(seg.start, 0.0), duration)  # Whisper sometimes predicts times past the end
        end = min(max(seg.end, start), duration)
        if seg.text.strip():
            segments.append({"id": len(segments), "start": round(start, 3), "end": round(end, 3),
                             "text": seg.text.strip()})
        if on_progress and duration:
            on_progress(end / duration)

    return {"language": info.language, "language_probability": round(info.language_probability, 4),
            "duration_seconds": round(duration, 3), "text": " ".join(s["text"] for s in segments),
            "segments": segments, "model": settings.whisper_model, "device": runtime.device,
            "compute_type": runtime.compute_type, "processing_seconds": round(time.perf_counter() - started, 3)}
