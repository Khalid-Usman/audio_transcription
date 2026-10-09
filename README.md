# Audio Transcription Service

A FastAPI service that transcribes audio with [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
and returns the text with start and end timestamps for each segment. Short recordings get their
transcription in the upload response. Long recordings get a job ID to poll. Job state and results
are stored in PostgreSQL, and uploaded audio in a local folder.

## How it works

```
POST /v1/transcriptions
   │  1. check file type, save upload to data/audio/, check it's real audio (PyAV)
   │  2. insert job row (status = queued) in PostgreSQL
   │  3. hand the job to the transcription thread pool
   │
   ├── short recording (≤ SYNC_MAX_DURATION_SECONDS): wait for the result → 200 + transcription
   └── long recording: return immediately                               → 202 + job ID

Thread pool (MAX_CONCURRENT_TRANSCRIPTIONS threads, one shared Whisper model)
   queued → processing → completed (result JSON stored in PostgreSQL) | failed (error stored)
```

- **The model is loaded once** at start-up and shared by all requests.
- **Transcription never runs on the asyncio event loop.** It runs in a thread pool. CTranslate2
  (the engine inside faster-whisper) releases Python's GIL while it computes, so threads run in
  parallel and the API keeps answering while jobs run.
- **Concurrency is bounded.** At most `MAX_CONCURRENT_TRANSCRIPTIONS` jobs run at once. Beyond
  `MAX_PENDING_JOBS` waiting or running jobs, new uploads get `503` instead of piling up.
- **Long recordings.** faster-whisper processes audio in 30-second windows over the whole file.
  With the Silero VAD filter on, it transcribes only speech and maps every timestamp back to the
  original recording. There's no manual cutting, so no words are split at chunk boundaries.
- **Restarts are safe.** Jobs interrupted by a crash or restart are re-queued on the next start.
  A status change only succeeds from the expected previous status, so a job is never processed
  twice. Partial uploads are written to a temporary name and renamed only when complete.

```
app/
  main.py         app factory and start-up/shutdown (database, model, job recovery)
  config.py       settings from environment variables / .env
  routes.py       the four endpoints and their JSON schemas
  transcriber.py  device selection, model loading, audio validation, transcription
  jobs.py         thread-pool job runner with a concurrency limit and restart recovery
  db.py           PostgreSQL table and queries (plain SQL)
```

The code is kept deliberately small: about 600 lines of application code in six modules, with no
base classes, dependency-injection framework or plug-in layers. Each module does one job, and the
comments explain *why* a choice was made rather than restating the code.

## Design decisions

Each choice below lists what was decided, why, and what it costs.

**1. faster-whisper as the speech engine.**
It runs the same Whisper weights as OpenAI's reference implementation, but on CTranslate2, which is
several times faster and uses much less memory. It supports `int8` on CPU, so CPU-only machines are
practical. It also includes Silero VAD and returns segment timestamps.
*Trade-off:* CTranslate2 has no Apple GPU (Metal) backend, so Apple Silicon runs on the CPU.

**2. One process with a thread pool, not Celery/Redis or a process pool.**
The model is loaded once and shared. Threads give real parallelism here because CTranslate2
releases Python's GIL while it computes. A process pool would load a full copy of the model per
process. Celery or Redis would add infrastructure this service doesn't need.
*Trade-off:* the queue lives in one process. Scaling means a bigger machine or GPU, not more
machines.

**3. Bounded concurrency with backpressure.**
At most `MAX_CONCURRENT_TRANSCRIPTIONS` jobs run at once. Once `MAX_PENDING_JOBS` are waiting or
running, new uploads get `503` with `Retry-After`. An unbounded queue would grow memory and waiting
times without limit, and every client would just get slower. A fast, explicit "busy" is easier to
handle.

**4. Short files answer directly; long files return a job ID.**
The duration is read from the file header before any decoding. Recordings up to
`SYNC_MAX_DURATION_SECONDS` are awaited, so simple clients get the transcription in one call. The
wait is `asyncio.shield`ed and limited by `SYNC_TIMEOUT_SECONDS`: if it runs long, or the client
disconnects, the job keeps running and the response becomes `202`. Long recordings return `202`
straight away, so no HTTP connection is held open for minutes.

**5. Long audio: faster-whisper's own windowing plus VAD, no manual chunking.**
faster-whisper already walks the whole recording in 30-second windows. With VAD on, it skips
silence and maps every timestamp back to the original timeline. Cutting the file into fixed chunks
ourselves would split words at the cut points and require timestamp stitching.
`condition_on_previous_text` is off, so one bad segment can't snowball into repeated text over a
long recording. Timestamps are clamped to the recording's length, because Whisper occasionally
predicts times past the end.
*Trade-off:* the decoded audio is held in memory (about 230 MB per hour), so
`MAX_AUDIO_DURATION_SECONDS` caps the length.

**6. Audio on local disk; job state and results in PostgreSQL.**
Results are small, structured and read back by job ID, so they fit in one row per job as `JSONB`.
Audio files can be hundreds of MB. Storing them in the database would bloat it and slow backups,
while a folder plus a path column costs nothing.

**7. Job state that survives crashes.**
- A status change is a single `UPDATE … WHERE status = <expected>`. A job can only move from
  `queued` to `processing` once, so it is never transcribed twice, and a failed job can't later be
  marked completed.
- On start-up, jobs left `queued` or `processing` by a crash are re-queued, because their audio is
  still on disk.
- Uploads are written to a temporary name and renamed into place only when complete. Leftover
  temporary files are deleted at start-up.

**8. Validate files twice: by extension, then by content.**
The extension allowlist rejects obviously wrong files immediately (`415`) without reading them. The
content check opens the file with PyAV and decodes one frame. It catches renamed or corrupt files
(`422`) before a job is created, instead of failing later in the worker.

**9. No system FFmpeg; PyAV pinned below 19.**
faster-whisper decodes audio with PyAV, whose wheels include FFmpeg, so users don't install FFmpeg
on any platform. PyAV is pinned below 19 because faster-whisper 1.2.1 passes an argument that
PyAV 19 removed, which breaks every decode. The tests caught this.

**10. Explicit device selection that fails loudly.**
- `auto` uses the GPU when CTranslate2 can see one, otherwise the CPU.
- `cuda` means "I expect a GPU": without one, start-up stops with an explanation instead of quietly
  running many times slower on the CPU.
- The compute type is checked against what the hardware actually supports.
- A one-second warm-up transcription at start-up makes missing CUDA or cuDNN libraries fail
  immediately, not on the first user's upload.

## Setup

### 1. Prerequisites

- **Python 3.10–3.13**
- **PostgreSQL** (tested with 16; any currently supported version should work). Install it and create a database:
  - macOS: `brew install postgresql@16 && brew services start postgresql@16 && createdb transcriptions`
  - Ubuntu/Debian: `sudo apt install postgresql && sudo -u postgres createdb transcriptions`
  - Windows: install from postgresql.org, then `createdb -U postgres transcriptions`
- **FFmpeg / FFprobe: not needed.** faster-whisper decodes audio with PyAV, and PyAV's wheels
  include the FFmpeg libraries. You'd only need FFmpeg's development libraries if pip has no PyAV
  wheel for your platform and builds it from source.

### 2. Virtual environment

macOS / Linux:
```bash
cd audio_transcription_service
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # then edit DATABASE_URL etc.
```

Windows (PowerShell):
```powershell
cd audio_transcription_service
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

(Conda users: `conda env create -f environment.yml`. It installs Python from conda and the same
`requirements.txt`.)

### 3. CPU configuration (default, every platform)

```ini
WHISPER_DEVICE=cpu            # or auto
WHISPER_COMPUTE_TYPE=int8     # or auto
WHISPER_MODEL=small
MAX_CONCURRENT_TRANSCRIPTIONS=1
WHISPER_CPU_THREADS=0         # 0 = CTranslate2 default
```

On CPU, total throughput is set by your cores. Raising `MAX_CONCURRENT_TRANSCRIPTIONS` above 1
helps only when there are spare cores. A rough start is cores ÷ 4 concurrent jobs, each with
`WHISPER_CPU_THREADS=4`.

### 4. NVIDIA GPU configuration (Linux x86_64 / Windows x64)

The GPU libraries are *not* in `requirements.txt`, so CPU users don't download them. You need an
NVIDIA driver plus the CUDA 12 **cuBLAS** and **cuDNN 9** libraries. faster-whisper's documented
options:

- **Linux, via pip** (inside the venv):
  ```bash
  pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"
  export LD_LIBRARY_PATH=$(python -c 'import os, nvidia.cublas.lib, nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))')
  ```
- **Windows / Linux, system-wide:** install the CUDA 12 toolkit and cuDNN 9 for CUDA 12 from
  NVIDIA, and make sure their `bin` (Windows) or `lib` (Linux) folders are on `PATH` /
  `LD_LIBRARY_PATH`.
- Older setups: CUDA 12 with cuDNN 8 needs `ctranslate2==4.4.0`; CUDA 11 needs `ctranslate2==3.24.0`.

Then:
```ini
WHISPER_DEVICE=cuda           # cuda = refuse to start without a GPU; auto = fall back to CPU
WHISPER_COMPUTE_TYPE=float16  # or auto; int8_float16 uses less GPU memory
WHISPER_MODEL=large-v3        # or turbo for speed
MAX_CONCURRENT_TRANSCRIPTIONS=2
```

At start-up the service runs a one-second warm-up transcription. Missing CUDA libraries therefore
show up immediately with a message pointing here, rather than on the first upload. The GPU path
was **not** run on the build machine (it has no GPU).

## Running

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Run a **single** uvicorn process (the default; don't add `--workers`). Concurrency comes from the
thread pool. Each extra process would load another copy of the model, and start-up job recovery
assumes one process owns the queue. The first start downloads the model into `MODEL_DOWNLOAD_DIR`.
Interactive API docs: http://localhost:8000/docs

## API

### `POST /v1/transcriptions`
Multipart upload. The field `file` is required; `language` (e.g. `en`) is optional and is
auto-detected if omitted.
```bash
curl -F "file=@meeting.mp3" http://localhost:8000/v1/transcriptions
curl -F "file=@lecture.m4a" -F "language=en" http://localhost:8000/v1/transcriptions
```

| Status | When | Body |
|---|---|---|
| `200` | Short recording, finished | Transcription (below) |
| `202` | Long recording (or a short one still running after `SYNC_TIMEOUT_SECONDS`) | Job status; `Location` header |
| `413` | File larger than `MAX_UPLOAD_MB` | `{"detail": {"code": "file_too_large", "message": ...}}` |
| `415` | Extension not .wav .mp3 .m4a .flac .ogg .opus .webm .aac | `unsupported_format` |
| `422` | Not real audio, empty, too long, unknown language, or transcription failed | `invalid_audio`, `empty_file`, `audio_too_long`, `unsupported_language`, ... |
| `503` | `MAX_PENDING_JOBS` reached | `server_busy`, with `Retry-After` |

Example transcription (200, abridged):
```json
{
  "job_id": "5a1f0f3e-9a52-4c4e-8f61-2d7c3b1e9a10",
  "status": "completed",
  "language": "en",
  "language_probability": 0.98,
  "duration_seconds": 10.0,
  "text": "Hello and welcome. Today we will look at the quarterly numbers.",
  "segments": [
    {"id": 0, "start": 0.52, "end": 2.9, "text": "Hello and welcome."},
    {"id": 1, "start": 3.4, "end": 7.1, "text": "Today we will look at the quarterly numbers."}
  ],
  "model": "small", "device": "cpu", "compute_type": "int8",
  "processing_seconds": 1.84
}
```

### `GET /v1/transcriptions/{job_id}`
```bash
curl http://localhost:8000/v1/transcriptions/5a1f0f3e-9a52-4c4e-8f61-2d7c3b1e9a10
```
```json
{
  "job_id": "5a1f0f3e-...", "status": "processing", "filename": "lecture.m4a",
  "duration_seconds": 3600.0, "progress": 0.42, "error": null,
  "created_at": "...", "updated_at": "...", "started_at": "...", "finished_at": null,
  "result_url": "/v1/transcriptions/5a1f0f3e-.../result"
}
```
`status` is `queued`, `processing`, `completed` or `failed`. The response is `404` for an unknown
ID and `422` for a malformed one.

### `GET /v1/transcriptions/{job_id}/result`
```bash
curl http://localhost:8000/v1/transcriptions/5a1f0f3e-9a52-4c4e-8f61-2d7c3b1e9a10/result
```
Returns `200` with the transcription once completed. Returns `409` with `not_ready` (and
`Retry-After`) while queued or processing, `409` with `job_failed` if it failed, and `404` for an
unknown ID.

### `GET /health`
```bash
curl http://localhost:8000/health
```
```json
{"status": "ok", "database": "ok", "model": "small", "device": "cpu", "compute_type": "int8"}
```
Returns `503` with `"database": "unreachable"` if PostgreSQL is down.

## Environment variables

Set them in the environment or in `.env` (see `.env.example`).

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `postgresql://postgres:postgres@localhost:5432/transcriptions` | PostgreSQL connection; the table is created automatically |
| `WHISPER_MODEL` | `small` | `tiny`, `base`, `small`, `medium`, `large-v3`, `turbo`, ... or a local model directory |
| `WHISPER_DEVICE` | `auto` | `auto` (GPU if usable, else CPU), `cpu`, `cuda` |
| `WHISPER_COMPUTE_TYPE` | `auto` | `auto` = `float16` on GPU, `int8` on CPU. Unsupported values stop start-up and list the supported ones |
| `WHISPER_CPU_THREADS` | `0` | CPU threads per transcription (0 = CTranslate2 default) |
| `WHISPER_BEAM_SIZE` | `5` | Beam search width (1 = fastest) |
| `WHISPER_VAD_FILTER` | `true` | Skip silence with Silero VAD |
| `MODEL_DOWNLOAD_DIR` | `models` | Where models are downloaded |
| `MAX_CONCURRENT_TRANSCRIPTIONS` | `1` | Transcriptions running at the same time |
| `MAX_PENDING_JOBS` | `20` | Queued + running jobs before uploads get `503` |
| `SYNC_MAX_DURATION_SECONDS` | `30` | Recordings up to this length return the transcription directly |
| `SYNC_TIMEOUT_SECONDS` | `120` | Longest wait for a direct result before falling back to `202` |
| `MAX_UPLOAD_MB` | `500` | Upload size limit |
| `MAX_AUDIO_DURATION_SECONDS` | `10800` | Longest accepted recording (3 h) |
| `DATA_DIR` | `data` | Uploaded audio is stored in `DATA_DIR/audio` |
| `LOG_LEVEL` | `INFO` | Logging level |
