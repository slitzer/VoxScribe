# VoxScribe

## Description
Dockerized web app for audio/video transcription using WhisperX. Supports GPU/CPU, diarization, multiple files, TXT/SRT output. Built with Gradio frontend.

## Features
- Upload multiple audio/video files (WAV, MP3, M4A, MP4, MKV)
- Model selection: base, small, medium
- Pre-select language via dropdown (avoids auto-detection delay)
- Output formats: TXT (with timestamps/speakers), SRT
- Speaker diarization (requires HF token)
- GPU/CPU auto-detect (fallback to CPU)
- Queue processing for batches
- Temporary WAV conversion/cleanup
- Status updates
- Persistent output directory (`/outputs` in container)
- Docker Compose support for CPU and GPU profiles

## Requirements
- Docker
- Docker Compose v2 (`docker compose`)
- NVIDIA GPU + NVIDIA Container Toolkit (optional, for CUDA)
- Hugging Face token for diarization

## Quick Start (Docker Compose)
1. Clone repo: `git clone <repo-url>`
2. Enter repo: `cd VoxScribe`
3. Start CPU service:
   ```bash
   docker compose up --build -d voxscribe
   ```
4. Open `http://localhost:7860`

### GPU mode
Start the GPU profile/service:
```bash
docker compose --profile gpu up --build -d voxscribe-gpu
```

## Persistent Outputs
- Compose mounts `./outputs` on host to `/outputs` in container.
- Every transcript is written into `/outputs` and survives container recreation.
- Downloads in the UI point to files in this mounted folder.

You can change output location by setting `OUTPUT_DIR` environment variable for the container.

## Docker Run (without Compose)
You can still run directly with Docker:

```bash
# CPU
docker build -t whisperx-web .
docker run --rm -p 7860:7860 -e OUTPUT_DIR=/outputs -v "$PWD/outputs:/outputs" whisperx-web

# GPU
docker run --rm --gpus all -p 7860:7860 -e OUTPUT_DIR=/outputs -v "$PWD/outputs:/outputs" whisperx-web
```

## Usage
- Open web UI
- Upload files
- Select model, language (auto-detect or dropdown), format, diarization (enter HF token)
- Click "Transcribe"
- Download generated outputs from the app
- Find persisted files in local `./outputs`

## Configuration
- Edit `app.py` for defaults (e.g., force CPU by setting `device = "cpu"`)
- Set `OUTPUT_DIR` env var to change where transcript files are written
- Rebuild after code changes: `docker compose up --build -d`

## Troubleshooting
- Port in use: `docker ps -a | grep 7860; docker stop <id>`
- No space: `docker system prune -af --volumes`
- Old GPU (e.g., 980 Ti): Set `compute_type = "float32"`, `batch_size=4` in `app.py`
- Diarization error: Ensure valid HF token; update `DiarizationPipeline` args
- GPU compose issues: Verify NVIDIA Container Toolkit and `docker run --gpus all` works on your host
- Remote/non-localhost environments: set `GRADIO_SHARE=true` to force a shareable Gradio URL if localhost checks fail
- Gradio schema error (`TypeError: argument of type 'bool' is not iterable`): this build launches Gradio with `show_api=False` to avoid schema parsing on startup
- Localhost accessibility error (`ValueError: When localhost is not accessible...`): app now retries automatically with `share=True`; you can still force this with `GRADIO_SHARE=true`

## Future/Wishlist
- Browser extension: Record live audio in chunks, send to server for real-time transcription. For groups (e.g., 6-player DnD sessions): Join session with usernames for speaker ID. Transcribe collaboratively.
- Auto-summarize transcripts: Generate notes/key points, retain original
- Real-time streaming: WebSocket for live updates
- Multi-user auth/sessions
- API endpoints for integration
- Enhanced UI: Progress bars, ETA, previews
