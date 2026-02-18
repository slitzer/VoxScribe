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

## Requirements
- Docker
- NVIDIA GPU (optional, for CUDA)
- Hugging Face token for diarization

## Installation
1. Clone repo: `git clone <repo-url>`
2. Build: `docker build -t whisperx-web .`
3. Run: `docker run --gpus all -p 7860:7860 whisperx-web` (or `--rm -p 7860:7860` for CPU-only)

Access at `http://localhost:7860` or server IP.

## Usage
- Open web UI
- Upload files
- Select model, language (auto-detect or dropdown), format, diarization (enter HF token)
- Click "Transcribe"
- Download outputs from /tmp (or mount volume: `-v /host/output:/tmp`)

## Configuration
- Edit `app.py` for defaults (e.g., force CPU by setting `device = "cpu"`)
- Rebuild after changes

## Troubleshooting
- Port in use: `docker ps -a | grep 7860; docker stop <id>`
- No space: `docker system prune -af --volumes`
- Old GPU (e.g., 980 Ti): Set `compute_type = "float32"`, batch_size=4 in `app.py`
- Diarization error: Ensure valid HF token; update `DiarizationPipeline` args

## Future/Wishlist
- Browser extension: Record live audio in chunks, send to server for real-time transcription. For groups (e.g., 6-player DnD sessions): Join session with usernames for speaker ID. Transcribe collaboratively.
- Auto-summarize transcripts: Generate notes/key points, retain original
- Persistent storage: Mount volumes for outputs
- Real-time streaming: WebSocket for live updates
- Multi-user auth/sessions
- API endpoints for integration
- Enhanced UI: Progress bars, ETA, previews
