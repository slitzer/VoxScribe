# VoxScribe Update Instructions (SSH Terminal)

This guide shows a full, repeatable process to update a running VoxScribe deployment from an SSH session.

## 1) Connect to your server

```bash
ssh <user>@<server-ip>
```

## 2) Go to the VoxScribe project directory

```bash
cd /path/to/VoxScribe
```

Optional checks:

```bash
git remote -v
git branch --show-current
```

## 3) Pull the latest code

If you deploy from `main`:

```bash
git fetch --all --prune
git checkout main
git pull --ff-only
```

## 4) (Optional but recommended) Backup outputs

VoxScribe persists transcripts in `./outputs` (mounted to `/outputs` in the container). You can still back this up before updating:

```bash
mkdir -p ~/voxscribe-backups
tar -czf ~/voxscribe-backups/voxscribe-outputs-$(date +%F-%H%M%S).tar.gz outputs
```

## 5) Rebuild and restart the app

### CPU deployment

```bash
docker compose up --build -d voxscribe
```

### GPU deployment

```bash
docker compose --profile gpu up --build -d voxscribe-gpu
```

## 6) Verify container status and logs

CPU service:

```bash
docker compose ps
docker compose logs --tail=200 voxscribe
```

GPU service:

```bash
docker compose logs --tail=200 voxscribe-gpu
```

## 7) Confirm app is reachable

Open in your browser:

```text
http://<server-ip>:7860
```

## 8) Alternative: Run without Docker Compose

CPU:

```bash
docker build -t whisperx-web .
docker run --rm -p 7860:7860 -e OUTPUT_DIR=/outputs -v "$PWD/outputs:/outputs" whisperx-web
```

GPU:

```bash
docker run --rm --gpus all -p 7860:7860 -e OUTPUT_DIR=/outputs -v "$PWD/outputs:/outputs" whisperx-web
```

## 9) Common troubleshooting after update

### Port conflict

```bash
docker ps -a | grep 7860
docker stop <id>
```

### Disk cleanup

```bash
docker system prune -af --volumes
```

### Remote/non-localhost Gradio access issues

Set `GRADIO_SHARE=true` and restart the container.

### GPU issues

Verify NVIDIA Container Toolkit is installed and that this works on the host:

```bash
docker run --rm --gpus all nvidia/cuda:12.3.2-base-ubuntu22.04 nvidia-smi
```

## 10) One-shot update command (CPU)

```bash
cd /path/to/VoxScribe && \
  git fetch --all --prune && \
  git checkout main && \
  git pull --ff-only && \
  docker compose up --build -d voxscribe && \
  docker compose ps
```

