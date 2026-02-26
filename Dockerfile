FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3.10 python3-pip ffmpeg libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libswresample-dev && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install --no-cache-dir -r requirements.txt
# torchcodec is optional for this app path and can emit noisy FFmpeg ABI errors
# on base images that ship FFmpeg 4.x only.
RUN pip uninstall -y torchcodec || true

COPY app.py .

ENV OUTPUT_DIR=/outputs
RUN mkdir -p /outputs

EXPOSE 7860

CMD ["python3", "app.py"]
