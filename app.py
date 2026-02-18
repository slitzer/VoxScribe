import gradio as gr
import whisperx
import torch
import os
from pydub import AudioSegment
import tempfile
from whisperx.diarize import DiarizationPipeline

def transcribe_files(files, model_size, use_diarization, hf_token, output_format):
    if not files: return None, "No files"
    device = "cpu"
    compute_type = "float32"
    batch_size = 4
    model = whisperx.load_model(model_size, device, compute_type=compute_type)
    results = []
    for file in files:
        try:
            audio = AudioSegment.from_file(file)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                audio.export(tmp.name, format="wav")
                wav_path = tmp.name
            result = model.transcribe(wav_path, batch_size=4)
            model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
            result = whisperx.align(result["segments"], model_a, metadata, wav_path, device)
            if use_diarization and hf_token:
                diarize_model = DiarizationPipeline(use_auth_token=hf_token, device=device)
                diarize_segments = diarize_model(wav_path)
                result = whisperx.assign_word_speakers(diarize_segments, result)
            base = os.path.splitext(os.path.basename(file))[0]
            out_path = f"/tmp/{base}_transcription.{output_format.lower()}"
            with open(out_path, "w", encoding="utf-8") as f:
                if output_format == "TXT":
                    for seg in result["segments"]:
                        spk = seg.get("speaker", "Unknown")
                        f.write(f"[{seg['start']:.2f}s - {seg['end']:.2f}s] {spk}: {seg['text'].strip()}\n")
                else:
                    for i, seg in enumerate(result["segments"], 1):
                        spk = seg.get("speaker", "Unknown")
                        s = f"{int(seg['start']//3600):02d}:{int((seg['start']%3600)//60):02d}:{int(seg['start']%60):02d},{int(seg['start']%1*1000):03d}"
                        e = f"{int(seg['end']//3600):02d}:{int((seg['end']%3600)//60):02d}:{int(seg['end']%60):02d},{int(seg['end']%1*1000):03d}"
                        f.write(f"{i}\n{s} --> {e}\n{spk}: {seg['text'].strip()}\n\n")
            results.append(out_path)
            os.unlink(wav_path)
        except Exception as e:
            results.append(str(e))
    return results, f"Done on {device.upper()}"

with gr.Blocks(title="WhisperX") as demo:
    gr.Markdown("# WhisperX Transcription")
    files = gr.File(label="Files", file_count="multiple", type="filepath")
    with gr.Row():
        model_size = gr.Dropdown(["base","small","medium"], value="base", label="Model")
        output_format = gr.Radio(["TXT","SRT"], value="TXT", label="Format")
    with gr.Row():
        use_diarization = gr.Checkbox(value=True, label="Diarization")
        hf_token = gr.Textbox(label="HF Token", type="password")
    btn = gr.Button("Transcribe", variant="primary")
    out = gr.Files(label="Downloads")
    status = gr.Textbox(label="Status")
    btn.click(transcribe_files, [files, model_size, use_diarization, hf_token, output_format], [out, status], queue=True)

demo.queue(max_size=20).launch(server_name="0.0.0.0", server_port=7860)
