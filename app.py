import os
import tempfile

import gradio as gr
import torch
import whisperx
from gradio_client import utils as gradio_client_utils
from pydub import AudioSegment
from whisperx.diarize import DiarizationPipeline

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

LANGUAGE_OPTIONS = [
    ("Auto-detect", "auto"),
    ("English", "en"),
    ("Spanish", "es"),
    ("French", "fr"),
    ("German", "de"),
    ("Italian", "it"),
    ("Portuguese", "pt"),
    ("Dutch", "nl"),
    ("Russian", "ru"),
    ("Japanese", "ja"),
    ("Korean", "ko"),
    ("Chinese", "zh"),
]


def patch_gradio_schema_parser():
    """Handle boolean JSON Schema nodes produced by pydantic/gradio combinations.

    Some gradio_client versions assume every schema node is a dict, but valid JSON
    Schema allows boolean nodes (e.g. ``additionalProperties: false``). When those
    appear, Gradio's /info endpoint can fail with:
    ``TypeError: argument of type 'bool' is not iterable``.
    """

    original_get_type = gradio_client_utils.get_type
    original_json_schema_to_python_type = gradio_client_utils._json_schema_to_python_type

    def safe_get_type(schema):
        if isinstance(schema, bool):
            # ``True`` means unconstrained schema and ``False`` means disallowed.
            # For API docs generation we can safely treat both as an empty schema.
            return {}
        return original_get_type(schema)

    def safe_json_schema_to_python_type(schema, defs):
        if isinstance(schema, bool):
            return "Any"
        return original_json_schema_to_python_type(schema, defs)

    gradio_client_utils.get_type = safe_get_type
    gradio_client_utils._json_schema_to_python_type = safe_json_schema_to_python_type


def transcribe_files(files, model_size, language, use_diarization, hf_token, output_format):
    if not files:
        return None, "No files"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "float32"
    batch_size = 16 if device == "cuda" else 4
    model = whisperx.load_model(model_size, device, compute_type=compute_type)

    results = []
    for file in files:
        wav_path = None
        try:
            audio = AudioSegment.from_file(file)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                audio.export(tmp.name, format="wav")
                wav_path = tmp.name

            transcribe_kwargs = {"batch_size": batch_size}
            if language != "auto":
                transcribe_kwargs["language"] = language

            result = model.transcribe(wav_path, **transcribe_kwargs)
            model_a, metadata = whisperx.load_align_model(
                language_code=result["language"], device=device
            )
            result = whisperx.align(result["segments"], model_a, metadata, wav_path, device)

            if use_diarization and hf_token:
                diarize_model = DiarizationPipeline(use_auth_token=hf_token, device=device)
                diarize_segments = diarize_model(wav_path)
                result = whisperx.assign_word_speakers(diarize_segments, result)

            base = os.path.splitext(os.path.basename(file))[0]
            out_path = os.path.join(OUTPUT_DIR, f"{base}_transcription.{output_format.lower()}")

            with open(out_path, "w", encoding="utf-8") as output_file:
                if output_format == "TXT":
                    for seg in result["segments"]:
                        spk = seg.get("speaker", "Unknown")
                        output_file.write(
                            f"[{seg['start']:.2f}s - {seg['end']:.2f}s] {spk}: {seg['text'].strip()}\\n"
                        )
                else:
                    for i, seg in enumerate(result["segments"], 1):
                        spk = seg.get("speaker", "Unknown")
                        s = f"{int(seg['start']//3600):02d}:{int((seg['start']%3600)//60):02d}:{int(seg['start']%60):02d},{int(seg['start']%1*1000):03d}"
                        e = f"{int(seg['end']//3600):02d}:{int((seg['end']%3600)//60):02d}:{int(seg['end']%60):02d},{int(seg['end']%1*1000):03d}"
                        output_file.write(f"{i}\\n{s} --> {e}\\n{spk}: {seg['text'].strip()}\\n\\n")

            results.append(out_path)
        except Exception as error:
            failed_file = os.path.basename(file)
            results.append(f"{failed_file}: {error}")
        finally:
            if wav_path and os.path.exists(wav_path):
                os.unlink(wav_path)

    return results, f"Done on {device.upper()}"


with gr.Blocks(title="WhisperX") as demo:
    gr.Markdown("# WhisperX Transcription")
    files = gr.File(label="Files", file_count="multiple", type="filepath")
    with gr.Row():
        model_size = gr.Dropdown(["base", "small", "medium"], value="base", label="Model")
        language = gr.Dropdown(
            choices=LANGUAGE_OPTIONS,
            value="auto",
            label="Language",
            info="Use auto-detect for mixed/unknown input, or select to speed up transcription.",
        )
        output_format = gr.Radio(["TXT", "SRT"], value="TXT", label="Format")
    with gr.Row():
        use_diarization = gr.Checkbox(value=True, label="Diarization")
        hf_token = gr.Textbox(label="HF Token", type="password")
    btn = gr.Button("Transcribe", variant="primary")
    out = gr.Files(label="Downloads")
    status = gr.Textbox(label="Status")
    btn.click(
        transcribe_files,
        [files, model_size, language, use_diarization, hf_token, output_format],
        [out, status],
        queue=True,
    )


def launch_app():
    patch_gradio_schema_parser()

    share_enabled = (
        os.getenv("GRADIO_SHARE", "false").strip().lower() in {"1", "true", "yes"}
    )

    launch_kwargs = {
        "server_name": "0.0.0.0",
        "server_port": 7860,
        "share": share_enabled,
        # Keep API docs hidden in production UI.
        "show_api": False,
    }

    try:
        demo.queue(max_size=20).launch(**launch_kwargs)
    except ValueError as error:
        if "localhost is not accessible" not in str(error):
            raise

        print("Localhost check failed, retrying with GRADIO_SHARE enabled.")
        launch_kwargs["share"] = True
        demo.queue(max_size=20).launch(**launch_kwargs)


if __name__ == "__main__":
    launch_app()
