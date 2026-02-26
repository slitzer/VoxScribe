import os
import tempfile
import uuid
import logging
import inspect
from datetime import datetime

import gradio as gr
import torch
import whisperx
from gradio_client import utils as gradio_client_utils
from pydub import AudioSegment
from huggingface_hub.utils import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError
from whisperx.diarize import DiarizationPipeline

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

logger = logging.getLogger(__name__)
DEV_MODE = os.getenv("VOXSCRIBE_DEV_MODE", "false").strip().lower() in {"1", "true", "yes"}


def _log_exception(context, error):
    logger.error("%s failed: %s: %s", context, type(error).__name__, error)
    print(f"[transcribe_files] {context} failed: {type(error).__name__}: {error}")


def _create_diarization_pipeline(hf_token, device):
    """Build DiarizationPipeline across whisperx/pyannote API variants."""

    init_parameters = inspect.signature(DiarizationPipeline.__init__).parameters
    token_kwarg = "token" if "token" in init_parameters else "use_auth_token"
    return DiarizationPipeline(**{token_kwarg: hf_token}, device=device)

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


def transcribe_files(
    files,
    model_size,
    language,
    device_mode,
    use_diarization,
    hf_token,
    output_format,
):
    def build_status(header, queued_items, active_item=None, completed_items=None, failed_items=None):
        completed_items = completed_items or []
        failed_items = failed_items or []

        lines = [header, "", "Queue:"]
        lines.extend([f"- {item}" for item in queued_items])

        if active_item:
            lines.extend(["", f"Active: {active_item}"])

        if completed_items:
            lines.extend(["", "Completed:"])
            lines.extend([f"- {item}" for item in completed_items])

        if failed_items:
            lines.extend(["", "Failed:"])
            lines.extend([f"- {item}" for item in failed_items])

        return "\n".join(lines)

    if not files:
        history_choices, selected_history, transcript_preview = get_transcript_history_state()
        return (
            None,
            "No files",
            gr.update(choices=history_choices, value=selected_history),
            transcript_preview,
        )

    hf_token = (hf_token or "").strip()
    if use_diarization and not hf_token:
        history_choices, selected_history, transcript_preview = get_transcript_history_state()
        return (
            None,
            "Diarization needs a non-empty Hugging Face token with pyannote model access.",
            gr.update(choices=history_choices, value=selected_history),
            transcript_preview,
        )

    if use_diarization:
        try:
            # Lightweight preflight to fail fast on auth/access issues before loading
            # transcription/alignment models and processing all files.
            _create_diarization_pipeline(hf_token, device="cpu")
        except (GatedRepoError, RepositoryNotFoundError, HfHubHTTPError, PermissionError) as error:
            _log_exception("diarization preflight auth/access", error)
            history_choices, selected_history, transcript_preview = get_transcript_history_state()
            return (
                None,
                "Diarization unavailable: auth/access issue for required Hugging Face model.",
                gr.update(choices=history_choices, value=selected_history),
                transcript_preview,
            )
        except (RuntimeError, OSError, ValueError) as error:
            _log_exception("diarization preflight model load", error)
            history_choices, selected_history, transcript_preview = get_transcript_history_state()
            return (
                None,
                "Diarization unavailable: model load failure.",
                gr.update(choices=history_choices, value=selected_history),
                transcript_preview,
            )
        except Exception as error:
            _log_exception("diarization preflight unexpected", error)
            if DEV_MODE:
                raise
            history_choices, selected_history, transcript_preview = get_transcript_history_state()
            return (
                None,
                "Diarization unavailable: unexpected internal error.",
                gr.update(choices=history_choices, value=selected_history),
                transcript_preview,
            )

    device_mode = (device_mode or "CPU").upper()
    gpu_fallback = False
    if device_mode == "GPU":
        if not torch.cuda.is_available():
            device = "cpu"
            gpu_fallback = True
        else:
            device = "cuda"
    else:
        device = "cpu"
    compute_type = "float16" if device == "cuda" else "float32"
    batch_size = 16 if device == "cuda" else 4

    try:
        model = whisperx.load_model(model_size, device, compute_type=compute_type)
    except ValueError as error:
        # Some older/consumer GPUs expose CUDA but do not support efficient
        # float16 for faster-whisper. Fallback to float32 automatically.
        if device != "cuda" or "float16" not in str(error).lower():
            raise

        compute_type = "float32"
        batch_size = 4
        model = whisperx.load_model(model_size, device, compute_type=compute_type)

    results = []
    failures = []
    completed = []
    total_files = len(files)
    queued_items = [
        f"Queued {i}/{total_files}: {os.path.basename(path)}"
        for i, path in enumerate(files, start=1)
    ]

    yield (
        results,
        build_status(
            (
                "GPU unavailable; using CPU. "
                if gpu_fallback
                else ""
            )
            + f"Initialized on {device.upper()}. {total_files} file(s) added to queue.",
            queued_items,
        ),
        gr.update(),
        gr.update(),
    )

    diarize_model = None
    if use_diarization:
        try:
            diarize_model = _create_diarization_pipeline(hf_token, device=device)
        except (GatedRepoError, RepositoryNotFoundError, HfHubHTTPError, PermissionError) as error:
            _log_exception("diarization initialization auth/access", error)
            history_choices, selected_history, transcript_preview = get_transcript_history_state()
            return (
                None,
                "Diarization unavailable: auth/access issue for required Hugging Face model.",
                gr.update(choices=history_choices, value=selected_history),
                transcript_preview,
            )
        except (RuntimeError, OSError, ValueError) as error:
            _log_exception("diarization initialization model load", error)
            history_choices, selected_history, transcript_preview = get_transcript_history_state()
            return (
                None,
                "Diarization unavailable: model load failure.",
                gr.update(choices=history_choices, value=selected_history),
                transcript_preview,
            )
        except Exception as error:
            _log_exception("diarization initialization unexpected", error)
            if DEV_MODE:
                raise
            history_choices, selected_history, transcript_preview = get_transcript_history_state()
            return (
                None,
                "Diarization unavailable: unexpected internal error.",
                gr.update(choices=history_choices, value=selected_history),
                transcript_preview,
            )

    for idx, file in enumerate(files, start=1):
        wav_path = None
        file_name = os.path.basename(file)
        active_item = f"Processing {idx}/{total_files}: {file_name}"

        yield (
            results,
            build_status(
                ("GPU unavailable; using CPU. " if gpu_fallback else "")
                + f"Running on {device.upper()}.",
                queued_items,
                active_item=active_item,
                completed_items=completed,
                failed_items=failures,
            ),
            gr.update(),
            gr.update(),
        )

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

            if diarize_model:
                diarize_segments = diarize_model(wav_path)
                result = whisperx.assign_word_speakers(diarize_segments, result)

            base = os.path.splitext(os.path.basename(file))[0]
            extension = "txt" if output_format == "TXT" else "srt"
            suffix = f"{datetime.utcnow().strftime('%Y%m%dT%H%M%S%f')}_{uuid.uuid4().hex[:8]}"
            output_name = f"{base}_transcription_{suffix}.{extension}"
            out_path = os.path.join(OUTPUT_DIR, output_name)

            with open(out_path, "w", encoding="utf-8") as output_file:
                if output_format == "TXT":
                    for seg in result["segments"]:
                        spk = seg.get("speaker", "Unknown")
                        output_file.write(
                            f"[{seg['start']:.2f}s - {seg['end']:.2f}s] {spk}: {seg['text'].strip()}\n"
                        )
                else:
                    for i, seg in enumerate(result["segments"], 1):
                        spk = seg.get("speaker", "Unknown")
                        s = f"{int(seg['start']//3600):02d}:{int((seg['start']%3600)//60):02d}:{int(seg['start']%60):02d},{int(seg['start']%1*1000):03d}"
                        e = f"{int(seg['end']//3600):02d}:{int((seg['end']%3600)//60):02d}:{int(seg['end']%60):02d},{int(seg['end']%1*1000):03d}"
                        output_file.write(f"{i}\n{s} --> {e}\n{spk}: {seg['text'].strip()}\n\n")

            results.append(out_path)
            completed.append(f"{file_name} -> {output_name}")
        except Exception as error:
            _log_exception(f"per-file processing ({file_name})", error)
            failures.append(
                f"{file_name} (transcription failure): {type(error).__name__}: {error}"
            )
            if DEV_MODE and not isinstance(
                error,
                (
                    ValueError,
                    RuntimeError,
                    OSError,
                    FileNotFoundError,
                ),
            ):
                raise
        finally:
            if wav_path and os.path.exists(wav_path):
                os.unlink(wav_path)

        yield (
            results,
            build_status(
                f"Processed {idx}/{total_files} file(s) on {device.upper()}.",
                queued_items,
                completed_items=completed,
                failed_items=failures,
            ),
            gr.update(),
            gr.update(),
        )

    successful_names = [os.path.basename(path) for path in results]
    failed_names = [entry.split(" (transcription failure):", 1)[0] for entry in failures]
    summary_lines = [
        f"Done on {device.upper()}.",
        f"Success: {len(results)}/{total_files}",
        f"Failed: {len(failures)}/{total_files}",
        f"Successful files: {', '.join(successful_names) if successful_names else 'None'}",
        f"Failed files: {', '.join(failed_names) if failed_names else 'None'}",
    ]
    if failures:
        summary_lines.append("Failure details: " + " | ".join(failures))

    history_choices, selected_history, transcript_preview = get_transcript_history_state()
    return (
        results,
        "\n".join(summary_lines),
        gr.update(choices=history_choices, value=selected_history),
        transcript_preview,
    )


def list_transcript_files():
    transcript_files = []
    for name in os.listdir(OUTPUT_DIR):
        path = os.path.join(OUTPUT_DIR, name)
        if not os.path.isfile(path):
            continue
        extension = os.path.splitext(name)[1].lower()
        if extension not in {".txt", ".srt"}:
            continue
        transcript_files.append(path)

    transcript_files.sort(key=lambda path: os.path.getmtime(path), reverse=True)
    return transcript_files


def get_transcript_history_state(selected_path=None):
    transcript_paths = list_transcript_files()
    if not transcript_paths:
        return [], None, "No transcripts found in OUTPUT_DIR yet."

    dropdown_choices = []
    for path in transcript_paths:
        modified_at = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M:%S")
        dropdown_choices.append((f"{os.path.basename(path)} (updated {modified_at})", path))

    active_path = selected_path if selected_path in transcript_paths else transcript_paths[0]
    with open(active_path, "r", encoding="utf-8") as transcript_file:
        content = transcript_file.read()
    return dropdown_choices, active_path, content


def refresh_transcript_history(selected_path=None):
    choices, active_path, content = get_transcript_history_state(selected_path)
    return (
        gr.update(choices=choices, value=active_path),
        content,
    )


def load_selected_transcript(selected_path):
    if not selected_path:
        return ""

    resolved_output_dir = os.path.realpath(OUTPUT_DIR)
    resolved_selected = os.path.realpath(selected_path)
    if os.path.commonpath([resolved_output_dir, resolved_selected]) != resolved_output_dir:
        return "Invalid transcript selection. Click Refresh."

    extension = os.path.splitext(resolved_selected)[1].lower()
    if extension not in {".txt", ".srt"}:
        return "Invalid transcript selection. Click Refresh."

    if not os.path.exists(resolved_selected):
        return "Selected transcript no longer exists. Click Refresh."

    if resolved_selected not in {os.path.realpath(path) for path in list_transcript_files()}:
        return "Invalid transcript selection. Click Refresh."

    with open(resolved_selected, "r", encoding="utf-8") as transcript_file:
        return transcript_file.read()


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
        device_mode = gr.Radio(
            ["CPU", "GPU"],
            value="CPU",
            label="Compute Device",
            info="CPU is safest on older GPUs. Select GPU only if CUDA is supported by your card.",
        )
        output_format = gr.Radio(["TXT", "SRT"], value="TXT", label="Format")
    with gr.Row():
        use_diarization = gr.Checkbox(value=True, label="Diarization")
        hf_token = gr.Textbox(
            label="HF Token",
            type="password",
            info="Required for diarization and must have access to Hugging Face pyannote diarization models.",
        )
    btn = gr.Button("Transcribe", variant="primary")
    out = gr.Files(label="Downloads")
    status = gr.Textbox(label="Status")

    with gr.Row():
        transcript_history = gr.Dropdown(
            label="Transcript History",
            choices=[],
            value=None,
            info="TXT/SRT files from OUTPUT_DIR sorted by most recently modified.",
        )
        refresh_history = gr.Button("Refresh History")
    transcript_preview = gr.Code(label="Transcript Preview", value="", language="markdown", interactive=False)

    btn.click(
        transcribe_files,
        [files, model_size, language, device_mode, use_diarization, hf_token, output_format],
        [out, status, transcript_history, transcript_preview],
        queue=True,
    )

    refresh_history.click(
        refresh_transcript_history,
        [transcript_history],
        [transcript_history, transcript_preview],
    )

    transcript_history.change(
        load_selected_transcript,
        [transcript_history],
        [transcript_preview],
    )

    demo.load(
        refresh_transcript_history,
        [transcript_history],
        [transcript_history, transcript_preview],
    )


def launch_app():
    patch_gradio_schema_parser()

    # In containerized/proxied environments, Gradio's localhost healthcheck can
    # be routed through an HTTP proxy and fail even when the server is healthy.
    # Make sure localhost never goes through a proxy.
    no_proxy_values = [v.strip() for v in os.getenv("NO_PROXY", "").split(",") if v.strip()]
    for host in ("127.0.0.1", "localhost"):
        if host not in no_proxy_values:
            no_proxy_values.append(host)
    if no_proxy_values:
        os.environ["NO_PROXY"] = ",".join(no_proxy_values)
        os.environ["no_proxy"] = os.environ["NO_PROXY"]

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

    queue_max_size = int(os.getenv("GRADIO_QUEUE_MAX_SIZE", "20"))
    queue_concurrency = int(os.getenv("GRADIO_QUEUE_CONCURRENCY", "2"))
    queued_demo = demo.queue(
        max_size=queue_max_size,
        default_concurrency_limit=queue_concurrency,
    )

    try:
        queued_demo.launch(**launch_kwargs)
    except ValueError as error:
        if "localhost is not accessible" not in str(error):
            raise

        print("Localhost check failed, retrying with GRADIO_SHARE enabled.")
        launch_kwargs["share"] = True
        queued_demo.launch(**launch_kwargs)


if __name__ == "__main__":
    launch_app()
