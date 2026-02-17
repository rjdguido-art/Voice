from __future__ import annotations

import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Generator, Optional, Tuple

import gradio as gr
import whisper

# When running as a bundled executable, keep app data near the .exe
# while loading bundled resources from PyInstaller's extraction dir.
if getattr(sys, "frozen", False):
    APP_DIR = Path(sys.executable).resolve().parent
    RESOURCE_DIR = Path(getattr(sys, "_MEIPASS", APP_DIR))
else:
    APP_DIR = Path(__file__).resolve().parent
    RESOURCE_DIR = APP_DIR

APP_TITLE = "Echo by Concentrix"
DEFAULT_LANGUAGE = "es"
DEFAULT_OUTPUT_DIR = APP_DIR / "outputs"
FFMPEG_EXE_NAME = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
FFMPEG_EXE = RESOURCE_DIR / "ffmpeg" / "bin" / FFMPEG_EXE_NAME
FFMPEG_BIN_DIR = FFMPEG_EXE.parent

MODEL_CACHE: Dict[Tuple[str, Optional[str]], whisper.Whisper] = {}
_DEVNULL_STREAM = None


def _ensure_standard_streams() -> None:
    """Provide stdout/stderr objects for windowed EXE runs where they may be None."""
    global _DEVNULL_STREAM
    if sys.stdout is None or sys.stderr is None:
        if _DEVNULL_STREAM is None:
            _DEVNULL_STREAM = open(os.devnull, "w", encoding="utf-8")
        if sys.stdout is None:
            sys.stdout = _DEVNULL_STREAM
        if sys.stderr is None:
            sys.stderr = _DEVNULL_STREAM


def _friendly_error(prefix: str, exc: Exception) -> str:
    message = str(exc).strip() or exc.__class__.__name__
    return f"{prefix}: {message}"


def _configure_ffmpeg_path() -> None:
    """Add portable ffmpeg folder to PATH when a bundled ffmpeg binary exists there."""
    if not FFMPEG_EXE.exists():
        return

    ffmpeg_path = str(FFMPEG_BIN_DIR)
    current_path = os.environ.get("PATH", "")
    parts = current_path.split(os.pathsep) if current_path else []
    normalized = {os.path.normcase(os.path.normpath(part)) for part in parts if part}
    normalized_ffmpeg_path = os.path.normcase(os.path.normpath(ffmpeg_path))
    if normalized_ffmpeg_path not in normalized:
        os.environ["PATH"] = ffmpeg_path + (os.pathsep + current_path if current_path else "")


def _is_ffmpeg_available() -> bool:
    return FFMPEG_EXE.exists() or shutil.which("ffmpeg") is not None


def _resolve_model_cache_dir(model_cache_dir: str) -> Optional[str]:
    trimmed = model_cache_dir.strip()
    if not trimmed:
        return None

    cache_path = Path(trimmed).expanduser()
    if not cache_path.is_absolute():
        cache_path = (APP_DIR / cache_path).resolve()
    cache_path.mkdir(parents=True, exist_ok=True)
    return str(cache_path)


def _resolve_output_dir(output_dir: str) -> Path:
    trimmed = output_dir.strip()
    target_output_dir = Path(trimmed).expanduser() if trimmed else DEFAULT_OUTPUT_DIR
    if not target_output_dir.is_absolute():
        target_output_dir = (APP_DIR / target_output_dir).resolve()
    target_output_dir.mkdir(parents=True, exist_ok=True)
    return target_output_dir


def _get_model(model_name: str, model_cache_dir: Optional[str]):
    cache_key = (model_name, model_cache_dir)
    if cache_key not in MODEL_CACHE:
        MODEL_CACHE[cache_key] = whisper.load_model(model_name, model_dir=model_cache_dir)
    return MODEL_CACHE[cache_key]


def _format_timestamp(seconds: float, always_include_hours: bool = False, decimal_marker: str = ".") -> str:
    milliseconds = max(0, int(round(seconds * 1000)))
    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000
    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000
    seconds_only = milliseconds // 1000
    milliseconds -= seconds_only * 1000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return f"{hours_marker}{minutes:02d}:{seconds_only:02d}{decimal_marker}{milliseconds:03d}"


def _to_srt(result: dict) -> str:
    lines = []
    for idx, segment in enumerate(result.get("segments", []), start=1):
        start = _format_timestamp(float(segment["start"]), always_include_hours=True, decimal_marker=",")
        end = _format_timestamp(float(segment["end"]), always_include_hours=True, decimal_marker=",")
        text = segment.get("text", "").strip()
        lines.append(f"{idx}\n{start} --> {end}\n{text}\n")
    return "\n".join(lines).strip()


def _to_vtt(result: dict) -> str:
    lines = ["WEBVTT", ""]
    for segment in result.get("segments", []):
        start = _format_timestamp(float(segment["start"]), always_include_hours=True)
        end = _format_timestamp(float(segment["end"]), always_include_hours=True)
        text = segment.get("text", "").strip()
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines).strip()


def _build_output_contents(result: dict) -> Dict[str, str]:
    return {
        "txt": result.get("text", "").strip(),
        "srt": _to_srt(result),
        "vtt": _to_vtt(result),
    }


def _save_outputs(
    target_output_dir: Path,
    source_stem: str,
    timestamp: str,
    output_format: str,
    contents_by_ext: Dict[str, str],
) -> list[Path]:
    extensions = ["txt", "srt", "vtt"] if output_format == "all" else [output_format]
    saved_files: list[Path] = []
    for ext in extensions:
        content = contents_by_ext.get(ext)
        if content is None:
            raise ValueError(f"Unsupported output format: {output_format}")
        output_file = target_output_dir / f"{source_stem}_{timestamp}.{ext}"
        output_file.write_text(content, encoding="utf-8")
        saved_files.append(output_file)
    return saved_files


def _create_transcript_textbox() -> gr.Textbox:
    # Some Gradio versions don't support show_copy_button.
    try:
        return gr.Textbox(label="Transcription text", lines=12, show_copy_button=True)
    except TypeError:
        return gr.Textbox(label="Transcription text", lines=12)


def predownload_model(model_name: str, model_cache_dir: str) -> Generator[str, None, None]:
    _configure_ffmpeg_path()
    resolved_model_dir = _resolve_model_cache_dir(model_cache_dir)

    status = [f"Pre-downloading model '{model_name}'..."]
    if resolved_model_dir:
        status.append(f"Using model cache directory: {resolved_model_dir}")
    else:
        status.append("Using default Whisper cache directory.")
    yield "\n".join(status)

    start_time = time.perf_counter()
    try:
        _get_model(model_name, resolved_model_dir)
    except Exception as exc:
        yield _friendly_error(f"Failed to pre-download model '{model_name}'", exc)
        return

    elapsed = time.perf_counter() - start_time
    status.append("Model is ready and cached.")
    status.append("You can now transcribe offline after this download completes.")
    status.append(f"Processing time: {elapsed:.2f}s")
    yield "\n".join(status)


def transcribe(
    audio_file: str,
    model_name: str,
    language: str,
    auto_detect_language: bool,
    output_format: str,
    output_dir: str,
    model_cache_dir: str,
    word_timestamps: bool,
) -> Generator[tuple[str, str], None, None]:
    if not audio_file:
        yield "", "Please upload an audio file first."
        return

    source_path = Path(audio_file)
    if not source_path.exists() or not source_path.is_file():
        yield "", f"Audio file not found: {audio_file}"
        return

    _configure_ffmpeg_path()
    if not _is_ffmpeg_available():
        ffmpeg_hint = RESOURCE_DIR / "ffmpeg" / "bin" / FFMPEG_EXE_NAME
        yield "", f"ffmpeg not found. Add ffmpeg to PATH or place it at: {ffmpeg_hint}"
        return

    try:
        target_output_dir = _resolve_output_dir(output_dir)
    except Exception as exc:
        yield "", _friendly_error("Could not prepare output directory", exc)
        return

    try:
        resolved_model_dir = _resolve_model_cache_dir(model_cache_dir)
    except Exception as exc:
        yield "", _friendly_error("Could not prepare model cache directory", exc)
        return

    selected_language: Optional[str] = None if auto_detect_language else language.strip() or DEFAULT_LANGUAGE

    status = [f"Loading model '{model_name}'..."]
    if resolved_model_dir:
        status.append(f"Using model cache directory: {resolved_model_dir}")
    yield "", "\n".join(status)

    try:
        model = _get_model(model_name, resolved_model_dir)
    except Exception as exc:
        yield "", _friendly_error(f"Failed to load model '{model_name}'", exc)
        return

    status.append("Transcribing audio...")
    status.append(f"Word timestamps: {'on' if word_timestamps else 'off'}")
    yield "", "\n".join(status)

    start_time = time.perf_counter()
    try:
        result = model.transcribe(audio_file, language=selected_language, word_timestamps=word_timestamps)
    except Exception as exc:
        yield "", _friendly_error("Transcription failed", exc)
        return

    transcript_text = result.get("text", "").strip() or "(No speech detected)"
    contents_by_ext = _build_output_contents(result)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        saved_files = _save_outputs(target_output_dir, source_path.stem, timestamp, output_format, contents_by_ext)
    except Exception as exc:
        yield transcript_text, _friendly_error("Transcription succeeded but saving output failed", exc)
        return

    elapsed = time.perf_counter() - start_time
    status.append("Saved file(s):")
    status.extend(f"- {path}" for path in saved_files)
    status.append(f"Processing time: {elapsed:.2f}s")
    yield transcript_text, "\n".join(status)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title=APP_TITLE) as demo:
        gr.Markdown(f"# {APP_TITLE}\nOffline local transcription with OpenAI Whisper.")

        audio_input = gr.Audio(label="Audio file", type="filepath")

        with gr.Row():
            model_input = gr.Dropdown(
                choices=["tiny", "base", "small", "medium", "large"],
                value="base",
                label="Model",
            )
            output_format_input = gr.Dropdown(
                choices=["txt", "srt", "vtt", "all"],
                value="txt",
                label="Output format",
            )

        with gr.Row():
            language_input = gr.Textbox(value=DEFAULT_LANGUAGE, label="Language")
            auto_detect_input = gr.Checkbox(value=False, label="Auto-detect language")
            word_timestamps_input = gr.Checkbox(value=False, label="Word timestamps")

        output_dir_input = gr.Textbox(value="./outputs", label="Output directory")
        model_cache_dir_input = gr.Textbox(value="", label="Model cache directory (optional)")

        with gr.Row():
            transcribe_btn = gr.Button("Transcribe", variant="primary")
            predownload_btn = gr.Button("Pre-download model")

        transcript_output = _create_transcript_textbox()
        status_output = gr.Textbox(label="Status", lines=10)

        transcribe_btn.click(
            fn=transcribe,
            inputs=[
                audio_input,
                model_input,
                language_input,
                auto_detect_input,
                output_format_input,
                output_dir_input,
                model_cache_dir_input,
                word_timestamps_input,
            ],
            outputs=[transcript_output, status_output],
        )

        predownload_btn.click(
            fn=predownload_model,
            inputs=[model_input, model_cache_dir_input],
            outputs=[status_output],
        )

    return demo


if __name__ == "__main__":
    _ensure_standard_streams()
    app = build_ui()
    app.queue().launch()
