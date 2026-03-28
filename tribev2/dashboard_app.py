from __future__ import annotations

import base64
from datetime import datetime
import hashlib
import html
import io
import json
import logging
import mimetypes
import os
from pathlib import Path
import pickle
import shutil
import textwrap
import traceback
import typing as tp
import uuid

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components

from tribev2.runtime import apply_warning_filters, configure_file_logging
from tribev2.eventstransforms import ExtractWordsFromAudio

LOGGER = logging.getLogger("tribev2.dashboard")


def _apply_warning_filters() -> None:
    apply_warning_filters()
    logging.getLogger("neuralset.extractors.base").setLevel(logging.ERROR)


_apply_warning_filters()

from tribev2.easy import (
    DEFAULT_TEXT_MODEL,
    EMOTION_AXES,
    EMOTION_LABELS,
    build_comparison_display_reference,
    ImageComparisonRun,
    MultiModalRun,
    PredictionRun,
    build_emotion_hypothesis_frame,
    build_browser_media_proxy,
    build_run_roi_frame,
    build_run_zone_frame,
    build_timestep_zone_frame,
    collect_timestep_metadata,
    export_prediction_video,
    load_model,
    predict_multimodal_from_prepared_events,
    predict_from_prepared_events,
    build_multimodal_events,
    prepare_events,
    render_animated_brain_3d_html,
    render_brain_panel_bytes,
    render_prediction_gif,
    render_prediction_mosaic,
    segment_preview,
    summarize_predictions,
)
from tribev2.openai_chat import (
    DEFAULT_OPENAI_CHAT_MODEL,
    build_openai_context_bundle,
    build_raw_timestep_frame,
    request_openai_run_explanation,
)


def configure_runtime_noise() -> None:
    _apply_warning_filters()


def configure_dashboard_logging(cache_folder: Path) -> Path:
    log_path = configure_file_logging(cache_folder / "logs" / "tribev2-dashboard.log")
    logging.getLogger("tribev2.openai_chat").setLevel(logging.INFO)
    LOGGER.info("Dashboard bootstrap complete | cache=%s", cache_folder)
    return log_path


def apply_theme() -> None:
    st.set_page_config(
        page_title="TRIBE v2",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
          :root {
            --bg: #f4f4f5;
            --ink: #09090b;
            --muted: #71717a;
            --line: #e4e4e7;
            --accent: #ea580c;
            --sidebar-bg: #18181b;
            --sidebar-fg: #fafafa;
            --panel: #ffffff;
          }
          html, body, [class*="css"] {
            font-family: "Inter", ui-sans-serif, system-ui, sans-serif;
            font-size: 14px;
          }
          h1, h2, h3, h4, h5, h6 {
            font-family: "Inter", ui-sans-serif, system-ui, sans-serif;
            letter-spacing: -0.02em;
            font-weight: 600;
          }
          .stApp {
            background: var(--bg);
            color: var(--ink);
          }
          .block-container {
            max-width: 1680px;
            padding-top: 0.5rem;
            padding-bottom: 0.75rem;
            padding-left: 1rem;
            padding-right: 1rem;
          }
          [data-testid="stSidebar"] {
            background: var(--sidebar-bg);
            border-right: 1px solid #27272a;
          }
          [data-testid="stSidebar"] h1,
          [data-testid="stSidebar"] h2,
          [data-testid="stSidebar"] h3,
          [data-testid="stSidebar"] label,
          [data-testid="stSidebar"] p,
          [data-testid="stSidebar"] span,
          [data-testid="stSidebar"] small,
          [data-testid="stSidebar"] code {
            color: var(--sidebar-fg);
          }
          [data-testid="stSidebar"] .stTextInput input,
          [data-testid="stSidebar"] .stTextArea textarea,
          [data-testid="stSidebar"] .stNumberInput input,
          [data-testid="stSidebar"] [data-baseweb="select"] > div {
            background: #fafafa;
            border: 1px solid #3f3f46;
            color: var(--ink);
            border-radius: 6px;
          }
          [data-testid="stSidebar"] .stTextInput input::placeholder,
          [data-testid="stSidebar"] .stTextArea textarea::placeholder,
          [data-testid="stSidebar"] .stNumberInput input::placeholder {
            color: #71717a;
            opacity: 1;
          }
          [data-testid="stSidebar"] [data-baseweb="select"] * {
            color: var(--ink);
          }
          [data-baseweb="popover"] [role="option"],
          [data-baseweb="popover"] [role="listbox"] *,
          [data-baseweb="popover"] [data-testid="stMarkdownContainer"] * {
            color: var(--ink);
          }
          [data-testid="stSidebar"] .stSlider [data-baseweb="slider"] {
            padding-top: 0.05rem;
          }
          .tribe-topbar {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            justify-content: space-between;
            gap: 0.5rem 1rem;
            padding: 0.45rem 0.65rem;
            margin-bottom: 0.55rem;
            border: 1px solid var(--line);
            border-radius: 8px;
            background: var(--panel);
          }
          .tribe-brand {
            font-size: 0.8125rem;
            font-weight: 700;
            letter-spacing: -0.03em;
            color: var(--ink);
          }
          .tribe-meta {
            font-size: 0.75rem;
            font-variant-numeric: tabular-nums;
            color: var(--muted);
            line-height: 1.35;
            text-align: right;
            max-width: min(100%, 72ch);
          }
          .tribe-meta b {
            color: var(--ink);
            font-weight: 600;
          }
          .tribe-kicker {
            font-size: 0.65rem;
            font-weight: 600;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: var(--muted);
            margin-bottom: 0.2rem;
          }
          .tribe-sectionhead {
            margin-bottom: 0.45rem;
          }
          .tribe-sectionhead h3 {
            margin: 0;
            font-size: 0.8125rem;
            font-weight: 600;
            line-height: 1.2;
            color: var(--ink);
          }
          .tribe-sectionhead p {
            margin: 0.12rem 0 0 0;
            color: var(--muted);
            font-size: 0.75rem;
            line-height: 1.35;
          }
          div[data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            box-shadow: none;
          }
          div[data-testid="stMetric"] {
            background: #fafafa;
            border: 1px solid var(--line);
            border-radius: 6px;
            padding: 0.25rem 0.45rem;
            min-height: 0;
          }
          div[data-testid="stMetricLabel"] p {
            font-size: 0.65rem;
            font-weight: 600;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            color: var(--muted);
          }
          .stTabs [data-baseweb="tab-list"] {
            gap: 0.2rem;
            padding-bottom: 0.1rem;
            min-height: 0;
          }
          .stTabs [data-baseweb="tab"] {
            height: 2rem;
            border-radius: 6px;
            padding: 0 0.65rem;
            font-size: 0.8125rem;
            background: transparent;
            border: 1px solid transparent;
          }
          .stTabs [aria-selected="true"] {
            background: var(--ink);
            color: #fafafa;
            border-color: var(--ink);
          }
          .stButton > button, .stDownloadButton > button {
            height: 2.25rem;
            border-radius: 6px;
            border: 1px solid var(--line);
            font-weight: 500;
            font-size: 0.8125rem;
          }
          .stButton > button[kind="primary"] {
            background: var(--accent);
            border-color: var(--accent);
            color: white;
            box-shadow: none;
          }
          .stButton > button:hover, .stDownloadButton > button:hover {
            border-color: #a1a1aa;
          }
          .stTextInput input, .stTextArea textarea, .stNumberInput input, [data-baseweb="select"] > div {
            background: #fafafa;
            border-radius: 6px;
            border: 1px solid var(--line);
          }
          div[data-testid="stFileUploader"] section {
            border-radius: 8px;
            border: 1px dashed #a1a1aa;
            background: #fafafa;
          }
          div[data-testid="stDataFrame"] {
            border-radius: 6px;
            overflow: hidden;
            border: 1px solid var(--line);
          }
          .tribe-runrow {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            justify-content: space-between;
            gap: 0.5rem;
            margin-top: 0.45rem;
            padding: 0.4rem 0.5rem;
            border-radius: 6px;
            border: 1px solid var(--line);
            background: #fafafa;
            font-size: 0.75rem;
            color: var(--muted);
          }
          .tribe-runrow strong {
            color: var(--ink);
            font-weight: 600;
          }
          .tribe-progress-caption {
            margin-top: 0.35rem;
            font-size: 0.7rem;
            color: var(--muted);
          }
          .tribe-busybutton {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 0.45rem;
            width: 100%;
            min-height: 2.25rem;
            padding: 0.5rem 0.75rem;
            border-radius: 6px;
            border: 1px solid var(--line);
            background: #f4f4f5;
            color: var(--muted);
            font-weight: 500;
            font-size: 0.8125rem;
            user-select: none;
            pointer-events: none;
          }
          .tribe-busyspinner {
            width: 0.85rem;
            height: 0.85rem;
            border-radius: 999px;
            border: 2px solid #e4e4e7;
            border-top-color: var(--ink);
            animation: tribe-spin 0.7s linear infinite;
          }
          @keyframes tribe-spin {
            to { transform: rotate(360deg); }
          }
          .tribe-sidebar-label {
            font-size: 0.65rem;
            font-weight: 600;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: #a1a1aa;
            margin: 0.35rem 0 0.15rem 0;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _build_status_meta(
    request: dict[str, object] | None,
    options: dict[str, object] | None,
    cache_folder: Path | None,
) -> str:
    request = request or {}
    options = options or {}
    ckpt = str(options.get("checkpoint", "facebook/tribev2")).split("/")[-1]
    cache_name = ""
    if cache_folder is not None:
        cache_name = Path(cache_folder).name or str(cache_folder)
    parts = [
        _format_request_label(request),
        str(options.get("device", "cuda")).upper(),
        ckpt,
        str(options.get("openai_model", DEFAULT_OPENAI_CHAT_MODEL)),
        f"WX {'on' if options.get('transcribe') else 'off'}",
        f"{float(options.get('image_duration', 4.0)):.1f}s×{int(options.get('image_fps', 6))}fps",
    ]
    if cache_name:
        parts.append(cache_name)
    inner = " · ".join(html.escape(p) for p in parts)
    return f'<div class="tribe-meta">{inner}</div>'


def hero(
    request: dict[str, object] | None = None,
    options: dict[str, object] | None = None,
    cache_folder: Path | None = None,
) -> None:
    st.markdown(
        f"""
        <div class="tribe-topbar">
          <div class="tribe-brand">TRIBE v2</div>
          {_build_status_meta(request, options, cache_folder)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_head(
    title: str,
    caption: str | None = None,
    *,
    kicker: str | None = None,
) -> None:
    kicker_html = f'<div class="tribe-kicker">{html.escape(kicker)}</div>' if kicker else ""
    cap_html = f"<p>{html.escape(caption)}</p>" if caption else ""
    st.markdown(
        f"""
        <div class="tribe-sectionhead">
          {kicker_html}
          <h3>{html.escape(title)}</h3>
          {cap_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _format_request_label(request: dict[str, object]) -> str:
    labels: list[str] = []
    video_paths = tp.cast(list[Path], request.get("video_paths") or [])
    audio_paths = tp.cast(list[Path], request.get("audio_paths") or [])
    texts = tp.cast(list[str], request.get("texts") or [])
    if request.get("video_path"):
        labels.append("Vidéo")
    if len(video_paths) == 2:
        labels.append("Vidéos x2")
    if request.get("audio_path"):
        labels.append("Audio")
    if len(audio_paths) == 2:
        labels.append("Audios x2")
    if request.get("text"):
        labels.append("Texte")
    if len(texts) == 2:
        labels.append("Textes x2")
    if request.get("image_paths"):
        count = len(tp.cast(list[Path], request["image_paths"]))
        labels.append("Image" if count == 1 else f"Images x{count}")
    return " + ".join(labels) if labels else "Aucune source"


def _validate_request_modalities(request: dict[str, object]) -> str | None:
    image_paths = tp.cast(list[Path], request.get("image_paths") or [])
    video_paths = tp.cast(list[Path], request.get("video_paths") or [])
    audio_paths = tp.cast(list[Path], request.get("audio_paths") or [])
    texts = tp.cast(list[str], request.get("texts") or [])
    has_video = bool(request.get("video_path"))
    has_audio = bool(request.get("audio_path"))
    has_text = bool(request.get("text"))
    comparison_modes = {
        "image": len(image_paths) == 2,
        "video": len(video_paths) == 2,
        "audio": len(audio_paths) == 2,
        "text": len(texts) == 2,
    }
    active_comparisons = [name for name, active in comparison_modes.items() if active]
    if len(active_comparisons) > 1:
        return "Choisissez un seul mode comparaison Ã  la fois."
    if active_comparisons:
        compare_kind = active_comparisons[0]
        extra_sources = {
            "video": has_video or bool(audio_paths) or bool(texts) or has_audio or has_text or bool(image_paths),
            "audio": has_audio or bool(video_paths) or bool(texts) or has_video or has_text or bool(image_paths),
            "text": has_text or bool(video_paths) or bool(audio_paths) or has_video or has_audio or bool(image_paths),
            "image": bool(video_paths) or bool(audio_paths) or bool(texts) or has_video or has_audio or has_text or len(image_paths) == 1,
        }
        if extra_sources.get(compare_kind, False):
            return "Le mode comparaison doit rester sur une seule modalité."
        return None
    if has_video and image_paths:
        return "Choisissez soit une vidéo, soit une image comme source visuelle."
    if len(image_paths) > 2:
        return "Chargez au maximum 2 images."
    if len(video_paths) > 2 or len(audio_paths) > 2 or len(texts) > 2:
        return "Chargez au maximum 2 éléments pour un mode comparaison."
    if len(image_paths) == 2 and (has_audio or has_text or has_video):
        return "Le mode 2 images reste séparé. Utilisez une seule image si vous voulez la combiner avec audio ou texte."
    return None


def _format_run_input_kind(run: PredictionRun | ImageComparisonRun) -> str:
    if isinstance(run, ImageComparisonRun):
        mapping = {
            "image": "Images",
            "video": "Videos",
            "audio": "Audios",
            "text": "Textes",
        }
        return f"{mapping.get(run.compare_kind, run.compare_kind.title())} x{len(run.runs)}"
    if isinstance(run, MultiModalRun):
        ordered = []
        for modality in ("video", "image", "audio", "text"):
            if modality in run.component_runs:
                ordered.append({"video": "Vidéo", "image": "Image", "audio": "Audio", "text": "Texte"}[modality])
        return " + ".join(ordered) if ordered else "Multimodal"
    return {"video": "Vidéo", "audio": "Audio", "text": "Texte", "image": "Image"}.get(
        run.input_kind,
        run.input_kind.title(),
    )


def render_action_progress(
    progress_host,
    steps: list[tuple[int, str]],
) -> tuple[object, tp.Callable[[int], None]]:
    bar = progress_host.progress(0, text=steps[0][1] if steps else "Preparation...")

    def update(step_index: int) -> None:
        index = max(0, min(step_index, len(steps) - 1))
        value, message = steps[index]
        bar.progress(value, text=message)

    return bar, update


def render_busy_prediction_button(label: str = "Inference en cours...") -> None:
    st.markdown(
        f"""
        <div class="tribe-busybutton" aria-disabled="true">
          <span class="tribe-busyspinner"></span>
          <span>{html.escape(label)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def clear_prediction_job_state() -> None:
    for key in (
        "prediction_running",
        "pending_prediction_request",
        "pending_prediction_options",
        "pending_prediction_cache_folder",
    ):
        st.session_state.pop(key, None)


@st.cache_resource(show_spinner=False)
def get_model(
    checkpoint: str,
    cache_folder: str,
    device: str,
    num_workers: int,
    text_model_name: str,
) -> object:
    return load_model(
        checkpoint=checkpoint,
        cache_folder=cache_folder,
        device=device,
        num_workers=num_workers,
        text_model_name=text_model_name,
    )


def save_upload(uploaded_file, folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    suffix = Path(uploaded_file.name).suffix
    safe_name = f"{uuid.uuid4().hex}{suffix}"
    path = folder / safe_name
    path.write_bytes(uploaded_file.getbuffer())
    return path


def build_npy_download(preds: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, preds)
    buffer.seek(0)
    return buffer.getvalue()


def fit_image_bytes_to_height(raw_bytes: bytes, max_height: int = 320) -> tuple[bytes, int]:
    image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    if image.height > max_height:
        scale = max_height / image.height
        image = image.resize((max(1, int(image.width * scale)), max_height), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue(), image.width


def fit_image_array_to_height(image_array: np.ndarray, max_height: int = 320) -> tuple[np.ndarray, int]:
    image = Image.fromarray(image_array)
    if image.height > max_height:
        scale = max_height / image.height
        image = image.resize((max(1, int(image.width * scale)), max_height), Image.Resampling.LANCZOS)
    return np.array(image), image.width


def render_bounded_image(
    image: bytes | np.ndarray,
    *,
    caption: str | None = None,
    max_height: int = 320,
) -> None:
    if isinstance(image, bytes):
        resized, width = fit_image_bytes_to_height(image, max_height=max_height)
        st.image(resized, caption=caption, width=width)
    else:
        resized, width = fit_image_array_to_height(image, max_height=max_height)
        st.image(resized, caption=caption, width=width)


def render_source_links(sources: list[tuple[str, str]] | tuple[tuple[str, str], ...]) -> None:
    links = [f"[{label}]({url})" for label, url in sources]
    st.caption("Sources: " + " | ".join(links))


def render_explainability_report(report: dict[str, object]) -> None:
    title = str(report.get("title", "Explication"))
    st.markdown(f"**{title}**")
    for section in report.get("sections", []):
        section_dict = tp.cast(dict[str, object], section)
        st.markdown(f"**{section_dict['title']}**")
        for bullet in tp.cast(list[str], section_dict["bullets"]):
            st.markdown(f"- {bullet}")
    render_source_links(tp.cast(list[tuple[str, str]], report.get("sources", [])))


def build_data_uri(raw_bytes: bytes, mime_type: str) -> str:
    encoded = base64.b64encode(raw_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _render_html_fragment(markup: str) -> None:
    html_renderer = getattr(st, "html", None)
    if callable(html_renderer):
        html_renderer(markup)
        return
    st.markdown(markup, unsafe_allow_html=True)


def build_base64_payload(raw_bytes: bytes) -> str:
    return base64.b64encode(raw_bytes).decode("ascii")


def guess_media_mime(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(path.name)
    if guessed:
        return guessed
    if path.suffix.lower() in {".mp4", ".mov", ".m4v"}:
        return "video/mp4"
    if path.suffix.lower() in {".webm"}:
        return "video/webm"
    if path.suffix.lower() in {".mp3"}:
        return "audio/mpeg"
    if path.suffix.lower() in {".wav"}:
        return "audio/wav"
    if path.suffix.lower() in {".ogg"}:
        return "audio/ogg"
    return "application/octet-stream"


def get_run_cache_key(run: PredictionRun) -> str:
    source_key = "none"
    if run.source_path and run.source_path.exists():
        stat = run.source_path.stat()
        source_key = f"{run.source_path}:{stat.st_size}:{stat.st_mtime_ns}"
    signal_key = f"{run.input_kind}:{run.preds.shape}:{float(np.abs(run.preds).mean()):.6f}:{float(np.abs(run.preds).max()):.6f}"
    return f"{source_key}:{signal_key}"


def get_dashboard_run_key(run: PredictionRun | ImageComparisonRun) -> str:
    if isinstance(run, ImageComparisonRun):
        return f"comparison:{run.compare_kind}:" + "|".join(get_run_cache_key(item) for item in run.runs)
    if isinstance(run, MultiModalRun):
        parts = [get_run_cache_key(run)]
        parts.extend(
            f"{modality}:{get_run_cache_key(component)}"
            for modality, component in sorted(run.component_runs.items())
        )
        return "multimodal:" + "|".join(parts)
    return get_run_cache_key(run)


def get_saved_runs_folder(cache_folder: Path) -> Path:
    return cache_folder / "saved_runs"


def get_saved_run_id(run: PredictionRun | ImageComparisonRun) -> str:
    run_key = get_dashboard_run_key(run)
    return hashlib.sha1(run_key.encode("utf-8")).hexdigest()[:16]


def _truncate_saved_text(value: str, *, limit: int = 88) -> str:
    compact = " ".join(value.split())
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 1)].rstrip() + "â€¦"


def _image_array_to_png_bytes(image_array: np.ndarray, *, max_height: int = 96) -> bytes:
    bounded_array, _ = fit_image_array_to_height(image_array, max_height=max_height)
    image = Image.fromarray(np.asarray(bounded_array, dtype=np.uint8))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _extract_visual_preview_bytes(run: PredictionRun | ImageComparisonRun) -> bytes | None:
    visual_run: PredictionRun | None = None
    if isinstance(run, ImageComparisonRun):
        visual_run = next((item for item in run.runs if item.input_kind in {"image", "video"}), None)
    elif isinstance(run, MultiModalRun):
        visual_run = run.component_runs.get("image") or run.component_runs.get("video")
    elif run.input_kind in {"image", "video"}:
        visual_run = run

    if visual_run is None:
        return None
    if visual_run.input_kind == "image" and visual_run.source_path and visual_run.source_path.exists():
        preview_bytes, _ = fit_image_bytes_to_height(visual_run.source_path.read_bytes(), max_height=96)
        return preview_bytes
    if visual_run.input_kind == "video":
        try:
            if visual_run.segments:
                preview = segment_preview(visual_run, 0)
                frame = preview.get("frame")
                if frame is not None:
                    return _image_array_to_png_bytes(np.asarray(frame, dtype=np.uint8), max_height=96)
        except Exception:
            LOGGER.exception("Failed to build saved run video preview | source=%s", visual_run.source_path)
    return None


def _saved_run_card_text(run: PredictionRun | ImageComparisonRun) -> tuple[str, str]:
    kind_label = _format_run_input_kind(run)
    if isinstance(run, ImageComparisonRun):
        if run.compare_kind == "text":
            texts = [_truncate_saved_text(item.raw_text or "") for item in run.runs[:2]]
            return kind_label, " / ".join(text for text in texts if text) or kind_label
        if run.compare_kind == "audio":
            names = [item.source_path.name for item in run.runs if item.source_path is not None]
            return kind_label, " / ".join(names) or kind_label
        if run.compare_kind in {"image", "video"}:
            names = [item.source_path.name for item in run.runs if item.source_path is not None]
            return kind_label, " / ".join(names) or kind_label
        return kind_label, kind_label
    if isinstance(run, MultiModalRun):
        if run.raw_text:
            return kind_label, _truncate_saved_text(run.raw_text)
        names = [path.name for _, path in sorted(run.source_paths.items())]
        return kind_label, " · ".join(names) or kind_label
    if run.input_kind == "text":
        return kind_label, _truncate_saved_text(run.raw_text or "")
    if run.input_kind == "audio" and run.source_path is not None:
        return kind_label, run.source_path.name
    if run.input_kind in {"video", "image"} and run.source_path is not None:
        return kind_label, run.source_path.name
    return kind_label, kind_label


def _serialize_prediction_run(run: PredictionRun) -> dict[str, tp.Any]:
    payload: dict[str, tp.Any] = {
        "kind": "prediction",
        "events": run.events,
        "preds": run.preds,
        "segments": run.segments,
        "input_kind": run.input_kind,
        "source_path": str(run.source_path) if run.source_path is not None else None,
        "raw_text": run.raw_text,
    }
    if isinstance(run, MultiModalRun):
        payload["kind"] = "multimodal"
        payload["component_runs"] = {
            modality: _serialize_prediction_run(component)
            for modality, component in run.component_runs.items()
        }
        payload["source_paths"] = {
            modality: str(path) for modality, path in run.source_paths.items()
        }
        payload["primary_input_kind"] = run.primary_input_kind
    return payload


def _serialize_saved_run(run: PredictionRun | ImageComparisonRun) -> dict[str, tp.Any]:
    if isinstance(run, ImageComparisonRun):
        return {
            "kind": "comparison",
            "compare_kind": run.compare_kind,
            "runs": [_serialize_prediction_run(item) for item in run.runs],
        }
    return _serialize_prediction_run(run)


def _deserialize_prediction_run(payload: dict[str, tp.Any]) -> PredictionRun:
    source_path = payload.get("source_path")
    common_kwargs = {
        "events": payload["events"],
        "preds": payload["preds"],
        "segments": payload.get("segments", []),
        "input_kind": str(payload["input_kind"]),
        "source_path": Path(source_path) if source_path else None,
        "raw_text": payload.get("raw_text"),
    }
    if payload.get("kind") == "multimodal":
        return MultiModalRun(
            **common_kwargs,
            component_runs={
                modality: _deserialize_prediction_run(component_payload)
                for modality, component_payload in tp.cast(
                    dict[str, dict[str, tp.Any]], payload.get("component_runs", {})
                ).items()
            },
            source_paths={
                modality: Path(path)
                for modality, path in tp.cast(
                    dict[str, str], payload.get("source_paths", {})
                ).items()
            },
            primary_input_kind=tp.cast(str | None, payload.get("primary_input_kind")),
        )
    return PredictionRun(**common_kwargs)


def _deserialize_saved_run(payload: dict[str, tp.Any]) -> PredictionRun | ImageComparisonRun:
    if payload.get("kind") == "comparison":
        return ImageComparisonRun(
            runs=[
                _deserialize_prediction_run(item)
                for item in tp.cast(list[dict[str, tp.Any]], payload.get("runs", []))
            ],
            compare_kind=str(payload.get("compare_kind", "image")),
        )
    return _deserialize_prediction_run(payload)


def persist_saved_run(cache_folder: Path, run: PredictionRun | ImageComparisonRun) -> str:
    saved_root = get_saved_runs_folder(cache_folder)
    saved_root.mkdir(parents=True, exist_ok=True)
    run_key = get_dashboard_run_key(run)
    run_id = get_saved_run_id(run)
    run_folder = saved_root / run_id
    run_folder.mkdir(parents=True, exist_ok=True)

    title, subtitle = _saved_run_card_text(run)
    preview_bytes = _extract_visual_preview_bytes(run)
    preview_file = None
    if preview_bytes is not None:
        preview_path = run_folder / "preview.png"
        preview_path.write_bytes(preview_bytes)
        preview_file = preview_path.name

    metadata_path = run_folder / "metadata.json"
    created_at = datetime.now().astimezone().isoformat(timespec="seconds")
    if metadata_path.exists():
        try:
            previous = json.loads(metadata_path.read_text(encoding="utf-8"))
            created_at = str(previous.get("created_at", created_at))
        except Exception:
            LOGGER.exception("Failed to read saved run metadata | path=%s", metadata_path)

    metadata = {
        "id": run_id,
        "run_key": run_key,
        "created_at": created_at,
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "kind_label": title,
        "subtitle": subtitle,
        "preview_file": preview_file,
        "input_kind": run.compare_kind if isinstance(run, ImageComparisonRun) else run.input_kind,
        "is_comparison": isinstance(run, ImageComparisonRun),
        "is_multimodal": isinstance(run, MultiModalRun),
        "timesteps": len(run.runs[0].preds) if isinstance(run, ImageComparisonRun) and run.runs else len(run.preds),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    payload = {
        "version": 2,
        "saved_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run": _serialize_saved_run(run),
    }
    with open(run_folder / "run.pkl", "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return run_id


def ensure_saved_run_folder(cache_folder: Path, run: PredictionRun | ImageComparisonRun) -> Path:
    run_id = persist_saved_run(cache_folder, run)
    return get_saved_runs_folder(cache_folder) / run_id


def get_saved_run_artifacts_folder(cache_folder: Path, run: PredictionRun | ImageComparisonRun) -> Path:
    folder = ensure_saved_run_folder(cache_folder, run) / "artifacts"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _artifact_bytes_path(
    cache_folder: Path,
    run: PredictionRun | ImageComparisonRun,
    *,
    filename: str,
) -> Path:
    return get_saved_run_artifacts_folder(cache_folder, run) / filename


def load_saved_run(cache_folder: Path, run_id: str) -> PredictionRun | ImageComparisonRun:
    run_path = get_saved_runs_folder(cache_folder) / run_id / "run.pkl"
    with open(run_path, "rb") as handle:
        payload = pickle.load(handle)
    if isinstance(payload, dict) and "run" in payload:
        return _deserialize_saved_run(tp.cast(dict[str, tp.Any], payload["run"]))
    return tp.cast(PredictionRun | ImageComparisonRun, payload)


def list_saved_runs(cache_folder: Path) -> list[dict[str, tp.Any]]:
    saved_root = get_saved_runs_folder(cache_folder)
    if not saved_root.exists():
        return []
    entries: list[dict[str, tp.Any]] = []
    for metadata_path in saved_root.glob("*/metadata.json"):
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            LOGGER.exception("Failed to parse saved run metadata | path=%s", metadata_path)
            continue
        payload["folder"] = str(metadata_path.parent)
        entries.append(payload)
    entries.sort(key=lambda item: str(item.get("updated_at", "")), reverse=True)
    return entries


def _saved_run_preview_path(entry: dict[str, tp.Any]) -> Path | None:
    preview_file = str(entry.get("preview_file") or "").strip()
    if not preview_file:
        return None
    return Path(str(entry["folder"])) / preview_file


def _build_saved_runs_gallery_html(
    entries: list[dict[str, tp.Any]],
    *,
    active_saved_id: str | None = None,
) -> str:
    cards: list[str] = []
    for entry in entries:
        preview_path = _saved_run_preview_path(entry)
        if preview_path is not None and preview_path.exists():
            preview_markup = (
                f'<img class="saved-run-thumb" src="{build_data_uri(preview_path.read_bytes(), "image/png")}" '
                f'alt="{html.escape(str(entry.get("kind_label", "Run")))}" />'
            )
        else:
            preview_markup = (
                '<div class="saved-run-placeholder">'
                f'{html.escape(str(entry.get("kind_label", "Run")))}'
                "</div>"
            )

        card_classes = "saved-run-card"
        if active_saved_id == entry.get("id"):
            card_classes += " is-active"
        meta_bits = [
            str(entry.get("updated_at", ""))[:16].replace("T", " "),
            f"{int(entry.get('timesteps', 0))} t",
        ]
        if active_saved_id == entry.get("id"):
            meta_bits.append("actif")
        cards.append(
            f"""
            <a class="{card_classes}" href="?saved_run={html.escape(str(entry['id']))}" target="_self">
              <div class="saved-run-visual">{preview_markup}</div>
              <div class="saved-run-kind">{html.escape(str(entry.get("kind_label", "Run")))}</div>
              <div class="saved-run-subtitle">{html.escape(str(entry.get("subtitle", "")))}</div>
              <div class="saved-run-meta">{html.escape(" · ".join(bit for bit in meta_bits if bit))}</div>
            </a>
            """
        )
    return textwrap.dedent(
        f"""
        <style>
          .saved-run-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(148px, 1fr));
            gap: 10px;
          }}
          .saved-run-card {{
            display: grid;
            gap: 6px;
            padding: 8px;
            border: 1px solid #e4e4e7;
            border-radius: 10px;
            background: #fafafa;
            color: #09090b;
            text-decoration: none;
            transition: border-color 140ms ease, transform 140ms ease, background 140ms ease;
          }}
          .saved-run-card:hover {{
            border-color: #a1a1aa;
            background: #ffffff;
            transform: translateY(-1px);
          }}
          .saved-run-card.is-active {{
            border-color: #ea580c;
            box-shadow: inset 0 0 0 1px rgba(234, 88, 12, 0.14);
          }}
          .saved-run-visual {{
            min-height: 72px;
          }}
          .saved-run-thumb {{
            width: 100%;
            height: 72px;
            object-fit: cover;
            border-radius: 8px;
            display: block;
            background: #18181b;
          }}
          .saved-run-placeholder {{
            height: 72px;
            border-radius: 8px;
            display: grid;
            place-items: center;
            background: linear-gradient(180deg, #f5f5f5 0%, #ededed 100%);
            color: #71717a;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 700;
          }}
          .saved-run-kind {{
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #71717a;
            font-weight: 700;
          }}
          .saved-run-subtitle {{
            font-size: 12px;
            font-weight: 600;
            line-height: 1.35;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            overflow: hidden;
            min-height: 2.8em;
          }}
          .saved-run-meta {{
            font-size: 11px;
            color: #71717a;
            line-height: 1.3;
          }}
        </style>
        <div class="saved-run-grid">
          {"".join(cards)}
        </div>
        """
    ).strip()


def sync_saved_run_from_query(cache_folder: Path) -> None:
    raw_value = st.query_params.get("saved_run", "")
    if isinstance(raw_value, list):
        saved_run_id = str(raw_value[0]) if raw_value else ""
    else:
        saved_run_id = str(raw_value)
    saved_run_id = saved_run_id.strip()
    if not saved_run_id or st.session_state.get("active_saved_run_id") == saved_run_id:
        return
    try:
        loaded_run = load_saved_run(cache_folder, saved_run_id)
    except Exception as exc:
        LOGGER.exception("Failed to load saved run from query | id=%s", saved_run_id)
        st.session_state["prediction_error"] = {
            "message": f"Impossible de charger le run sauvegardÃ©: {exc}",
            "traceback": traceback.format_exc(),
        }
        return
    st.session_state["prediction_run"] = loaded_run
    st.session_state["active_saved_run_id"] = saved_run_id
    st.session_state["prediction_notice"] = "Run chargé depuis la bibliothèque."


def render_saved_runs_panel(cache_folder: Path) -> None:
    entries = list_saved_runs(cache_folder)
    with st.container(border=True):
        section_head(
            "Bibliotheque",
            "Cliquez sur une carte pour rouvrir directement un resultat sauvegarde.",
            kicker="Runs",
        )
        if not entries:
            st.caption("Aucun run sauvegardé pour ce dossier cache.")
            return
        active_saved_id = st.session_state.get("active_saved_run_id")
        _render_html_fragment(
            _build_saved_runs_gallery_html(entries, active_saved_id=tp.cast(str | None, active_saved_id))
        )


def _figure_to_png_bytes(fig) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return buffer.getvalue()


def _artifact_variant_suffix(cache_tag: str | None) -> str:
    if not cache_tag:
        return ""
    return "_" + hashlib.sha1(cache_tag.encode("utf-8")).hexdigest()[:10]


def get_cached_prediction_gif(
    run: PredictionRun,
    *,
    cache_folder: Path,
    max_frames: int = 72,
    display_reference: np.ndarray | None = None,
    cache_tag: str | None = None,
) -> bytes:
    gif_cache = st.session_state.setdefault("brain_gifs", {})
    variant_suffix = _artifact_variant_suffix(cache_tag)
    cache_scope = cache_tag or "default"
    gif_key = f"{get_run_cache_key(run)}:gif:{max_frames}:{cache_scope}"
    if gif_key in gif_cache:
        return tp.cast(bytes, gif_cache[gif_key])
    artifact_path = _artifact_bytes_path(
        cache_folder,
        run,
        filename=f"brainplayback_{max_frames:03d}f{variant_suffix}.gif",
    )
    if artifact_path.exists():
        gif_bytes = artifact_path.read_bytes()
        gif_cache[gif_key] = gif_bytes
        return gif_bytes
    with st.spinner("Generation de l'animation cerebrale..."):
        gif_bytes = render_prediction_gif(
            run,
            max_frames=max_frames,
            display_reference=display_reference,
        )
    artifact_path.write_bytes(gif_bytes)
    gif_cache[gif_key] = gif_bytes
    return gif_bytes


def get_cached_prediction_mosaic_png(
    run: PredictionRun,
    *,
    cache_folder: Path,
    max_timesteps: int = 6,
    cache_tag: str | None = None,
) -> bytes | None:
    variant_suffix = _artifact_variant_suffix(cache_tag)
    artifact_path = _artifact_bytes_path(
        cache_folder,
        run,
        filename=f"mosaic_{max_timesteps:02d}t{variant_suffix}.png",
    )
    if artifact_path.exists():
        return artifact_path.read_bytes()
    return None


def ensure_cached_prediction_mosaic_png(
    run: PredictionRun,
    *,
    cache_folder: Path,
    max_timesteps: int = 6,
    display_reference: np.ndarray | None = None,
    cache_tag: str | None = None,
) -> bytes:
    mosaic_cache = st.session_state.setdefault("mosaic_png", {})
    variant_suffix = _artifact_variant_suffix(cache_tag)
    cache_scope = cache_tag or "default"
    cache_key = f"{get_run_cache_key(run)}:mosaic:{max_timesteps}:{cache_scope}"
    if cache_key in mosaic_cache:
        return tp.cast(bytes, mosaic_cache[cache_key])
    artifact_path = _artifact_bytes_path(
        cache_folder,
        run,
        filename=f"mosaic_{max_timesteps:02d}t{variant_suffix}.png",
    )
    if artifact_path.exists():
        png_bytes = artifact_path.read_bytes()
        mosaic_cache[cache_key] = png_bytes
        return png_bytes
    with st.spinner("Generation de la mosaique..."):
        fig = render_prediction_mosaic(
            run,
            max_timesteps=max_timesteps,
            display_reference=display_reference,
        )
        png_bytes = _figure_to_png_bytes(fig)
    artifact_path.write_bytes(png_bytes)
    mosaic_cache[cache_key] = png_bytes
    return png_bytes


def get_cached_animated_3d_html(
    run: PredictionRun,
    *,
    cache_folder: Path,
    max_frames: int = 30,
    height: int = 760,
    spinner_label: str = "Generation de la vue 3D animee...",
    display_reference: np.ndarray | None = None,
    cache_tag: str | None = None,
) -> str:
    html_cache = st.session_state.setdefault("animated_3d_html", {})
    render_version = "smoothv2"
    cache_scope = f"{cache_tag or 'default'}:{render_version}"
    variant_suffix = _artifact_variant_suffix(cache_scope)
    html_key = f"{get_run_cache_key(run)}:3d:{max_frames}:{height}:{cache_scope}"
    artifact_path = _artifact_bytes_path(
        cache_folder,
        run,
        filename=f"brain3d_{max_frames:03d}f_{height:04d}h{variant_suffix}.html",
    )
    if html_key not in html_cache:
        if artifact_path.exists():
            html_cache[html_key] = artifact_path.read_text(encoding="utf-8")
        else:
            LOGGER.info("Generating 3D animation HTML | key=%s | max_frames=%s", html_key, max_frames)
            with st.spinner(spinner_label):
                html_cache[html_key] = render_animated_brain_3d_html(
                    run,
                    max_frames=max_frames,
                    height=height,
                    display_reference=display_reference,
                )
            artifact_path.write_text(tp.cast(str, html_cache[html_key]), encoding="utf-8")
    return tp.cast(str, html_cache[html_key])


def render_media_preview(
    path: Path,
    *,
    kind: str,
    preview_folder: Path | None = None,
) -> None:
    preview_path = path
    try:
        output_folder = preview_folder or (path.parent / "_preview_media")
        preview_path = build_browser_media_proxy(
            source_path=path,
            output_folder=output_folder,
        )
    except Exception:
        LOGGER.exception("Failed to build media preview proxy | path=%s", path)
    payload = preview_path.read_bytes()
    if kind == "video":
        st.video(payload)
    else:
        st.audio(payload)


def render_raw_timestep_table(run: PredictionRun, *, height: int = 430) -> None:
    st.dataframe(build_raw_timestep_frame(run), width="stretch", height=height)


def get_cached_zone_bundle(run: PredictionRun) -> dict[str, pd.DataFrame]:
    cache = st.session_state.setdefault("zone_analysis_bundle", {})
    cache_key = get_run_cache_key(run)
    if cache_key not in cache:
        cache[cache_key] = {
            "emotion": build_emotion_hypothesis_frame(run),
            "run_zone": build_run_zone_frame(run),
            "run_roi": build_run_roi_frame(run),
            "zone_timeseries": build_timestep_zone_frame(run),
        }
    return tp.cast(dict[str, pd.DataFrame], cache[cache_key])


def render_emotion_radar(frame: pd.DataFrame, *, chart_key: str, height: int = 340) -> None:
    import plotly.graph_objects as go

    if frame.empty:
        st.caption("Radar indisponible.")
        return
    theta = frame["label"].tolist()
    radius = frame["score_pct"].astype(float).tolist()
    theta.append(theta[0])
    radius.append(radius[0])
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=radius,
            theta=theta,
            fill="toself",
            line={"color": "#ea580c", "width": 3},
            fillcolor="rgba(234, 88, 12, 0.18)",
            marker={"size": 6, "color": "#c2410c"},
            hovertemplate="%{theta}: %{r:.1f}<extra></extra>",
            name="Hypothese",
        )
    )
    fig.update_layout(
        margin={"l": 24, "r": 24, "t": 8, "b": 8},
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        polar={
            "bgcolor": "rgba(0,0,0,0)",
            "radialaxis": {
                "range": [0, 100],
                "tickvals": [25, 50, 75, 100],
                "tickfont": {"size": 10, "color": "#71717a"},
                "gridcolor": "rgba(113, 113, 122, 0.18)",
                "linecolor": "rgba(113, 113, 122, 0.18)",
            },
            "angularaxis": {
                "tickfont": {"size": 12, "color": "#111827"},
                "gridcolor": "rgba(113, 113, 122, 0.12)",
                "linecolor": "rgba(113, 113, 122, 0.12)",
            },
        },
        showlegend=False,
    )
    st.plotly_chart(
        fig,
        width="stretch",
        theme=None,
        key=chart_key,
        config={"displayModeBar": False},
    )


def render_zone_analysis_panel(run: PredictionRun, *, panel_key: str) -> None:
    bundle = get_cached_zone_bundle(run)
    emotion_frame = bundle["emotion"]
    run_zone_frame = bundle["run_zone"]
    run_roi_frame = bundle["run_roi"]
    zone_timeseries_frame = bundle["zone_timeseries"]
    top_zone_frame = run_zone_frame.head(6).copy()

    st.caption(
        "Atlas HCP-MMP sur fsaverage5. Les zones visibles ici sont corticales. "
        "Le radar emotionnel est une hypothese de stimulus, pas un decodeur clinique."
    )

    overview_left, overview_right = st.columns([1.0, 1.05], gap="large")
    with overview_left:
        render_emotion_radar(emotion_frame, chart_key=f"emotion_radar_{panel_key}")
        st.dataframe(
            emotion_frame[["label", "score_pct", "top_zone_drivers", "lexical_hits"]],
            width="stretch",
            height=220,
        )
    with overview_right:
        if not top_zone_frame.empty:
            st.bar_chart(
                top_zone_frame.set_index("zone")["share"],
                height=320,
                width="stretch",
            )
        st.dataframe(
            top_zone_frame[["zone", "share", "roi_count", "systems"]],
            width="stretch",
            height=220,
        )

    zone_tabs = st.tabs(["Zones", "ROI HCP", "Timeline zones"])
    with zone_tabs[0]:
        st.dataframe(
            run_zone_frame[["zone", "share", "value", "roi_count", "systems"]],
            width="stretch",
            height=320,
        )
    with zone_tabs[1]:
        st.dataframe(
            run_roi_frame[["roi", "zone", "share", "value", "signed_value", "systems"]],
            width="stretch",
            height=360,
        )
    with zone_tabs[2]:
        if not zone_timeseries_frame.empty:
            pivot = zone_timeseries_frame.pivot(
                index="timestep",
                columns="zone",
                values="share",
            ).fillna(0.0)
            pivot_values = pivot.to_numpy(dtype=float)
            if pivot.shape[0] > 0 and np.isfinite(pivot_values).all():
                st.line_chart(pivot, height=260, width="stretch")
        st.dataframe(
            zone_timeseries_frame[["timestep", "start_s", "duration_s", "zone", "share", "value", "systems"]],
            width="stretch",
            height=320,
        )


def build_workspace_zone_timeseries_frame(
    run: PredictionRun,
    *,
    bundle: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    zone_bundle = bundle or get_cached_zone_bundle(run)
    run_zone_frame = zone_bundle["run_zone"]
    zone_timeseries_frame = zone_bundle["zone_timeseries"]
    if run_zone_frame.empty or zone_timeseries_frame.empty:
        return pd.DataFrame()
    pivot = zone_timeseries_frame.pivot(
        index="timestep",
        columns="zone",
        values="share",
    ).fillna(0.0)
    if pivot.empty:
        return pd.DataFrame()
    ordered_columns = [zone for zone in run_zone_frame["zone"].tolist() if zone in pivot.columns]
    if ordered_columns:
        pivot = pivot.reindex(columns=ordered_columns)
    pivot_values = pivot.to_numpy(dtype=float)
    if pivot.shape[0] == 0 or not np.isfinite(pivot_values).all():
        return pd.DataFrame()
    return pivot


def get_cached_openai_context_bundle(
    run: PredictionRun | ImageComparisonRun,
    *,
    image_detail: str,
    max_images: int,
) -> tuple[str, list[dict[str, str]], list[str]]:
    cache = st.session_state.setdefault("openai_context_bundle", {})
    cache_key = f"{get_dashboard_run_key(run)}:{image_detail}:{max_images}"
    if cache_key not in cache:
        cache[cache_key] = build_openai_context_bundle(
            run,
            image_detail=image_detail,
            max_images=max_images,
        )
    return tp.cast(tuple[str, list[dict[str, str]], list[str]], cache[cache_key])


def render_openai_chat_panel(
    run: PredictionRun | ImageComparisonRun,
    *,
    model: str,
    api_key: str,
    reasoning_effort: str,
    image_detail: str,
    max_images: int,
) -> None:
    with st.container(border=True):
        section_head("Analyse GPT", kicker="Chat")
        if not api_key.strip():
            st.info("Clé OpenAI : sidebar ou `OPENAI_API_KEY`.")
            return

        run_key = get_dashboard_run_key(run)
        sessions = st.session_state.setdefault("openai_chat_sessions", {})
        session = sessions.setdefault(
            run_key,
            {
                "messages": [],
                "previous_response_id": None,
            },
        )
        st.caption(f"{model} · {int(max_images)} img")
        if st.button("Nouvelle conversation", width="stretch", key=f"chat_reset_{run_key}"):
            LOGGER.info("OpenAI chat reset | run=%s", run_key)
            sessions[run_key] = {"messages": [], "previous_response_id": None}
            st.rerun()

        context_bundle = get_cached_openai_context_bundle(
            run,
            image_detail=image_detail,
            max_images=max_images,
        )
        context_text, _, labels = context_bundle
        with st.expander("Contexte", expanded=False):
            st.markdown(
                "\n".join(f"- {label}" for label in labels)
                if labels
                else "Aucune image de timestep selectionnee."
            )
            st.code(context_text, language="json")

        for message in session["messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        prompt = st.chat_input("Messageâ€¦", key=f"chat_input_{run_key}")
        if not prompt:
            return

        session["messages"].append({"role": "user", "content": prompt})
        LOGGER.info(
            "OpenAI chat prompt | run=%s | model=%s | chars=%s",
            run_key,
            model,
            len(prompt),
        )
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                with st.spinner(f"{model} analyse le run..."):
                    reply, response_id, labels = request_openai_run_explanation(
                        api_key=api_key.strip(),
                        model=model.strip(),
                        reasoning_effort=reasoning_effort,
                        user_prompt=prompt,
                        run=run,
                        previous_response_id=session.get("previous_response_id"),
                        include_context=session.get("previous_response_id") is None,
                        image_detail=image_detail,
                        max_images=max_images,
                        context_bundle=context_bundle,
                    )
            except Exception as exc:
                LOGGER.exception("OpenAI chat request failed | run=%s | model=%s", run_key, model)
                session["messages"].append(
                    {
                        "role": "assistant",
                        "content": f"Echec de l'appel OpenAI: {exc}",
                    }
                )
                st.exception(exc)
                return
            if labels and session.get("previous_response_id") is None:
                st.caption(" · ".join(labels))
            st.markdown(reply)
            LOGGER.info(
                "OpenAI chat reply received | run=%s | model=%s | chars=%s",
                run_key,
                model,
                len(reply),
            )

        session["messages"].append({"role": "assistant", "content": reply})
        session["previous_response_id"] = response_id


def build_synced_player_html(run: PredictionRun, *, cache_folder: Path) -> str:
    if run.source_path is None or run.input_kind not in {"video", "audio"}:
        raise ValueError("Synced playback is only available for audio and video runs with a source file.")
    media_path = run.source_path
    try:
        media_path = build_browser_media_proxy(
            source_path=run.source_path,
            output_folder=cache_folder / "sync_media",
        )
    except Exception:
        LOGGER.exception("Failed to build browser media proxy | source=%s", run.source_path)
    media_bytes = media_path.read_bytes()
    media_mime = guess_media_mime(media_path)
    timeline = collect_timestep_metadata(run)
    frame_uris = [
        build_data_uri(
            render_brain_panel_bytes(run.preds, timestep=item["index"]),
            "image/jpeg",
        )
        for item in timeline
    ]
    emotion_labels = [EMOTION_LABELS[emotion] for emotion in EMOTION_AXES]
    emotion_frames: list[dict[str, tp.Any]] = []
    for item in timeline:
        emotion_frame = build_emotion_hypothesis_frame(run, timestep=item["index"])
        scores_by_emotion = {
            str(row.emotion): float(row.score_pct)
            for row in emotion_frame.itertuples(index=False)
        }
        top_emotion = emotion_frame.iloc[0] if not emotion_frame.empty else None
        emotion_frames.append(
            {
                "scores": [
                    round(scores_by_emotion.get(emotion, 0.0), 1)
                    for emotion in EMOTION_AXES
                ],
                "topLabel": str(top_emotion["label"]) if top_emotion is not None else "Indetermine",
                "topScore": float(top_emotion["score_pct"]) if top_emotion is not None else 0.0,
            }
        )
    payload = {
        "timesteps": [
            {
                "index": item["index"],
                "start": item["start"],
                "duration": item["duration"],
                "text": item["text"] or "",
                "brainUri": frame_uris[idx],
            }
            for idx, item in enumerate(timeline)
        ],
        "emotion": {
            "labels": emotion_labels,
            "frames": emotion_frames,
        },
        "media": {
            "mime": media_mime,
            "uri": build_data_uri(media_bytes, media_mime),
            "kind": run.input_kind,
            "name": media_path.name,
        },
    }
    payload_json = json.dumps(payload)
    brain_markup = """
        <div class="sync-brain-stack">
          <div class="sync-brain-wrap">
            <img id="tribe-brain" class="sync-brain" alt="TRIBE v2 predicted brain activity" />
            <div id="tribe-badge" class="sync-badge">Timestep 1</div>
          </div>
          <div class="sync-radar-card">
            <div class="sync-radar-head">
              <div class="sync-radar-title">Radar affectif</div>
              <div id="tribe-radar-top" class="sync-radar-top">Initialisation</div>
            </div>
            <svg id="tribe-radar" class="sync-radar" viewBox="0 0 260 220" aria-label="Radar emotionnel synchronise">
              <g id="tribe-radar-grid" class="sync-radar-grid"></g>
              <polygon id="tribe-radar-fill" class="sync-radar-fill" points=""></polygon>
              <polygon id="tribe-radar-stroke" class="sync-radar-stroke" points=""></polygon>
              <g id="tribe-radar-points"></g>
              <g id="tribe-radar-labels"></g>
            </svg>
          </div>
        </div>
    """
    if run.input_kind == "video":
        stage_class = "sync-stage sync-stage-video"
        media_markup = """
          <video id="tribe-media" class="sync-media sync-media-video" preload="metadata" playsinline controls></video>
        """
        stage_markup = media_markup + brain_markup
    else:
        stage_class = "sync-stage sync-stage-audio"
        media_markup = """
          <audio id="tribe-media" class="sync-media sync-media-audio" preload="metadata" controls></audio>
        """
        stage_markup = media_markup + brain_markup
    return f"""
    <div style="font-family: Inter, ui-sans-serif, system-ui, sans-serif; color: #09090b;">
      <style>
        .sync-card {{
          border: 1px solid #e4e4e7;
          border-radius: 8px;
          padding: 10px 12px;
          background: #fff;
        }}
        .sync-meta {{
          color: #71717a;
          font-size: 11px;
          margin-top: 6px;
        }}
        .sync-text {{
          margin-top: 8px;
          min-height: 44px;
          color: #09090b;
          line-height: 1.4;
          font-size: 12px;
          white-space: pre-wrap;
        }}
        .sync-stage {{
          margin-top: 6px;
        }}
        .sync-stage-video {{
          display: grid;
          grid-template-columns: minmax(240px, 0.72fr) minmax(0, 1fr);
          gap: 10px;
          align-items: start;
        }}
        .sync-stage-audio {{
          display: grid;
          gap: 10px;
        }}
        .sync-media {{
          width: 100%;
          display: block;
          border-radius: 6px;
          border: 1px solid #e4e4e7;
          background: #09090b;
        }}
        .sync-media-video {{
          aspect-ratio: 16 / 9;
          object-fit: cover;
        }}
        .sync-media-audio {{
          background: #fafafa;
        }}
        .sync-brain-wrap {{
          position: relative;
        }}
        .sync-brain-stack {{
          display: grid;
          gap: 10px;
        }}
        .sync-brain {{
          width: 100%;
          border-radius: 6px;
          background: #fafafa;
          border: 1px solid #e4e4e7;
          display: block;
        }}
        .sync-radar-card {{
          border: 1px solid #e4e4e7;
          border-radius: 8px;
          background:
            radial-gradient(circle at top, rgba(244,114,182,0.10), transparent 56%),
            linear-gradient(180deg, #0f172a 0%, #111827 100%);
          padding: 10px 10px 8px;
          color: #f8fafc;
        }}
        .sync-radar-head {{
          display: flex;
          justify-content: space-between;
          gap: 12px;
          align-items: baseline;
          margin-bottom: 6px;
        }}
        .sync-radar-title {{
          font-size: 11px;
          text-transform: uppercase;
          letter-spacing: 0.12em;
          color: rgba(248, 250, 252, 0.72);
          font-weight: 700;
        }}
        .sync-radar-top {{
          font-size: 12px;
          color: #fda4af;
          font-weight: 600;
          white-space: nowrap;
        }}
        .sync-radar {{
          width: 100%;
          display: block;
        }}
        .sync-radar-grid line {{
          stroke: rgba(255,255,255,0.16);
          stroke-width: 1;
        }}
        .sync-radar-grid polygon {{
          fill: none;
          stroke: rgba(255,255,255,0.12);
          stroke-width: 1;
        }}
        .sync-radar-fill {{
          fill: rgba(251, 113, 133, 0.28);
        }}
        .sync-radar-stroke {{
          fill: none;
          stroke: #fb7185;
          stroke-width: 2.2;
          stroke-linejoin: round;
        }}
        .sync-radar-point {{
          fill: #fde68a;
          stroke: #0f172a;
          stroke-width: 1.5;
        }}
        .sync-radar-label {{
          fill: rgba(248, 250, 252, 0.92);
          font-size: 10px;
          font-weight: 600;
        }}
        .sync-badge {{
          position: absolute;
          left: 8px;
          bottom: 8px;
          padding: 4px 8px;
          border-radius: 4px;
          background: #18181b;
          color: #fafafa;
          font-size: 11px;
          font-weight: 600;
        }}
        .sync-controls {{
          display: grid;
          grid-template-columns: auto auto minmax(140px, 1fr) auto;
          gap: 8px;
          align-items: center;
          margin-top: 8px;
        }}
        .sync-btn {{
          border: 1px solid #e4e4e7;
          border-radius: 6px;
          background: #fff;
          padding: 6px 12px;
          font-size: 12px;
          font-weight: 500;
          cursor: pointer;
        }}
        .sync-btn:hover {{
          border-color: #a1a1aa;
        }}
        .sync-slider {{
          width: 100%;
        }}
        .sync-time {{
          color: #71717a;
          font-size: 11px;
          font-variant-numeric: tabular-nums;
          white-space: nowrap;
        }}
        .sync-error {{
          margin-top: 8px;
          color: #dc2626;
          font-size: 12px;
        }}
        @media (max-width: 760px) {{
          .sync-stage-video {{
            grid-template-columns: 1fr;
          }}
          .sync-controls {{
            grid-template-columns: 1fr 1fr;
          }}
          .sync-slider {{
            grid-column: 1 / -1;
          }}
          .sync-time {{
            grid-column: 1 / -1;
          }}
        }}
      </style>
      <div class="sync-card">
        <div class="{stage_class}">
          {stage_markup}
        </div>
        <div id="tribe-meta" class="sync-meta"></div>
        <div id="tribe-text" class="sync-text"></div>
        <div class="sync-controls">
          <button id="tribe-play" class="sync-btn" type="button">Play</button>
          <button id="tribe-pause" class="sync-btn" type="button">Pause</button>
          <input id="tribe-scrub" class="sync-slider" type="range" min="0" max="1" step="0.01" value="0" />
          <div id="tribe-time" class="sync-time">0.00s / 0.00s</div>
        </div>
        <div class="sync-meta">{html.escape(media_path.name)} · {html.escape(run.input_kind)}</div>
        <div id="tribe-error" class="sync-error" style="display:none;"></div>
      </div>
      <script>
        const payload = {payload_json};
        const steps = payload.timesteps;
        const player = document.getElementById("tribe-media");
        const brain = document.getElementById("tribe-brain");
        const badge = document.getElementById("tribe-badge");
        const meta = document.getElementById("tribe-meta");
        const text = document.getElementById("tribe-text");
        const radarGrid = document.getElementById("tribe-radar-grid");
        const radarLabels = document.getElementById("tribe-radar-labels");
        const radarFill = document.getElementById("tribe-radar-fill");
        const radarStroke = document.getElementById("tribe-radar-stroke");
        const radarPoints = document.getElementById("tribe-radar-points");
        const radarTop = document.getElementById("tribe-radar-top");
        const playButton = document.getElementById("tribe-play");
        const pauseButton = document.getElementById("tribe-pause");
        const scrub = document.getElementById("tribe-scrub");
        const time = document.getElementById("tribe-time");
        const error = document.getElementById("tribe-error");
        const emotionLabels = (payload.emotion && payload.emotion.labels) ? payload.emotion.labels : [];
        const emotionFrames = (payload.emotion && payload.emotion.frames) ? payload.emotion.frames : [];
        let activeIndex = -1;
        const svgNs = "http://www.w3.org/2000/svg";
        const radarCenterX = 130;
        const radarCenterY = 108;
        const radarRadius = 68;

        function showError(message) {{
          error.style.display = "block";
          error.textContent = message;
        }}

        function hideError() {{
          error.style.display = "none";
          error.textContent = "";
        }}

        function formatTime(value) {{
          const safe = Number.isFinite(value) ? Number(value) : 0;
          return `${{safe.toFixed(2)}}s`;
        }}

        function polarPoint(axisIndex, axisCount, valueRatio) {{
          const angle = (-Math.PI / 2) + ((axisIndex / axisCount) * Math.PI * 2);
          const radius = radarRadius * valueRatio;
          return [
            radarCenterX + (Math.cos(angle) * radius),
            radarCenterY + (Math.sin(angle) * radius),
          ];
        }}

        function buildRadarPoints(scores) {{
          const axisCount = Math.max(emotionLabels.length, 1);
          return scores.map((score, idx) => {{
            const ratio = Math.max(0, Math.min(1, Number(score || 0) / 100));
            const [x, y] = polarPoint(idx, axisCount, ratio);
            return `${{x.toFixed(2)}},${{y.toFixed(2)}}`;
          }}).join(" ");
        }}

        function initRadar() {{
          if (!emotionLabels.length) {{
            radarTop.textContent = "Radar indisponible";
            return;
          }}
          const axisCount = emotionLabels.length;
          for (let level = 1; level <= 4; level += 1) {{
            const ratio = level / 4;
            const polygon = document.createElementNS(svgNs, "polygon");
            polygon.setAttribute("points", buildRadarPoints(new Array(axisCount).fill(ratio * 100)));
            radarGrid.appendChild(polygon);
          }}
          emotionLabels.forEach((label, idx) => {{
            const [lineX, lineY] = polarPoint(idx, axisCount, 1);
            const axis = document.createElementNS(svgNs, "line");
            axis.setAttribute("x1", String(radarCenterX));
            axis.setAttribute("y1", String(radarCenterY));
            axis.setAttribute("x2", String(lineX));
            axis.setAttribute("y2", String(lineY));
            radarGrid.appendChild(axis);

            const [labelX, labelY] = polarPoint(idx, axisCount, 1.18);
            const labelNode = document.createElementNS(svgNs, "text");
            labelNode.textContent = label;
            labelNode.setAttribute("x", String(labelX));
            labelNode.setAttribute("y", String(labelY));
            labelNode.setAttribute("class", "sync-radar-label");
            const anchor = Math.abs(labelX - radarCenterX) < 12 ? "middle" : (labelX < radarCenterX ? "end" : "start");
            labelNode.setAttribute("text-anchor", anchor);
            labelNode.setAttribute("dominant-baseline", labelY < radarCenterY ? "ideographic" : "hanging");
            radarLabels.appendChild(labelNode);
          }});
        }}

        function renderRadar(index) {{
          if (!emotionFrames.length || !emotionLabels.length) return;
          const frame = emotionFrames[Math.max(0, Math.min(index, emotionFrames.length - 1))];
          const scores = Array.isArray(frame.scores) ? frame.scores : new Array(emotionLabels.length).fill(0);
          const points = buildRadarPoints(scores);
          radarFill.setAttribute("points", points);
          radarStroke.setAttribute("points", points);
          radarPoints.innerHTML = "";
          scores.forEach((score, idx) => {{
            const ratio = Math.max(0, Math.min(1, Number(score || 0) / 100));
            const [x, y] = polarPoint(idx, emotionLabels.length, ratio);
            const point = document.createElementNS(svgNs, "circle");
            point.setAttribute("cx", String(x));
            point.setAttribute("cy", String(y));
            point.setAttribute("r", "3.3");
            point.setAttribute("class", "sync-radar-point");
            radarPoints.appendChild(point);
          }});
          radarTop.textContent = `${{frame.topLabel || "Indetermine"}} · ${{Number(frame.topScore || 0).toFixed(1)}}%`;
        }}

        function findStepIndex(currentTime) {{
          if (!steps.length) return 0;
          for (let i = 0; i < steps.length; i += 1) {{
            const start = Number(steps[i].start || 0);
            const end = start + Number(steps[i].duration || 1);
            if (currentTime >= start && currentTime < end) {{
              return i;
            }}
          }}
          return currentTime >= Number(steps[steps.length - 1].start || 0) ? steps.length - 1 : 0;
        }}

        function renderStep(index) {{
          if (!steps.length || index === activeIndex) return;
          const step = steps[index];
          activeIndex = index;
          brain.src = step.brainUri;
          const end = Number(step.start) + Number(step.duration);
          meta.textContent = `Timestep ${{index + 1}} / ${{steps.length}} | ${{Number(step.start).toFixed(2)}}s - ${{end.toFixed(2)}}s`;
          badge.textContent = `Timestep ${{index + 1}}`;
          text.textContent = step.text ? step.text : "â€”";
          renderRadar(index);
        }}

        function syncToPlayer() {{
          const currentTime = player.currentTime || 0;
          renderStep(findStepIndex(currentTime));
          if (Number.isFinite(player.duration) && player.duration > 0) {{
            scrub.max = String(player.duration);
            time.textContent = `${{formatTime(currentTime)}} / ${{formatTime(player.duration)}}`;
          }} else {{
            time.textContent = `${{formatTime(currentTime)}} / 0.00s`;
          }}
          if (document.activeElement !== scrub) {{
            scrub.value = String(currentTime);
          }}
        }}

        if (!steps.length) {{
          showError("Aucun timestep synchronisable n'a ete genere pour ce run.");
        }}

        player.addEventListener("loadedmetadata", () => {{
          hideError();
          scrub.max = String(Number.isFinite(player.duration) ? player.duration : 1);
          syncToPlayer();
        }});
        player.addEventListener("loadeddata", syncToPlayer);
        player.addEventListener("canplay", hideError);
        player.addEventListener("seeked", syncToPlayer);
        player.addEventListener("timeupdate", syncToPlayer);
        player.addEventListener("play", syncToPlayer);
        player.addEventListener("error", () => {{
          showError("Impossible de lire le media source dans le navigateur.");
        }});
        scrub.addEventListener("input", () => {{
          player.currentTime = Number(scrub.value || 0);
          syncToPlayer();
        }});
        playButton.addEventListener("click", async () => {{
          try {{
            await player.play();
          }} catch (err) {{
            showError("La lecture a ete refusee ou le media est incompatible.");
          }}
        }});
        pauseButton.addEventListener("click", () => player.pause());

        player.src = payload.media.uri;
        initRadar();
        player.load();
        renderStep(0);
        syncToPlayer();
      </script>
    </div>
    """


def get_cached_sync_player_html(run: PredictionRun, *, cache_folder: Path) -> str | None:
    cache = st.session_state.setdefault("sync_player_html", {})
    cache_key = f"{get_run_cache_key(run)}:sync-visible-media-v3"
    if cache_key in cache:
        return tp.cast(str, cache[cache_key])
    artifact_path = _artifact_bytes_path(
        cache_folder,
        run,
        filename="sync_visible_media_v3.html",
    )
    if artifact_path.exists():
        html_value = artifact_path.read_text(encoding="utf-8")
        cache[cache_key] = html_value
        return html_value
    return None


def ensure_cached_sync_player_html(run: PredictionRun, *, cache_folder: Path) -> str:
    existing = get_cached_sync_player_html(run, cache_folder=cache_folder)
    if existing is not None:
        return existing
    cache = st.session_state.setdefault("sync_player_html", {})
    cache_key = f"{get_run_cache_key(run)}:sync-visible-media-v3"
    html_value = build_synced_player_html(run, cache_folder=cache_folder)
    artifact_path = _artifact_bytes_path(
        cache_folder,
        run,
        filename="sync_visible_media_v3.html",
    )
    artifact_path.write_text(html_value, encoding="utf-8")
    cache[cache_key] = html_value
    return html_value


def get_cached_prediction_video_path(
    run: PredictionRun,
    *,
    cache_folder: Path,
    max_timesteps: int,
    interpolated_fps: int | None,
) -> Path | None:
    artifact_folder = _artifact_bytes_path(
        cache_folder,
        run,
        filename="mp4",
    )
    artifact_folder.mkdir(parents=True, exist_ok=True)
    fps_tag = interpolated_fps or 1
    expected = artifact_folder / f"tribev2_prediction_{min(len(run.preds), max_timesteps):03d}t_{fps_tag:02d}fps.mp4"
    if expected.exists():
        return expected
    return None


def ensure_cached_prediction_video_path(
    run: PredictionRun,
    *,
    cache_folder: Path,
    max_timesteps: int,
    interpolated_fps: int | None,
) -> Path:
    existing = get_cached_prediction_video_path(
        run,
        cache_folder=cache_folder,
        max_timesteps=max_timesteps,
        interpolated_fps=interpolated_fps,
    )
    if existing is not None:
        return existing
    output_folder = _artifact_bytes_path(
        cache_folder,
        run,
        filename="mp4",
    )
    output_folder.mkdir(parents=True, exist_ok=True)
    return export_prediction_video(
        run,
        output_folder=output_folder,
        max_timesteps=max_timesteps,
        interpolated_fps=interpolated_fps,
    )


def input_panel(cache_folder: Path) -> tuple[dict, dict, bool]:
    whisperx_available = ExtractWordsFromAudio.whisperx_available()
    with st.sidebar:
        st.markdown('<p class="tribe-sidebar-label">ModÃ¨le</p>', unsafe_allow_html=True)
        checkpoint = st.text_input("Checkpoint", value="facebook/tribev2")
        device = st.selectbox("Device", options=["auto", "cuda", "cpu"], index=1)
        num_workers = st.number_input(
            "DataLoader workers",
            min_value=0,
            max_value=32,
            value=0,
            step=1,
            help="0 est la valeur la plus sure pour Windows et Streamlit.",
        )
        text_model_name = st.text_input(
            "Modele texte",
            value=DEFAULT_TEXT_MODEL,
            help="Repo Hugging Face public ou chemin local au format transformers.",
        )
        text_pipeline_mode = st.radio(
            "Pipeline texte",
            options=["paper", "direct"],
            index=0 if whisperx_available else 1,
            format_func=lambda value: (
                "Paper-like (TTS + ASR)"
                if value == "paper"
                else "Rapide (texte direct)"
            ),
            help="`Paper-like` est plus proche du notebook officiel; `Rapide` garde les timings synthetiques du fork.",
        )
        if text_pipeline_mode == "paper" and not whisperx_available:
            st.caption("WhisperX indisponible dans cet env: retour automatique au texte direct.")
            text_pipeline_mode = "direct"
        direct_text = text_pipeline_mode == "direct"
        seconds_per_word = 0.45
        max_context_words = 128
        if direct_text:
            seconds_per_word = st.slider(
                "Duree synthetique par mot (texte direct)",
                min_value=0.1,
                max_value=1.0,
                value=0.45,
                step=0.05,
            )
            max_context_words = st.slider(
                "Mots conserves dans le contexte texte",
                min_value=16,
                max_value=256,
                value=128,
                step=16,
            )
        else:
            st.caption("Mode texte le plus proche du notebook officiel: synthese vocale puis retranscription pour retrouver des timings de mots realistes.")
        transcribe = st.checkbox(
            "Transcrire l'audio via whisperx",
            value=False,
            disabled=not whisperx_available,
            help="Utilise `uvx whisperx`, `uv tool run whisperx`, ou `python -m whisperx`.",
        )
        image_duration = st.slider(
            "Duree clip image",
            min_value=1.0,
            max_value=8.0,
            value=4.0,
            step=0.5,
            help="Chaque image est convertie en mini-video statique avant inference.",
        )
        image_fps = st.slider(
            "FPS clip image",
            min_value=1,
            max_value=12,
            value=6,
            step=1,
        )
        st.divider()
        st.markdown('<p class="tribe-sidebar-label">OpenAI</p>', unsafe_allow_html=True)
        openai_model = st.text_input("ModÃ¨le chat", value=DEFAULT_OPENAI_CHAT_MODEL)
        openai_reasoning = st.selectbox(
            "Effort de raisonnement",
            options=["low", "medium", "high", "xhigh"],
            index=1,
        )
        openai_image_detail = st.selectbox(
            "Detail des images envoyees",
            options=["low", "auto", "high"],
            index=0,
        )
        openai_max_images = st.slider(
            "Images de timesteps envoyees",
            min_value=1,
            max_value=6,
            value=4,
            step=1,
        )
        openai_api_key = st.text_input(
            "OPENAI_API_KEY",
            value="",
            type="password",
            placeholder="Env si vide",
        )

    options = {
        "checkpoint": checkpoint,
        "device": device,
        "num_workers": int(num_workers),
        "text_model_name": text_model_name,
        "text_pipeline_mode": text_pipeline_mode,
        "transcribe": transcribe,
        "direct_text": direct_text,
        "seconds_per_word": seconds_per_word,
        "max_context_words": max_context_words,
        "image_duration": image_duration,
        "image_fps": image_fps,
        "openai_model": openai_model,
        "openai_reasoning": openai_reasoning,
        "openai_image_detail": openai_image_detail,
        "openai_max_images": openai_max_images,
        "openai_api_key": openai_api_key,
    }

    request: dict[str, object] = {}
    launch_pressed = False
    with st.container(border=True):
        section_head("Source", kicker="Entrée")
        tabs = st.tabs(["Video", "Audio", "Texte", "Images"])
        with tabs[0]:
            video_files = st.file_uploader(
                "Importer 1 ou 2 videos",
                type=["mp4", "mov", "avi", "mkv", "webm"],
                accept_multiple_files=True,
                key="video_files",
            )
            if video_files:
                selected = video_files[:2]
                if len(video_files) > 2:
                    st.warning("Seules les 2 premieres videos seront utilisees.")
                video_paths = [save_upload(file, cache_folder / "uploads") for file in selected]
                if len(video_paths) == 1:
                    request["video_path"] = video_paths[0]
                else:
                    request["video_paths"] = video_paths
                cols = st.columns(len(selected))
                for col, path in zip(cols, video_paths):
                    with col:
                        render_media_preview(
                            Path(path),
                            kind="video",
                            preview_folder=cache_folder / "preview_media",
                        )
        with tabs[1]:
            audio_files = st.file_uploader(
                "Importer 1 ou 2 audios",
                type=["wav", "mp3", "flac", "ogg"],
                accept_multiple_files=True,
                key="audio_files",
            )
            if audio_files:
                selected = audio_files[:2]
                if len(audio_files) > 2:
                    st.warning("Seuls les 2 premiers audios seront utilises.")
                audio_paths = [save_upload(file, cache_folder / "uploads") for file in selected]
                if len(audio_paths) == 1:
                    request["audio_path"] = audio_paths[0]
                else:
                    request["audio_paths"] = audio_paths
                cols = st.columns(len(selected))
                for col, path in zip(cols, audio_paths):
                    with col:
                        render_media_preview(
                            Path(path),
                            kind="audio",
                            preview_folder=cache_folder / "preview_media",
                        )
        with tabs[2]:
            text_file_inputs = st.file_uploader(
                "Importer jusqu'Ã  2 fichiers .txt",
                type=["txt"],
                accept_multiple_files=True,
                key="text_files",
            )
            text_values: list[str] = []
            if text_file_inputs:
                selected = text_file_inputs[:2]
                if len(text_file_inputs) > 2:
                    st.warning("Seuls les 2 premiers fichiers texte seront utilises.")
                text_values = [file.getvalue().decode("utf-8") for file in selected]
                cols = st.columns(len(text_values))
                for col, (file, text_value) in zip(cols, zip(selected, text_values)):
                    with col:
                        st.text_area(f"AperÃ§u {file.name}", value=text_value, height=180, disabled=True)
            else:
                text_col_1, text_col_2 = st.columns(2, gap="medium")
                with text_col_1:
                    text_a = st.text_area("Texte 1", placeholder="â€¦", height=160, key="text_input_a")
                with text_col_2:
                    text_b = st.text_area("Texte 2", placeholder="â€¦", height=160, key="text_input_b")
                text_values = [value.strip() for value in (text_a, text_b) if value.strip()]
            if len(text_values) == 1:
                request["text"] = text_values[0]
            elif len(text_values) == 2:
                request["texts"] = text_values
        with tabs[3]:
            image_files = st.file_uploader(
                "Importer jusqu'Ã  2 images",
                type=["png", "jpg", "jpeg", "webp", "bmp"],
                accept_multiple_files=True,
            )
            if image_files:
                selected = image_files[:2]
                if len(image_files) > 2:
                    st.warning("Seules les 2 premieres images seront utilisees.")
                image_paths = [save_upload(file, cache_folder / "uploads") for file in selected]
                request["image_paths"] = image_paths
                cols = st.columns(len(selected))
                for col, file in zip(cols, selected):
                    with col:
                        render_bounded_image(file.getvalue(), caption=file.name, max_height=240)

        text_candidates = []
        if request.get("text"):
            text_candidates.append(str(request["text"]))
        text_candidates.extend(str(value) for value in tp.cast(list[str], request.get("texts") or []))
        word_counts = [len(text.split()) for text in text_candidates if text.strip()]
        if word_counts and direct_text and min(word_counts) < 8:
            st.warning(
                "Le texte direct sur des micro-prompts reste pratique, mais s'ecarte du protocole du papier et produit souvent des cartes tres proches. Pour vous rapprocher du notebook officiel, passez en `Paper-like (TTS + ASR)` et utilisez des phrases ou paragraphes plus longs."
            )
        elif word_counts and not direct_text:
            st.caption(
                "Pipeline `Paper-like` actif: les textes seront convertis en audio puis retranscrits avant prediction, ce qui se rapproche davantage du protocole officiel."
            )

        action_col, button_col = st.columns([1.35, 0.72], gap="medium")
        with action_col:
            st.markdown(
                f"""
                <div class="tribe-runrow">
                  <span>Active : <strong>{html.escape(_format_request_label(request))}</strong></span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with button_col:
            if st.session_state.get("prediction_running", False):
                render_busy_prediction_button()
            else:
                launch_pressed = st.button(
                    "Lancer",
                    type="primary",
                    width="stretch",
                    key="launch_prediction",
                )
    return request, options, launch_pressed


def run_prediction_ui(cache_folder: Path, request: dict, options: dict, launch_pressed: bool) -> None:
    notice = st.session_state.pop("prediction_notice", None)
    if notice:
        st.success(str(notice))

    error_state = st.session_state.pop("prediction_error", None)
    if error_state:
        st.error(str(error_state.get("message", "La prediction a echoue.")))
        trace = str(error_state.get("traceback", "")).strip()
        if trace:
            with st.expander("Traceback", expanded=False):
                st.code(trace)

    if launch_pressed:
        if not request:
            st.error("Importez d'abord une video, un audio, un texte ou une image.")
            return
        validation_error = _validate_request_modalities(request)
        if validation_error:
            st.error(validation_error)
            return
        st.session_state["prediction_running"] = True
        st.session_state["pending_prediction_request"] = dict(request)
        st.session_state["pending_prediction_options"] = dict(options)
        st.session_state["pending_prediction_cache_folder"] = str(cache_folder)
        LOGGER.info(
            "Prediction queued | source=%s | checkpoint=%s | device=%s | workers=%s",
            _format_request_label(request),
            options["checkpoint"],
            options["device"],
            options["num_workers"],
        )
        st.rerun()

    if not st.session_state.get("prediction_running", False):
        return

    request = tp.cast(dict[str, object], st.session_state.get("pending_prediction_request") or {})
    options = tp.cast(dict[str, object], st.session_state.get("pending_prediction_options") or {})
    pending_cache = st.session_state.get("pending_prediction_cache_folder")
    cache_folder = Path(str(pending_cache)) if pending_cache else cache_folder
    if not request or not options:
        clear_prediction_job_state()
        st.session_state["prediction_error"] = {
            "message": "Le run en attente est incomplet. Relancez la prediction.",
            "traceback": "",
        }
        st.rerun()

    progress_host = st.empty()
    progress_steps = [
        (6, "Validation du run..."),
        (18, "Chargement du modele..."),
        (36, "Preparation des evenements..."),
        (68, "Inference GPU en cours..."),
        (84, "Mise en cache des vues du dashboard..."),
        (100, "Run termine."),
    ]
    progress_bar, update_progress = render_action_progress(progress_host, progress_steps)
    LOGGER.info(
        "Prediction started | source=%s | checkpoint=%s | device=%s | workers=%s",
        _format_request_label(request),
        options["checkpoint"],
        options["device"],
        options["num_workers"],
    )
    try:
        update_progress(1)
        model = get_model(
            options["checkpoint"],
            str(cache_folder),
            options["device"],
            options["num_workers"],
            options["text_model_name"],
        )
        image_paths = [Path(p) for p in tp.cast(list[Path], request.get("image_paths") or [])]
        video_paths = [Path(p) for p in tp.cast(list[Path], request.get("video_paths") or [])]
        audio_paths = [Path(p) for p in tp.cast(list[Path], request.get("audio_paths") or [])]
        texts = [str(value) for value in tp.cast(list[str], request.get("texts") or [])]
        has_multimodal = sum(
            int(bool(value))
            for value in (
                request.get("video_path"),
                request.get("audio_path"),
                request.get("text"),
                image_paths,
            )
        ) > 1
        compare_kind = None
        compare_payloads: list[dict[str, object]] = []
        if len(image_paths) == 2:
            compare_kind = "image"
            compare_payloads = [{"image_path": path} for path in image_paths]
        elif len(video_paths) == 2:
            compare_kind = "video"
            compare_payloads = [{"video_path": path} for path in video_paths]
        elif len(audio_paths) == 2:
            compare_kind = "audio"
            compare_payloads = [{"audio_path": path} for path in audio_paths]
        elif len(texts) == 2:
            compare_kind = "text"
            compare_payloads = [{"text": text_value} for text_value in texts]

        if compare_kind is not None:
            compare_runs: list[PredictionRun] = []
            total_items = max(1, len(compare_payloads))
            for idx, payload in enumerate(compare_payloads, start=1):
                LOGGER.info(
                    "Preparing comparison run | kind=%s | index=%s/%s",
                    compare_kind,
                    idx,
                    total_items,
                )
                update_progress(2)
                events, input_kind = prepare_events(
                    cache_folder=cache_folder,
                    text=tp.cast(str | None, payload.get("text")),
                    audio_path=tp.cast(str | Path | None, payload.get("audio_path")),
                    video_path=tp.cast(str | Path | None, payload.get("video_path")),
                    image_path=tp.cast(str | Path | None, payload.get("image_path")),
                    transcribe=options["transcribe"],
                    direct_text=options["direct_text"],
                    seconds_per_word=options["seconds_per_word"],
                    max_context_words=options["max_context_words"],
                    image_duration=options["image_duration"],
                    image_fps=options["image_fps"],
                )
                progress_bar.progress(
                    min(76, 40 + int((idx - 1) / total_items * 26)),
                    text=f"Inference {compare_kind} {idx}/{total_items}...",
                )
                source_path = payload.get("video_path") or payload.get("audio_path") or payload.get("image_path")
                raw_text = payload.get("text")
                compare_runs.append(
                    predict_from_prepared_events(
                        model,
                        events,
                        input_kind=input_kind,
                        source_path=Path(source_path) if source_path else None,
                        raw_text=str(raw_text) if raw_text is not None else None,
                        verbose=False,
                    )
                )
            run = compare_runs[0] if len(compare_runs) == 1 else ImageComparisonRun(
                runs=compare_runs,
                compare_kind=str(compare_kind),
            )
        elif has_multimodal:
            update_progress(2)
            combined_events, prepared_components = build_multimodal_events(
                cache_folder=cache_folder,
                text=tp.cast(str | None, request.get("text")),
                audio_path=tp.cast(str | Path | None, request.get("audio_path")),
                video_path=tp.cast(str | Path | None, request.get("video_path")),
                image_path=image_paths[0] if image_paths else None,
                transcribe=options["transcribe"],
                direct_text=options["direct_text"],
                seconds_per_word=options["seconds_per_word"],
                max_context_words=options["max_context_words"],
                image_duration=options["image_duration"],
                image_fps=options["image_fps"],
            )
            update_progress(3)
            run = predict_multimodal_from_prepared_events(
                model,
                combined_events,
                prepared_components=prepared_components,
            )
        else:
            update_progress(2)
            events, input_kind = prepare_events(
                cache_folder=cache_folder,
                transcribe=options["transcribe"],
                direct_text=options["direct_text"],
                seconds_per_word=options["seconds_per_word"],
                max_context_words=options["max_context_words"],
                text=tp.cast(str | None, request.get("text")),
                audio_path=tp.cast(str | Path | None, request.get("audio_path")),
                video_path=tp.cast(str | Path | None, request.get("video_path")),
                image_path=image_paths[0] if image_paths else None,
            )
            update_progress(3)
            source_path = request.get("video_path") or request.get("audio_path") or (image_paths[0] if image_paths else None)
            raw_text = request.get("text")
            run = predict_from_prepared_events(
                model,
                events,
                input_kind=input_kind,
                source_path=Path(source_path) if source_path else None,
                raw_text=str(raw_text) if raw_text is not None else None,
            )
        update_progress(4)
        saved_run_id = persist_saved_run(cache_folder, run)
        st.session_state["prediction_run"] = run
        st.session_state["active_saved_run_id"] = saved_run_id
        st.session_state["mosaic_requested"] = False
        st.session_state["interactive_html_by_timestep"] = {}
        st.session_state["video_exports"] = {}
        update_progress(5)
        if isinstance(run, ImageComparisonRun):
            LOGGER.info(
                "Prediction completed | comparison=%s | compare_kind=%s | runs=%s",
                True,
                run.compare_kind,
                len(run.runs),
            )
        elif isinstance(run, MultiModalRun):
            LOGGER.info(
                "Prediction completed | input_kind=%s | modalities=%s | timesteps=%s | vertices=%s",
                run.input_kind,
                ",".join(sorted(run.component_runs)),
                len(run.preds),
                run.preds.shape[1],
            )
        else:
            LOGGER.info(
                "Prediction completed | input_kind=%s | timesteps=%s | vertices=%s",
                run.input_kind,
                len(run.preds),
                run.preds.shape[1],
            )
        clear_prediction_job_state()
        st.session_state["prediction_notice"] = "Prêt. Run sauvegardé."
        st.rerun()
    except Exception as exc:
        LOGGER.exception("Prediction failed | source=%s", _format_request_label(request))
        clear_prediction_job_state()
        st.session_state["prediction_error"] = {
            "message": f"La prediction a echoue: {exc}",
            "traceback": traceback.format_exc(),
        }
        st.rerun()


def render_input_preview(run: PredictionRun, *, cache_folder: Path | None = None) -> None:
    if isinstance(run, MultiModalRun):
        labels: list[str] = []
        renderers: list[tp.Callable[[], None]] = []

        if "video" in run.source_paths:
            video_path = run.source_paths["video"]
            labels.append("Vidéo")
            renderers.append(
                lambda path=video_path: render_media_preview(
                    path,
                    kind="video",
                    preview_folder=(cache_folder / "preview_media") if cache_folder else None,
                )
            )
        if "audio" in run.source_paths:
            audio_path = run.source_paths["audio"]
            labels.append("Audio")
            renderers.append(
                lambda path=audio_path: render_media_preview(
                    path,
                    kind="audio",
                    preview_folder=(cache_folder / "preview_media") if cache_folder else None,
                )
            )
        if run.raw_text:
            labels.append("Texte")
            renderers.append(lambda text=run.raw_text: st.text_area("Texte", value=text, height=180))
        if "image" in run.source_paths:
            image_path = run.source_paths["image"]
            labels.append("Image")
            renderers.append(
                lambda path=image_path: render_bounded_image(
                    path.read_bytes(),
                    caption=path.name,
                    max_height=260,
                )
            )
        if not labels:
            st.caption("â€”")
            return
        tabs = st.tabs(labels)
        for tab, renderer in zip(tabs, renderers):
            with tab:
                renderer()
        return
    if run.raw_text:
        st.text_area("Texte", value=run.raw_text, height=180)
    elif run.source_path and run.input_kind == "video":
        render_media_preview(
            run.source_path,
            kind="video",
            preview_folder=(cache_folder / "preview_media") if cache_folder else None,
        )
    elif run.source_path and run.input_kind == "audio":
        render_media_preview(
            run.source_path,
            kind="audio",
            preview_folder=(cache_folder / "preview_media") if cache_folder else None,
        )
    elif run.source_path and run.input_kind == "image":
        render_bounded_image(
            run.source_path.read_bytes(),
            caption=run.source_path.name,
            max_height=280,
        )
    else:
        st.caption("â€”")


def results_panel(cache_folder: Path, run: PredictionRun | ImageComparisonRun) -> None:
    if isinstance(run, ImageComparisonRun):
        return comparison_results_panel(run, cache_folder=cache_folder)

    run_key = get_dashboard_run_key(run)
    summary = summarize_predictions(run.preds)
    zone_bundle = get_cached_zone_bundle(run)
    workspace_zone_chart = build_workspace_zone_timeseries_frame(run, bundle=zone_bundle)
    summary_chart = summary.set_index("timestep")[["mean_abs", "std"]]
    chart_values = summary_chart.to_numpy(dtype=float) if not summary_chart.empty else np.empty((0, 2))
    has_valid_summary_chart = summary_chart.shape[0] > 0 and np.isfinite(chart_values).all()

    with st.container(border=True):
        section_head("Run", kicker="Stats")
        metric_cols = st.columns(5, gap="small")
        metric_cols[0].metric("Timesteps", len(run.preds))
        metric_cols[1].metric("Vertices", run.preds.shape[1])
        metric_cols[2].metric("Modalite", _format_run_input_kind(run))
        metric_cols[3].metric("Evenements", len(run.events))
        metric_cols[4].metric("Mean abs", f"{float(summary['mean_abs'].mean()):.4f}")
        if isinstance(run, MultiModalRun):
            st.caption("Overlay multimodal: rouge = visuel, vert = audio, bleu = texte.")

    workspace_col, inspect_col = st.columns([1.72, 1.0], gap="large")
    with workspace_col:
        with st.container(border=True):
            section_head("Signal + playback", kicker="Vue")
            chart_col, animation_col = st.columns([0.9, 1.15], gap="large")
            with chart_col:
                chart_tabs = st.tabs(["Zones", "Global"])
                with chart_tabs[0]:
                    if not workspace_zone_chart.empty:
                        st.line_chart(workspace_zone_chart, height=248, width="stretch")
                        st.caption("Part relative par zone corticale et par timestep.")
                    else:
                        st.caption("Courbes par zone indisponibles.")
                with chart_tabs[1]:
                    if has_valid_summary_chart:
                        st.line_chart(summary_chart, height=248, width="stretch")
                        st.caption("Signal global moyen et dispersion.")
                    else:
                        st.caption("Courbe globale indisponible.")
            with animation_col:
                caption = "GIF fusion RGB" if isinstance(run, MultiModalRun) else "GIF"
                gif_bytes = get_cached_prediction_gif(run, cache_folder=cache_folder)
                st.image(gif_bytes, caption=caption, width="stretch")

        with st.container(border=True):
            section_head("Vues", kicker="Exports")
            include_mp4 = not isinstance(run, MultiModalRun)
            tab_labels = ["3D"]
            if include_mp4:
                tab_labels.append("MP4")
            tab_labels.extend(["Mosaique", "Zones", "Evenements"])
            synced_media_available = run.input_kind in {"video", "audio"} and run.source_path is not None
            if synced_media_available:
                tab_labels = ["Sync"] + tab_labels
            tabs = st.tabs(tab_labels)
            tab_offset = 0

            if synced_media_available:
                with tabs[0]:
                    sync_html = get_cached_sync_player_html(run, cache_folder=cache_folder)
                    if st.button("Preparer Sync", width="stretch", key=f"prepare_sync_{run_key}"):
                        LOGGER.info("Preparing synced player | run=%s", run_key)
                        sync_progress_host = st.empty()
                        _, sync_progress = render_action_progress(
                            sync_progress_host,
                            [
                                (12, "Assemblage du media source..."),
                                (56, "Generation des surfaces synchronisees..."),
                                (100, "Lecteur synchronise pret."),
                            ],
                        )
                        sync_progress(1)
                        sync_html = ensure_cached_sync_player_html(run, cache_folder=cache_folder)
                        sync_progress(2)
                    if sync_html:
                        sync_height = 860 if run.input_kind == "video" else 760
                        components.html(sync_html, height=sync_height, scrolling=False)
                    tab_offset = 1

            with tabs[tab_offset]:
                html_value = get_cached_animated_3d_html(run, cache_folder=cache_folder)
                components.html(html_value, height=820, scrolling=False)
                st.download_button(
                    "HTML 3D",
                    data=html_value.encode("utf-8"),
                    file_name=f"tribev2_{run.input_kind}_brain_animation.html",
                    mime="text/html",
                    width="stretch",
                )

            next_tab_index = tab_offset + 1

            if include_mp4:
                with tabs[next_tab_index]:
                    export_cols = st.columns(2)
                    max_timesteps = export_cols[0].slider(
                        "Timesteps a inclure",
                        min_value=1,
                        max_value=len(run.preds),
                        value=min(15, len(run.preds)),
                    )
                    fps_options = {
                        "1 fps brut": None,
                        "6 fps": 6,
                        "12 fps": 12,
                        "24 fps": 24,
                    }
                    fps_label = export_cols[1].selectbox("Fluidite", options=list(fps_options), index=2)
                    fps_value = tp.cast(int | None, fps_options[fps_label])
                    video_path = get_cached_prediction_video_path(
                        run,
                        cache_folder=cache_folder,
                        max_timesteps=max_timesteps,
                        interpolated_fps=fps_value,
                    )
                    if video_path is None and st.button("Generer MP4", width="stretch", key=f"export_mp4_{run_key}"):
                        LOGGER.info(
                            "Generating MP4 | run=%s | timesteps=%s | fps=%s",
                            run_key,
                            max_timesteps,
                            fps_value,
                        )
                        mp4_progress_host = st.empty()
                        _, mp4_progress = render_action_progress(
                            mp4_progress_host,
                            [
                                (10, "Preparation des frames..."),
                                (52, "Encodage de la video..."),
                                (100, "MP4 pret."),
                            ],
                        )
                        mp4_progress(1)
                        video_path = ensure_cached_prediction_video_path(
                            run,
                            cache_folder=cache_folder,
                            max_timesteps=max_timesteps,
                            interpolated_fps=fps_value,
                        )
                        LOGGER.info("MP4 generated | run=%s | path=%s", run_key, video_path)
                        mp4_progress(2)
                    if video_path is not None:
                        video_bytes = video_path.read_bytes()
                        st.video(video_bytes)
                        st.download_button(
                            "MP4",
                            data=video_bytes,
                            file_name=video_path.name,
                            mime="video/mp4",
                            width="stretch",
                        )
                next_tab_index += 1

            with tabs[next_tab_index]:
                mosaic_timesteps = min(6, len(run.preds))
                mosaic_png = get_cached_prediction_mosaic_png(
                    run,
                    cache_folder=cache_folder,
                    max_timesteps=mosaic_timesteps,
                )
                if mosaic_png is None and st.button("Mosaique", width="stretch", key=f"mosaic_{run_key}"):
                    LOGGER.info("Generating mosaic | run=%s", run_key)
                    mosaic_progress_host = st.empty()
                    _, mosaic_progress = render_action_progress(
                        mosaic_progress_host,
                        [
                            (14, "Selection des timesteps..."),
                            (62, "Rendu des panneaux corticaux..."),
                            (100, "Mosaique prete."),
                        ],
                    )
                    mosaic_progress(1)
                    mosaic_png = ensure_cached_prediction_mosaic_png(
                        run,
                        cache_folder=cache_folder,
                        max_timesteps=mosaic_timesteps,
                    )
                    LOGGER.info("Mosaic generated | run=%s", run_key)
                    mosaic_progress(2)
                if mosaic_png is not None:
                    st.image(mosaic_png, width="stretch")
                    st.download_button(
                        "PNG",
                        data=mosaic_png,
                        file_name=f"tribev2_{run.input_kind}_mosaic.png",
                        mime="image/png",
                        width="stretch",
                    )

            with tabs[next_tab_index + 1]:
                render_zone_analysis_panel(run, panel_key=f"{run_key}_workspace")

            with tabs[next_tab_index + 2]:
                st.dataframe(run.events, width="stretch", height=320)

    with inspect_col:
        with st.container(border=True):
            section_head("Inspecteur", kicker="Source")
            preview_tab, data_tab, zone_tab = st.tabs(["Aperçu", "Timesteps", "Zones"])
            with preview_tab:
                render_input_preview(run, cache_folder=cache_folder)
            with data_tab:
                render_raw_timestep_table(run, height=420)
            with zone_tab:
                render_zone_analysis_panel(run, panel_key=f"{run_key}_inspect")

        with st.container(border=True):
            section_head("Fichiers", kicker="DL")
            export_cols = st.columns(5)
            export_cols[0].download_button(
                ".npy",
                data=build_npy_download(run.preds),
                file_name="tribev2_predictions.npy",
                mime="application/octet-stream",
                width="stretch",
            )
            export_cols[1].download_button(
                "events.csv",
                data=run.events.to_csv(index=False).encode("utf-8"),
                file_name="tribev2_events.csv",
                mime="text/csv",
                width="stretch",
            )
            export_cols[2].download_button(
                "résumé.csv",
                data=summary.to_csv(index=False).encode("utf-8"),
                file_name="tribev2_summary.csv",
                mime="text/csv",
                width="stretch",
            )
            export_cols[3].download_button(
                "zones.csv",
                data=zone_bundle["run_zone"].to_csv(index=False).encode("utf-8"),
                file_name="tribev2_zone_summary.csv",
                mime="text/csv",
                width="stretch",
            )
            export_cols[4].download_button(
                "roi.csv",
                data=zone_bundle["run_roi"].to_csv(index=False).encode("utf-8"),
                file_name="tribev2_hcp_roi_summary.csv",
                mime="text/csv",
                width="stretch",
            )


def comparison_results_panel(run: ImageComparisonRun, *, cache_folder: Path | None = None) -> None:
    common_timesteps = min(len(item.preds) for item in run.runs)
    display_reference = build_comparison_display_reference(run)
    comparison_key = hashlib.sha1(get_dashboard_run_key(run).encode("utf-8")).hexdigest()[:12]
    kind_labels = {
        "image": "images",
        "video": "videos",
        "audio": "audios",
        "text": "textes",
    }
    item_labels = {
        "image": "Image",
        "video": "Video",
        "audio": "Audio",
        "text": "Texte",
    }
    kind_label = kind_labels.get(run.compare_kind, run.compare_kind)
    item_label = item_labels.get(run.compare_kind, run.compare_kind.title())
    LOGGER.info(
        "Rendering comparison results | kind=%s | runs=%s | common_timesteps=%s",
        run.compare_kind,
        len(run.runs),
        common_timesteps,
    )
    with st.container(border=True):
        section_head(f"Comparaison {kind_label}", kicker="2x")
        metric_cols = st.columns(4)
        metric_cols[0].metric("Runs", len(run.runs))
        metric_cols[1].metric("Timesteps communs", common_timesteps)
        metric_cols[2].metric("Vertices", run.runs[0].preds.shape[1])
        metric_cols[3].metric("Modalite", _format_run_input_kind(run))
        st.caption(
            "Les vues cerveau de cette comparaison utilisent maintenant une normalisation commune partagee entre les deux runs."
        )

    cols = st.columns(len(run.runs), gap="large")
    for idx, (col, item) in enumerate(zip(cols, run.runs), start=1):
        with col:
            with st.container(border=True):
                section_head(f"{item_label} {idx}", kicker="Cmp")
                render_input_preview(item, cache_folder=cache_folder)
                if cache_folder is not None:
                    gif_bytes = get_cached_prediction_gif(
                        item,
                        cache_folder=cache_folder,
                        display_reference=display_reference,
                        cache_tag=f"{comparison_key}:{idx}:shared_norm",
                    )
                else:
                    with st.spinner(f"Generation de l'animation {run.compare_kind} {idx}..."):
                        gif_bytes = render_prediction_gif(
                            item,
                            display_reference=display_reference,
                        )
                st.image(gif_bytes, caption="GIF", width="stretch")
                if cache_folder is not None:
                    image_html = get_cached_animated_3d_html(
                        item,
                        cache_folder=cache_folder,
                        max_frames=18,
                        height=620,
                        spinner_label=f"Generation de la 3D animee {run.compare_kind} {idx}...",
                        display_reference=display_reference,
                        cache_tag=f"{comparison_key}:{idx}:shared_norm",
                    )
                else:
                    with st.spinner(f"Generation de la 3D animee {run.compare_kind} {idx}..."):
                        image_html = render_animated_brain_3d_html(
                            item,
                            max_frames=18,
                            height=620,
                            display_reference=display_reference,
                        )
                components.html(image_html, height=690, scrolling=False)
                st.download_button(
                    f"HTML 3D · {idx}",
                    data=image_html.encode("utf-8"),
                    file_name=f"tribev2_{run.compare_kind}_{idx}_brain_animation.html",
                    mime="text/html",
                    width="stretch",
                )
                with st.expander("Timesteps", expanded=False):
                    render_raw_timestep_table(item, height=260)
                with st.expander("Zones / emotions", expanded=False):
                    render_zone_analysis_panel(item, panel_key=f"comparison_{idx}_{get_run_cache_key(item)}")
                st.download_button(
                    f".npy · {idx}",
                    data=build_npy_download(item.preds),
                    file_name=f"tribev2_image_{idx}_predictions.npy",
                    mime="application/octet-stream",
                    width="stretch",
                )
def main() -> None:
    configure_runtime_noise()
    apply_theme()
    hero_slot = st.empty()
    library_slot = st.empty()
    cache_folder = Path(st.sidebar.text_input("Cache", value="./cache"))
    log_path = configure_dashboard_logging(cache_folder)
    with st.sidebar:
        st.caption(str(log_path))
    sync_saved_run_from_query(cache_folder)
    request, options, launch_pressed = input_panel(cache_folder)
    with hero_slot.container():
        hero(request, options, cache_folder)
    with library_slot.container():
        render_saved_runs_panel(cache_folder)
    run_prediction_ui(cache_folder, request, options, launch_pressed)

    run = st.session_state.get("prediction_run")
    if run is not None:
        results_col, chat_col = st.columns([2.65, 1.15], gap="large")
        with results_col:
            results_panel(cache_folder, run)
        with chat_col:
            render_openai_chat_panel(
                run,
                model=options["openai_model"],
                api_key=options["openai_api_key"] or os.getenv("OPENAI_API_KEY", ""),
                reasoning_effort=options["openai_reasoning"],
                image_detail=options["openai_image_detail"],
                max_images=int(options["openai_max_images"]),
            )


if __name__ == "__main__":
    main()


