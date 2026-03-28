from __future__ import annotations

import base64
import html
import io
import json
import logging
import mimetypes
import os
from pathlib import Path
import shutil
import traceback
import typing as tp
import uuid

import numpy as np
from PIL import Image
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
    ImageComparisonRun,
    MultiModalRun,
    PredictionRun,
    build_browser_media_proxy,
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
    if request.get("video_path"):
        labels.append("Vidéo")
    if request.get("audio_path"):
        labels.append("Audio")
    if request.get("text"):
        labels.append("Texte")
    if request.get("image_paths"):
        count = len(tp.cast(list[Path], request["image_paths"]))
        labels.append("Image" if count == 1 else f"Images x{count}")
    return " + ".join(labels) if labels else "Aucune source"


def _validate_request_modalities(request: dict[str, object]) -> str | None:
    image_paths = tp.cast(list[Path], request.get("image_paths") or [])
    has_video = bool(request.get("video_path"))
    has_audio = bool(request.get("audio_path"))
    has_text = bool(request.get("text"))
    if has_video and image_paths:
        return "Choisissez soit une vidéo, soit une image comme source visuelle."
    if len(image_paths) > 2:
        return "Chargez au maximum 2 images."
    if len(image_paths) == 2 and (has_audio or has_text or has_video):
        return "Le mode 2 images reste séparé. Utilisez une seule image si vous voulez la combiner avec audio ou texte."
    return None


def _format_run_input_kind(run: PredictionRun | ImageComparisonRun) -> str:
    if isinstance(run, ImageComparisonRun):
        return f"Images x{len(run.runs)}"
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
        return "comparison:" + "|".join(get_run_cache_key(item) for item in run.runs)
    if isinstance(run, MultiModalRun):
        parts = [get_run_cache_key(run)]
        parts.extend(
            f"{modality}:{get_run_cache_key(component)}"
            for modality, component in sorted(run.component_runs.items())
        )
        return "multimodal:" + "|".join(parts)
    return get_run_cache_key(run)


def get_cached_animated_3d_html(
    run: PredictionRun,
    *,
    max_frames: int = 30,
    height: int = 760,
    spinner_label: str = "Generation de la vue 3D animee...",
) -> str:
    html_cache = st.session_state.setdefault("animated_3d_html", {})
    html_key = f"{get_run_cache_key(run)}:3d:{max_frames}:{height}"
    if html_key not in html_cache:
        LOGGER.info("Generating 3D animation HTML | key=%s | max_frames=%s", html_key, max_frames)
        with st.spinner(spinner_label):
            html_cache[html_key] = render_animated_brain_3d_html(
                run,
                max_frames=max_frames,
                height=height,
            )
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

        prompt = st.chat_input("Message…", key=f"chat_input_{run_key}")
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
        "media": {
            "mime": media_mime,
            "uri": build_data_uri(media_bytes, media_mime),
            "kind": run.input_kind,
            "name": media_path.name,
        },
    }
    payload_json = json.dumps(payload)
    brain_markup = """
        <div class="sync-brain-wrap">
          <img id="tribe-brain" class="sync-brain" alt="TRIBE v2 predicted brain activity" />
          <div id="tribe-badge" class="sync-badge">Timestep 1</div>
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
        .sync-brain {{
          width: 100%;
          border-radius: 6px;
          background: #fafafa;
          border: 1px solid #e4e4e7;
          display: block;
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
        const playButton = document.getElementById("tribe-play");
        const pauseButton = document.getElementById("tribe-pause");
        const scrub = document.getElementById("tribe-scrub");
        const time = document.getElementById("tribe-time");
        const error = document.getElementById("tribe-error");
        let activeIndex = -1;

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
          text.textContent = step.text ? step.text : "—";
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
        player.load();
        renderStep(0);
        syncToPlayer();
      </script>
    </div>
    """


def input_panel(cache_folder: Path) -> tuple[dict, dict, bool]:
    whisperx_available = ExtractWordsFromAudio.whisperx_available()
    with st.sidebar:
        st.markdown('<p class="tribe-sidebar-label">Modèle</p>', unsafe_allow_html=True)
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
        direct_text = st.checkbox("Texte direct (sans TTS/ASR)", value=True)
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
        openai_model = st.text_input("Modèle chat", value=DEFAULT_OPENAI_CHAT_MODEL)
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
            video_file = st.file_uploader(
                "Importer une video", type=["mp4", "mov", "avi", "mkv", "webm"]
            )
            if video_file is not None:
                request["video_path"] = save_upload(video_file, cache_folder / "uploads")
                render_media_preview(
                    Path(tp.cast(str | Path, request["video_path"])),
                    kind="video",
                    preview_folder=cache_folder / "preview_media",
                )
        with tabs[1]:
            audio_file = st.file_uploader(
                "Importer un audio", type=["wav", "mp3", "flac", "ogg"]
            )
            if audio_file is not None:
                request["audio_path"] = save_upload(audio_file, cache_folder / "uploads")
                render_media_preview(
                    Path(tp.cast(str | Path, request["audio_path"])),
                    kind="audio",
                    preview_folder=cache_folder / "preview_media",
                )
        with tabs[2]:
            text_input = st.text_area("Texte", placeholder="…", height=140)
            text_file = st.file_uploader("ou importer un .txt", type=["txt"])
            if text_file is not None:
                text_value = text_file.getvalue().decode("utf-8")
                st.text_area("Apercu du fichier", value=text_value, height=160)
                request["text"] = text_value
            elif text_input.strip():
                request["text"] = text_input
        with tabs[3]:
            image_files = st.file_uploader(
                "Importer jusqu'a 2 images",
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
        has_multimodal = sum(
            int(bool(value))
            for value in (
                request.get("video_path"),
                request.get("audio_path"),
                request.get("text"),
                image_paths,
            )
        ) > 1
        if len(image_paths) == 2:
            image_runs: list[PredictionRun] = []
            total_images = max(1, len(image_paths))
            for idx, image_path in enumerate(image_paths, start=1):
                LOGGER.info("Preparing image run | index=%s/%s | path=%s", idx, total_images, image_path)
                update_progress(2)
                events, input_kind = prepare_events(
                    cache_folder=cache_folder,
                    image_path=image_path,
                    image_duration=options["image_duration"],
                    image_fps=options["image_fps"],
                )
                progress_bar.progress(
                    min(76, 40 + int((idx - 1) / total_images * 26)),
                    text=f"Inference image {idx}/{total_images}...",
                )
                image_runs.append(
                    predict_from_prepared_events(
                        model,
                        events,
                        input_kind=input_kind,
                        source_path=image_path,
                        verbose=False,
                    )
                )
            run = image_runs[0] if len(image_runs) == 1 else ImageComparisonRun(runs=image_runs)
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
        st.session_state["prediction_run"] = run
        st.session_state["mosaic_requested"] = False
        st.session_state["interactive_html_by_timestep"] = {}
        st.session_state["video_exports"] = {}
        update_progress(5)
        if isinstance(run, ImageComparisonRun):
            LOGGER.info(
                "Prediction completed | comparison=%s | runs=%s",
                True,
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
        st.session_state["prediction_notice"] = "Prêt."
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
            st.caption("—")
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
        st.caption("—")


def results_panel(cache_folder: Path, run: PredictionRun | ImageComparisonRun) -> None:
    if isinstance(run, ImageComparisonRun):
        return comparison_results_panel(run)

    run_key = get_dashboard_run_key(run)
    summary = summarize_predictions(run.preds)
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
                if has_valid_summary_chart:
                    st.line_chart(
                        summary_chart,
                        height=248,
                        width="stretch",
                    )
                else:
                    st.caption("Courbe indisponible.")
            animation_cache = st.session_state.setdefault("brain_gif_bytes", {})
            animation_key = get_dashboard_run_key(run)
            if animation_key not in animation_cache:
                with st.spinner("Generation de l'animation cerebrale..."):
                    animation_cache[animation_key] = render_prediction_gif(run)
            with animation_col:
                caption = "GIF fusion RGB" if isinstance(run, MultiModalRun) else "GIF"
                st.image(animation_cache[animation_key], caption=caption, width="stretch")

        with st.container(border=True):
            section_head("Vues", kicker="Exports")
            include_mp4 = not isinstance(run, MultiModalRun)
            tab_labels = ["3D", "MP4", "Mosaique", "Evenements"]
            synced_media_available = run.input_kind in {"video", "audio"} and run.source_path is not None
            if synced_media_available:
                tab_labels = ["Sync"] + tab_labels
            tabs = st.tabs(tab_labels)
            tab_offset = 0

            if synced_media_available:
                with tabs[0]:
                    sync_cache = st.session_state.setdefault("sync_player_html", {})
                    sync_key = f"{run_key}:sync-visible-media-v2"
                    if st.button("Préparer Sync", width="stretch", key=f"prepare_sync_{run_key}"):
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
                        sync_cache[sync_key] = build_synced_player_html(run, cache_folder=cache_folder)
                        sync_progress(2)
                    sync_html = sync_cache.get(sync_key)
                    if sync_html:
                        components.html(sync_html, height=640, scrolling=False)
                    tab_offset = 1

            with tabs[tab_offset]:
                html = get_cached_animated_3d_html(run)
                components.html(html, height=820, scrolling=False)
                st.download_button(
                    "HTML 3D",
                    data=html.encode("utf-8"),
                    file_name=f"tribev2_{run.input_kind}_brain_animation.html",
                    mime="text/html",
                    width="stretch",
                )

            with tabs[tab_offset + 1]:
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
                fps_label = export_cols[1].selectbox(
                    "Fluidite",
                    options=list(fps_options),
                    index=2,
                )
                export_key = f"{max_timesteps}:{fps_options[fps_label]}"
                video_cache = st.session_state.setdefault("video_exports", {})
                if st.button("Générer MP4", width="stretch", key=f"export_mp4_{run_key}"):
                    LOGGER.info(
                        "Generating MP4 | run=%s | timesteps=%s | fps=%s",
                        run_key,
                        max_timesteps,
                        fps_options[fps_label],
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
                    video_path = export_prediction_video(
                        run,
                        output_folder=cache_folder / "exports",
                        max_timesteps=max_timesteps,
                        interpolated_fps=fps_options[fps_label],
                    )
                    video_cache[export_key] = str(video_path)
                    LOGGER.info("MP4 generated | run=%s | path=%s", run_key, video_path)
                    mp4_progress(2)
                video_path_str = video_cache.get(export_key)
                if video_path_str:
                    video_path = Path(video_path_str)
                    st.video(video_path.read_bytes())
                    st.download_button(
                        "MP4",
                        data=video_path.read_bytes(),
                        file_name=video_path.name,
                        mime="video/mp4",
                        width="stretch",
                    )
            with tabs[tab_offset + 2]:
                mosaic_cache = st.session_state.setdefault("mosaic_figures", {})
                mosaic_key = f"mosaic:{run_key}"
                if st.button("Mosaïque", width="stretch", key=f"mosaic_{run_key}"):
                    st.session_state["mosaic_requested"] = True
                    if mosaic_key not in mosaic_cache:
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
                        mosaic_cache[mosaic_key] = render_prediction_mosaic(run)
                        LOGGER.info("Mosaic generated | run=%s", run_key)
                        mosaic_progress(2)
                if st.session_state.get("mosaic_requested") and mosaic_key in mosaic_cache:
                    st.pyplot(mosaic_cache[mosaic_key], clear_figure=False, width="stretch")

            with tabs[tab_offset + 3]:
                st.dataframe(run.events, width="stretch", height=320)

    with inspect_col:
        with st.container(border=True):
            section_head("Inspecteur", kicker="Source")
            preview_tab, data_tab = st.tabs(["Aperçu", "Timesteps"])
            with preview_tab:
                render_input_preview(run, cache_folder=cache_folder)
            with data_tab:
                render_raw_timestep_table(run, height=420)

        with st.container(border=True):
            section_head("Fichiers", kicker="DL")
            export_cols = st.columns(3)
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


def comparison_results_panel(run: ImageComparisonRun) -> None:
    common_timesteps = min(len(item.preds) for item in run.runs)
    LOGGER.info(
        "Rendering comparison results | images=%s | common_timesteps=%s",
        len(run.runs),
        common_timesteps,
    )
    with st.container(border=True):
        section_head("Comparaison images", kicker="2×")
        metric_cols = st.columns(4)
        metric_cols[0].metric("Images", len(run.runs))
        metric_cols[1].metric("Timesteps communs", common_timesteps)
        metric_cols[2].metric("Vertices", run.runs[0].preds.shape[1])
        metric_cols[3].metric("Modalite", "Images")

    cols = st.columns(len(run.runs), gap="large")
    for idx, (col, item) in enumerate(zip(cols, run.runs), start=1):
        with col:
            with st.container(border=True):
                section_head(f"Image {idx}", kicker="Img")
                render_input_preview(item)
                gif_cache = st.session_state.setdefault("brain_gif_bytes", {})
                gif_key = get_run_cache_key(item)
                if gif_key not in gif_cache:
                    with st.spinner(f"Generation de l'animation image {idx}..."):
                        gif_cache[gif_key] = render_prediction_gif(item)
                st.image(gif_cache[gif_key], caption="GIF", width="stretch")
                image_html = get_cached_animated_3d_html(
                    item,
                    max_frames=18,
                    height=620,
                    spinner_label=f"Generation de la 3D animee image {idx}...",
                )
                components.html(image_html, height=690, scrolling=False)
                st.download_button(
                    f"HTML 3D · {idx}",
                    data=image_html.encode("utf-8"),
                    file_name=f"tribev2_image_{idx}_brain_animation.html",
                    mime="text/html",
                    width="stretch",
                )
                with st.expander("Timesteps", expanded=False):
                    render_raw_timestep_table(item, height=260)
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
    cache_folder = Path(st.sidebar.text_input("Cache", value="./cache"))
    log_path = configure_dashboard_logging(cache_folder)
    with st.sidebar:
        st.caption(str(log_path))
    request, options, launch_pressed = input_panel(cache_folder)
    with hero_slot.container():
        hero(request, options, cache_folder)
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
