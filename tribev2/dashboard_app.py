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
    PredictionRun,
    collect_timestep_metadata,
    export_prediction_video,
    load_model,
    predict_from_prepared_events,
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
        page_title="TRIBE v2 Easy",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
          @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Space+Grotesk:wght@500;700&display=swap');
          :root {
            --bg: #edf1f5;
            --bg-deep: #dfe5eb;
            --panel: rgba(255, 255, 255, 0.78);
            --panel-solid: #ffffff;
            --ink: #0f1722;
            --muted: #617082;
            --line: rgba(15, 23, 34, 0.10);
            --accent: #ff6a1f;
            --accent-strong: #e55610;
            --accent-soft: rgba(255, 106, 31, 0.12);
            --navy: #0e1724;
            --navy-soft: #182335;
          }
          html, body, [class*="css"] {
            font-family: "IBM Plex Sans", sans-serif;
          }
          h1, h2, h3, h4, h5, h6 {
            font-family: "Space Grotesk", sans-serif;
            letter-spacing: -0.03em;
          }
          .stApp {
            background:
              radial-gradient(circle at top right, rgba(255, 106, 31, 0.11), transparent 22%),
              linear-gradient(180deg, var(--bg) 0%, var(--bg-deep) 100%);
            color: var(--ink);
          }
          .block-container {
            max-width: 1700px;
            padding-top: 0.9rem;
            padding-bottom: 1.1rem;
            padding-left: 1.2rem;
            padding-right: 1.2rem;
          }
          [data-testid="stSidebar"] {
            background:
              linear-gradient(180deg, var(--navy) 0%, #121d2b 55%, var(--navy-soft) 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.06);
          }
          [data-testid="stSidebar"] h1,
          [data-testid="stSidebar"] h2,
          [data-testid="stSidebar"] h3,
          [data-testid="stSidebar"] label,
          [data-testid="stSidebar"] p,
          [data-testid="stSidebar"] span,
          [data-testid="stSidebar"] small,
          [data-testid="stSidebar"] code {
            color: #eef3f8;
          }
          [data-testid="stSidebar"] .stTextInput input,
          [data-testid="stSidebar"] .stTextArea textarea,
          [data-testid="stSidebar"] .stNumberInput input,
          [data-testid="stSidebar"] [data-baseweb="select"] > div {
            background: rgba(246, 248, 251, 0.96);
            border: 1px solid rgba(255, 255, 255, 0.14);
            color: #0f1722;
          }
          [data-testid="stSidebar"] .stTextInput input::placeholder,
          [data-testid="stSidebar"] .stTextArea textarea::placeholder,
          [data-testid="stSidebar"] .stNumberInput input::placeholder {
            color: #6b7787;
            opacity: 1;
          }
          [data-testid="stSidebar"] .stTextInput input,
          [data-testid="stSidebar"] .stTextArea textarea,
          [data-testid="stSidebar"] .stNumberInput input {
            caret-color: #0f1722;
          }
          [data-testid="stSidebar"] [data-baseweb="select"] * {
            color: #0f1722;
          }
          [data-baseweb="popover"] [role="option"],
          [data-baseweb="popover"] [role="listbox"] *,
          [data-baseweb="popover"] [data-testid="stMarkdownContainer"] * {
            color: #0f1722;
          }
          [data-testid="stSidebar"] .stSlider [data-baseweb="slider"] {
            padding-top: 0.1rem;
          }
          .tribe-shellbar {
            display: grid;
            grid-template-columns: minmax(340px, 1.2fr) auto;
            gap: 18px;
            align-items: end;
            padding: 1rem 1.05rem 0.85rem 1.05rem;
            margin-bottom: 0.9rem;
            border: 1px solid var(--line);
            border-radius: 20px;
            background:
              linear-gradient(135deg, rgba(255,255,255,0.72), rgba(255,255,255,0.54)),
              radial-gradient(circle at top right, rgba(255, 106, 31, 0.14), transparent 30%);
            box-shadow: 0 18px 40px rgba(15, 23, 34, 0.07);
            backdrop-filter: blur(18px);
          }
          .tribe-shellbar h1 {
            margin: 0;
            font-size: clamp(1.45rem, 2vw, 2.15rem);
            line-height: 1.02;
          }
          .tribe-shellbar p {
            margin: 0.2rem 0 0 0;
            max-width: 64ch;
            color: var(--muted);
            font-size: 0.92rem;
            line-height: 1.45;
          }
          .tribe-kicker {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            margin-bottom: 0.4rem;
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: var(--accent-strong);
          }
          .tribe-pills {
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
            justify-content: flex-end;
            align-items: center;
          }
          .tribe-pill {
            display: inline-flex;
            flex-direction: column;
            align-items: flex-start;
            gap: 0.08rem;
            min-width: 118px;
            padding: 0.5rem 0.72rem 0.54rem 0.72rem;
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.48);
            border: 1px solid rgba(15, 23, 34, 0.08);
            box-shadow: 0 10px 18px rgba(15, 23, 34, 0.04);
          }
          .tribe-pill-label {
            font-size: 0.62rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: var(--muted);
          }
          .tribe-pill-value {
            font-size: 0.86rem;
            color: #18212d;
            font-weight: 600;
            line-height: 1.15;
          }
          .tribe-sectionhead {
            margin-bottom: 0.6rem;
          }
          .tribe-sectionhead h3 {
            margin: 0;
            font-size: 1rem;
            line-height: 1.1;
          }
          .tribe-sectionhead p {
            margin: 0.16rem 0 0 0;
            color: var(--muted);
            font-size: 0.85rem;
            line-height: 1.4;
          }
          div[data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 18px;
            box-shadow: 0 10px 26px rgba(15, 23, 34, 0.05);
            backdrop-filter: blur(16px);
          }
          div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.54);
            border: 1px solid rgba(15, 23, 34, 0.08);
            border-radius: 14px;
            padding: 0.35rem 0.6rem;
            min-height: 0;
          }
          div[data-testid="stMetricLabel"] p {
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--muted);
          }
          div[data-testid="stMetricValue"] {
            font-family: "Space Grotesk", sans-serif;
          }
          .stTabs [data-baseweb="tab-list"] {
            gap: 0.35rem;
            padding-bottom: 0.15rem;
          }
          .stTabs [data-baseweb="tab"] {
            height: 2.25rem;
            border-radius: 999px;
            padding: 0 0.95rem;
            background: rgba(15, 23, 34, 0.05);
            border: 1px solid rgba(15, 23, 34, 0.06);
          }
          .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #111827, #202b39);
            color: white;
            box-shadow: 0 10px 18px rgba(15, 23, 34, 0.14);
          }
          .stButton > button, .stDownloadButton > button {
            height: 2.6rem;
            border-radius: 12px;
            border: 1px solid rgba(15, 23, 34, 0.08);
            font-weight: 600;
            letter-spacing: 0.01em;
          }
          .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, var(--accent), var(--accent-strong));
            border: none;
            color: white;
            box-shadow: 0 12px 22px rgba(229, 86, 16, 0.24);
          }
          .stButton > button:hover, .stDownloadButton > button:hover {
            border-color: rgba(15, 23, 34, 0.16);
          }
          .stTextInput input, .stTextArea textarea, .stNumberInput input, [data-baseweb="select"] > div {
            background: rgba(255, 255, 255, 0.82);
            border-radius: 12px;
            border: 1px solid rgba(15, 23, 34, 0.08);
          }
          div[data-testid="stFileUploader"] section {
            border-radius: 16px;
            border: 1px dashed rgba(15, 23, 34, 0.18);
            background: rgba(255, 255, 255, 0.46);
          }
          div[data-testid="stDataFrame"] {
            border-radius: 14px;
            overflow: hidden;
            border: 1px solid rgba(15, 23, 34, 0.08);
          }
          .tribe-inline-note {
            margin-top: 0.3rem;
            color: var(--muted);
            font-size: 0.8rem;
          }
          .tribe-slimbar {
            display: grid;
            grid-template-columns: minmax(0, 1fr) auto;
            gap: 0.9rem;
            align-items: center;
            padding: 0.78rem 0.9rem;
            margin-top: 0.7rem;
            border-radius: 16px;
            border: 1px solid rgba(15, 23, 34, 0.08);
            background: rgba(255, 255, 255, 0.5);
          }
          .tribe-slimbar-title {
            font-size: 0.9rem;
            font-weight: 600;
            color: var(--ink);
          }
          .tribe-slimbar-copy {
            margin-top: 0.15rem;
            font-size: 0.8rem;
            color: var(--muted);
          }
          .tribe-progress-caption {
            margin-top: 0.4rem;
            font-size: 0.78rem;
            color: var(--muted);
          }
          .tribe-busybutton {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 0.58rem;
            width: 100%;
            min-height: 2.6rem;
            padding: 0.7rem 1rem;
            border-radius: 12px;
            border: 1px solid rgba(15, 23, 34, 0.10);
            background: linear-gradient(135deg, rgba(112, 123, 140, 0.28), rgba(86, 95, 111, 0.22));
            color: rgba(15, 23, 34, 0.78);
            font-weight: 600;
            user-select: none;
            pointer-events: none;
            opacity: 0.95;
          }
          .tribe-busyspinner {
            width: 0.95rem;
            height: 0.95rem;
            border-radius: 999px;
            border: 2px solid rgba(15, 23, 34, 0.16);
            border-top-color: rgba(15, 23, 34, 0.82);
            animation: tribe-spin 0.75s linear infinite;
          }
          @keyframes tribe-spin {
            to { transform: rotate(360deg); }
          }
          .tribe-keyline {
            margin: 0.15rem 0 0.7rem 0;
            height: 1px;
            background: linear-gradient(90deg, rgba(15,23,34,0.14), rgba(15,23,34,0));
          }
          @media (max-width: 960px) {
            .tribe-shellbar {
              grid-template-columns: 1fr;
            }
            .tribe-pills {
              justify-content: flex-start;
            }
            .tribe-pill {
              min-width: 0;
            }
            .tribe-slimbar {
              grid-template-columns: 1fr;
            }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _build_shell_pills(
    request: dict[str, object] | None,
    options: dict[str, object] | None,
    cache_folder: Path | None,
) -> str:
    request = request or {}
    options = options or {}
    pills = [
        ("Source", _format_request_label(request)),
        ("Device", str(options.get("device", "cuda")).upper()),
        ("Checkpoint", str(options.get("checkpoint", "facebook/tribev2")).split("/")[-1]),
        ("Chat", str(options.get("openai_model", DEFAULT_OPENAI_CHAT_MODEL))),
        ("WhisperX", "on" if options.get("transcribe") else "off"),
    ]
    if cache_folder is not None:
        pills.append(("Cache", Path(cache_folder).name or str(cache_folder)))
    pills.append(
        (
            "Image clip",
            f"{float(options.get('image_duration', 4.0)):.1f}s @ {int(options.get('image_fps', 6))} fps",
        )
    )
    return "".join(
        f"""
        <div class="tribe-pill">
          <div class="tribe-pill-label">{html.escape(label)}</div>
          <div class="tribe-pill-value">{html.escape(value)}</div>
        </div>
        """
        for label, value in pills
    )


def hero(
    request: dict[str, object] | None = None,
    options: dict[str, object] | None = None,
    cache_folder: Path | None = None,
) -> None:
    st.markdown(
        f"""
        <div class="tribe-shellbar">
          <div>
            <div class="tribe-kicker">TRIBE V2 EASY WORKSPACE</div>
            <h1>Multimodal brain prediction dashboard</h1>
            <p>
              Importez une source, lancez l'inference locale GPU, inspectez les surfaces corticales et demandez l'analyse du run au chat OpenAI.
            </p>
          </div>
          <div class="tribe-pills">
            {_build_shell_pills(request, options, cache_folder)}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_head(title: str, caption: str, *, kicker: str | None = None) -> None:
    kicker_html = (
        f'<div class="tribe-kicker" style="margin-bottom:0.18rem;">{kicker}</div>'
        if kicker
        else ""
    )
    st.markdown(
        f"""
        <div class="tribe-sectionhead">
          {kicker_html}
          <h3>{title}</h3>
          <p>{caption}</p>
        </div>
        <div class="tribe-keyline"></div>
        """,
        unsafe_allow_html=True,
    )


def _format_request_label(request: dict[str, object]) -> str:
    if request.get("video_path"):
        return "Video"
    if request.get("audio_path"):
        return "Audio"
    if request.get("text"):
        return "Texte"
    if request.get("image_paths"):
        count = len(tp.cast(list[Path], request["image_paths"]))
        return f"Images x{count}"
    return "Aucune source"


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


def render_raw_timestep_table(run: PredictionRun, *, height: int = 430) -> None:
    st.markdown("**Donnees par timestep**")
    st.dataframe(build_raw_timestep_frame(run), width="stretch", height=height)
    st.caption(
        "Table brute envoyable au chat: temps, texte aligne quand il existe, et statistiques numeriques de prediction."
    )


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
        section_head(
            "OpenAI analyst",
            "Chat lateral pour interpreter le run avec GPT-5.4 a partir des timesteps et des stats.",
            kicker="Chat",
        )
        if not api_key.strip():
            st.info(
                "Renseignez `OPENAI_API_KEY` dans la sidebar ou dans l'environnement pour activer le chat."
            )
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
        chat_meta = st.columns(2, gap="small")
        chat_meta[0].metric("Model", model)
        chat_meta[1].metric("Images", int(max_images))
        st.caption("Le prompt systeme demande une lecture prudente: patterns corticaux, valence plausible, emotions candidates et incertitudes obligatoires.")
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
        with st.expander("Contexte envoye au modele", expanded=False):
            st.markdown(
                "\n".join(f"- {label}" for label in labels)
                if labels
                else "Aucune image de timestep selectionnee."
            )
            st.code(context_text, language="json")

        for message in session["messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        prompt = st.chat_input(
            "Demandez a GPT d'expliquer ce run...",
            key=f"chat_input_{run_key}",
        )
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
                st.caption("Contexte envoye: " + " | ".join(labels))
            st.markdown(reply)
            LOGGER.info(
                "OpenAI chat reply received | run=%s | model=%s | chars=%s",
                run_key,
                model,
                len(reply),
            )

        session["messages"].append({"role": "assistant", "content": reply})
        session["previous_response_id"] = response_id


def build_synced_player_html(run: PredictionRun) -> str:
    if run.source_path is None or run.input_kind not in {"video", "audio"}:
        raise ValueError("Synced playback is only available for audio and video runs with a source file.")
    media_uri = build_data_uri(run.source_path.read_bytes(), guess_media_mime(run.source_path))
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
    }
    payload_json = json.dumps(payload)
    media_tag = "video" if run.input_kind == "video" else "audio"
    return f"""
    <div style="font-family: ui-sans-serif, system-ui, sans-serif; color: #171717;">
      <style>
        .sync-wrap {{
          display: grid;
          grid-template-columns: minmax(320px, 1.2fr) minmax(320px, 1fr);
          gap: 18px;
          align-items: start;
        }}
        .sync-card {{
          border: 1px solid rgba(23, 23, 23, 0.10);
          border-radius: 18px;
          padding: 14px;
          background: rgba(255, 250, 242, 0.92);
          box-shadow: 0 8px 20px rgba(23, 23, 23, 0.05);
        }}
        .sync-card h4 {{
          margin: 0 0 10px 0;
          font-size: 15px;
        }}
        .sync-meta {{
          color: #665f57;
          font-size: 13px;
          margin-top: 10px;
        }}
        .sync-text {{
          margin-top: 10px;
          min-height: 56px;
          color: #171717;
          line-height: 1.45;
          font-size: 14px;
          white-space: pre-wrap;
        }}
        .sync-slider {{
          width: 100%;
          margin-top: 10px;
        }}
        .sync-brain {{
          width: 100%;
          border-radius: 14px;
          background: white;
          border: 1px solid rgba(23, 23, 23, 0.08);
        }}
      </style>
      <div class="sync-wrap">
        <div class="sync-card">
          <h4>{'Video source' if run.input_kind == 'video' else 'Audio source'}</h4>
          <{media_tag} id="tribe-media" src="{media_uri}" controls loop playsinline style="width:100%; border-radius: 14px; {'max-height: 340px; object-fit: contain;' if run.input_kind == 'video' else ''}"></{media_tag}>
          <div id="tribe-meta" class="sync-meta"></div>
          <div id="tribe-text" class="sync-text"></div>
        </div>
        <div class="sync-card">
          <h4>Cerveau predit en temps reel</h4>
          <img id="tribe-brain" class="sync-brain" alt="TRIBE v2 predicted brain activity" />
          <div class="sync-meta">La carte change avec le temps de lecture du media. Le pas temporel suit les segments predits du modele.</div>
        </div>
      </div>
      <script>
        const payload = {payload_json};
        const steps = payload.timesteps;
        const player = document.getElementById("tribe-media");
        const brain = document.getElementById("tribe-brain");
        const meta = document.getElementById("tribe-meta");
        const text = document.getElementById("tribe-text");
        let activeIndex = -1;

        function findStepIndex(currentTime) {{
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
          text.textContent = step.text ? step.text : "Pas de texte aligne pour ce segment.";
        }}

        function syncToPlayer() {{
          if (!steps.length) return;
          renderStep(findStepIndex(player.currentTime || 0));
          requestAnimationFrame(syncToPlayer);
        }}

        player.addEventListener("seeked", () => renderStep(findStepIndex(player.currentTime || 0)));
        player.addEventListener("timeupdate", () => renderStep(findStepIndex(player.currentTime || 0)));
        renderStep(0);
        syncToPlayer();
        player.play().catch(() => {{}});
      </script>
    </div>
    """


def input_panel(cache_folder: Path) -> tuple[dict, dict, bool]:
    whisperx_available = ExtractWordsFromAudio.whisperx_available()
    with st.sidebar:
        st.header("Configuration")
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
        if not whisperx_available:
            st.info(
                "Transcription desactivee: WhisperX n'est pas detecte dans l'environnement actif."
            )
        st.divider()
        st.header("Chat OpenAI")
        openai_model = st.text_input(
            "Modele OpenAI",
            value=DEFAULT_OPENAI_CHAT_MODEL,
            help="Le guide modeles OpenAI recommande gpt-5.4 pour les cas complexes et multimodaux.",
        )
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
            help="Cle utilisee par le panneau de chat lateral. Laissez vide pour utiliser la variable d'environnement.",
            placeholder="Utilise OPENAI_API_KEY si laisse vide",
        )
        if os.getenv("OPENAI_API_KEY", ""):
            st.caption("Variable d'environnement OPENAI_API_KEY detectee. Laissez le champ vide pour l'utiliser.")
        st.caption(
            "Le dashboard prefere `meta-llama/Llama-3.2-3B` pour rester aligné "
            "avec le repo d'origine, puis repasse automatiquement sur "
            "`unsloth/Llama-3.2-3B` si le repo gated n'est pas accessible "
            "dans l'environnement courant."
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
        section_head(
            "Input workspace",
            "Choisissez une seule modalite par run. L'etat actif remonte dans le header, pas dans des panneaux lateraux redondants.",
            kicker="Source",
        )
        tabs = st.tabs(["Video", "Audio", "Texte", "Images"])
        with tabs[0]:
            video_file = st.file_uploader(
                "Importer une video", type=["mp4", "mov", "avi", "mkv", "webm"]
            )
            if video_file is not None:
                request["video_path"] = save_upload(video_file, cache_folder / "uploads")
                st.video(video_file.getvalue())
        with tabs[1]:
            audio_file = st.file_uploader(
                "Importer un audio", type=["wav", "mp3", "flac", "ogg"]
            )
            if audio_file is not None:
                request["audio_path"] = save_upload(audio_file, cache_folder / "uploads")
                st.audio(audio_file.getvalue())
        with tabs[2]:
            text_input = st.text_area(
                "Texte brut",
                placeholder="Collez un script, une transcription ou un prompt narratif.",
                height=176,
            )
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

        action_col, button_col = st.columns([1.45, 0.78], gap="large")
        with action_col:
            st.markdown(
                """
                <div class="tribe-slimbar">
                  <div>
                    <div class="tribe-slimbar-title">Run en un clic</div>
                    <div class="tribe-slimbar-copy">
                      Une seule source active a la fois. Preparation, inference, vues 2D, 3D et exports se chainent automatiquement.
                    </div>
                  </div>
                  <div class="tribe-pill">
                    <div class="tribe-pill-label">Source active</div>
                    <div class="tribe-pill-value">"""
                + html.escape(_format_request_label(request))
                + """</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with button_col:
            if st.session_state.get("prediction_running", False):
                render_busy_prediction_button()
            else:
                launch_pressed = st.button(
                    "Lancer la prediction",
                    type="primary",
                    width="stretch",
                    key="launch_prediction",
                )
            st.markdown(
                f"""
                <div class="tribe-progress-caption">
                  Checkpoint <strong>{html.escape(str(checkpoint).split('/')[-1])}</strong><br/>
                  Chat <strong>{html.escape(openai_model)}</strong>
                </div>
                """,
                unsafe_allow_html=True,
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
            st.error("Importez d'abord une video, un audio ou un texte.")
            return
        active_inputs = [key for key, value in request.items() if value]
        if len(active_inputs) != 1:
            st.error("Choisissez une seule modalite a la fois: video, audio, texte ou images.")
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
        if "image_paths" in request:
            image_paths = [Path(p) for p in request["image_paths"]]
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
        else:
            update_progress(2)
            events, input_kind = prepare_events(
                cache_folder=cache_folder,
                transcribe=options["transcribe"],
                direct_text=options["direct_text"],
                seconds_per_word=options["seconds_per_word"],
                max_context_words=options["max_context_words"],
                **request,
            )
            update_progress(3)
            source_path = next(
                (value for key, value in request.items() if key.endswith("_path")),
                None,
            )
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
        else:
            LOGGER.info(
                "Prediction completed | input_kind=%s | timesteps=%s | vertices=%s",
                run.input_kind,
                len(run.preds),
                run.preds.shape[1],
            )
        clear_prediction_job_state()
        st.session_state["prediction_notice"] = "Prediction prete. Les vues et exports sont disponibles plus bas."
        st.rerun()
    except Exception as exc:
        LOGGER.exception("Prediction failed | source=%s", _format_request_label(request))
        clear_prediction_job_state()
        st.session_state["prediction_error"] = {
            "message": f"La prediction a echoue: {exc}",
            "traceback": traceback.format_exc(),
        }
        st.rerun()


def render_input_preview(run: PredictionRun) -> None:
    if run.raw_text:
        st.text_area("Texte", value=run.raw_text, height=220)
    elif run.source_path and run.input_kind == "video":
        st.video(run.source_path.read_bytes())
    elif run.source_path and run.input_kind == "audio":
        st.audio(run.source_path.read_bytes())
    elif run.source_path and run.input_kind == "image":
        render_bounded_image(
            run.source_path.read_bytes(),
            caption=run.source_path.name,
            max_height=320,
        )
    else:
        st.caption("Apercu non disponible pour cette entree.")


def results_panel(cache_folder: Path, run: PredictionRun | ImageComparisonRun) -> None:
    if isinstance(run, ImageComparisonRun):
        return comparison_results_panel(run)

    run_key = get_run_cache_key(run)
    summary = summarize_predictions(run.preds)
    with st.container(border=True):
        section_head(
            "Run status",
            "Resume numerique du run courant.",
            kicker="Metrics",
        )
        metric_cols = st.columns(5, gap="small")
        metric_cols[0].metric("Timesteps", len(run.preds))
        metric_cols[1].metric("Vertices", run.preds.shape[1])
        metric_cols[2].metric("Modalite", run.input_kind.capitalize())
        metric_cols[3].metric("Evenements", len(run.events))
        metric_cols[4].metric("Mean abs", f"{float(summary['mean_abs'].mean()):.4f}")

    workspace_col, inspect_col = st.columns([1.72, 1.0], gap="large")
    with workspace_col:
        with st.container(border=True):
            section_head(
                "Timeline surface",
                "Courbe globale et lecture cerebrale en boucle dans le meme espace de travail.",
                kicker="Workspace",
            )
            chart_col, animation_col = st.columns([0.9, 1.15], gap="large")
            with chart_col:
                st.line_chart(
                    summary.set_index("timestep")[["mean_abs", "std"]],
                    height=248,
                    width="stretch",
                )
            animation_cache = st.session_state.setdefault("brain_gif_bytes", {})
            animation_key = get_run_cache_key(run)
            if animation_key not in animation_cache:
                with st.spinner("Generation de l'animation cerebrale..."):
                    animation_cache[animation_key] = render_prediction_gif(run)
            with animation_col:
                st.image(
                    animation_cache[animation_key],
                    caption="Brain playback",
                    width="stretch",
                )

        with st.container(border=True):
            section_head(
                "Deep views",
                "Vues avancees et exports media.",
                kicker="Views",
            )
            tab_labels = ["3D anime", "Animation MP4", "Figure multi-timesteps", "Table des evenements"]
            synced_media_available = run.input_kind in {"video", "audio"} and run.source_path is not None
            if synced_media_available:
                tab_labels = ["Lecteur synchronise"] + tab_labels
            tabs = st.tabs(tab_labels)
            tab_offset = 0

            if synced_media_available:
                with tabs[0]:
                    sync_cache = st.session_state.setdefault("sync_player_html", {})
                    sync_key = run_key
                    if st.button(
                        "Preparer le lecteur synchronise",
                        width="stretch",
                        key=f"prepare_sync_{run_key}",
                    ):
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
                        sync_cache[sync_key] = build_synced_player_html(run)
                        sync_progress(2)
                    sync_html = sync_cache.get(sync_key)
                    if sync_html:
                        components.html(sync_html, height=860, scrolling=False)
                    else:
                        st.caption("Associe le temps du media a la carte cerebrale predite.")
                    tab_offset = 1

            with tabs[tab_offset]:
                html = get_cached_animated_3d_html(run)
                components.html(html, height=860, scrolling=False)
                st.caption(
                    "Lecture auto active. Faites tourner le cerveau avec la souris, puis utilisez Pause pour figer un instant precis."
                )
                st.download_button(
                    "Telecharger la vue 3D HTML",
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
                if st.button("Generer le MP4", width="stretch", key=f"export_mp4_{run_key}"):
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
                        "Telecharger le MP4",
                        data=video_path.read_bytes(),
                        file_name=video_path.name,
                        mime="video/mp4",
                        width="stretch",
                    )
                else:
                    st.caption("Le MP4 reprend la surface cerebrale predite a chaque timestep.")

            with tabs[tab_offset + 2]:
                mosaic_cache = st.session_state.setdefault("mosaic_figures", {})
                mosaic_key = f"mosaic:{run_key}"
                if st.button("Generer la mosaique", width="stretch", key=f"mosaic_{run_key}"):
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
            section_head(
                "Inspector",
                "Apercu de la source et donnees brutes.",
                kicker="Inspector",
            )
            preview_tab, data_tab = st.tabs(["Preview", "Timesteps"])
            with preview_tab:
                render_input_preview(run)
            with data_tab:
                render_raw_timestep_table(run, height=420)

        with st.container(border=True):
            section_head(
                "Exports",
                "Telechargements rapides.",
                kicker="Output",
            )
            export_cols = st.columns(3)
            export_cols[0].download_button(
                "Predictions .npy",
                data=build_npy_download(run.preds),
                file_name="tribev2_predictions.npy",
                mime="application/octet-stream",
                width="stretch",
            )
            export_cols[1].download_button(
                "Events .csv",
                data=run.events.to_csv(index=False).encode("utf-8"),
                file_name="tribev2_events.csv",
                mime="text/csv",
                width="stretch",
            )
            export_cols[2].download_button(
                "Resume .csv",
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
        section_head(
            "Image comparison",
            "Deux colonnes synchrones pour comparer rapidement les patterns.",
            kicker="Compare",
        )
        metric_cols = st.columns(4)
        metric_cols[0].metric("Images", len(run.runs))
        metric_cols[1].metric("Timesteps communs", common_timesteps)
        metric_cols[2].metric("Vertices", run.runs[0].preds.shape[1])
        metric_cols[3].metric("Modalite", "Images")

    cols = st.columns(len(run.runs), gap="large")
    for idx, (col, item) in enumerate(zip(cols, run.runs), start=1):
        with col:
            with st.container(border=True):
                section_head(
                    f"Image {idx}",
                    "Preview, playback cortical et 3D orientable.",
                    kicker="Panel",
                )
                render_input_preview(item)
                gif_cache = st.session_state.setdefault("brain_gif_bytes", {})
                gif_key = get_run_cache_key(item)
                if gif_key not in gif_cache:
                    with st.spinner(f"Generation de l'animation image {idx}..."):
                        gif_cache[gif_key] = render_prediction_gif(item)
                st.image(
                    gif_cache[gif_key],
                    caption="Brain playback",
                    width="stretch",
                )
                image_html = get_cached_animated_3d_html(
                    item,
                    max_frames=18,
                    height=620,
                    spinner_label=f"Generation de la 3D animee image {idx}...",
                )
                components.html(image_html, height=690, scrolling=False)
                st.download_button(
                    f"3D HTML image {idx}",
                    data=image_html.encode("utf-8"),
                    file_name=f"tribev2_image_{idx}_brain_animation.html",
                    mime="text/html",
                    width="stretch",
                )
                with st.expander("Donnees par timestep", expanded=False):
                    render_raw_timestep_table(item, height=260)
                st.download_button(
                    f"Predictions image {idx}",
                    data=build_npy_download(item.preds),
                    file_name=f"tribev2_image_{idx}_predictions.npy",
                    mime="application/octet-stream",
                    width="stretch",
                )


def main() -> None:
    configure_runtime_noise()
    apply_theme()
    hero_slot = st.empty()
    cache_folder = Path(st.sidebar.text_input("Dossier cache", value="./cache"))
    log_path = configure_dashboard_logging(cache_folder)
    with st.sidebar:
        st.caption(f"Log file: `{log_path}`")
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
