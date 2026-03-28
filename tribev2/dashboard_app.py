from __future__ import annotations

import base64
import io
import json
import logging
import mimetypes
import os
from pathlib import Path
import shutil
import typing as tp
import uuid
import warnings

import numpy as np
from PIL import Image
import streamlit as st
import streamlit.components.v1 as components

from tribev2.eventstransforms import ExtractWordsFromAudio


def _apply_warning_filters() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r"`torch\.cuda\.amp\.autocast\(args\.\.\.\)` is deprecated.*",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"LabelEncoder: event_types has not been set.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.*",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"The events dataframe contains an `Index` column.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"LabelEncoder has only found one label.*",
        category=UserWarning,
    )
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


def apply_theme() -> None:
    st.set_page_config(
        page_title="TRIBE v2 Easy",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
          :root {
            --bg: #f6f2ea;
            --panel: #fffaf2;
            --ink: #171717;
            --muted: #665f57;
            --accent: #b74316;
            --accent-soft: #f2d2c5;
          }
          .stApp {
            background:
              radial-gradient(circle at top right, rgba(183, 67, 22, 0.10), transparent 30%),
              linear-gradient(180deg, #fbf8f2 0%, var(--bg) 100%);
            color: var(--ink);
          }
          .tribe-hero {
            padding: 1.2rem 1.4rem;
            border: 1px solid rgba(23, 23, 23, 0.08);
            border-radius: 20px;
            background: linear-gradient(135deg, rgba(255,250,242,0.95), rgba(242,210,197,0.85));
            box-shadow: 0 12px 30px rgba(23, 23, 23, 0.06);
            margin-bottom: 1rem;
          }
          .tribe-caption {
            color: var(--muted);
            font-size: 0.95rem;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero() -> None:
    st.markdown(
        """
        <div class="tribe-hero">
          <h1 style="margin:0 0 0.25rem 0;">TRIBE v2 Easy Dashboard</h1>
          <div class="tribe-caption">
            Chargez un texte, un audio ou une video, lancez l'inference,
            puis explorez les activations cerebrales predites directement dans le dashboard.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
    st.markdown("### Chat OpenAI")
    st.caption(
        "Le chat envoie automatiquement des images de timesteps clefs et les donnees numeriques du run a l'API."
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
    if st.button("Nouvelle conversation", width="stretch", key=f"chat_reset_{run_key}"):
        sessions[run_key] = {"messages": [], "previous_response_id": None}
        st.rerun()

    context_bundle = get_cached_openai_context_bundle(
        run,
        image_detail=image_detail,
        max_images=max_images,
    )
    context_text, _, labels = context_bundle
    with st.expander("Contexte envoye au modele", expanded=not session["messages"]):
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


def input_panel(cache_folder: Path) -> tuple[dict, dict]:
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
            value=os.getenv("OPENAI_API_KEY", ""),
            type="password",
            help="Cle utilisee par le panneau de chat lateral. Laissez vide pour utiliser seulement la variable d'environnement.",
        )
        st.caption(
            "Le dashboard utilise par defaut `unsloth/Llama-3.2-3B`, "
            "un backbone public non gated compatible avec le codeur texte du projet."
        )

    tabs = st.tabs(["Video", "Audio", "Texte", "Images"])
    request: dict[str, object] = {}
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
            height=180,
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
                    render_bounded_image(file.getvalue(), caption=file.name, max_height=260)

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
    return request, options


def run_prediction_ui(cache_folder: Path, request: dict, options: dict) -> None:
    if st.button("Lancer la prediction", type="primary", width="stretch"):
        if not request:
            st.error("Importez d'abord une video, un audio ou un texte.")
            return
        active_inputs = [key for key, value in request.items() if value]
        if len(active_inputs) != 1:
            st.error("Choisissez une seule modalite a la fois: video, audio, texte ou images.")
            return
        try:
            with st.spinner("Chargement du modele..."):
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
                with st.spinner("Preparation des evenements image..."):
                    for image_path in image_paths:
                        events, input_kind = prepare_events(
                            cache_folder=cache_folder,
                            image_path=image_path,
                            image_duration=options["image_duration"],
                            image_fps=options["image_fps"],
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
                with st.spinner("Preparation des evenements..."):
                    events, input_kind = prepare_events(
                        cache_folder=cache_folder,
                        transcribe=options["transcribe"],
                        direct_text=options["direct_text"],
                        seconds_per_word=options["seconds_per_word"],
                        max_context_words=options["max_context_words"],
                        **request,
                    )
                with st.spinner("Inference en cours..."):
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
            st.session_state["prediction_run"] = run
            st.session_state["mosaic_requested"] = False
            st.session_state["interactive_html_by_timestep"] = {}
            st.session_state["video_exports"] = {}
        except Exception as exc:
            st.exception(exc)


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

    summary = summarize_predictions(run.preds)
    metric_cols = st.columns(4)
    metric_cols[0].metric("Timesteps gardes", len(run.preds))
    metric_cols[1].metric("Vertices", run.preds.shape[1])
    metric_cols[2].metric("Modalite", run.input_kind.capitalize())
    metric_cols[3].metric("Evenements", len(run.events))

    chart_col, preview_col = st.columns([1.3, 1.0], gap="large")
    with chart_col:
        st.subheader("Resume temporel")
        st.line_chart(
            summary.set_index("timestep")[["mean_abs", "std"]],
            height=260,
            width="stretch",
        )
    with preview_col:
        st.subheader("Entree")
        render_input_preview(run)

    st.subheader("Lecture animee")
    animation_cols = st.columns([1.25, 1.0], gap="large")
    animation_cache = st.session_state.setdefault("brain_gif_bytes", {})
    animation_key = get_run_cache_key(run)
    if animation_key not in animation_cache:
        with st.spinner("Generation de l'animation cerebrale..."):
            animation_cache[animation_key] = render_prediction_gif(run)
    with animation_cols[0]:
        st.image(
            animation_cache[animation_key],
            caption="Animation cerebrale en boucle",
            width="stretch",
        )
    with animation_cols[1]:
        render_raw_timestep_table(run)

    st.subheader("Exports")
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

    tab_labels = ["3D anime", "Animation MP4", "Figure multi-timesteps", "Table des evenements"]
    synced_media_available = run.input_kind in {"video", "audio"} and run.source_path is not None
    if synced_media_available:
        tab_labels = ["Lecteur synchronise"] + tab_labels
    tabs = st.tabs(tab_labels)
    tab_offset = 0

    if synced_media_available:
        with tabs[0]:
            sync_cache = st.session_state.setdefault("sync_player_html", {})
            sync_key = get_run_cache_key(run)
            if st.button("Preparer le lecteur synchronise", width="stretch"):
                with st.spinner("Preparation du lecteur synchronise..."):
                    sync_cache[sync_key] = build_synced_player_html(run)
            sync_html = sync_cache.get(sync_key)
            if sync_html:
                components.html(sync_html, height=860, scrolling=False)
            else:
                st.caption("Ce lecteur associe le temps de lecture du media a la carte cerebrale predite correspondante.")
            tab_offset = 1

    with tabs[tab_offset]:
        html = get_cached_animated_3d_html(run)
        components.html(html, height=900, scrolling=False)
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
        if st.button("Generer le MP4", width="stretch"):
            with st.spinner("Generation du MP4..."):
                video_path = export_prediction_video(
                    run,
                    output_folder=cache_folder / "exports",
                    max_timesteps=max_timesteps,
                    interpolated_fps=fps_options[fps_label],
                )
            video_cache[export_key] = str(video_path)
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
        if st.button("Generer la mosaique", width="stretch"):
            st.session_state["mosaic_requested"] = True
        if st.session_state.get("mosaic_requested"):
            mosaic = render_prediction_mosaic(run)
            st.pyplot(mosaic, clear_figure=True, width="stretch")

    with tabs[tab_offset + 3]:
        st.dataframe(run.events, width="stretch", height=320)


def comparison_results_panel(run: ImageComparisonRun) -> None:
    st.subheader("Comparaison d'images")
    common_timesteps = min(len(item.preds) for item in run.runs)
    metric_cols = st.columns(4)
    metric_cols[0].metric("Images", len(run.runs))
    metric_cols[1].metric("Timesteps communs", common_timesteps)
    metric_cols[2].metric("Vertices", run.runs[0].preds.shape[1])
    metric_cols[3].metric("Modalite", "Images")

    st.markdown("**Comparaison animee**")
    st.caption(
        "Chaque cerveau 3D demarre automatiquement en boucle. Sur une image statique, une rotation douce de camera rend la lecture plus visible."
    )
    cols = st.columns(len(run.runs), gap="large")
    for idx, (col, item) in enumerate(zip(cols, run.runs), start=1):
        with col:
            st.markdown(f"**Image {idx}**")
            render_input_preview(item)
            gif_cache = st.session_state.setdefault("brain_gif_bytes", {})
            gif_key = get_run_cache_key(item)
            if gif_key not in gif_cache:
                with st.spinner(f"Generation de l'animation image {idx}..."):
                    gif_cache[gif_key] = render_prediction_gif(item)
            st.image(
                gif_cache[gif_key],
                caption="Animation cerebrale en boucle",
                width="stretch",
            )
            image_html = get_cached_animated_3d_html(
                item,
                max_frames=18,
                height=620,
                spinner_label=f"Generation de la 3D animee image {idx}...",
            )
            st.markdown("**Vue 3D animee**")
            components.html(image_html, height=690, scrolling=False)
            st.caption(
                "Auto-play actif. Faites glisser la scene pour orienter le cerveau, ou utilisez Pause pour inspecter une pose."
            )
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
    hero()
    cache_folder = Path(st.sidebar.text_input("Dossier cache", value="./cache"))
    request, options = input_panel(cache_folder)
    run_prediction_ui(cache_folder, request, options)

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
