from __future__ import annotations

import base64
import io
import json
import logging
import mimetypes
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
    build_timestep_report_frame,
    build_explainability_report,
    build_image_comparison_guide,
    build_result_interpretation,
    collect_timestep_metadata,
    describe_timestep,
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
    report_frame = build_timestep_report_frame(run)
    with animation_cols[1]:
        st.markdown("**Lecture par timestep**")
        st.dataframe(report_frame, width="stretch", height=430)
        st.caption(
            "Chaque ligne donne une lecture d'un resultat du modele: zone probable, valence du stimulus, emotions textuelles quand elles existent, et resume spatial."
        )

    key_timestep = int(summary["mean_abs"].idxmax())
    key_meta = collect_timestep_metadata(run)[key_timestep]
    description = describe_timestep(run.preds, timestep=key_timestep)
    interpretation = build_result_interpretation(
        run,
        timestep=key_timestep,
        description=description,
        segment_text=key_meta["text"],
    )
    spotlight_cols = st.columns([0.95, 1.35], gap="large")
    with spotlight_cols[0]:
        st.markdown("**Timestep le plus saillant**")
        st.write(
            {
                "timestep": key_timestep,
                "start": round(float(key_meta["start"]), 3),
                "duration": round(float(key_meta["duration"]), 3),
            }
        )
        if key_meta["text"]:
            st.text_area("Texte du segment", value=key_meta["text"], height=120)
        spotlight_metrics = st.columns(2)
        spotlight_metrics[0].metric("Zone probable", interpretation["zone"])
        spotlight_metrics[1].metric(
            "Valence stimulus",
            interpretation["affect"]["valence"].capitalize(),
        )
    with spotlight_cols[1]:
        st.markdown("**Interpretation du resultat**")
        st.caption(interpretation["summary"])
        st.markdown("- " + interpretation["modality_hint"])
        st.markdown("- " + interpretation["lateral_note"])
        st.markdown("- Fonctions plausibles: " + ", ".join(interpretation["systems"]))
        if interpretation["affect"]["evidence"]:
            st.markdown(
                "- Indices affectifs reperes dans le stimulus: "
                + ", ".join(interpretation["affect"]["evidence"])
            )
        else:
            st.markdown(
                "- Aucun indice textuel assez net pour qualifier le resultat en peur, desir, joie, etc."
            )
        st.caption(
            "Les etiquettes affectives viennent du stimulus textuel aligne quand il existe; ce n'est pas une lecture directe de l'etat mental depuis la carte."
        )

    with st.expander("Explication", expanded=False):
        report = build_explainability_report(
            run,
            timestep=key_timestep,
            duration=float(key_meta["duration"]) if key_meta["duration"] is not None else None,
            description=description,
        )
        render_explainability_report(report)
        st.markdown("- Utilisez la vue 3D pour tourner autour du cortex et verifier si le foyer est plutot lateral, medial, dorsal ou ventral.")
        st.markdown("- Utilisez l'animation MP4 pour voir comment la prediction evolue dans le temps et la rapprocher du texte, de l'audio ou de la video source.")

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
        html_cache = st.session_state.setdefault("animated_3d_html", {})
        html_key = get_run_cache_key(run)
        if st.button("Generer la vue 3D animee", width="stretch"):
            with st.spinner("Generation de la vue 3D animee..."):
                html_cache[html_key] = render_animated_brain_3d_html(run)
        html = html_cache.get(html_key)
        if html:
            components.html(html, height=900, scrolling=False)
            st.download_button(
                "Telecharger la vue 3D HTML",
                data=html.encode("utf-8"),
                file_name=f"tribev2_{run.input_kind}_brain_animation.html",
                mime="text/html",
                width="stretch",
            )
        else:
            st.caption("Generez la vue 3D animee pour obtenir un cerveau rotatable avec bouton Play/Pause.")

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
    common_mean_abs = np.mean(
        [np.abs(item.preds[:common_timesteps]).mean(axis=1) for item in run.runs],
        axis=0,
    )
    key_timestep = int(np.argmax(common_mean_abs))
    descriptions = [describe_timestep(item.preds, timestep=key_timestep) for item in run.runs]

    with st.expander("Explication", expanded=False):
        guide = build_image_comparison_guide(
            run,
            timestep=key_timestep,
            descriptions=descriptions,
        )
        st.markdown(f"**{guide['title']}**")
        for bullet in guide["bullets"]:
            st.markdown(f"- {bullet}")
        render_source_links(guide["sources"])

    cols = st.columns(len(run.runs), gap="large")
    for idx, (col, item, description) in enumerate(zip(cols, run.runs, descriptions), start=1):
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
            interpretation = build_result_interpretation(
                item,
                timestep=key_timestep,
                description=description,
            )
            st.markdown("**Interpretation du resultat**")
            st.caption(interpretation["summary"])
            st.markdown(f"- Zone probable: {interpretation['zone']}")
            st.markdown("- Fonctions plausibles: " + ", ".join(interpretation["systems"]))
            st.markdown("- " + interpretation["modality_hint"])
            st.caption(
                "Sur une image seule, le dashboard peut proposer une lecture fonctionnelle grossiere de la zone, mais pas une emotion fiable decodee depuis la carte."
            )
            with st.expander("Lecture par timestep", expanded=False):
                st.dataframe(build_timestep_report_frame(item), width="stretch", height=260)
            animated_3d_cache = st.session_state.setdefault("animated_3d_html", {})
            image_3d_key = get_run_cache_key(item)
            if st.button(f"Generer la 3D animee image {idx}", width="stretch", key=f"image3d_{idx}"):
                with st.spinner(f"Generation de la 3D animee image {idx}..."):
                    animated_3d_cache[image_3d_key] = render_animated_brain_3d_html(item)
            image_html = animated_3d_cache.get(image_3d_key)
            if image_html:
                components.html(image_html, height=760, scrolling=False)
            with st.expander(f"Pourquoi l'image {idx} genere cette carte", expanded=False):
                render_explainability_report(
                    build_explainability_report(
                        item,
                        timestep=key_timestep,
                        description=description,
                    )
                )
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
        results_panel(cache_folder, run)


if __name__ == "__main__":
    main()
