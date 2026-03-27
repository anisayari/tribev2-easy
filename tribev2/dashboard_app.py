from __future__ import annotations

import io
import logging
from pathlib import Path
import shutil
import uuid
import warnings

import numpy as np
import streamlit as st
import streamlit.components.v1 as components


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
    describe_timestep,
    export_prediction_video,
    load_model,
    predict_from_prepared_events,
    prepare_events,
    render_brain_figure,
    render_interactive_brain_html,
    render_prediction_mosaic,
    segment_preview,
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


def render_reading_guide(
    *,
    input_kind: str,
    timestep: int,
    duration: float | None,
) -> None:
    st.markdown("**Comment lire cette carte**")
    notes = [
        "La surface affiche le cortex sur le maillage `fsaverage5`, pas un cerveau individuel.",
        "Les couleurs chaudes indiquent une reponse predite plus forte a ce timestep.",
        "Ce signal est une prediction du modele TRIBE, pas une mesure fMRI reelle.",
    ]
    if duration is not None:
        notes.append(
            f"Le timestep {timestep} couvre environ {duration:.2f}s du stimulus {input_kind}."
        )
    for note in notes:
        st.markdown(f"- {note}")


def input_panel(cache_folder: Path) -> tuple[dict, dict]:
    uvx_available = shutil.which("uvx") is not None
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
            disabled=not uvx_available,
            help="Necessite `uvx whisperx` dans le PATH.",
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
        if not uvx_available:
            st.info("Transcription desactivee: `uvx whisperx` n'est pas installe.")
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
                    st.image(file.getvalue(), caption=file.name, width="stretch")

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
        st.image(run.source_path.read_bytes(), caption=run.source_path.name, width="stretch")
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

    st.subheader("Exploration par timestep")
    timestep = st.slider(
        "Choisir un timestep",
        min_value=0,
        max_value=len(run.preds) - 1,
        value=0,
        step=1,
    )
    preview = segment_preview(run, timestep)
    description = describe_timestep(run.preds, timestep=timestep)
    fig_col, info_col = st.columns([1.7, 1.0], gap="large")
    with fig_col:
        fig = render_brain_figure(run.preds, timestep=timestep, vmin=0.5)
        st.pyplot(fig, clear_figure=True, width="stretch")
    with info_col:
        st.write(
            {
                "start": round(float(preview["start"]), 3) if preview["start"] is not None else None,
                "duration": round(float(preview["duration"]), 3) if preview["duration"] is not None else None,
            }
        )
        if preview["frame"] is not None:
            st.image(preview["frame"], caption="Frame associee", width="stretch")
        if preview["text"]:
            st.text_area("Texte du segment", value=preview["text"], height=160)
        st.markdown("**Lecture rapide**")
        st.caption(description["summary"])
        exp_cols = st.columns(2)
        exp_cols[0].metric("Lateralite", description["laterality"].capitalize())
        exp_cols[1].metric("Orientation", description["dorso_ventral"].capitalize())
        exp_cols = st.columns(2)
        exp_cols[0].metric("Zone AP", description["antero_posterior"].capitalize())
        exp_cols[1].metric("Top 1% du signal", f"{description['focus_share']:.1%}")

    with st.expander("Explication", expanded=False):
        render_reading_guide(
            input_kind=run.input_kind,
            timestep=timestep,
            duration=float(preview["duration"]) if preview["duration"] is not None else None,
        )
        st.markdown(
            "- Utilisez la vue 3D pour tourner autour du cortex et verifier si le foyer est plutot lateral, medial, dorsal ou ventral."
        )
        st.markdown(
            "- Utilisez l'animation MP4 pour voir comment la prediction evolue dans le temps et la rapprocher du texte, de l'audio ou de la video source."
        )

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

    tabs = st.tabs(["3D interactif", "Animation MP4", "Figure multi-timesteps", "Table des evenements"])

    with tabs[0]:
        html_cache = st.session_state.setdefault("interactive_html_by_timestep", {})
        if st.button("Generer la scene 3D", width="stretch"):
            with st.spinner("Generation de la scene 3D..."):
                html_cache[timestep] = render_interactive_brain_html(
                    run.preds,
                    timestep=timestep,
                    vmin=0.5,
                )
        html = html_cache.get(timestep)
        if html:
            components.html(html, height=760, scrolling=False)
            st.download_button(
                "Telecharger la scene HTML",
                data=html.encode("utf-8"),
                file_name=f"tribev2_timestep_{timestep:03d}.html",
                mime="text/html",
                width="stretch",
            )
        else:
            st.caption("Generez la scene pour obtenir un cerveau 3D que vous pouvez tourner librement.")

    with tabs[1]:
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

    with tabs[2]:
        if st.button("Generer la mosaique", width="stretch"):
            st.session_state["mosaic_requested"] = True
        if st.session_state.get("mosaic_requested"):
            mosaic = render_prediction_mosaic(run)
            st.pyplot(mosaic, clear_figure=True, width="stretch")

    with tabs[3]:
        st.dataframe(run.events, width="stretch", height=320)


def comparison_results_panel(run: ImageComparisonRun) -> None:
    st.subheader("Comparaison d'images")
    common_timesteps = min(len(item.preds) for item in run.runs)
    timestep = st.slider(
        "Choisir un timestep commun",
        min_value=0,
        max_value=common_timesteps - 1,
        value=0,
        step=1,
        key="image_compare_timestep",
    )
    metric_cols = st.columns(4)
    metric_cols[0].metric("Images", len(run.runs))
    metric_cols[1].metric("Timesteps communs", common_timesteps)
    metric_cols[2].metric("Vertices", run.runs[0].preds.shape[1])
    metric_cols[3].metric("Modalite", "Images")

    cols = st.columns(len(run.runs), gap="large")
    for idx, (col, item) in enumerate(zip(cols, run.runs), start=1):
        with col:
            st.markdown(f"**Image {idx}**")
            render_input_preview(item)
            fig = render_brain_figure(item.preds, timestep=timestep, vmin=0.5)
            st.pyplot(fig, clear_figure=True, width="stretch")
            description = describe_timestep(item.preds, timestep=timestep)
            st.caption(description["summary"])
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
