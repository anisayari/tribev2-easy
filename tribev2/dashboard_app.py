from __future__ import annotations

import io
from pathlib import Path
import shutil
import uuid

import numpy as np
import streamlit as st

from tribev2.easy import (
    DEFAULT_TEXT_MODEL,
    PredictionRun,
    load_model,
    predict_from_prepared_events,
    prepare_events,
    render_brain_figure,
    render_prediction_mosaic,
    segment_preview,
    summarize_predictions,
)


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
        if not uvx_available:
            st.info("Transcription desactivee: `uvx whisperx` n'est pas installe.")
        st.caption(
            "Le dashboard utilise par defaut `unsloth/Llama-3.2-3B`, "
            "un backbone public non gated compatible avec le codeur texte du projet."
        )

    tabs = st.tabs(["Video", "Audio", "Texte"])
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

    options = {
        "checkpoint": checkpoint,
        "device": device,
        "num_workers": int(num_workers),
        "text_model_name": text_model_name,
        "transcribe": transcribe,
        "direct_text": direct_text,
        "seconds_per_word": seconds_per_word,
        "max_context_words": max_context_words,
    }
    return request, options


def run_prediction_ui(cache_folder: Path, request: dict, options: dict) -> None:
    if st.button("Lancer la prediction", type="primary", use_container_width=True):
        if not request:
            st.error("Importez d'abord une video, un audio ou un texte.")
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
        except Exception as exc:
            st.exception(exc)


def results_panel(run: PredictionRun) -> None:
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
            use_container_width=True,
        )
    with preview_col:
        st.subheader("Entree")
        if run.raw_text:
            st.text_area("Texte", value=run.raw_text, height=220)
        elif run.source_path and run.input_kind == "video":
            st.video(run.source_path.read_bytes())
        elif run.source_path and run.input_kind == "audio":
            st.audio(run.source_path.read_bytes())
        else:
            st.caption("Apercu non disponible pour cette entree.")

    st.subheader("Exploration par timestep")
    timestep = st.slider(
        "Choisir un timestep",
        min_value=0,
        max_value=len(run.preds) - 1,
        value=0,
        step=1,
    )
    fig_col, info_col = st.columns([1.7, 1.0], gap="large")
    with fig_col:
        fig = render_brain_figure(run.preds, timestep=timestep, vmin=0.5)
        st.pyplot(fig, clear_figure=True, use_container_width=True)
    with info_col:
        preview = segment_preview(run, timestep)
        st.write(
            {
                "start": round(float(preview["start"]), 3) if preview["start"] is not None else None,
                "duration": round(float(preview["duration"]), 3) if preview["duration"] is not None else None,
            }
        )
        if preview["frame"] is not None:
            st.image(preview["frame"], caption="Frame associee", use_container_width=True)
        if preview["text"]:
            st.text_area("Texte du segment", value=preview["text"], height=160)

    st.subheader("Exports")
    export_cols = st.columns(3)
    export_cols[0].download_button(
        "Predictions .npy",
        data=build_npy_download(run.preds),
        file_name="tribev2_predictions.npy",
        mime="application/octet-stream",
        use_container_width=True,
    )
    export_cols[1].download_button(
        "Events .csv",
        data=run.events.to_csv(index=False).encode("utf-8"),
        file_name="tribev2_events.csv",
        mime="text/csv",
        use_container_width=True,
    )
    export_cols[2].download_button(
        "Resume .csv",
        data=summary.to_csv(index=False).encode("utf-8"),
        file_name="tribev2_summary.csv",
        mime="text/csv",
        use_container_width=True,
    )

    with st.expander("Figure multi-timesteps", expanded=False):
        if st.button("Generer la mosaique", use_container_width=True):
            st.session_state["mosaic_requested"] = True
        if st.session_state.get("mosaic_requested"):
            mosaic = render_prediction_mosaic(run)
            st.pyplot(mosaic, clear_figure=True, use_container_width=True)

    with st.expander("Table des evenements", expanded=False):
        st.dataframe(run.events, use_container_width=True, height=320)


def main() -> None:
    apply_theme()
    hero()
    cache_folder = Path(st.sidebar.text_input("Dossier cache", value="./cache"))
    request, options = input_panel(cache_folder)
    run_prediction_ui(cache_folder, request, options)

    run = st.session_state.get("prediction_run")
    if run is not None:
        results_panel(run)


if __name__ == "__main__":
    main()
