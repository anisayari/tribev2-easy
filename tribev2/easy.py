from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import sys
import tempfile
import typing as tp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from tribev2.demo_utils import (
    TextToEvents,
    TribeModel,
    _cuda_runtime_supported,
    build_text_events_from_text,
    get_audio_and_text_events,
)
from tribev2.plotting.cortical import PlotBrainNilearn
from tribev2.plotting.utils import get_clip, get_text, has_audio, has_video

DEFAULT_TEXT_MODEL = "unsloth/Llama-3.2-3B"


@dataclass
class PredictionRun:
    events: pd.DataFrame
    preds: np.ndarray
    segments: list
    input_kind: str
    source_path: Path | None = None
    raw_text: str | None = None


def resolve_device(device: str = "auto") -> str:
    """Resolve the requested runtime device for TRIBE."""
    if device == "auto":
        return "cuda" if _cuda_runtime_supported() else "cpu"
    if device == "cuda" and not _cuda_runtime_supported():
        raise RuntimeError(
            "CUDA was requested but the installed PyTorch runtime cannot execute "
            "this GPU. Install a Blackwell-compatible build, for example "
            "`torch>=2.7` with `cu128`."
        )
    if device not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported device: {device}")
    return device


def resolve_text_model_name(text_model_name: str | None = None) -> str:
    """Resolve the text backbone used by the TRIBE text extractor."""
    candidate = text_model_name or os.environ.get("TRIBEV2_TEXT_MODEL")
    if candidate is None:
        candidate = DEFAULT_TEXT_MODEL
    candidate = str(candidate).strip()
    if not candidate:
        raise ValueError("Text model name must not be empty.")
    return candidate


def load_model(
    *,
    checkpoint: str = "facebook/tribev2",
    cache_folder: str | Path = "./cache",
    device: str = "auto",
    num_workers: int = 0,
    text_model_name: str | None = None,
    config_update: dict | None = None,
) -> TribeModel:
    """Load TRIBE with settings that are safe for local dashboard usage."""
    cache_folder = Path(cache_folder)
    cache_folder.mkdir(parents=True, exist_ok=True)
    merged_update = {"data.num_workers": int(num_workers)}
    if config_update:
        merged_update.update(config_update)
    if text_model_name is None:
        merged_update.setdefault(
            "data.text_feature.model_name",
            resolve_text_model_name(),
        )
    else:
        merged_update["data.text_feature.model_name"] = resolve_text_model_name(
            text_model_name
        )
    return TribeModel.from_pretrained(
        checkpoint,
        cache_folder=cache_folder,
        device=resolve_device(device),
        config_update=merged_update,
    )


def prepare_events(
    *,
    cache_folder: str | Path,
    text: str | None = None,
    text_path: str | Path | None = None,
    audio_path: str | Path | None = None,
    video_path: str | Path | None = None,
    transcribe: bool = False,
    direct_text: bool = True,
    seconds_per_word: float = 0.45,
    max_context_words: int = 128,
) -> tuple[pd.DataFrame, str]:
    """Prepare a standardised events dataframe from one user input."""
    provided = {
        "text": text,
        "text_path": text_path,
        "audio_path": audio_path,
        "video_path": video_path,
    }
    active = [name for name, value in provided.items() if value]
    if len(active) != 1:
        raise ValueError(
            "Exactly one of text, text_path, audio_path or video_path must be provided."
        )

    cache_folder = Path(cache_folder)
    cache_folder.mkdir(parents=True, exist_ok=True)

    if text is not None or text_path is not None:
        raw_text = text if text is not None else Path(text_path).read_text(encoding="utf-8")
        if direct_text:
            return (
                build_text_events_from_text(
                    raw_text,
                    seconds_per_word=seconds_per_word,
                    max_context_words=max_context_words,
                ),
                "text",
            )
        return (
            TextToEvents(
                text=raw_text,
                infra={"folder": str(cache_folder), "mode": "retry"},
            ).get_events(),
            "text",
        )

    event_type = "Audio" if audio_path is not None else "Video"
    path = Path(audio_path or video_path)  # type: ignore[arg-type]
    events = pd.DataFrame(
        [
            {
                "type": event_type,
                "filepath": str(path),
                "start": 0,
                "timeline": "default",
                "subject": "default",
            }
        ]
    )
    return get_audio_and_text_events(events, audio_only=not transcribe), event_type.lower()


def predict_from_prepared_events(
    model: TribeModel,
    events: pd.DataFrame,
    *,
    input_kind: str,
    source_path: Path | None = None,
    raw_text: str | None = None,
    verbose: bool = True,
) -> PredictionRun:
    preds, segments = model.predict(events=events, verbose=verbose)
    return PredictionRun(
        events=events,
        preds=preds,
        segments=segments,
        input_kind=input_kind,
        source_path=source_path,
        raw_text=raw_text,
    )


def summarize_predictions(preds: np.ndarray) -> pd.DataFrame:
    """Build a lightweight summary dataframe for charts."""
    return pd.DataFrame(
        {
            "timestep": np.arange(len(preds)),
            "mean": preds.mean(axis=1),
            "std": preds.std(axis=1),
            "mean_abs": np.abs(preds).mean(axis=1),
            "max_abs": np.abs(preds).max(axis=1),
        }
    )


def render_brain_figure(
    preds: np.ndarray,
    *,
    timestep: int,
    mesh: str = "fsaverage5",
    views: tuple[str, ...] = ("left", "right", "dorsal"),
    cmap: str = "fire",
    norm_percentile: int = 99,
    vmin: float | None = None,
) -> plt.Figure:
    """Render a static cortical figure for one predicted timestep."""
    if timestep < 0 or timestep >= len(preds):
        raise IndexError(f"Invalid timestep {timestep} for predictions of length {len(preds)}.")

    plotter = PlotBrainNilearn(mesh=mesh)
    fig, axes = plotter.get_fig_axes(list(views))
    plotter.plot_surf(
        preds[timestep],
        axes=axes,
        views=list(views),
        cmap=cmap,
        norm_percentile=norm_percentile,
        vmin=vmin,
    )
    fig.suptitle(f"Predicted activity at timestep {timestep}", fontsize=14, y=0.98)
    fig.tight_layout()
    return fig


def render_prediction_mosaic(
    run: PredictionRun,
    *,
    max_timesteps: int = 6,
    mesh: str = "fsaverage5",
    show_stimuli: bool = True,
) -> plt.Figure:
    """Render a compact multi-timestep figure for the dashboard."""
    plotter = PlotBrainNilearn(mesh=mesh)
    n_timesteps = min(max_timesteps, len(run.preds))
    allow_stimuli = False
    if run.segments:
        first_segment = run.segments[0]
        allow_stimuli = has_audio(first_segment) or has_video(first_segment)
    fig = plotter.plot_timesteps(
        run.preds[:n_timesteps],
        segments=run.segments[:n_timesteps],
        show_stimuli=show_stimuli and allow_stimuli,
        views="left",
        cmap="fire",
        norm_percentile=99,
        vmin=0.5,
    )
    fig.tight_layout()
    return fig


def segment_preview(run: PredictionRun, timestep: int) -> dict[str, tp.Any]:
    """Extract lightweight preview data for one segment."""
    segment = run.segments[timestep]
    preview: dict[str, tp.Any] = {
        "start": getattr(segment, "start", None),
        "duration": getattr(segment, "duration", None),
        "text": get_text(segment),
        "frame": None,
    }
    clip = get_clip(segment)
    if clip is not None:
        try:
            sample_time = min(max(clip.duration / 2, 0), max(clip.duration - 1e-3, 0))
            preview["frame"] = clip.get_frame(sample_time)
        finally:
            clip.close()
    return preview


def resolve_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    env_ffmpeg = Path(sys.executable).parent / "Library" / "bin" / "ffmpeg.exe"
    if env_ffmpeg.exists():
        return str(env_ffmpeg)
    raise FileNotFoundError("ffmpeg executable not found")


def write_text_to_temp_file(text: str, folder: str | Path) -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix="tribev2_text_", suffix=".txt", dir=folder)
    os.close(fd)
    path = Path(temp_path)
    path.write_text(text, encoding="utf-8")
    return path
