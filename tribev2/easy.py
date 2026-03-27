from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
import shutil
import subprocess
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
from tribev2.plotting.cortical_pv import PlotBrainPyvista
from tribev2.plotting.utils import (
    get_clip,
    get_cmap,
    get_scalar_mappable,
    get_text,
    has_audio,
    has_video,
    robust_normalize,
)

DEFAULT_TEXT_MODEL = "unsloth/Llama-3.2-3B"
VALID_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


@dataclass
class PredictionRun:
    events: pd.DataFrame
    preds: np.ndarray
    segments: list
    input_kind: str
    source_path: Path | None = None
    raw_text: str | None = None


@dataclass
class ImageComparisonRun:
    runs: list[PredictionRun]

    @property
    def input_kind(self) -> str:
        return "image"


@lru_cache(maxsize=4)
def get_pyvista_plotter(mesh: str = "fsaverage5") -> PlotBrainPyvista:
    """Cache the heavier PyVista plotter/mesh backend for dashboard reuse."""
    return PlotBrainPyvista(mesh=mesh)


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
    image_path: str | Path | None = None,
    transcribe: bool = False,
    direct_text: bool = True,
    seconds_per_word: float = 0.45,
    max_context_words: int = 128,
    image_duration: float = 4.0,
    image_fps: int = 6,
) -> tuple[pd.DataFrame, str]:
    """Prepare a standardised events dataframe from one user input."""
    provided = {
        "text": text,
        "text_path": text_path,
        "audio_path": audio_path,
        "video_path": video_path,
        "image_path": image_path,
    }
    active = [name for name, value in provided.items() if value]
    if len(active) != 1:
        raise ValueError(
            "Exactly one of text, text_path, audio_path, video_path or image_path must be provided."
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

    if image_path is not None:
        image_path = Path(image_path)
        if image_path.suffix.lower() not in VALID_IMAGE_SUFFIXES:
            raise ValueError(
                f"Unsupported image format '{image_path.suffix}'. "
                f"Expected one of {sorted(VALID_IMAGE_SUFFIXES)}."
            )
        if not image_path.is_file():
            raise FileNotFoundError(f"Image file does not exist: {image_path}")
        video_path = build_video_from_image(
            image_path=image_path,
            output_folder=cache_folder / "image_clips",
            duration=image_duration,
            fps=image_fps,
        )
        events = pd.DataFrame(
            [
                {
                    "type": "Video",
                    "filepath": str(video_path),
                    "start": 0,
                    "timeline": "default",
                    "subject": "default",
                }
            ]
        )
        return get_audio_and_text_events(events, audio_only=True), "image"

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
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.90, wspace=0.02, hspace=0.02)
    return fig


def render_prediction_mosaic(
    run: PredictionRun,
    *,
    max_timesteps: int = 6,
    mesh: str = "fsaverage5",
    show_stimuli: bool = True,
) -> plt.Figure:
    """Render a compact multi-timestep figure for the dashboard."""
    plotter = get_pyvista_plotter(mesh)
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
    return fig


def _get_surface_render_data(
    signal: np.ndarray,
    *,
    mesh: str = "fsaverage5",
    cmap: str = "fire",
    norm_percentile: int | None = 99,
    vmin: float | None = None,
    vmax: float | None = None,
    threshold: float | None = None,
    symmetric_cbar: bool = False,
) -> tuple[PlotBrainPyvista, np.ndarray, np.ndarray, np.ndarray]:
    plotter = get_pyvista_plotter(mesh)
    if norm_percentile is not None:
        signal = robust_normalize(signal, percentile=norm_percentile)
    mesh_data = plotter._mesh["both"]
    stat_map = plotter.get_stat_map(signal)["both"]
    sm = get_scalar_mappable(
        signal,
        get_cmap(cmap),
        vmin=vmin,
        vmax=vmax,
        threshold=threshold,
        symmetric_cbar=symmetric_cbar,
    )
    rgba = sm.to_rgba(stat_map)
    bg_map = mesh_data["bg_map"]
    bg_norm = (bg_map - bg_map.min()) / (bg_map.max() - bg_map.min() + 1e-8)
    bg_rgb = 1 - np.column_stack(
        [plotter.bg_darkness + bg_norm * (1 - plotter.bg_darkness)] * 3
    )
    colors = rgba[:, 3:4] * rgba[:, :3] + (1 - rgba[:, 3:4]) * bg_rgb
    return plotter, mesh_data["coords"], mesh_data["faces"], colors


def render_interactive_brain_html(
    preds: np.ndarray,
    *,
    timestep: int,
    mesh: str = "fsaverage5",
    cmap: str = "fire",
    norm_percentile: int | None = 99,
    vmin: float | None = None,
    vmax: float | None = None,
    threshold: float | None = None,
    symmetric_cbar: bool = False,
    width: int = 980,
    height: int = 700,
) -> str:
    """Render one timestep as an interactive PyVista HTML scene."""
    if timestep < 0 or timestep >= len(preds):
        raise IndexError(f"Invalid timestep {timestep} for predictions of length {len(preds)}.")

    import pyvista as pv

    _, vertices, faces, colors = _get_surface_render_data(
        preds[timestep],
        mesh=mesh,
        cmap=cmap,
        norm_percentile=norm_percentile,
        vmin=vmin,
        vmax=vmax,
        threshold=threshold,
        symmetric_cbar=symmetric_cbar,
    )
    pv_faces = np.column_stack([np.full(len(faces), 3), faces]).astype(np.int64)
    surface = pv.PolyData(vertices, pv_faces)
    surface.point_data["colors"] = colors

    plotter = pv.Plotter(off_screen=True, window_size=[width, height])
    plotter.add_mesh(
        surface,
        scalars="colors",
        rgb=True,
        smooth_shading=True,
        ambient=0.3,
    )
    plotter.set_background("white")
    plotter.view_vector([-1, 0, 0], viewup=[0, 0, 1])
    plotter.camera.zoom(1.35)
    html = plotter.export_html(None).getvalue()
    plotter.close()
    return html


def export_prediction_video(
    run: PredictionRun,
    *,
    output_folder: str | Path,
    max_timesteps: int | None = None,
    mesh: str = "fsaverage5",
    interpolated_fps: int | None = 12,
    cmap: str = "fire",
    norm_percentile: int = 99,
    vmin: float | None = 0.5,
) -> Path:
    """Render a brain prediction animation to MP4."""
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    n_timesteps = len(run.preds) if max_timesteps is None else min(len(run.preds), max_timesteps)
    fps_tag = interpolated_fps or 1
    output_path = output_folder / f"tribev2_prediction_{n_timesteps:03d}t_{fps_tag:02d}fps.mp4"
    plotter = get_pyvista_plotter(mesh)
    plotter.plot_timesteps_mp4(
        run.preds[:n_timesteps],
        filepath=output_path,
        segments=run.segments[:n_timesteps],
        interpolated_fps=interpolated_fps,
        cmap=cmap,
        norm_percentile=norm_percentile,
        vmin=vmin,
    )
    return output_path


def describe_timestep(
    preds: np.ndarray,
    *,
    timestep: int,
    mesh: str = "fsaverage5",
    top_percent: float = 1.0,
) -> dict[str, tp.Any]:
    """Summarize where the strongest predicted activity sits on the cortex."""
    if timestep < 0 or timestep >= len(preds):
        raise IndexError(f"Invalid timestep {timestep} for predictions of length {len(preds)}.")
    if not 0 < top_percent <= 100:
        raise ValueError("top_percent must be within (0, 100].")

    signal = np.asarray(preds[timestep], dtype=float)
    abs_signal = np.abs(signal)
    n_vertices = len(abs_signal)
    k = max(32, int(np.ceil(n_vertices * top_percent / 100)))
    idx = np.argpartition(abs_signal, -k)[-k:]
    plotter = get_pyvista_plotter(mesh)
    coords = plotter._mesh["both"]["coords"]
    focus_coords = coords[idx]
    weights = abs_signal[idx]
    weighted_center = np.average(focus_coords, axis=0, weights=weights)
    coord_scale = np.maximum(np.max(np.abs(coords), axis=0), 1e-6)
    x_score, y_score, z_score = weighted_center / coord_scale

    def classify_axis(
        score: float,
        negative: str,
        neutral: str,
        positive: str,
        threshold: float = 0.12,
    ) -> str:
        if score <= -threshold:
            return negative
        if score >= threshold:
            return positive
        return neutral

    laterality = classify_axis(
        x_score,
        negative="gauche",
        neutral="bilaterale",
        positive="droite",
    )
    antero_posterior = classify_axis(
        y_score,
        negative="posterieure",
        neutral="centrale",
        positive="anterieure",
    )
    dorso_ventral = classify_axis(
        z_score,
        negative="ventrale",
        neutral="intermediaire",
        positive="dorsale",
    )
    focus_share = float(weights.sum() / max(abs_signal.sum(), 1e-8))
    mean_abs = float(abs_signal.mean())
    peak_abs = float(abs_signal.max())
    summary = (
        f"Les sommets les plus saillants sont surtout {laterality}, "
        f"plutot {antero_posterior} et {dorso_ventral}. "
        f"Le top {top_percent:.1f}% des sommets concentre {focus_share:.1%} "
        f"de l'amplitude absolue a ce timestep."
    )
    return {
        "laterality": laterality,
        "antero_posterior": antero_posterior,
        "dorso_ventral": dorso_ventral,
        "focus_share": focus_share,
        "mean_abs": mean_abs,
        "peak_abs": peak_abs,
        "summary": summary,
    }


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


def resolve_video_encoder(ffmpeg_bin: str | None = None) -> str:
    ffmpeg_bin = ffmpeg_bin or resolve_ffmpeg()
    result = subprocess.run(
        [ffmpeg_bin, "-hide_banner", "-encoders"],
        capture_output=True,
        text=True,
        check=True,
    )
    available = result.stdout + result.stderr
    for encoder in ("libx264", "libopenh264", "mpeg4"):
        if encoder in available:
            return encoder
    raise RuntimeError("No supported MP4 video encoder found in ffmpeg.")


def build_video_from_image(
    *,
    image_path: str | Path,
    output_folder: str | Path,
    duration: float = 4.0,
    fps: int = 6,
) -> Path:
    image_path = Path(image_path)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    if duration <= 0:
        raise ValueError("Image clip duration must be strictly positive.")
    if fps < 1:
        raise ValueError("Image clip fps must be at least 1.")

    ffmpeg_bin = resolve_ffmpeg()
    video_encoder = resolve_video_encoder(ffmpeg_bin)
    safe_stem = "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in image_path.stem)
    output_path = output_folder / f"{safe_stem}_{duration:.2f}s_{fps}fps.mp4"
    if output_path.exists():
        return output_path

    cmd = [
        ffmpeg_bin,
        "-y",
        "-loop",
        "1",
        "-framerate",
        str(fps),
        "-i",
        str(image_path),
        "-f",
        "lavfi",
        "-i",
        "anullsrc=channel_layout=mono:sample_rate=16000",
        "-t",
        f"{duration:.3f}",
        "-shortest",
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
        "-c:v",
        video_encoder,
    ]
    if video_encoder == "libx264":
        cmd.extend(["-crf", "18"])
    elif video_encoder == "libopenh264":
        cmd.extend(["-b:v", "4M"])
    else:
        cmd.extend(["-q:v", "3"])
    cmd.extend(
        [
            "-c:a",
            "aac",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]
    )
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    return output_path


def write_text_to_temp_file(text: str, folder: str | Path) -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix="tribev2_text_", suffix=".txt", dir=folder)
    os.close(fd)
    path = Path(temp_path)
    path.write_text(text, encoding="utf-8")
    return path
