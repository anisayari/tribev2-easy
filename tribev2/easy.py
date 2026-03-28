from __future__ import annotations

from contextlib import redirect_stderr
from dataclasses import dataclass, field
from functools import lru_cache
import io
import json
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import typing as tp
import unicodedata

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
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

logger = logging.getLogger(__name__)

PRIMARY_TEXT_MODEL = "meta-llama/Llama-3.2-3B"
FALLBACK_TEXT_MODEL = "unsloth/Llama-3.2-3B"
DEFAULT_TEXT_MODEL = PRIMARY_TEXT_MODEL
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


@dataclass
class MultiModalRun(PredictionRun):
    component_runs: dict[str, PredictionRun] = field(default_factory=dict)
    source_paths: dict[str, Path] = field(default_factory=dict)
    primary_input_kind: str | None = None


MULTIMODAL_CHANNEL_ORDER: tuple[str, ...] = ("visual", "audio", "text")
MULTIMODAL_CHANNEL_LABELS: dict[str, str] = {
    "visual": "Visuel",
    "audio": "Audio",
    "text": "Texte",
}
MULTIMODAL_SOURCE_LABELS: dict[str, str] = {
    "video": "Vidéo",
    "image": "Image",
    "audio": "Audio",
    "text": "Texte",
}


EXPLAINABILITY_SOURCES: tuple[tuple[str, str], ...] = (
    (
        "Meta AI blog (2026-03-26)",
        "https://ai.meta.com/blog/tribe-v2-brain-predictive-foundation-model/",
    ),
    (
        "Meta AI publication (2026-03-26)",
        "https://ai.meta.com/research/publications/a-foundation-model-of-vision-audition-and-language-for-in-silico-neuroscience/",
    ),
    (
        "Official demo notebook",
        "https://github.com/facebookresearch/tribev2/blob/main/tribe_demo.ipynb",
    ),
)

VALENCE_CUE_LEXICON: dict[str, set[str]] = {
    "positive": {
        "good", "great", "joy", "happy", "love", "safe", "calm", "hope", "beautiful",
        "success", "excited", "wonderful", "bien", "heureux", "heureuse", "joie",
        "amour", "calme", "espoir", "belle", "beau", "positif", "positive",
    },
    "negative": {
        "bad", "sad", "hate", "pain", "danger", "death", "violent", "loss", "cry",
        "terrible", "awful", "mal", "triste", "haine", "douleur", "danger", "mort",
        "violent", "perte", "pleure", "negatif", "negative",
    },
    "joy": {
        "joy", "happy", "delight", "smile", "laugh", "celebrate", "relief", "fun",
        "joie", "heureux", "heureuse", "sourire", "rire", "celebre", "soulagement",
    },
    "fear": {
        "fear", "afraid", "scared", "terror", "panic", "threat", "danger", "worry",
        "worried", "anxiety", "anxious", "peur", "effraye", "terrifie", "panique",
        "menace", "inquiet", "inquiete", "anxiete",
    },
    "desire": {
        "want", "need", "wish", "hope", "desire", "crave", "longing", "dream",
        "envie", "veux", "veut", "vouloir", "besoin", "souhaite", "desir", "reve",
    },
    "anger": {
        "anger", "angry", "rage", "furious", "hate", "fight", "attack", "mad",
        "colere", "furieux", "rage", "haine", "attaque", "frappe",
    },
    "sadness": {
        "sad", "grief", "cry", "tears", "lonely", "loss", "mourning",
        "triste", "chagrin", "pleure", "larmes", "seul", "perte", "deuil",
    },
    "calm": {
        "calm", "quiet", "peace", "soft", "slow", "rest", "gentle",
        "calme", "paisible", "doux", "douce", "lent", "lente", "repos",
    },
}


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


def resolve_text_model_candidates(text_model_name: str | None = None) -> list[str]:
    """Resolve a preferred->fallback chain for the TRIBE text extractor."""
    candidate = resolve_text_model_name(text_model_name)
    if candidate == PRIMARY_TEXT_MODEL:
        return [PRIMARY_TEXT_MODEL, FALLBACK_TEXT_MODEL]
    return [candidate]


def _is_text_model_access_error(exc: Exception) -> bool:
    """Return True when the preferred text repo is unavailable in this env."""
    message = f"{type(exc).__name__}: {exc}".lower()
    patterns = (
        "gated repo",
        "cannot access gated repo",
        "trying to access a gated repo",
        "awaiting a review",
        "403 client error",
        "401 client error",
        "repository not found",
        "access to model",
    )
    return any(pattern in message for pattern in patterns)


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
    requested_text_model = (
        text_model_name
        if text_model_name is not None
        else tp.cast(str | None, merged_update.get("data.text_feature.model_name"))
    )
    candidates = resolve_text_model_candidates(requested_text_model)
    last_error: Exception | None = None
    for index, candidate in enumerate(candidates):
        current_update = dict(merged_update)
        current_update["data.text_feature.model_name"] = candidate
        try:
            model = TribeModel.from_pretrained(
                checkpoint,
                cache_folder=cache_folder,
                device=resolve_device(device),
                config_update=current_update,
            )
            setattr(model, "_tribev2_text_model_name", candidate)
            if index > 0:
                logger.warning(
                    "Loaded TRIBE text backbone fallback '%s' after preferred model was unavailable.",
                    candidate,
                )
            return model
        except Exception as exc:
            last_error = exc
            is_last = index == len(candidates) - 1
            if is_last or not _is_text_model_access_error(exc):
                raise
            logger.warning(
                "Preferred text backbone '%s' unavailable, retrying with fallback '%s'.",
                candidate,
                candidates[index + 1],
            )
    assert last_error is not None
    raise last_error


def _prepare_media_events_with_quiet_stderr(
    events: pd.DataFrame,
    *,
    audio_only: bool,
) -> pd.DataFrame:
    """Shield Streamlit runs from tqdm/stderr issues inside event transforms."""
    stderr_buffer = io.StringIO()
    try:
        with redirect_stderr(stderr_buffer):
            return get_audio_and_text_events(events, audio_only=audio_only)
    finally:
        stderr_buffer.close()


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
        return _prepare_media_events_with_quiet_stderr(events, audio_only=True), "image"

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
    return (
        _prepare_media_events_with_quiet_stderr(events, audio_only=not transcribe),
        event_type.lower(),
    )


def predict_from_prepared_events(
    model: TribeModel,
    events: pd.DataFrame,
    *,
    input_kind: str,
    source_path: Path | None = None,
    raw_text: str | None = None,
    verbose: bool = True,
) -> PredictionRun:
    stderr_buffer = io.StringIO()
    try:
        with redirect_stderr(stderr_buffer):
            preds, segments = model.predict(events=events, verbose=verbose)
    finally:
        stderr_buffer.close()
    return PredictionRun(
        events=events,
        preds=preds,
        segments=segments,
        input_kind=input_kind,
        source_path=source_path,
        raw_text=raw_text,
    )


def build_multimodal_events(
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
) -> tuple[pd.DataFrame, dict[str, dict[str, tp.Any]]]:
    """Prepare one combined events frame plus per-modality metadata."""
    component_specs: list[tuple[str, dict[str, tp.Any]]] = []
    if video_path is not None:
        component_specs.append(("video", {"video_path": video_path}))
    if image_path is not None:
        component_specs.append(("image", {"image_path": image_path}))
    if audio_path is not None:
        component_specs.append(("audio", {"audio_path": audio_path}))
    if text is not None or text_path is not None:
        component_specs.append(("text", {"text": text, "text_path": text_path}))
    if len(component_specs) < 2:
        raise ValueError("At least two modalities are required to build a multimodal run.")

    prepared_components: dict[str, dict[str, tp.Any]] = {}
    frames: list[pd.DataFrame] = []
    cache_folder = Path(cache_folder)
    for modality, payload in component_specs:
        events, input_kind = prepare_events(
            cache_folder=cache_folder,
            transcribe=transcribe,
            direct_text=direct_text,
            seconds_per_word=seconds_per_word,
            max_context_words=max_context_words,
            image_duration=image_duration,
            image_fps=image_fps,
            **payload,
        )
        raw_text_value = None
        if modality == "text":
            raw_text_value = text if text is not None else Path(tp.cast(str | Path, text_path)).read_text(encoding="utf-8")
        source_path = payload.get("video_path") or payload.get("audio_path") or payload.get("image_path")
        prepared_components[modality] = {
            "events": events.copy(),
            "input_kind": input_kind,
            "source_path": Path(source_path) if source_path else None,
            "raw_text": raw_text_value,
        }
        frames.append(events)

    combined = pd.concat(frames, ignore_index=True, sort=False)
    sort_columns = [column for column in ("start", "timeline", "type") if column in combined.columns]
    if sort_columns:
        combined = combined.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    return combined, prepared_components


def predict_multimodal_from_prepared_events(
    model: TribeModel,
    combined_events: pd.DataFrame,
    *,
    prepared_components: dict[str, dict[str, tp.Any]],
    verbose: bool = True,
) -> MultiModalRun:
    """Run one fused prediction and one prediction per modality."""
    component_runs: dict[str, PredictionRun] = {}
    for modality, spec in prepared_components.items():
        component_runs[modality] = predict_from_prepared_events(
            model,
            tp.cast(pd.DataFrame, spec["events"]),
            input_kind=str(spec["input_kind"]),
            source_path=tp.cast(Path | None, spec["source_path"]),
            raw_text=tp.cast(str | None, spec["raw_text"]),
            verbose=False,
        )

    combined_source_path = None
    primary_input_kind = None
    for candidate in ("video", "image", "audio", "text"):
        if candidate in prepared_components:
            primary_input_kind = candidate
            combined_source_path = tp.cast(Path | None, prepared_components[candidate]["source_path"])
            break

    fused = predict_from_prepared_events(
        model,
        combined_events,
        input_kind="multimodal",
        source_path=combined_source_path,
        raw_text=tp.cast(str | None, prepared_components.get("text", {}).get("raw_text")),
        verbose=verbose,
    )
    return MultiModalRun(
        events=fused.events,
        preds=fused.preds,
        segments=fused.segments,
        input_kind=fused.input_kind,
        source_path=fused.source_path,
        raw_text=fused.raw_text,
        component_runs=component_runs,
        source_paths={
            modality: spec["source_path"]
            for modality, spec in prepared_components.items()
            if spec.get("source_path") is not None
        },
        primary_input_kind=primary_input_kind,
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


def list_run_channels(run: PredictionRun) -> list[str]:
    """Describe which signal channels are represented in the prepared events."""
    if isinstance(run, MultiModalRun):
        ordered: list[str] = []
        for modality in ("video", "image", "audio", "text"):
            if modality in run.component_runs:
                ordered.append(MULTIMODAL_SOURCE_LABELS[modality].lower())
        if ordered:
            return ordered
    channels: list[str] = []
    event_types = {str(value).strip().lower() for value in run.events.get("type", pd.Series(dtype=object)).dropna()}
    if run.input_kind == "image":
        return ["image statique", "clip video synthetique"]
    if run.input_kind == "text":
        return ["texte aligne"]
    if "video" in event_types or run.input_kind == "video":
        channels.append("video")
    if "audio" in event_types or run.input_kind == "audio":
        channels.append("audio")
    if {"word", "text"} & event_types or run.raw_text:
        channels.append("texte aligne")
    if not channels:
        channels.append(run.input_kind)
    return channels


def normalize_signal_for_display(
    signal: np.ndarray,
    *,
    percentile: int | None = 99,
) -> np.ndarray:
    """Normalize a signal robustly and silence divide-by-zero edge cases."""
    signal = np.asarray(signal, dtype=float)
    if percentile is None:
        return signal
    with np.errstate(divide="ignore", invalid="ignore"):
        out = robust_normalize(signal, percentile=percentile)
    return np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)


def select_animation_indices(n_timesteps: int, max_frames: int = 72) -> list[int]:
    """Select evenly spaced timestep indices while preserving endpoints."""
    if n_timesteps <= 0:
        return []
    if n_timesteps <= max_frames:
        return list(range(n_timesteps))
    values = np.linspace(0, n_timesteps - 1, num=max_frames)
    indices = sorted({int(round(value)) for value in values})
    if indices[0] != 0:
        indices[0] = 0
    if indices[-1] != n_timesteps - 1:
        indices[-1] = n_timesteps - 1
    return indices


def normalize_text_for_cues(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKD", text or "")
    normalized = normalized.encode("ascii", "ignore").decode("ascii").lower()
    return re.findall(r"[a-z']+", normalized)


def infer_affective_cues(text: str | None) -> dict[str, tp.Any]:
    """Estimate coarse valence/emotion cues from stimulus text, not from the map alone."""
    if not text or not text.strip():
        return {
            "available": False,
            "valence": "indeterminee",
            "emotions": [],
            "evidence": [],
            "summary": "Pas assez de contenu textuel pour estimer une valence ou une emotion.",
        }

    tokens = normalize_text_for_cues(text)
    if not tokens:
        return {
            "available": False,
            "valence": "indeterminee",
            "emotions": [],
            "evidence": [],
            "summary": "Le texte du segment ne contient pas assez d'indices lexicaux exploitables.",
        }

    scores = {
        name: sum(token in lexicon for token in tokens)
        for name, lexicon in VALENCE_CUE_LEXICON.items()
    }
    evidence = sorted(
        {
            token
            for token in tokens
            if any(token in lexicon for lexicon in VALENCE_CUE_LEXICON.values())
        }
    )
    positive_score = scores["positive"] + scores["joy"] + scores["calm"] + max(scores["desire"] - 1, 0)
    negative_score = scores["negative"] + scores["fear"] + scores["anger"] + scores["sadness"]
    if positive_score == 0 and negative_score == 0:
        valence = "indeterminee"
    elif positive_score >= negative_score * 1.5:
        valence = "plutot positive"
    elif negative_score >= positive_score * 1.5:
        valence = "plutot negative"
    else:
        valence = "mixte"

    emotion_scores = {
        key: scores[key]
        for key in ("joy", "fear", "desire", "anger", "sadness", "calm")
        if scores[key] > 0
    }
    emotions = [key for key, _ in sorted(emotion_scores.items(), key=lambda item: (-item[1], item[0]))[:2]]
    if emotions:
        summary = (
            f"Le texte du segment suggere une valence {valence}. "
            f"Indices emotionnels dominants: {', '.join(emotions)}."
        )
    else:
        summary = (
            "Le texte du segment n'apporte pas assez d'indices lexicaux pour "
            "isoler une emotion dominante, meme si une valence globale peut etre suggeree."
        )
    return {
        "available": bool(emotions or valence != "indeterminee"),
        "valence": valence,
        "emotions": emotions,
        "evidence": evidence[:8],
        "summary": summary,
    }


def infer_region_profile(
    description: dict[str, tp.Any],
    *,
    input_kind: str,
) -> dict[str, tp.Any]:
    """Translate coarse spatial axes into conservative functional hypotheses."""
    ap = description["antero_posterior"]
    dv = description["dorso_ventral"]
    lat = description["laterality"]

    if ap == "posterieure":
        if dv == "dorsale":
            zone = "cortex occipito-parietal dorsal"
            systems = ["vision spatiale", "mouvement visuel", "attention visuo-spatiale"]
        elif dv == "ventrale":
            zone = "cortex occipito-temporal ventral"
            systems = ["voie visuelle ventrale", "formes, objets, scenes ou visages"]
        else:
            zone = "cortex occipital posterieur"
            systems = ["traitement visuel precoce", "structure globale de l'image ou de la scene"]
    elif ap == "centrale":
        if dv == "dorsale":
            zone = "cortex parietal dorsal"
            systems = ["attention", "integration multisensorielle", "coordination perception-action"]
        elif dv == "ventrale":
            zone = "cortex temporal lateral/ventral"
            systems = ["audition ou parole", "semantique", "indices sociaux ou narratifs"]
        else:
            zone = "jonction temporo-parietale"
            systems = ["integration de contexte", "passage entre contenu sensoriel et interpretation"]
    else:
        if dv == "dorsale":
            zone = "cortex frontal dorsal"
            systems = ["controle attentionnel", "planification", "maintien du contexte"]
        elif dv == "ventrale":
            zone = "cortex fronto-temporal ventral"
            systems = ["evaluation de signification", "contexte socio-affectif", "integration de valeur"]
        else:
            zone = "cortex prefrontal"
            systems = ["integration de haut niveau", "contexte", "prediction ou decision"]

    modality_hint = {
        "video": "Le profil est compatible avec un melange d'indices visuels, sonores et linguistiques.",
        "audio": "Le profil doit surtout etre lu comme une reponse au contenu auditif et, si disponible, a la parole.",
        "text": "Le profil doit surtout etre lu comme une reponse au contenu linguistique et au contexte semantique.",
        "image": "Le profil doit surtout etre lu comme une reponse au contenu visuel fixe.",
        "multimodal": "Le profil combine ici plusieurs modalites et doit etre lu comme une superposition visuelle, auditive et/ou linguistique.",
    }[input_kind]

    if lat == "gauche":
        lateral_note = "La lateralisation gauche peut etre compatible avec du langage ou un traitement plus sequentiel, sans que ce soit specifique."
    elif lat == "droite":
        lateral_note = "La lateralisation droite peut etre compatible avec prosodie, scene globale ou indices socio-affectifs, sans que ce soit specifique."
    else:
        lateral_note = "La bilateralite suggere plutot un traitement distribue ou multisensoriel."

    return {
        "zone": f"{zone}, plutot {lat}",
        "systems": systems,
        "modality_hint": modality_hint,
        "lateral_note": lateral_note,
    }


def build_result_interpretation(
    run: PredictionRun,
    *,
    timestep: int,
    description: dict[str, tp.Any] | None = None,
    segment_text: str | None = None,
) -> dict[str, tp.Any]:
    """Explain one result with cautious functional and affective hypotheses."""
    description = description or describe_timestep(run.preds, timestep=timestep)
    region = infer_region_profile(description, input_kind=run.input_kind)
    if segment_text is None:
        if timestep < len(run.segments):
            segment_text = get_segment_text(run.segments[timestep])
        elif run.input_kind == "text":
            segment_text = run.raw_text or ""
    affect = infer_affective_cues(segment_text)

    if affect["available"]:
        affect_summary = (
            f"Le stimulus courant semble {affect['valence']}; "
            f"indices emotionnels: {', '.join(affect['emotions']) if affect['emotions'] else 'non dominants'}."
        )
    else:
        affect_summary = (
            "Aucune lecture fiable de type peur/desir/joie ne doit etre deduite de la carte seule; "
            "il manque ici des indices textuels assez nets."
        )

    summary = (
        f"Zone probablement dominante: {region['zone']}. "
        f"Cette configuration est compatible avec {', '.join(region['systems'][:2])}. "
        f"{affect_summary}"
    )
    cautions = [
        "Les zones et fonctions proposees ici sont des hypotheses grossieres a partir d'une topographie predite, pas une localisation clinique.",
        "Les etiquettes affectives proviennent du stimulus textuel aligne quand il existe; elles ne sont pas decodees directement depuis la carte cerebrale.",
    ]
    return {
        "zone": region["zone"],
        "systems": region["systems"],
        "modality_hint": region["modality_hint"],
        "lateral_note": region["lateral_note"],
        "affect": affect,
        "summary": summary,
        "cautions": cautions,
    }


def build_timestep_reports(
    run: PredictionRun,
    *,
    indices: list[int] | None = None,
) -> list[dict[str, tp.Any]]:
    """Build a structured interpretation row for each timestep."""
    metadata = collect_timestep_metadata(run)
    if indices is None:
        indices = list(range(len(run.preds)))
    rows: list[dict[str, tp.Any]] = []
    if isinstance(run, MultiModalRun):
        summary = summarize_predictions(run.preds).set_index("timestep")
        for idx in indices:
            meta = metadata[idx]
            _, channel_labels, _, matched_indices = _build_multimodal_overlay_signals(
                run,
                timestep=idx,
            )
            channel_text = ", ".join(MULTIMODAL_CHANNEL_LABELS[label] for label in channel_labels) or "Fusion"
            aligned_text = ", ".join(
                f"{MULTIMODAL_SOURCE_LABELS.get(modality, modality)} t{matched_idx}"
                for modality, matched_idx in matched_indices.items()
            )
            row = summary.loc[idx]
            rows.append(
                {
                    "timestep": idx,
                    "start_s": round(float(meta["start"]), 3),
                    "duration_s": round(float(meta["duration"]), 3),
                    "text": meta["text"],
                    "summary": f"Overlay multimodal {channel_text}.",
                    "zone": "Fusion multimodale",
                    "systems": channel_text,
                    "valence": "indeterminee",
                    "emotions": "",
                    "evidence": aligned_text or f"mean_abs={float(row['mean_abs']):.4f}",
                }
            )
        return rows
    for idx in indices:
        meta = metadata[idx]
        description = describe_timestep(run.preds, timestep=idx)
        interpretation = build_result_interpretation(
            run,
            timestep=idx,
            description=description,
            segment_text=meta["text"],
        )
        rows.append(
            {
                "timestep": idx,
                "start_s": round(float(meta["start"]), 3),
                "duration_s": round(float(meta["duration"]), 3),
                "text": meta["text"],
                "summary": description["summary"],
                "zone": interpretation["zone"],
                "systems": ", ".join(interpretation["systems"]),
                "valence": interpretation["affect"]["valence"],
                "emotions": ", ".join(interpretation["affect"]["emotions"]),
                "evidence": ", ".join(interpretation["affect"]["evidence"]),
            }
        )
    return rows


def build_timestep_report_frame(run: PredictionRun) -> pd.DataFrame:
    """Tabular view of per-timestep interpretations."""
    return pd.DataFrame(build_timestep_reports(run))


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


def _get_timestep_row(run: PredictionRun, timestep: int) -> dict[str, tp.Any]:
    metadata = collect_timestep_metadata(run)
    if not metadata:
        return {"index": timestep, "start": float(timestep), "duration": 1.0, "text": ""}
    safe_index = max(0, min(timestep, len(metadata) - 1))
    return metadata[safe_index]


def _get_multimodal_channel(modality: str) -> str:
    if modality in {"video", "image"}:
        return "visual"
    if modality == "audio":
        return "audio"
    return "text"


def _build_multimodal_overlay_signals(
    run: MultiModalRun,
    *,
    timestep: int,
    norm_percentile: int | None = 99,
) -> tuple[list[np.ndarray], list[str], np.ndarray, dict[str, int]]:
    target = _get_timestep_row(run, timestep)
    target_center = float(target["start"]) + (float(target["duration"]) / 2.0)
    grouped: dict[str, list[np.ndarray]] = {channel: [] for channel in MULTIMODAL_CHANNEL_ORDER}
    matched_indices: dict[str, int] = {}

    for modality, component_run in run.component_runs.items():
        metadata = collect_timestep_metadata(component_run)
        if len(component_run.preds) == 0:
            continue
        if metadata:
            centers = np.array(
                [float(row["start"]) + (float(row["duration"]) / 2.0) for row in metadata],
                dtype=float,
            )
            component_index = int(np.argmin(np.abs(centers - target_center)))
        else:
            component_index = min(timestep, len(component_run.preds) - 1)
        matched_indices[modality] = component_index
        signal = np.abs(np.asarray(component_run.preds[component_index], dtype=float))
        if norm_percentile is not None:
            signal = normalize_signal_for_display(signal, percentile=norm_percentile)
        signal = np.clip(np.nan_to_num(signal, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        grouped[_get_multimodal_channel(modality)].append(signal)

    channel_signals: list[np.ndarray] = []
    channel_labels: list[str] = []
    for channel in MULTIMODAL_CHANNEL_ORDER:
        signals = grouped[channel]
        if not signals:
            continue
        if len(signals) == 1:
            merged = signals[0]
        else:
            merged = np.mean(np.stack(signals, axis=0), axis=0)
        channel_signals.append(np.clip(merged, 0.0, 1.0))
        channel_labels.append(channel)

    if channel_signals:
        alpha_signal = np.max(np.stack(channel_signals, axis=0), axis=0)
    else:
        alpha_signal = np.zeros(run.preds.shape[1], dtype=float)
    return channel_signals, channel_labels, np.clip(alpha_signal, 0.0, 1.0), matched_indices


def _get_multimodal_surface_render_data(
    run: MultiModalRun,
    *,
    timestep: int,
    mesh: str = "fsaverage5",
    norm_percentile: int | None = 99,
) -> tuple[PlotBrainPyvista, np.ndarray, np.ndarray, np.ndarray, list[str], dict[str, int]] | None:
    channel_signals, channel_labels, alpha_signal, matched_indices = _build_multimodal_overlay_signals(
        run,
        timestep=timestep,
        norm_percentile=norm_percentile,
    )
    if len(channel_signals) < 2:
        return None

    plotter = get_pyvista_plotter(mesh)
    hemis = [plotter.get_hemis(signal) for signal in channel_signals]
    alpha_hemis = plotter.get_hemis(alpha_signal)
    both_colors: np.ndarray | None = None
    bg_cmap = plt.get_cmap("gray_r")
    for selected_hemi in ("left", "right", "both"):
        stat_maps = [hemi[selected_hemi]["stat_map"] for hemi in hemis]
        colors = np.stack(stat_maps, axis=1)
        if colors.shape[1] == 2:
            colors = np.concatenate([colors, np.zeros((colors.shape[0], 1))], axis=1)
        colors = np.clip(colors, 0.0, 1.0)
        alpha = np.clip(alpha_hemis[selected_hemi]["stat_map"][:, None], 0.0, 1.0)
        bg_map = hemis[0][selected_hemi]["bg_map"]
        bg_norm = (bg_map - bg_map.min()) / (bg_map.max() - bg_map.min() + 1e-8)
        bg_rgba = bg_cmap(bg_norm)
        rgba = np.concatenate([colors, np.ones((colors.shape[0], 1))], axis=1)
        rgba[:, :3] = rgba[:, :3] * alpha + bg_rgba[:, :3] * (1.0 - alpha)
        if selected_hemi == "both":
            both_colors = rgba

    mesh_data = plotter._mesh["both"]
    assert both_colors is not None
    return (
        plotter,
        mesh_data["coords"],
        mesh_data["faces"],
        both_colors,
        channel_labels,
        matched_indices,
    )


def render_multimodal_brain_panel_image(
    run: MultiModalRun,
    *,
    timestep: int,
    mesh: str = "fsaverage5",
    views: tuple[str, ...] = ("left", "right", "dorsal"),
    norm_percentile: int = 99,
) -> np.ndarray:
    plotter = get_pyvista_plotter(mesh)
    channel_signals, _, alpha_signal, _ = _build_multimodal_overlay_signals(
        run,
        timestep=timestep,
        norm_percentile=norm_percentile,
    )
    if len(channel_signals) < 2:
        return render_brain_panel_image(
            run.preds,
            timestep=timestep,
            mesh=mesh,
            views=views,
            cmap="fire",
            norm_percentile=norm_percentile,
            vmin=0.5,
        )
    fig, axes = plt.subplots(
        1,
        len(views),
        figsize=(2.2 * len(views), 2.1),
        squeeze=False,
    )
    flat_axes = list(axes.flatten())
    plotter.plot_surf_rgb(
        channel_signals,
        alpha_signals=alpha_signal,
        norm_percentile=None,
        axes=flat_axes,
        views=list(views),
    )
    for ax in flat_axes:
        ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.01, hspace=0.01)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)
    buffer.seek(0)
    return np.array(Image.open(buffer).convert("RGB"))


def render_run_panel_image(
    run: PredictionRun,
    *,
    timestep: int,
    mesh: str = "fsaverage5",
    views: tuple[str, ...] = ("left", "right", "dorsal"),
    cmap: str = "fire",
    norm_percentile: int = 99,
    vmin: float | None = 0.5,
) -> np.ndarray:
    if isinstance(run, MultiModalRun):
        return render_multimodal_brain_panel_image(
            run,
            timestep=timestep,
            mesh=mesh,
            views=views,
            norm_percentile=norm_percentile,
        )
    return render_brain_panel_image(
        run.preds,
        timestep=timestep,
        mesh=mesh,
        views=views,
        cmap=cmap,
        norm_percentile=norm_percentile,
        vmin=vmin,
    )


def render_run_panel_bytes(
    run: PredictionRun,
    *,
    timestep: int,
    mesh: str = "fsaverage5",
    views: tuple[str, ...] = ("left", "right", "dorsal"),
    cmap: str = "fire",
    norm_percentile: int = 99,
    vmin: float | None = 0.5,
    image_format: str = "JPEG",
    quality: int = 84,
) -> bytes:
    image = Image.fromarray(
        render_run_panel_image(
            run,
            timestep=timestep,
            mesh=mesh,
            views=views,
            cmap=cmap,
            norm_percentile=norm_percentile,
            vmin=vmin,
        )
    )
    buffer = io.BytesIO()
    save_kwargs: dict[str, tp.Any] = {}
    if image_format.upper() == "JPEG":
        image = image.convert("RGB")
        save_kwargs.update({"quality": quality, "optimize": True})
    image.save(buffer, format=image_format.upper(), **save_kwargs)
    return buffer.getvalue()


def render_prediction_mosaic(
    run: PredictionRun,
    *,
    max_timesteps: int = 6,
    mesh: str = "fsaverage5",
    show_stimuli: bool = True,
) -> plt.Figure:
    """Render a compact multi-timestep figure for the dashboard."""
    n_timesteps = min(max_timesteps, len(run.preds))
    if n_timesteps < 1:
        raise ValueError("Cannot render a mosaic for an empty prediction run.")
    allow_frames = bool(
        show_stimuli and run.segments and any(has_video(seg) for seg in run.segments[:n_timesteps])
    )
    n_rows = 2 if allow_frames else 1
    row_heights = [1.2, 0.9] if allow_frames else [1.0]
    fig, axes = plt.subplots(
        n_rows,
        n_timesteps,
        figsize=(3.15 * n_timesteps, 2.85 * n_rows),
        squeeze=False,
        gridspec_kw={"height_ratios": row_heights},
    )
    for idx in range(n_timesteps):
        brain_img = render_run_panel_image(
            run,
            timestep=idx,
            mesh=mesh,
            vmin=0.5,
        )
        brain_ax = axes[0, idx]
        brain_ax.imshow(brain_img)
        brain_ax.axis("off")
        brain_ax.set_title(f"t={idx}s", fontsize=10, pad=6)

        if allow_frames:
            stim_ax = axes[1, idx]
            stim_ax.axis("off")
            clip = get_clip(run.segments[idx]) if idx < len(run.segments) else None
            if clip is not None:
                try:
                    sample_time = min(max(clip.duration / 2, 0), max(clip.duration - 1e-3, 0))
                    stim_ax.imshow(clip.get_frame(sample_time))
                finally:
                    clip.close()
            text = get_segment_text(run.segments[idx]) if idx < len(run.segments) else None
            if text:
                trimmed = text.strip()
                if len(trimmed) > 64:
                    trimmed = trimmed[:61] + "..."
                stim_ax.text(
                    0.5,
                    -0.08,
                    trimmed,
                    ha="center",
                    va="top",
                    transform=stim_ax.transAxes,
                    fontsize=8,
                    wrap=True,
                )

    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.07, wspace=0.04, hspace=0.18)
    return fig


def render_brain_panel_image(
    preds: np.ndarray,
    *,
    timestep: int,
    mesh: str = "fsaverage5",
    views: tuple[str, ...] = ("left", "right", "dorsal"),
    cmap: str = "fire",
    norm_percentile: int = 99,
    vmin: float | None = 0.5,
) -> np.ndarray:
    """Render one timestep as a compact RGB image for mosaics and synced playback."""
    if timestep < 0 or timestep >= len(preds):
        raise IndexError(f"Invalid timestep {timestep} for predictions of length {len(preds)}.")
    plotter = get_pyvista_plotter(mesh)
    signal = preds[timestep]
    if norm_percentile is not None:
        signal = normalize_signal_for_display(signal, percentile=norm_percentile)
    fig, axes = plt.subplots(
        1,
        len(views),
        figsize=(2.2 * len(views), 2.1),
        squeeze=False,
    )
    flat_axes = list(axes.flatten())
    plotter.plot_surf(
        signal,
        axes=flat_axes,
        views=list(views),
        cmap=cmap,
        norm_percentile=None,
        vmin=vmin,
    )
    for ax in flat_axes:
        ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.01, hspace=0.01)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)
    buffer.seek(0)
    return np.array(Image.open(buffer).convert("RGB"))


def render_brain_panel_bytes(
    preds: np.ndarray,
    *,
    timestep: int,
    mesh: str = "fsaverage5",
    views: tuple[str, ...] = ("left", "right", "dorsal"),
    cmap: str = "fire",
    norm_percentile: int = 99,
    vmin: float | None = 0.5,
    image_format: str = "JPEG",
    quality: int = 84,
) -> bytes:
    """Render one timestep to bytes for browser playback widgets."""
    image = Image.fromarray(
        render_brain_panel_image(
            preds,
            timestep=timestep,
            mesh=mesh,
            views=views,
            cmap=cmap,
            norm_percentile=norm_percentile,
            vmin=vmin,
        )
    )
    buffer = io.BytesIO()
    save_kwargs: dict[str, tp.Any] = {}
    if image_format.upper() == "JPEG":
        image = image.convert("RGB")
        save_kwargs.update({"quality": quality, "optimize": True})
    image.save(buffer, format=image_format.upper(), **save_kwargs)
    return buffer.getvalue()


def _annotate_frame_with_timestep(image: Image.Image, timestep: int) -> Image.Image:
    """Overlay a compact timestep badge in the lower-left corner."""
    frame = image.convert("RGBA")
    draw = ImageDraw.Draw(frame, "RGBA")
    font = ImageFont.load_default()
    label = f"t={timestep}"
    bbox = draw.textbbox((0, 0), label, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    pad_x = max(8, frame.width // 110)
    pad_y = max(5, frame.height // 160)
    margin_x = max(10, frame.width // 36)
    margin_y = max(10, frame.height // 30)
    x = margin_x
    y = max(0, frame.height - margin_y - text_height - (pad_y * 2))
    draw.rounded_rectangle(
        (
            x - pad_x,
            y - pad_y,
            x + text_width + pad_x,
            y + text_height + pad_y,
        ),
        radius=max(8, frame.height // 80),
        fill=(15, 23, 34, 190),
        outline=(255, 255, 255, 42),
        width=1,
    )
    draw.text((x, y), label, font=font, fill=(255, 248, 242, 255))
    return frame.convert("RGB")


def render_prediction_gif(
    run: PredictionRun,
    *,
    mesh: str = "fsaverage5",
    max_frames: int = 72,
    vmin: float | None = 0.5,
) -> bytes:
    """Render a looping animated GIF of predicted brain activity."""
    indices = select_animation_indices(len(run.preds), max_frames=max_frames)
    if not indices:
        raise ValueError("Cannot render an animation for an empty prediction run.")
    frames = [
        _annotate_frame_with_timestep(
            Image.fromarray(
                render_run_panel_image(
                    run,
                    timestep=idx,
                    mesh=mesh,
                    vmin=vmin,
                )
            ),
            idx,
        ).convert("P", palette=Image.ADAPTIVE)
        for idx in indices
    ]
    timeline = collect_timestep_metadata(run)
    durations = [
        int(max(160, min(1200, round(float(timeline[idx]["duration"]) * 1000))))
        for idx in indices
    ]
    buffer = io.BytesIO()
    frames[0].save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        disposal=2,
    )
    return buffer.getvalue()


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
        signal = normalize_signal_for_display(signal, percentile=norm_percentile)
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


def _cmap_to_plotly_colorscale(cmap_name: str = "fire", n: int = 12) -> list[list[tp.Any]]:
    cmap = get_cmap(cmap_name)
    scale: list[list[tp.Any]] = []
    for idx, value in enumerate(np.linspace(0, 1, n)):
        rgba = cmap(value)
        rgb = tuple(int(channel * 255) for channel in rgba[:3])
        scale.append([round(idx / max(n - 1, 1), 6), f"rgb{rgb}"])
    return scale


def render_animated_brain_3d_html(
    run: PredictionRun,
    *,
    mesh: str = "fsaverage5",
    max_frames: int = 30,
    norm_percentile: int = 99,
    vmin: float | None = 0.5,
    width: int = 980,
    height: int = 760,
) -> str:
    """Render a rotatable 3D brain animation with play/pause controls."""
    from plotly.offline import get_plotlyjs

    indices = select_animation_indices(len(run.preds), max_frames=max_frames)
    if not indices:
        raise ValueError("Cannot render a 3D animation for an empty prediction run.")

    plotter = get_pyvista_plotter(mesh)
    mesh_data = plotter._mesh["both"]
    coords = np.round(mesh_data["coords"], 3)
    faces = mesh_data["faces"].astype(int)
    timeline = collect_timestep_metadata(run)
    frames: list[dict[str, tp.Any]] = []
    reports = build_timestep_reports(run, indices=indices) if not isinstance(run, MultiModalRun) else None
    for frame_offset, idx in enumerate(indices):
        if isinstance(run, MultiModalRun):
            multimodal_surface = _get_multimodal_surface_render_data(
                run,
                timestep=idx,
                mesh=mesh,
                norm_percentile=norm_percentile,
            )
            if multimodal_surface is None:
                _, _, _, colors = _get_surface_render_data(
                    run.preds[idx],
                    mesh=mesh,
                    norm_percentile=norm_percentile,
                    cmap="fire",
                    vmin=vmin,
                )
                report = {
                    "text": timeline[idx]["text"],
                    "summary": "Fusion multimodale disponible, mais une seule famille de modalite contribue visiblement a ce timestep.",
                    "zone": "Fusion multimodale",
                    "valence": "indeterminee",
                }
            else:
                _, _, _, colors, channel_labels, matched_indices = multimodal_surface
                label_text = ", ".join(MULTIMODAL_CHANNEL_LABELS[label] for label in channel_labels)
                matched_text = ", ".join(
                    f"{MULTIMODAL_SOURCE_LABELS.get(modality, modality)} t{matched_idx}"
                    for modality, matched_idx in matched_indices.items()
                )
                report = {
                    "text": timeline[idx]["text"],
                    "summary": f"Overlay multimodal {label_text}. Alignement courant: {matched_text}.",
                    "zone": "Fusion multimodale",
                    "valence": "indeterminee",
                }
        else:
            report = tp.cast(list[dict[str, tp.Any]], reports)[frame_offset]
            _, _, _, colors = _get_surface_render_data(
                run.preds[idx],
                mesh=mesh,
                norm_percentile=norm_percentile,
                cmap="fire",
                vmin=vmin,
            )
        frames.append(
            {
                "index": idx,
                "start": timeline[idx]["start"],
                "duration": timeline[idx]["duration"],
                "text": report["text"],
                "summary": report["summary"],
                "zone": report["zone"],
                "valence": report["valence"],
                "vertexcolor": np.round(colors * 255).astype(int).tolist(),
            }
        )

    payload = {
        "x": coords[:, 0].tolist(),
        "y": coords[:, 1].tolist(),
        "z": coords[:, 2].tolist(),
        "i": faces[:, 0].tolist(),
        "j": faces[:, 1].tolist(),
        "k": faces[:, 2].tolist(),
        "frames": frames,
        "frameDurationMs": max(
            180,
            int(
                np.median(
                    [
                        max(160, min(1200, round(float(frame["duration"]) * 1000)))
                        for frame in frames
                    ]
                )
            ),
        ),
        "height": height,
    }
    payload_json = json.dumps(payload)
    plotly_js = get_plotlyjs()

    return f"""
    <div style="font-family: ui-sans-serif, system-ui, sans-serif; color: #171717;">
      <style>
        .brain3d-wrap {{
          border: 1px solid rgba(23, 23, 23, 0.10);
          border-radius: 18px;
          padding: 14px;
          background: rgba(255, 250, 242, 0.94);
          box-shadow: 0 8px 20px rgba(23, 23, 23, 0.05);
        }}
        .brain3d-toolbar {{
          display: flex;
          justify-content: space-between;
          gap: 14px;
          flex-wrap: wrap;
          margin-bottom: 12px;
          align-items: center;
        }}
        .brain3d-controls {{
          display: flex;
          gap: 10px;
          align-items: center;
          flex-wrap: wrap;
        }}
        .brain3d-btn {{
          border: 1px solid rgba(23, 23, 23, 0.12);
          border-radius: 999px;
          background: white;
          padding: 8px 14px;
          cursor: pointer;
          transition: transform 120ms ease, box-shadow 120ms ease, background 120ms ease;
          box-shadow: 0 4px 10px rgba(23, 23, 23, 0.04);
        }}
        .brain3d-btn:hover {{
          transform: translateY(-1px);
          box-shadow: 0 8px 18px rgba(23, 23, 23, 0.09);
        }}
        .brain3d-badge {{
          display: inline-flex;
          align-items: center;
          gap: 8px;
          border-radius: 999px;
          padding: 8px 12px;
          font-size: 12px;
          font-weight: 600;
          letter-spacing: 0.02em;
          color: #74341a;
          background: rgba(183, 67, 22, 0.10);
          border: 1px solid rgba(183, 67, 22, 0.15);
        }}
        .brain3d-badge-live {{
          color: #8d2500;
          background: linear-gradient(135deg, rgba(255, 155, 92, 0.28), rgba(183, 67, 22, 0.14));
          box-shadow: 0 0 0 1px rgba(183, 67, 22, 0.07), 0 8px 20px rgba(183, 67, 22, 0.12);
        }}
        .brain3d-badge-dot {{
          width: 8px;
          height: 8px;
          border-radius: 999px;
          background: #c84f1c;
          box-shadow: 0 0 0 0 rgba(200, 79, 28, 0.45);
          animation: brain3d-pulse 1.6s infinite;
        }}
        .brain3d-meta {{
          color: #665f57;
          font-size: 13px;
          line-height: 1.5;
        }}
        .brain3d-progress {{
          width: 100%;
          height: 8px;
          border-radius: 999px;
          background: rgba(23, 23, 23, 0.08);
          overflow: hidden;
          margin-bottom: 12px;
        }}
        .brain3d-progress-fill {{
          height: 100%;
          width: 0%;
          border-radius: inherit;
          background: linear-gradient(90deg, #ffbf80, #b74316);
          transition: width 160ms linear;
        }}
        @keyframes brain3d-pulse {{
          0% {{ box-shadow: 0 0 0 0 rgba(200, 79, 28, 0.45); }}
          70% {{ box-shadow: 0 0 0 10px rgba(200, 79, 28, 0.0); }}
          100% {{ box-shadow: 0 0 0 0 rgba(200, 79, 28, 0.0); }}
        }}
      </style>
      <div class="brain3d-wrap">
        <div class="brain3d-toolbar">
          <div class="brain3d-controls">
            <button id="brain3d-play" class="brain3d-btn">Play</button>
            <button id="brain3d-pause" class="brain3d-btn">Pause</button>
            <div id="brain3d-status" class="brain3d-badge">
              <span class="brain3d-badge-dot"></span>
              Auto-play actif
            </div>
          </div>
          <div id="brain3d-label" class="brain3d-meta"></div>
        </div>
        <div class="brain3d-progress">
          <div id="brain3d-progress-fill" class="brain3d-progress-fill"></div>
        </div>
        <div id="brain3d-plot" style="width:100%; height:{height}px;"></div>
        <div id="brain3d-summary" class="brain3d-meta"></div>
      </div>
      <script>{plotly_js}</script>
      <script>
        const payload = {payload_json};
        const frames = payload.frames;
        const plotDiv = document.getElementById("brain3d-plot");
        const label = document.getElementById("brain3d-label");
        const summary = document.getElementById("brain3d-summary");
        const status = document.getElementById("brain3d-status");
        const progressFill = document.getElementById("brain3d-progress-fill");
        let currentIndex = 0;
        let timer = null;
        let orbitTick = 0;

        const trace = {{
          type: "mesh3d",
          x: payload.x,
          y: payload.y,
          z: payload.z,
          i: payload.i,
          j: payload.j,
          k: payload.k,
          vertexcolor: frames[0].vertexcolor,
          flatshading: false,
          hoverinfo: "skip",
          showscale: false,
          lighting: {{ambient: 0.45, diffuse: 0.65, specular: 0.15, roughness: 0.7}},
          lightposition: {{x: -100, y: 0, z: 200}},
        }};
        const layout = {{
          margin: {{l: 0, r: 0, b: 0, t: 0}},
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
          scene: {{
            bgcolor: "rgba(0,0,0,0)",
            xaxis: {{visible: false}},
            yaxis: {{visible: false}},
            zaxis: {{visible: false}},
            aspectmode: "data",
            camera: {{eye: {{x: -1.55, y: 0.08, z: 0.82}}}},
          }},
          uirevision: "brain3d-fixed",
        }};

        function cameraEye(stepSeed) {{
          const angle = -1.18 + stepSeed * 0.24;
          const radius = 1.55;
          return {{
            x: -Math.cos(angle) * radius,
            y: Math.sin(angle) * radius * 0.45,
            z: 0.82 + 0.07 * Math.sin(angle * 0.55),
          }};
        }}

        function setPlaying(isPlaying) {{
          status.innerHTML = isPlaying
            ? '<span class="brain3d-badge-dot"></span>Auto-play actif'
            : '<span class="brain3d-badge-dot"></span>Pause';
          status.classList.toggle("brain3d-badge-live", isPlaying);
        }}

        function renderFrame(index) {{
          currentIndex = ((index % frames.length) + frames.length) % frames.length;
          const frame = frames[currentIndex];
          orbitTick += 1;
          Plotly.restyle(plotDiv, {{vertexcolor: [frame.vertexcolor]}}, [0]);
          Plotly.relayout(plotDiv, {{"scene.camera.eye": cameraEye(orbitTick)}});
          label.textContent = `Timestep ${{frame.index + 1}} / ${{frames.length}} | ${{Number(frame.start).toFixed(2)}}s`;
          progressFill.style.width = `${{((currentIndex + 1) / Math.max(frames.length, 1)) * 100}}%`;
          const parts = [frame.summary, `Zone probable: ${{frame.zone}}`, `Valence: ${{frame.valence}}`];
          if (frame.text) {{
            parts.push(`Texte: ${{frame.text}}`);
          }}
          summary.textContent = parts.join(" | ");
        }}

        function play() {{
          if (timer !== null) return;
          setPlaying(true);
          timer = window.setInterval(() => renderFrame(currentIndex + 1), payload.frameDurationMs);
        }}

        function pause() {{
          if (timer !== null) {{
            window.clearInterval(timer);
            timer = null;
          }}
          setPlaying(false);
        }}

        document.getElementById("brain3d-play").addEventListener("click", play);
        document.getElementById("brain3d-pause").addEventListener("click", pause);
        Plotly.newPlot(plotDiv, [trace], layout, {{
          displayModeBar: true,
          responsive: true,
          scrollZoom: true,
        }}).then(() => {{
          renderFrame(0);
          play();
        }});
      </script>
    </div>
    """


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
    if float(weights.sum()) <= 1e-12:
        weighted_center = focus_coords.mean(axis=0)
    else:
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


def collect_timestep_metadata(run: PredictionRun) -> list[dict[str, tp.Any]]:
    """Collect lightweight timing/text metadata for synced playback."""
    out: list[dict[str, tp.Any]] = []
    for idx in range(len(run.preds)):
        segment = run.segments[idx] if idx < len(run.segments) else None
        start = getattr(segment, "start", None)
        duration = getattr(segment, "duration", None)
        if start is None:
            start = float(idx)
        if duration is None:
            if idx + 1 < len(run.segments):
                next_start = getattr(run.segments[idx + 1], "start", None)
                duration = (float(next_start) - float(start)) if next_start is not None else 1.0
            else:
                duration = 1.0
        duration = max(float(duration), 1e-3)
        out.append(
            {
                "index": idx,
                "start": float(start),
                "duration": duration,
                "text": get_segment_text(segment) if segment is not None else "",
            }
        )
    return out


def get_segment_text(segment: tp.Any) -> str:
    """Best-effort text extraction that tolerates lightweight fake segments in tests."""
    if segment is None:
        return ""
    try:
        return get_text(segment)
    except Exception:
        return ""


def build_explainability_report(
    run: PredictionRun,
    *,
    timestep: int,
    duration: float | None = None,
    description: dict[str, tp.Any] | None = None,
) -> dict[str, tp.Any]:
    """Build a source-backed explanation for one run and timestep."""
    description = description or describe_timestep(run.preds, timestep=timestep)
    channels = list_run_channels(run)
    channel_text = ", ".join(channels)
    shared_section = {
        "title": "Ce que TRIBE v2 fait ici",
        "bullets": [
            "Le papier et le billet Meta presentent TRIBE v2 comme un modele tri-modal vision/audio/langage qui predit une reponse corticale, pas une pensee decodee ni un diagnostic.",
            "La publication indique plus de 1 000 heures de fMRI sur 720 sujets; le billet parle de plus de 700 volontaires sains et met en avant un gain de resolution d'environ 70x par rapport a des modeles similaires anterieurs.",
            f"Pour cette entree, le pipeline exploite surtout: {channel_text}.",
        ],
    }
    reading_bullets = [
        "La surface affiche `fsaverage5`, donc un cortex de reference partage, pas un cerveau individuel.",
        "Les couleurs chaudes indiquent ici une reponse predite plus forte a ce timestep.",
        description["summary"],
    ]
    if duration is not None:
        reading_bullets.append(
            f"Le timestep {timestep} couvre environ {duration:.2f}s du stimulus prepare."
        )
    reading_section = {
        "title": "Comment lire cette carte",
        "bullets": reading_bullets,
    }

    if run.input_kind == "video":
        modality_section = {
            "title": "Pourquoi cette video est interpretable",
            "bullets": [
                "Le notebook officiel decompose une video en indices visuels, audio et textuels alignes dans le temps, puis le modele predit une carte corticale a chaque seconde environ.",
                "Le notebook cite des extracteurs distincts pour la vision, l'audio et le texte, ensuite fusionnes par un Transformer commun avant la prediction sur le cortex.",
                "Sur une video, les changements de carte viennent souvent d'un melange entre changement de scene, variation sonore et contenu verbal au meme timestep.",
            ],
        }
        limits_section = {
            "title": "Limites",
            "bullets": [
                "Cette carte estime une reponse moyenne plausible selon le modele, pas l'activite mesuree d'un sujet reel.",
                "Une zone plus chaude n'implique pas a elle seule une interpretation cognitive unique; plusieurs indices sensoriels peuvent contribuer en meme temps.",
            ],
        }
    elif run.input_kind == "audio":
        has_text_alignment = "texte aligne" in channels
        modality_section = {
            "title": "Pourquoi cet audio est interpretable",
            "bullets": [
                "TRIBE v2 inclut explicitement l'audition parmi ses trois modalites; sur un fichier audio, la carte reflete surtout la structure sonore et, si disponible, les mots alignes.",
                (
                    "Cette execution contient aussi un canal textuel aligne, donc les mots peuvent aider a expliquer certaines variations temporelles."
                    if has_text_alignment
                    else "Cette execution fonctionne sans transcription obligatoire; la prediction repose alors principalement sur le contenu acoustique."
                ),
                "Le plus utile est de rapprocher chaque pic temporel d'un changement de rythme, d'intonation, de timbre ou de parole.",
            ],
        }
        limits_section = {
            "title": "Limites",
            "bullets": [
                "Sans transcription alignee, l'explication reste surtout acoustique et moins semantique.",
                "Le modele simule une reponse corticale probable; il ne localise pas de maniere definitive une fonction cerebrale humaine.",
            ],
        }
    elif run.input_kind == "text":
        direct_text = run.raw_text is not None and "texte aligne" in channels and "audio" not in channels
        modality_section = {
            "title": "Pourquoi ce texte est interpretable",
            "bullets": [
                "Le notebook officiel indique que la branche texte passe par le codeur langage du modele et que les predictions sont emises sur le cortex au fil d'une timeline alignee.",
                (
                    "Dans le notebook officiel, le texte est converti en parole puis retranscrit pour retrouver des timings de mots, car le modele a ete entraine sur des stimuli naturalistes audio/video."
                    if not direct_text
                    else "Dans ce fork, le mode `Texte direct` cree des timings synthetiques par mot pour rendre l'usage local plus simple; c'est un raccourci pratique, distinct de la demarche exacte du notebook."
                ),
                "Quand vous lisez la carte, reliez les changements de signal a l'arrivee de nouveaux mots, a la densite du contexte et aux ruptures de phrase.",
            ],
        }
        limits_section = {
            "title": "Limites",
            "bullets": [
                "En mode texte direct, les timings sont artificiels: ils servent a piloter le modele localement, pas a reproduire une presentation experimentale reelle.",
                "La carte reste une prediction de reponse cerebrale plausible a partir du texte, pas une preuve qu'une region code uniquement ce contenu semantique.",
            ],
        }
    else:
        modality_section = {
            "title": "Pourquoi cette image est interpretable",
            "bullets": [
                "Le billet Meta insiste sur la capacite du modele a predire des reponses a ce que le cerveau voit; cette branche de votre fork reutilise donc la voie visuelle du modele.",
                "Techniquement, une image statique est convertie ici en court clip video silencieux pour alimenter le pipeline du projet sans changer les poids du modele.",
                "Si la carte reste stable entre plusieurs timesteps, cela signifie surtout que le contenu visuel fixe domine; les petites variations restantes viennent du fenetrage temporel du pipeline, pas d'une nouvelle scene.",
            ],
        }
        limits_section = {
            "title": "Limites",
            "bullets": [
                "Le support des images statiques est un ajout de ce fork; ce n'est pas le protocole principal mis en avant dans le notebook officiel.",
                "Comme l'image est repetee dans le temps, il faut interpreter les differences temporelles avec prudence et se concentrer surtout sur le patron spatial global.",
            ],
        }

    return {
        "title": f"Explication {run.input_kind}",
        "sections": [shared_section, modality_section, reading_section, limits_section],
        "sources": EXPLAINABILITY_SOURCES,
    }


def build_image_comparison_guide(
    run: ImageComparisonRun,
    *,
    timestep: int,
    descriptions: list[dict[str, tp.Any]] | None = None,
) -> dict[str, tp.Any]:
    """Explain how to compare two static-image predictions."""
    descriptions = descriptions or [
        describe_timestep(item.preds, timestep=timestep) for item in run.runs
    ]
    bullets = [
        "Les deux images passent par le meme pipeline visuel et le meme maillage cortical, donc la comparaison est surtout spatiale.",
        "Comparez d'abord la lateralite, l'axe antero-posterieur, l'axe dorso-ventral et la concentration du top 1% du signal.",
        "Comme chaque image est transformee en clip silencieux statique dans ce fork, les differences entre colonnes viennent du contenu visuel, pas d'un changement de son ou de scene.",
    ]
    if len(descriptions) >= 2:
        left = descriptions[0]
        right = descriptions[1]
        bullets.append(
            "Au timestep observe, l'image 1 est surtout "
            f"{left['laterality']}, {left['antero_posterior']} et {left['dorso_ventral']}; "
            "l'image 2 est surtout "
            f"{right['laterality']}, {right['antero_posterior']} et {right['dorso_ventral']}."
        )
    return {
        "title": "Comment comparer ces images",
        "bullets": bullets,
        "sources": EXPLAINABILITY_SOURCES,
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


def build_browser_media_proxy(
    *,
    source_path: str | Path,
    output_folder: str | Path,
) -> Path:
    """Transcode source media to a browser-friendly proxy for dashboard playback."""
    source_path = Path(source_path)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    suffix = source_path.suffix.lower()
    stat = source_path.stat()
    safe_stem = "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in source_path.stem)
    fingerprint = f"{stat.st_size}_{stat.st_mtime_ns}"
    ffmpeg_bin = resolve_ffmpeg()

    if suffix in {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}:
        video_encoder = resolve_video_encoder(ffmpeg_bin)
        output_path = output_folder / f"{safe_stem}_{fingerprint}_sync.mp4"
        if output_path.exists():
            return output_path
        cmd = [
            ffmpeg_bin,
            "-y",
            "-i",
            str(source_path),
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
            "-c:v",
            video_encoder,
        ]
        if video_encoder == "libx264":
            cmd.extend(["-crf", "20"])
        elif video_encoder == "libopenh264":
            cmd.extend(["-b:v", "4M"])
        else:
            cmd.extend(["-q:v", "3"])
        cmd.extend(
            [
                "-c:a",
                "aac",
                "-movflags",
                "+faststart",
                str(output_path),
            ]
        )
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return output_path

    if suffix in {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}:
        output_path = output_folder / f"{safe_stem}_{fingerprint}_sync.wav"
        if output_path.exists():
            return output_path
        cmd = [
            ffmpeg_bin,
            "-y",
            "-i",
            str(source_path),
            "-vn",
            "-ac",
            "2",
            "-ar",
            "44100",
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ]
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return output_path

    return source_path


def write_text_to_temp_file(text: str, folder: str | Path) -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix="tribev2_text_", suffix=".txt", dir=folder)
    os.close(fd)
    path = Path(temp_path)
    path.write_text(text, encoding="utf-8")
    return path
