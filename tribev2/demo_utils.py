# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""TribeModel for inference and utilities for building event DataFrames."""

import gc
import logging
import os
import re
import typing as tp
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
import pydantic
import requests
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
import yaml
from einops import rearrange
from exca import ConfDict, TaskInfra
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    logger.addHandler(_handler)
from neuralset.events.transforms import (
    AddContextToWords,
    AddSentenceToWords,
    AddText,
    ChunkEvents,
    ExtractAudioFromVideo,
    RemoveMissing,
)
from neuralset.events.utils import standardize_events

from tribev2.eventstransforms import ExtractWordsFromAudio
from tribev2.main import TribeExperiment

VALID_SUFFIXES: dict[str, set[str]] = {
    "text_path": {".txt"},
    "audio_path": {".wav", ".mp3", ".flac", ".ogg"},
    "video_path": {".mp4", ".avi", ".mkv", ".mov", ".webm"},
}
_HIDDEN_STATE_CPU_OFFLOAD_BYTES = 768 * 1024 * 1024
_MEMORY_SAFE_PATCHED = False
_OFFLOAD_LOGGED_LABELS: set[str] = set()


class _PortableUnsafeLoader(yaml.UnsafeLoader):
    """UnsafeLoader variant that can deserialize POSIX paths on Windows."""


def _construct_posix_path(
    loader: yaml.UnsafeLoader, node: yaml.SequenceNode
) -> str:
    parts = loader.construct_sequence(node)
    return str(PurePosixPath(*parts))


_PortableUnsafeLoader.add_constructor(
    "tag:yaml.org,2002:python/object/apply:pathlib.PosixPath",
    _construct_posix_path,
)


def _cuda_runtime_supported() -> bool:
    """Return True only when the installed torch build can run this GPU."""
    if not torch.cuda.is_available():
        return False
    try:
        major, minor = torch.cuda.get_device_capability()
        supported_arches = {
            arch.removeprefix("sm_")
            for arch in torch.cuda.get_arch_list()
            if arch.startswith("sm_")
        }
        return f"{major}{minor}" in supported_arches
    except Exception:
        return False


def _estimate_tensor_bytes(tensors: tp.Iterable[torch.Tensor]) -> int:
    return sum(int(tensor.numel() * tensor.element_size()) for tensor in tensors)


def _concat_hidden_states_cpu(states: tp.Sequence[torch.Tensor]) -> torch.Tensor:
    cpu_states = [
        state.detach().to(device="cpu", copy=True).unsqueeze(1) for state in states
    ]
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return torch.cat(cpu_states, axis=1)


def _dedupe_items_by_uid(
    items: tp.Sequence[tp.Any],
    item_uid: tp.Callable[[tp.Any], str],
) -> tuple[list[tp.Any], int]:
    deduped: list[tp.Any] = []
    seen: set[str] = set()
    duplicate_count = 0
    for item in items:
        uid = item_uid(item)
        if uid in seen:
            duplicate_count += 1
            continue
        seen.add(uid)
        deduped.append(item)
    return deduped, duplicate_count


def _concat_hidden_states_memory_safe(
    states: tp.Sequence[torch.Tensor],
    *,
    label: str,
) -> tuple[torch.Tensor, bool]:
    state_list = list(states)
    if not state_list:
        raise RuntimeError(f"Cannot concatenate empty {label}.")
    estimated_bytes = _estimate_tensor_bytes(state_list)
    should_offload = any(state.is_cuda for state in state_list) and (
        estimated_bytes >= _HIDDEN_STATE_CPU_OFFLOAD_BYTES
    )
    if should_offload:
        if label not in _OFFLOAD_LOGGED_LABELS:
            logger.info(
                "Offloading %s concat to CPU to avoid a %.2f GiB CUDA allocation spike.",
                label,
                estimated_bytes / (1024**3),
            )
            _OFFLOAD_LOGGED_LABELS.add(label)
        else:
            logger.debug(
                "Reusing CPU offload path for %s (%.2f GiB estimated).",
                label,
                estimated_bytes / (1024**3),
            )
        return _concat_hidden_states_cpu(state_list), True
    try:
        return torch.cat([state.unsqueeze(1) for state in state_list], axis=1), False
    except torch.OutOfMemoryError:
        logger.warning(
            "CUDA OOM during %s concat; retrying on CPU. "
            "A full Streamlit restart may still be needed to defragment memory.",
            label,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return _concat_hidden_states_cpu(state_list), True


def _apply_memory_safe_extractor_patches() -> None:
    global _MEMORY_SAFE_PATCHED
    if _MEMORY_SAFE_PATCHED:
        return

    from exca import map as exca_map
    from neuralset.extractors import image as ns_image
    from neuralset.extractors import video as ns_video

    original_call_and_store = exca_map.MapInfra._call_and_store

    def _patched_call_and_store(self, items, use_cache_dict: bool = True):
        item_list = list(items)
        imethod = getattr(self, "_infra_method", None)
        item_uid = getattr(imethod, "item_uid", None)
        if item_uid is not None and len(item_list) > 1:
            item_list, duplicate_count = _dedupe_items_by_uid(item_list, item_uid)
            if duplicate_count:
                logger.warning(
                    "Deduplicated %d extractor items with identical cache keys before compute.",
                    duplicate_count,
                )
        return original_call_and_store(self, item_list, use_cache_dict=use_cache_dict)

    def _patched_predict_hidden_states(
        self, images: np.ndarray, audio: np.ndarray | None = None
    ) -> torch.Tensor:
        pred = self.predict(images, audio)
        if "xclip" in self.model_name:
            is_mit = self.layer_type == "mit"
            pred = pred.mit_output if is_mit else pred.vision_model_output
        states = pred.hidden_states
        out, used_cpu_offload = _concat_hidden_states_memory_safe(
            states,
            label=f"video hidden states ({self.model_name})",
        )
        if "xclip" in self.model_name and not self.layer_type:
            out = out[[-1], ...]
        del states
        del pred
        if used_cpu_offload and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return out

    def _patched_extract_batched_latents(self, images: torch.Tensor) -> torch.Tensor:
        states = self._get_hidden_states(images)
        out, used_cpu_offload = _concat_hidden_states_memory_safe(
            states,
            label=f"image hidden states ({self.model_name})",
        )
        del states
        if used_cpu_offload and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return out

    exca_map.MapInfra._call_and_store = _patched_call_and_store
    ns_video._HFVideoModel.predict_hidden_states = _patched_predict_hidden_states
    ns_image.HuggingFaceImage._extract_batched_latents = (
        _patched_extract_batched_latents
    )
    _MEMORY_SAFE_PATCHED = True


def download_file(url: str, path: str | Path) -> Path:
    """Download a file from *url* and save it to *path*.

    Raises ``requests.HTTPError`` on non-2xx responses.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=128 * 1024):
                if chunk:
                    f.write(chunk)
    logger.info(f"Downloaded {url} -> {path}")
    return path


def get_audio_and_text_events(
    events: pd.DataFrame, audio_only: bool = False
) -> pd.DataFrame:
    """Run the audio/video-to-text pipeline on an events DataFrame.

    Extracts audio from video, chunks long clips, transcribes words, and
    attaches sentence/context annotations.  Set *audio_only* to ``True``
    to skip the transcription and text stages.
    """
    transforms = [
        ExtractAudioFromVideo(),
        ChunkEvents(event_type_to_chunk="Audio", max_duration=60, min_duration=30),
        ChunkEvents(event_type_to_chunk="Video", max_duration=60, min_duration=30),
    ]
    if not audio_only:
        transforms.extend(
            [
                ExtractWordsFromAudio(),
                AddText(),
                AddSentenceToWords(max_unmatched_ratio=0.05),
                AddContextToWords(
                    sentence_only=False, max_context_len=1024, split_field=""
                ),
                RemoveMissing(),
            ]
        )
    events = standardize_events(events)
    for transform in transforms:
        events = transform(events)
    return standardize_events(events)


def build_text_events_from_text(
    text: str,
    *,
    seconds_per_word: float = 0.45,
    max_context_words: int = 128,
    timeline: str = "default",
    subject: str = "default",
    language: str | None = None,
) -> pd.DataFrame:
    """Create Word events directly from raw text without TTS or ASR."""
    normalized = text.replace("\r\n", "\n").strip()
    if not normalized:
        raise ValueError("Text input is empty.")
    if seconds_per_word <= 0:
        raise ValueError("seconds_per_word must be strictly positive.")
    if max_context_words < 1:
        raise ValueError("max_context_words must be at least 1.")

    sentence_splitter = re.compile(r"(?<=[.!?])(?:\s+|\n+)|\n+")
    word_pattern = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*", flags=re.UNICODE)
    sentences = [
        sentence.strip()
        for sentence in sentence_splitter.split(normalized)
        if sentence.strip()
    ]
    if not sentences:
        sentences = [normalized]

    rows: list[dict[str, tp.Any]] = []
    context_words: list[str] = []
    current_start = 0.0
    for sequence_id, sentence in enumerate(sentences):
        words = word_pattern.findall(sentence)
        if not words:
            continue
        for word in words:
            context_words.append(word)
            row = {
                "type": "Word",
                "text": word,
                "start": current_start,
                "duration": seconds_per_word,
                "sequence_id": sequence_id,
                "sentence": sentence,
                "context": " ".join(context_words[-max_context_words:]),
                "timeline": timeline,
                "subject": subject,
            }
            if language is not None:
                row["language"] = language
            rows.append(row)
            current_start += seconds_per_word

    if not rows:
        raise ValueError("No valid words were found in the provided text.")
    return standardize_events(pd.DataFrame(rows))


class TextToEvents(pydantic.BaseModel):
    """Convert raw text to an events DataFrame via text-to-speech + transcription.

    The text is synthesised to audio with gTTS, then processed through
    :func:`get_audio_and_text_events` to obtain word-level events.
    """

    text: str
    infra: TaskInfra = TaskInfra()

    def model_post_init(self, __context: tp.Any) -> None:
        if self.infra.folder is None:
            raise ValueError("A folder must be specified to save the audio file.")

    @infra.apply()
    def get_events(self) -> pd.DataFrame:
        from gtts import gTTS
        from langdetect import detect

        audio_path = Path(self.infra.uid_folder(create=True)) / "audio.mp3"
        lang = detect(self.text)
        tts = gTTS(self.text, lang=lang)
        tts.save(str(audio_path))
        logger.info(f"Wrote TTS audio to {audio_path}")

        audio_event = {
            "type": "Audio",
            "filepath": str(audio_path),
            "start": 0,
            "timeline": "default",
            "subject": "default",
        }
        return get_audio_and_text_events(pd.DataFrame([audio_event]))


class TribeModel(TribeExperiment):
    """High-level inference wrapper around :class:`TribeExperiment`.

    Provides a simple ``from_pretrained`` / ``predict`` interface for
    generating fMRI-like brain-activity predictions from text, audio,
    or video inputs.

    Typical usage::

        model = TribeModel.from_pretrained("facebook/tribev2")
        events = model.get_events_dataframe(video_path="clip.mp4")
        preds, segments = model.predict(events)
    """

    cache_folder: str = "./cache"
    remove_empty_segments: bool = True

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_dir: str | Path,
        checkpoint_name: str = "best.ckpt",
        cache_folder: str | Path = None,
        cluster: str = None,
        device: str = "auto",
        config_update: dict | None = None,
    ) -> "TribeModel":
        """Load a trained model from a checkpoint directory or HuggingFace Hub repo.

        ``checkpoint_dir`` can be either a local path containing
        ``config.yaml`` and ``<checkpoint_name>``, or a HuggingFace Hub
        repo id (e.g. ``"facebook/tribev2"``).

        Parameters
        ----------
        checkpoint_dir:
            Local directory or HuggingFace Hub repo id that contains
            ``config.yaml`` and the checkpoint file.
        checkpoint_name:
            Filename of the checkpoint inside *checkpoint_dir*.
        cache_folder:
            Directory used to cache extracted features. Created if it
            does not exist.  Defaults to ``"./cache"`` when ``None``.
        cluster:
            Cluster backend forwarded to feature-extractor infra
            (``"auto"`` by default).
        device:
            Torch device string.  ``"auto"`` selects CUDA when available.
        config_update:
            Optional dictionary of config overrides applied after the
            YAML config is loaded.

        Returns
        -------
        TribeModel
            A ready-to-use model instance with weights loaded in eval mode.
        """
        _apply_memory_safe_extractor_patches()
        if cache_folder is not None:
            Path(cache_folder).mkdir(parents=True, exist_ok=True)
        if device == "auto":
            device = "cuda" if _cuda_runtime_supported() else "cpu"
        checkpoint_dir_str = str(checkpoint_dir)
        local_checkpoint_dir = Path(checkpoint_dir_str)
        if local_checkpoint_dir.exists():
            config_path = local_checkpoint_dir / "config.yaml"
            ckpt_path = local_checkpoint_dir / checkpoint_name
        else:
            from huggingface_hub import hf_hub_download

            # Keep Hugging Face repo ids POSIX-style on Windows; converting
            # through Path() turns "org/repo" into "org\\repo" and breaks
            # hf_hub_download validation.
            repo_id = checkpoint_dir_str.replace("\\", "/")
            config_path = hf_hub_download(repo_id, "config.yaml")
            ckpt_path = hf_hub_download(repo_id, checkpoint_name)
        with open(config_path, "r") as f:
            config = ConfDict(yaml.load(f, Loader=_PortableUnsafeLoader))
        for modality in ["text", "audio", "video"]:
            config[f"data.{modality}_feature.infra.folder"] = cache_folder
            config[f"data.{modality}_feature.infra.cluster"] = cluster
        for key in [
            "data.text_feature.device",
            "data.audio_feature.device",
            "data.image_feature.image.device",
            "data.video_feature.image.device",
        ]:
            if key in config:
                config[key] = device

        for param in [
            "infra.workdir",
            "data.study.infra_timelines",
            "data.neuro.infra",
            "data.image_feature.infra",
        ]:
            config.pop(param)
        config["data.study.path"] = "."
        config["average_subjects"] = True
        config["checkpoint_path"] = str(config["infra.folder"]) + f"/{checkpoint_name}"
        config["cache_folder"] = (
            str(cache_folder) if cache_folder is not None else "./cache"
        )
        if config_update is not None:
            config.update(config_update)
        xp = cls(**config)

        logger.info(f"Loading model from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True, mmap=True)
        build_args = ckpt["model_build_args"]
        state_dict = {
            k.removeprefix("model."): v for k, v in ckpt["state_dict"].items()
        }
        del ckpt

        model = xp.brain_model_config.build(**build_args)
        model.load_state_dict(state_dict, strict=True, assign=True)
        del state_dict
        model.to(device)
        model.eval()
        xp._model = model
        return xp

    def get_events_dataframe(
        self,
        text_path: str | None = None,
        audio_path: str | None = None,
        video_path: str | None = None,
        *,
        direct_text: bool = False,
        seconds_per_word: float = 0.45,
        max_context_words: int = 128,
    ) -> pd.DataFrame:
        """Build an events DataFrame from exactly one input source.

        Parameters
        ----------
        text_path:
            Path to a ``.txt`` file. The text is converted to speech, then
            transcribed back to produce word-level events.
        audio_path:
            Path to an audio file (``.wav``, ``.mp3``, ``.flac``, ``.ogg``).
        video_path:
            Path to a video file (``.mp4``, ``.avi``, ``.mkv``, ``.mov``,
            ``.webm``).
        direct_text:
            When ``True`` and ``text_path`` is provided, build word events
            directly from the text instead of using TTS + ASR.
        seconds_per_word:
            Synthetic duration assigned to each word when ``direct_text=True``.
        max_context_words:
            Number of trailing words kept in each word context when
            ``direct_text=True``.

        Returns
        -------
        pd.DataFrame
            Standardised events DataFrame with columns such as ``type``,
            ``filepath``, ``start``, ``duration``, ``timeline``, and
            ``subject``.

        Raises
        ------
        ValueError
            If zero or more than one path is provided, or if the file
            extension does not match the expected suffixes.
        FileNotFoundError
            If the specified file does not exist.
        """
        provided = {
            name: value
            for name, value in [
                ("text_path", text_path),
                ("audio_path", audio_path),
                ("video_path", video_path),
            ]
            if value is not None
        }
        if len(provided) != 1:
            raise ValueError(
                f"Exactly one of text_path, audio_path, video_path must be "
                f"provided, got: {list(provided.keys()) or 'none'}"
            )

        name, value = next(iter(provided.items()))
        path = Path(value)
        suffix = path.suffix.lower()
        if suffix not in VALID_SUFFIXES[name]:
            raise ValueError(
                f"{name} must end with one of {sorted(VALID_SUFFIXES[name])}, "
                f"got '{suffix}'"
            )
        if not path.is_file():
            raise FileNotFoundError(f"{name} does not exist: {path}")

        if text_path is not None:
            text = path.read_text(encoding="utf-8")
            if not text.strip():
                raise ValueError(f"Text file is empty: {path}")
            if direct_text:
                return build_text_events_from_text(
                    text,
                    seconds_per_word=seconds_per_word,
                    max_context_words=max_context_words,
                )
            return TextToEvents(
                text=text,
                infra={"folder": self.cache_folder, "mode": "retry"},
            ).get_events()

        event_type = "Audio" if audio_path is not None else "Video"
        event = {
            "type": event_type,
            "filepath": str(path),
            "start": 0,
            "timeline": "default",
            "subject": "default",
        }
        return get_audio_and_text_events(pd.DataFrame([event]))

    def get_text_events_dataframe(
        self,
        text: str,
        *,
        seconds_per_word: float = 0.45,
        max_context_words: int = 128,
        timeline: str = "default",
        subject: str = "default",
        language: str | None = None,
    ) -> pd.DataFrame:
        """Build direct text events without passing through TTS + ASR."""
        return build_text_events_from_text(
            text,
            seconds_per_word=seconds_per_word,
            max_context_words=max_context_words,
            timeline=timeline,
            subject=subject,
            language=language,
        )

    def predict(
        self, events: pd.DataFrame, verbose: bool = True
    ) -> tuple[np.ndarray, list]:
        """Run inference on an events DataFrame and return per-TR predictions.

        Each batch is split into segments of length ``data.TR``.  When
        ``remove_empty_segments`` is ``True`` (the default), segments that
        contain no events are discarded.

        Parameters
        ----------
        events:
            Events DataFrame, typically produced by
            :meth:`get_events_dataframe`.
        verbose:
            If ``True`` (default), display a ``tqdm`` progress bar.

        Returns
        -------
        preds : np.ndarray
            Array of shape ``(n_kept_segments, n_vertices)`` with the
            predicted brain activity.
        all_segments : list
            Corresponding segment objects aligned with *preds*.

        Raises
        ------
        RuntimeError
            If the model has not been loaded via :meth:`from_pretrained`.
        """
        if self._model is None:
            raise RuntimeError(
                "TribeModel must be instantiated via the .from_pretrained method"
            )
        model = self._model
        loader = self.data.get_loaders(events=events, split_to_build="all")["all"]

        preds, all_segments = [], []
        n_samples, n_kept = 0, 0
        with torch.inference_mode():
            for batch in tqdm(loader, disable=not verbose):
                batch = batch.to(model.device)
                batch_segments = []
                for segment in batch.segments:
                    for t in np.arange(0, segment.duration - 1e-2, self.data.TR):
                        batch_segments.append(
                            segment.copy(offset=t, duration=self.data.TR)
                        )
                if self.remove_empty_segments:
                    keep = np.array([len(s.ns_events) > 0 for s in batch_segments])
                else:
                    keep = np.ones(len(batch_segments), dtype=bool)
                n_kept += keep.sum()
                n_samples += len(batch_segments)
                batch_segments = [s for i, s in enumerate(batch_segments) if keep[i]]
                y_pred = model(batch).detach().cpu().numpy()
                y_pred = rearrange(y_pred, "b d t -> (b t) d")[keep]
                preds.append(y_pred)
                all_segments.extend(batch_segments)
        preds = np.concatenate(preds)
        if len(all_segments) != preds.shape[0]:
            raise ValueError(
                f"Number of samples: {preds.shape[0]} != {len(all_segments)}"
            )
        logger.info(
            "Predicted %d / %d segments (%.1f%% kept)",
            n_kept,
            n_samples,
            100.0 * n_kept / max(n_samples, 1),
        )
        return preds, all_segments
