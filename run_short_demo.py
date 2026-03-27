from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from tribev2.demo_utils import TribeModel, download_file, get_audio_and_text_events


def resolve_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    env_ffmpeg = Path(sys.executable).parent / "Library" / "bin" / "ffmpeg.exe"
    if env_ffmpeg.exists():
        return str(env_ffmpeg)
    raise FileNotFoundError("ffmpeg executable not found")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device used for TRIBE and the modality extractors.",
    )
    parser.add_argument(
        "--keep-audio",
        action="store_true",
        help="Keep audio in the short clip so the audio extractor also runs.",
    )
    parser.add_argument(
        "--transcribe",
        action="store_true",
        help="Run the text/transcription pipeline. Requires `uvx whisperx`.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache = Path("cache")
    cache.mkdir(exist_ok=True)

    source_video = cache / "sample_video.mp4"
    short_video = cache / "sample_video_4s.mp4"

    print("Downloading demo video...", flush=True)
    download_file(
        "https://download.blender.org/durian/trailer/sintel_trailer-480p.mp4",
        source_video,
    )

    print("Creating a short 4s clip...", flush=True)
    subprocess.run(
        [
            resolve_ffmpeg(),
            "-y",
            "-i",
            str(source_video),
            "-t",
            "4",
        ]
        + ([] if args.keep_audio else ["-an"])
        + [str(short_video)],
        check=True,
        capture_output=True,
        text=True,
    )

    print("Loading TRIBE v2...", flush=True)
    model = TribeModel.from_pretrained(
        "facebook/tribev2",
        cache_folder=cache,
        device=args.device,
        config_update={"data.num_workers": 0},
    )
    print(f"Model device: {model._model.device}", flush=True)

    print("Building events...", flush=True)
    events = pd.DataFrame(
        [
            {
                "type": "Video",
                "filepath": str(short_video),
                "start": 0,
                "timeline": "default",
                "subject": "default",
            }
        ]
    )
    df = get_audio_and_text_events(events, audio_only=not args.transcribe)
    print(df[["type", "start", "duration", "filepath"]].to_string(), flush=True)

    print("Running inference...", flush=True)
    preds, segments = model.predict(events=df)
    print(f"Predictions shape: {preds.shape}", flush=True)
    print(f"Segments: {len(segments)}", flush=True)


if __name__ == "__main__":
    main()
