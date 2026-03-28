from pathlib import Path

import numpy as np
from PIL import Image

from tribev2.demo_utils import build_text_events_from_text
from tribev2 import easy as easy_module
from tribev2.easy import (
    DEFAULT_TEXT_MODEL,
    ImageComparisonRun,
    PredictionRun,
    build_explainability_report,
    build_image_comparison_guide,
    build_result_interpretation,
    collect_timestep_metadata,
    describe_timestep,
    infer_affective_cues,
    prepare_events,
    resolve_text_model_name,
)


def test_build_text_events_from_text_creates_contextual_word_rows():
    events = build_text_events_from_text(
        "Hello world. This is a test.",
        seconds_per_word=0.5,
        max_context_words=3,
    )

    assert list(events.type.unique()) == ["Word"]
    assert events.text.tolist() == ["Hello", "world", "This", "is", "a", "test"]
    assert events.context.iloc[0] == "Hello"
    assert events.context.iloc[1].endswith("world")
    assert events.start.iloc[2] == 1.0
    assert events.duration.iloc[0] == 0.5


def test_prepare_events_supports_direct_text(tmp_path: Path):
    events, input_kind = prepare_events(
        cache_folder=tmp_path,
        text="One more short example.",
        direct_text=True,
    )

    assert input_kind == "text"
    assert not events.empty
    assert set(events.type.unique()) == {"Word"}


def test_resolve_text_model_name_defaults_to_public_repo():
    assert resolve_text_model_name() == DEFAULT_TEXT_MODEL


def test_describe_timestep_returns_readable_summary():
    preds = np.zeros((2, 20484), dtype=float)
    preds[0, :128] = 2.0

    description = describe_timestep(preds, timestep=0)

    assert description["laterality"] in {"gauche", "droite", "bilaterale"}
    assert description["focus_share"] > 0
    assert "Les sommets les plus saillants" in description["summary"]


def test_prepare_events_supports_single_image(tmp_path: Path, monkeypatch):
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (48, 32), color=(220, 80, 40)).save(image_path)
    generated_video = tmp_path / "sample.mp4"
    generated_video.write_bytes(b"fake")

    monkeypatch.setattr(
        easy_module,
        "build_video_from_image",
        lambda **kwargs: generated_video,
    )
    monkeypatch.setattr(
        easy_module,
        "get_audio_and_text_events",
        lambda events, audio_only=False: events,
    )

    events, input_kind = prepare_events(cache_folder=tmp_path, image_path=image_path)

    assert input_kind == "image"
    assert events.iloc[0]["type"] == "Video"
    assert events.iloc[0]["filepath"] == str(generated_video)


def test_build_explainability_report_for_direct_text_mentions_fork_shortcut():
    events = build_text_events_from_text("Simple local text example.", seconds_per_word=0.5)
    preds = np.zeros((2, 20484), dtype=float)
    preds[0, :64] = 1.0
    run = PredictionRun(
        events=events,
        preds=preds,
        segments=[],
        input_kind="text",
        raw_text="Simple local text example.",
    )

    report = build_explainability_report(
        run,
        timestep=0,
        description={
            "summary": "resume test",
            "laterality": "gauche",
            "antero_posterior": "centrale",
            "dorso_ventral": "dorsale",
            "focus_share": 0.2,
            "mean_abs": 0.1,
            "peak_abs": 1.0,
        },
    )

    bullets = " ".join(
        bullet
        for section in report["sections"]
        for bullet in section["bullets"]
    )
    assert "Texte direct" in bullets
    assert "timings synthetiques" in bullets


def test_build_explainability_report_for_image_mentions_static_clip(tmp_path: Path):
    events = np.array([])
    preds = np.zeros((2, 20484), dtype=float)
    preds[0, :64] = 1.0
    run = PredictionRun(
        events=easy_module.pd.DataFrame([{"type": "Video"}]),
        preds=preds,
        segments=[],
        input_kind="image",
        source_path=tmp_path / "sample.png",
    )

    report = build_explainability_report(
        run,
        timestep=0,
        description={
            "summary": "resume test",
            "laterality": "gauche",
            "antero_posterior": "centrale",
            "dorso_ventral": "dorsale",
            "focus_share": 0.2,
            "mean_abs": 0.1,
            "peak_abs": 1.0,
        },
    )

    bullets = " ".join(
        bullet
        for section in report["sections"]
        for bullet in section["bullets"]
    )
    assert "clip video silencieux" in bullets
    assert "ajout de ce fork" in bullets


def test_build_image_comparison_guide_mentions_both_images():
    preds = np.zeros((2, 20484), dtype=float)
    events = build_text_events_from_text("placeholder")
    run = ImageComparisonRun(
        runs=[
            PredictionRun(events=events, preds=preds, segments=[], input_kind="image"),
            PredictionRun(events=events, preds=preds, segments=[], input_kind="image"),
        ]
    )

    guide = build_image_comparison_guide(
        run,
        timestep=0,
        descriptions=[
            {
                "laterality": "gauche",
                "antero_posterior": "posterieure",
                "dorso_ventral": "dorsale",
                "summary": "",
                "focus_share": 0.2,
                "mean_abs": 0.1,
                "peak_abs": 1.0,
            },
            {
                "laterality": "droite",
                "antero_posterior": "anterieure",
                "dorso_ventral": "ventrale",
                "summary": "",
                "focus_share": 0.2,
                "mean_abs": 0.1,
                "peak_abs": 1.0,
            },
        ],
    )

    text = " ".join(guide["bullets"])
    assert "image 1" in text
    assert "image 2" in text


def test_collect_timestep_metadata_uses_segment_timing():
    class FakeSegment:
        def __init__(self, start, duration):
            self.start = start
            self.duration = duration
            self.events = easy_module.pd.DataFrame()

    run = PredictionRun(
        events=build_text_events_from_text("placeholder"),
        preds=np.zeros((2, 20484), dtype=float),
        segments=[FakeSegment(0.0, 0.8), FakeSegment(0.8, 1.2)],
        input_kind="audio",
    )

    timeline = collect_timestep_metadata(run)

    assert timeline[0]["start"] == 0.0
    assert timeline[0]["duration"] == 0.8
    assert timeline[1]["start"] == 0.8
    assert timeline[1]["duration"] == 1.2


def test_infer_affective_cues_detects_negative_fear_signal():
    affect = infer_affective_cues("I am scared and worried about the danger ahead.")

    assert affect["valence"] == "plutot negative"
    assert "fear" in affect["emotions"]
    assert "danger" in affect["evidence"]


def test_build_result_interpretation_uses_text_cues_for_affect():
    preds = np.zeros((2, 20484), dtype=float)
    preds[0, :128] = 2.0
    run = PredictionRun(
        events=build_text_events_from_text("I want safety and love, not fear."),
        preds=preds,
        segments=[],
        input_kind="text",
        raw_text="I want safety and love, not fear.",
    )

    interpretation = build_result_interpretation(
        run,
        timestep=0,
        description=describe_timestep(preds, timestep=0),
        segment_text="I want safety and love, not fear.",
    )

    assert interpretation["zone"]
    assert interpretation["systems"]
    assert interpretation["affect"]["valence"] in {"plutot positive", "mixte", "plutot negative"}
