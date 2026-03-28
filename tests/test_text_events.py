from pathlib import Path

import numpy as np
from PIL import Image
import torch

from tribev2 import demo_utils as demo_utils_module
from tribev2.demo_utils import build_text_events_from_text
from tribev2 import easy as easy_module
from tribev2.eventstransforms import ExtractWordsFromAudio
from tribev2.easy import (
    DEFAULT_TEXT_MODEL,
    ImageComparisonRun,
    PredictionRun,
    build_explainability_report,
    build_image_comparison_guide,
    build_result_interpretation,
    build_timestep_report_frame,
    collect_timestep_metadata,
    describe_timestep,
    infer_affective_cues,
    prepare_events,
    render_animated_brain_3d_html,
    render_prediction_gif,
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


def test_whisperx_detection_finds_executable_next_to_env(monkeypatch, tmp_path: Path):
    scripts = tmp_path / "Scripts"
    scripts.mkdir()
    uvx = scripts / "uvx.exe"
    uvx.write_bytes(b"")

    monkeypatch.setattr(easy_module.sys, "prefix", str(tmp_path))
    monkeypatch.setattr("tribev2.eventstransforms.sys.prefix", str(tmp_path))
    monkeypatch.setattr("tribev2.eventstransforms.sys.executable", str(tmp_path / "python.exe"))
    monkeypatch.setattr("tribev2.eventstransforms.shutil.which", lambda name: None)

    cmd = ExtractWordsFromAudio.whisperx_command()

    assert cmd is not None
    assert cmd[0].endswith("uvx.exe")


def test_build_timestep_report_frame_contains_interpretation_columns():
    preds = np.zeros((2, 20484), dtype=float)
    preds[0, :128] = 1.0
    run = PredictionRun(
        events=build_text_events_from_text("hello hope"),
        preds=preds,
        segments=[],
        input_kind="text",
        raw_text="hello hope",
    )

    frame = build_timestep_report_frame(run)

    assert list(frame.columns)[:4] == ["timestep", "start_s", "duration_s", "text"]
    assert "zone" in frame.columns
    assert "valence" in frame.columns


def test_render_prediction_gif_returns_bytes(monkeypatch):
    preds = np.zeros((3, 20484), dtype=float)
    run = PredictionRun(
        events=build_text_events_from_text("hello hope"),
        preds=preds,
        segments=[],
        input_kind="text",
        raw_text="hello hope",
    )

    monkeypatch.setattr(
        easy_module,
        "render_brain_panel_image",
        lambda *args, **kwargs: np.full((24, 24, 3), 180, dtype=np.uint8),
    )

    gif_bytes = render_prediction_gif(run, max_frames=3)

    assert gif_bytes[:6] in {b"GIF87a", b"GIF89a"}


def test_render_animated_brain_3d_html_autoplays(monkeypatch):
    class FakePlotter:
        bg_darkness = 0.0
        _mesh = {
            "both": {
                "coords": np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ]
                ),
                "faces": np.array([[0, 1, 2], [0, 2, 3]], dtype=int),
                "bg_map": np.array([0.0, 0.35, 0.7, 1.0], dtype=float),
            }
        }

        def get_stat_map(self, signal):
            return {"both": np.asarray(signal, dtype=float)}

    monkeypatch.setattr(easy_module, "get_pyvista_plotter", lambda mesh: FakePlotter())
    monkeypatch.setattr(
        easy_module,
        "collect_timestep_metadata",
        lambda run: [
            {"index": 0, "start": 0.0, "duration": 0.6, "text": "frame 1"},
            {"index": 1, "start": 0.6, "duration": 0.6, "text": "frame 2"},
        ],
    )
    monkeypatch.setattr(
        easy_module,
        "build_timestep_reports",
        lambda run, indices=None: [
            {
                "text": f"frame {idx + 1}",
                "summary": "resume",
                "zone": "occipitale",
                "valence": "indeterminee",
            }
            for idx in (indices or [0, 1])
        ],
    )
    run = PredictionRun(
        events=build_text_events_from_text("hello hope"),
        preds=np.array([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]], dtype=float),
        segments=[],
        input_kind="image",
    )

    html = render_animated_brain_3d_html(run, max_frames=2, height=320)

    assert "Auto-play actif" in html
    assert "scene.camera.eye" in html
    assert "play();" in html
    assert "vertexcolor" in html


def test_concat_hidden_states_memory_safe_retries_on_cpu(monkeypatch):
    original_cat = demo_utils_module.torch.cat
    calls = {"count": 0}

    def flaky_cat(tensors, axis=0):
        calls["count"] += 1
        if calls["count"] == 1:
            raise torch.OutOfMemoryError("synthetic oom")
        return original_cat(tensors, axis=axis)

    monkeypatch.setattr(demo_utils_module.torch, "cat", flaky_cat)
    monkeypatch.setattr(demo_utils_module.torch.cuda, "is_available", lambda: False)

    out, used_cpu_offload = demo_utils_module._concat_hidden_states_memory_safe(
        [torch.zeros((1, 2, 3)), torch.ones((1, 2, 3))],
        label="test hidden states",
    )

    assert calls["count"] == 2
    assert used_cpu_offload is True
    assert out.shape == (1, 2, 2, 3)
