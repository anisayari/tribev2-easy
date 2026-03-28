import io
from pathlib import Path
import logging
import sys
import types
import warnings

import numpy as np
import pandas as pd
from PIL import Image, ImageSequence
import torch

from tribev2 import demo_utils as demo_utils_module
from tribev2 import dashboard_app as dashboard_app_module
from tribev2.demo_utils import build_text_events_from_text
from tribev2 import easy as easy_module
from tribev2 import openai_chat as openai_chat_module
from tribev2 import utils as utils_module
from tribev2 import eventstransforms as eventstransforms_module
from tribev2.eventstransforms import ExtractWordsFromAudio
from tribev2.easy import (
    DEFAULT_TEXT_MODEL,
    FALLBACK_TEXT_MODEL,
    ImageComparisonRun,
    MultiModalRun,
    PRIMARY_TEXT_MODEL,
    PredictionRun,
    build_comparison_display_reference,
    build_explainability_report,
    build_image_comparison_guide,
    build_emotion_hypothesis_frame,
    build_result_interpretation,
    build_run_roi_frame,
    build_run_zone_frame,
    build_timestep_report_frame,
    build_timestep_zone_frame,
    collect_timestep_metadata,
    describe_timestep,
    infer_affective_cues,
    normalize_signal_for_display,
    prepare_events,
    render_animated_brain_3d_html,
    render_prediction_gif,
    resolve_text_model_candidates,
    resolve_text_model_name,
)
from tribev2.openai_chat import (
    build_chat_system_prompt,
    build_openai_context_bundle,
    build_raw_timestep_frame,
)
from tribev2.runtime import apply_warning_filters, configure_file_logging


def _patch_synthetic_hcp(monkeypatch):
    label_map = {
        "V1": np.array([0, 1]),
        "A1": np.array([2, 3]),
        "TPOJ1": np.array([4, 5]),
        "10d": np.array([6, 7]),
    }

    def fake_labels(*args, **kwargs):
        return label_map

    def fake_summarize(data, hemi="both", mesh="fsaverage5"):
        signal = np.asarray(data, dtype=float)
        return np.array([signal[vertices].mean() for vertices in label_map.values()], dtype=float)

    monkeypatch.setattr(easy_module, "get_hcp_labels", fake_labels)
    monkeypatch.setattr(easy_module, "summarize_by_roi", fake_summarize)
    monkeypatch.setattr(utils_module, "get_hcp_labels", fake_labels)
    monkeypatch.setattr(utils_module, "summarize_by_roi", fake_summarize)


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


def test_dedupe_items_by_uid_preserves_first_occurrence():
    items = ["clip-a", "clip-b", "clip-a", "clip-c", "clip-b"]

    deduped, duplicate_count = demo_utils_module._dedupe_items_by_uid(items, lambda item: item)

    assert deduped == ["clip-a", "clip-b", "clip-c"]
    assert duplicate_count == 2


def test_apply_warning_filters_guards_tqdm_destructor():
    apply_warning_filters()
    from tqdm.std import tqdm as tqdm_cls

    orphan = object.__new__(tqdm_cls)

    tqdm_cls.__del__(orphan)


def test_prepare_events_supports_direct_text(tmp_path: Path):
    events, input_kind = prepare_events(
        cache_folder=tmp_path,
        text="One more short example.",
        direct_text=True,
    )

    assert input_kind == "text"
    assert not events.empty
    assert set(events.type.unique()) == {"Word"}


def test_dashboard_request_label_supports_multiple_modalities():
    label = dashboard_app_module._format_request_label(
        {
            "video_path": Path("clip.mp4"),
            "audio_path": Path("voice.wav"),
            "text": "hello world",
        }
    )

    assert label == "Vidéo + Audio + Texte"


def test_dashboard_request_label_supports_comparison_inputs():
    label = dashboard_app_module._format_request_label(
        {
            "video_paths": [Path("a.mp4"), Path("b.mp4")],
        }
    )

    assert label == "Vidéos x2"


def test_multimodal_run_key_and_prompt_include_overlay_context():
    events = build_text_events_from_text("hello world", seconds_per_word=1.0)
    preds = np.zeros((2, 20484), dtype=float)
    video_run = PredictionRun(events=events, preds=preds + 0.2, segments=[], input_kind="video")
    text_run = PredictionRun(
        events=events,
        preds=preds + 0.4,
        segments=[],
        input_kind="text",
        raw_text="hello world",
    )
    run = MultiModalRun(
        events=events,
        preds=preds + 0.1,
        segments=[],
        input_kind="multimodal",
        raw_text="hello world",
        component_runs={"video": video_run, "text": text_run},
    )

    key = dashboard_app_module.get_dashboard_run_key(run)
    prompt = build_chat_system_prompt(run)
    report = build_timestep_report_frame(run)

    assert key.startswith("multimodal:")
    assert "rouge pour le visuel" in prompt
    assert report.iloc[0]["zone"] == "Fusion multimodale"


def test_get_topk_rois_handles_dict_keys(monkeypatch):
    _patch_synthetic_hcp(monkeypatch)
    data = np.array([8.0, 8.0, 2.0, 2.0, 5.0, 5.0, 1.0, 1.0])

    rois = utils_module.get_topk_rois(data, hemi="both", mesh="fsaverage5", k=2)

    assert list(rois) == ["V1", "TPOJ1"]


def test_zone_and_emotion_frames_are_built_from_hcp_rois(monkeypatch):
    _patch_synthetic_hcp(monkeypatch)
    events = build_text_events_from_text("I am scared but hopeful.", seconds_per_word=1.0)
    preds = np.zeros((2, 8), dtype=float)
    preds[0, :2] = 4.0
    preds[0, 4:6] = 3.0
    preds[0, 6:8] = 2.0
    preds[1, 2:4] = 3.0
    run = PredictionRun(
        events=events,
        preds=preds,
        segments=[],
        input_kind="text",
        raw_text="I am scared but hopeful.",
    )

    zone_frame = build_run_zone_frame(run)
    roi_frame = build_run_roi_frame(run)
    timeseries_frame = build_timestep_zone_frame(run)
    emotion_frame = build_emotion_hypothesis_frame(run)

    assert {"zone", "share", "roi_count"}.issubset(zone_frame.columns)
    assert {"roi", "zone", "value"}.issubset(roi_frame.columns)
    assert {"timestep", "zone", "share"}.issubset(timeseries_frame.columns)
    assert {"label", "score_pct", "top_zone_drivers"}.issubset(emotion_frame.columns)
    assert "Peur" in emotion_frame["label"].tolist()


def test_openai_context_bundle_includes_zone_payload(monkeypatch):
    _patch_synthetic_hcp(monkeypatch)
    monkeypatch.setattr(openai_chat_module, "render_run_panel_bytes", lambda *args, **kwargs: b"fake-jpeg")
    events = build_text_events_from_text("A fearful social scene.", seconds_per_word=1.0)
    preds = np.zeros((2, 8), dtype=float)
    preds[0, :2] = 2.0
    preds[0, 4:6] = 3.0
    run = PredictionRun(
        events=events,
        preds=preds,
        segments=[],
        input_kind="text",
        raw_text="A fearful social scene.",
    )

    prompt = build_chat_system_prompt(run)
    context_text, _, _ = build_openai_context_bundle(run, max_images=1)

    assert "tableaux HCP par zone" in prompt
    assert "radar map" in prompt
    assert '"zone_payload"' in context_text
    assert '"emotion_hypotheses"' in context_text


def test_resolve_text_model_name_defaults_to_preferred_repo():
    assert resolve_text_model_name() == DEFAULT_TEXT_MODEL


def test_resolve_text_model_candidates_adds_fallback_for_meta_repo():
    assert resolve_text_model_candidates() == [PRIMARY_TEXT_MODEL, FALLBACK_TEXT_MODEL]
    assert resolve_text_model_candidates(PRIMARY_TEXT_MODEL) == [
        PRIMARY_TEXT_MODEL,
        FALLBACK_TEXT_MODEL,
    ]


def test_resolve_text_model_candidates_keeps_explicit_non_meta_model():
    local_model = "C:/models/custom-llama"
    assert resolve_text_model_candidates(FALLBACK_TEXT_MODEL) == [FALLBACK_TEXT_MODEL]
    assert resolve_text_model_candidates(local_model) == [local_model]


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


def test_build_raw_timestep_frame_contains_only_raw_metrics():
    preds = np.zeros((2, 20484), dtype=float)
    preds[0, :128] = 1.0
    run = PredictionRun(
        events=build_text_events_from_text("hello hope"),
        preds=preds,
        segments=[],
        input_kind="text",
        raw_text="hello hope",
    )

    frame = build_raw_timestep_frame(run)

    assert list(frame.columns) == [
        "timestep",
        "start_s",
        "duration_s",
        "text",
        "mean",
        "std",
        "mean_abs",
        "max_abs",
    ]
    assert "zone" not in frame.columns
    assert "valence" not in frame.columns


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


def test_render_prediction_gif_draws_timestep_badge(monkeypatch):
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
        lambda *args, **kwargs: np.full((120, 180, 3), 180, dtype=np.uint8),
    )

    gif_bytes = render_prediction_gif(run, max_frames=3)
    frame = next(ImageSequence.Iterator(Image.open(io.BytesIO(gif_bytes)))).convert("RGB")
    badge_region = np.asarray(frame.crop((0, frame.height - 40, 80, frame.height)))

    assert badge_region.std() > 0
    assert not np.all(badge_region == 180)


def test_build_synced_player_html_uses_single_media_card(tmp_path: Path, monkeypatch):
    media_path = tmp_path / "sample.mp4"
    media_path.write_bytes(b"fake-mp4")
    run = PredictionRun(
        events=build_text_events_from_text("hello hope"),
        preds=np.zeros((2, 20484), dtype=float),
        segments=[],
        input_kind="video",
        source_path=media_path,
    )

    monkeypatch.setattr(
        dashboard_app_module,
        "build_browser_media_proxy",
        lambda **kwargs: media_path,
    )
    monkeypatch.setattr(
        dashboard_app_module,
        "render_brain_panel_bytes",
        lambda *args, **kwargs: b"brain-jpeg",
    )
    monkeypatch.setattr(
        dashboard_app_module,
        "build_emotion_hypothesis_frame",
        lambda *args, **kwargs: pd.DataFrame(
            [
                {"emotion": "joy", "label": "Joie", "score_pct": 42.0},
                {"emotion": "fear", "label": "Peur", "score_pct": 18.0},
                {"emotion": "sadness", "label": "Tristesse", "score_pct": 11.0},
                {"emotion": "anger", "label": "Colere", "score_pct": 7.0},
                {"emotion": "desire", "label": "Desir", "score_pct": 15.0},
                {"emotion": "calm", "label": "Calme", "score_pct": 22.0},
            ]
        ),
    )

    html = dashboard_app_module.build_synced_player_html(run, cache_folder=tmp_path)

    assert 'id="tribe-play"' in html
    assert 'id="tribe-pause"' in html
    assert 'id="tribe-media"' in html
    assert 'id="tribe-radar"' in html
    assert 'id="tribe-radar-top"' in html
    assert 'class="sync-media sync-media-video"' in html
    assert "Video source" not in html
    assert "Cerveau predit en temps reel" not in html


def test_persist_saved_run_roundtrip_for_text(tmp_path: Path):
    run = PredictionRun(
        events=build_text_events_from_text("A calm hopeful text example."),
        preds=np.zeros((3, 20484), dtype=float),
        segments=[],
        input_kind="text",
        raw_text="A calm hopeful text example.",
    )

    run_id = dashboard_app_module.persist_saved_run(tmp_path, run)
    entries = dashboard_app_module.list_saved_runs(tmp_path)
    loaded = dashboard_app_module.load_saved_run(tmp_path, run_id)

    assert any(entry["id"] == run_id for entry in entries)
    assert isinstance(loaded, PredictionRun)
    assert loaded.raw_text == run.raw_text
    assert loaded.preds.shape == (3, 20484)


def test_persist_saved_run_stores_versioned_payload_not_live_run_object(tmp_path: Path):
    run = PredictionRun(
        events=build_text_events_from_text("Serialized payload"),
        preds=np.zeros((2, 20484), dtype=float),
        segments=[],
        input_kind="text",
        raw_text="Serialized payload",
    )

    run_id = dashboard_app_module.persist_saved_run(tmp_path, run)
    run_path = dashboard_app_module.get_saved_runs_folder(tmp_path) / run_id / "run.pkl"

    with open(run_path, "rb") as handle:
        payload = dashboard_app_module.pickle.load(handle)

    assert isinstance(payload, dict)
    assert payload["version"] == 2
    assert payload["run"]["kind"] == "prediction"
    assert payload["run"]["raw_text"] == "Serialized payload"


def test_persist_saved_run_roundtrip_for_text_comparison(tmp_path: Path):
    preds = np.zeros((2, 20484), dtype=float)
    run = ImageComparisonRun(
        runs=[
            PredictionRun(
                events=build_text_events_from_text("first text"),
                preds=preds.copy(),
                segments=[],
                input_kind="text",
                raw_text="first text",
            ),
            PredictionRun(
                events=build_text_events_from_text("second text"),
                preds=preds.copy(),
                segments=[],
                input_kind="text",
                raw_text="second text",
            ),
        ],
        compare_kind="text",
    )

    run_id = dashboard_app_module.persist_saved_run(tmp_path, run)
    loaded = dashboard_app_module.load_saved_run(tmp_path, run_id)

    assert isinstance(loaded, ImageComparisonRun)
    assert loaded.compare_kind == "text"
    assert len(loaded.runs) == 2
    assert loaded.runs[0].raw_text == "first text"
    assert loaded.runs[1].raw_text == "second text"


def test_persist_saved_run_generates_visual_preview_for_image(tmp_path: Path):
    image_path = tmp_path / "preview-source.png"
    Image.new("RGB", (96, 64), color=(220, 120, 80)).save(image_path)
    run = PredictionRun(
        events=build_text_events_from_text("visual snapshot"),
        preds=np.zeros((2, 20484), dtype=float),
        segments=[],
        input_kind="image",
        source_path=image_path,
    )

    run_id = dashboard_app_module.persist_saved_run(tmp_path, run)
    entries = dashboard_app_module.list_saved_runs(tmp_path)
    entry = next(item for item in entries if item["id"] == run_id)
    preview_path = Path(entry["folder"]) / str(entry["preview_file"])

    assert preview_path.exists()


def test_get_cached_animated_3d_html_reuses_saved_artifact(tmp_path: Path, monkeypatch):
    dashboard_app_module.st.session_state.clear()
    run = PredictionRun(
        events=build_text_events_from_text("Persistent 3D view"),
        preds=np.zeros((2, 20484), dtype=float),
        segments=[],
        input_kind="text",
        raw_text="Persistent 3D view",
    )
    calls = {"count": 0}

    def fake_render(*args, **kwargs):
        calls["count"] += 1
        return "<html>brain-3d</html>"

    monkeypatch.setattr(dashboard_app_module, "render_animated_brain_3d_html", fake_render)

    html_first = dashboard_app_module.get_cached_animated_3d_html(
        run,
        cache_folder=tmp_path,
        max_frames=5,
        height=320,
    )
    artifact_matches = list(
        dashboard_app_module.get_saved_run_artifacts_folder(tmp_path, run).glob("brain3d_005f_0320h*.html")
    )
    assert len(artifact_matches) == 1
    artifact_path = artifact_matches[0]

    dashboard_app_module.st.session_state.pop("animated_3d_html", None)
    monkeypatch.setattr(
        dashboard_app_module,
        "render_animated_brain_3d_html",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("3D should load from artifact")),
    )
    html_second = dashboard_app_module.get_cached_animated_3d_html(
        run,
        cache_folder=tmp_path,
        max_frames=5,
        height=320,
    )

    assert calls["count"] == 1
    assert artifact_path.exists()
    assert html_first == "<html>brain-3d</html>"
    assert html_second == html_first


def test_ensure_cached_sync_player_html_reuses_saved_artifact(tmp_path: Path, monkeypatch):
    dashboard_app_module.st.session_state.clear()
    media_path = tmp_path / "clip.mp4"
    media_path.write_bytes(b"fake-video")
    run = PredictionRun(
        events=build_text_events_from_text("sync artifact"),
        preds=np.zeros((2, 20484), dtype=float),
        segments=[],
        input_kind="video",
        source_path=media_path,
    )
    calls = {"count": 0}

    def fake_build(*args, **kwargs):
        calls["count"] += 1
        return "<html>sync-player</html>"

    monkeypatch.setattr(dashboard_app_module, "build_synced_player_html", fake_build)

    html_first = dashboard_app_module.ensure_cached_sync_player_html(run, cache_folder=tmp_path)
    artifact_path = dashboard_app_module.get_saved_run_artifacts_folder(tmp_path, run) / "sync_visible_media_v3.html"

    dashboard_app_module.st.session_state.pop("sync_player_html", None)
    monkeypatch.setattr(
        dashboard_app_module,
        "build_synced_player_html",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("sync player should load from artifact")),
    )
    html_second = dashboard_app_module.ensure_cached_sync_player_html(run, cache_folder=tmp_path)

    assert calls["count"] == 1
    assert artifact_path.exists()
    assert html_first == "<html>sync-player</html>"
    assert html_second == html_first


def test_build_workspace_zone_timeseries_frame_orders_columns_from_run_zone():
    run = PredictionRun(
        events=build_text_events_from_text("zone timeline"),
        preds=np.zeros((2, 20484), dtype=float),
        segments=[],
        input_kind="text",
        raw_text="zone timeline",
    )
    bundle = {
        "run_zone": pd.DataFrame(
            [
                {"zone": "Zone B", "share": 0.5},
                {"zone": "Zone A", "share": 0.3},
                {"zone": "Zone C", "share": 0.2},
            ]
        ),
        "zone_timeseries": pd.DataFrame(
            [
                {"timestep": 0, "zone": "Zone A", "share": 0.1},
                {"timestep": 0, "zone": "Zone B", "share": 0.6},
                {"timestep": 0, "zone": "Zone C", "share": 0.3},
                {"timestep": 1, "zone": "Zone A", "share": 0.2},
                {"timestep": 1, "zone": "Zone B", "share": 0.5},
                {"timestep": 1, "zone": "Zone C", "share": 0.3},
            ]
        ),
    }

    pivot = dashboard_app_module.build_workspace_zone_timeseries_frame(run, bundle=bundle)

    assert pivot.columns.tolist() == ["Zone B", "Zone A", "Zone C"]
    assert pivot.loc[0, "Zone B"] == 0.6
    assert pivot.loc[1, "Zone A"] == 0.2


def test_saved_runs_gallery_html_uses_clickable_cards(tmp_path: Path):
    preview_path = tmp_path / "preview.png"
    Image.new("RGB", (64, 48), color=(40, 120, 200)).save(preview_path)
    html = dashboard_app_module._build_saved_runs_gallery_html(
        [
            {
                "id": "run123",
                "kind_label": "Vidéo",
                "subtitle": "clip.mp4",
                "updated_at": "2026-03-28T14:22:00+01:00",
                "timesteps": 6,
                "folder": str(tmp_path),
                "preview_file": preview_path.name,
            }
        ],
        active_saved_id="run123",
    )

    assert '?saved_run=run123' in html
    assert 'saved-run-card is-active' in html
    assert 'saved-run-thumb' in html
    assert html.startswith("<style>")


def test_render_html_fragment_prefers_streamlit_html(monkeypatch):
    calls = []

    fake_st = types.SimpleNamespace(
        html=lambda markup: calls.append(("html", markup)),
        markdown=lambda markup, unsafe_allow_html=False: calls.append(
            ("markdown", markup, unsafe_allow_html)
        ),
    )
    monkeypatch.setattr(dashboard_app_module, "st", fake_st)

    dashboard_app_module._render_html_fragment("<div>gallery</div>")

    assert calls == [("html", "<div>gallery</div>")]


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


def test_build_openai_context_bundle_for_single_run(monkeypatch):
    preds = np.zeros((3, 20484), dtype=float)
    preds[1, :256] = 1.0
    run = PredictionRun(
        events=build_text_events_from_text("hello hope"),
        preds=preds,
        segments=[],
        input_kind="text",
        raw_text="hello hope",
    )
    monkeypatch.setattr(
        openai_chat_module,
        "render_brain_panel_bytes",
        lambda *args, **kwargs: b"fake-jpeg",
    )

    context_text, image_parts, labels = build_openai_context_bundle(run, max_images=2)

    assert "tribev2_prediction_run" in context_text
    assert len(image_parts) == 2
    assert len(labels) == 2
    assert image_parts[0]["type"] == "input_image"
    assert image_parts[0]["image_url"].startswith("data:image/jpeg;base64,")


def test_build_openai_context_bundle_for_image_comparison(monkeypatch):
    preds = np.zeros((2, 20484), dtype=float)
    run = ImageComparisonRun(
        runs=[
            PredictionRun(
                events=build_text_events_from_text("image one"),
                preds=preds,
                segments=[],
                input_kind="image",
            ),
            PredictionRun(
                events=build_text_events_from_text("image two"),
                preds=preds,
                segments=[],
                input_kind="image",
            ),
        ]
    )
    monkeypatch.setattr(
        openai_chat_module,
        "render_brain_panel_bytes",
        lambda *args, **kwargs: b"fake-jpeg",
    )

    context_text, image_parts, labels = build_openai_context_bundle(run, max_images=4)

    assert "tribev2_image_comparison" in context_text
    assert "image_1" in context_text
    assert "image_2" in context_text
    assert len(image_parts) >= 2
    assert any("image_1" in label for label in labels)
    assert any("image_2" in label for label in labels)


def test_build_openai_context_bundle_for_text_comparison(monkeypatch):
    preds = np.zeros((2, 20484), dtype=float)
    run = ImageComparisonRun(
        runs=[
            PredictionRun(
                events=build_text_events_from_text("text one"),
                preds=preds,
                segments=[],
                input_kind="text",
                raw_text="text one",
            ),
            PredictionRun(
                events=build_text_events_from_text("text two"),
                preds=preds,
                segments=[],
                input_kind="text",
                raw_text="text two",
            ),
        ],
        compare_kind="text",
    )
    monkeypatch.setattr(
        openai_chat_module,
        "render_brain_panel_bytes",
        lambda *args, **kwargs: b"fake-jpeg",
    )

    context_text, _, labels = build_openai_context_bundle(run, max_images=2)

    assert "tribev2_text_comparison" in context_text
    assert "text_1" in context_text
    assert "text_2" in context_text
    assert any("text_1" in label for label in labels)


def test_build_chat_system_prompt_for_image_mentions_static_clip_and_uncertainties(tmp_path: Path):
    run = PredictionRun(
        events=easy_module.pd.DataFrame([{"type": "Video"}]),
        preds=np.zeros((3, 20484), dtype=float),
        segments=[],
        input_kind="image",
        source_path=tmp_path / "image.png",
    )

    prompt = build_chat_system_prompt(run)

    assert "court clip video silencieux statique" in prompt
    assert "Emotion ou ressenti plausible" in prompt
    assert "Ce qui reste incertain" in prompt
    assert "predictions TRIBE v2 et non de mesures fMRI reelles" in prompt


def test_build_openai_context_bundle_includes_interpretation_contract_for_text(monkeypatch):
    run = PredictionRun(
        events=build_text_events_from_text("A quiet but tense scene."),
        preds=np.zeros((2, 20484), dtype=float),
        segments=[],
        input_kind="text",
        raw_text="A quiet but tense scene.",
    )
    monkeypatch.setattr(
        openai_chat_module,
        "render_brain_panel_bytes",
        lambda *args, **kwargs: b"fake-jpeg",
    )

    context_text, _, _ = build_openai_context_bundle(run, max_images=2)
    payload = openai_chat_module.json.loads(context_text)

    assert payload["run"]["interpretation_contract"]["format_attendu"][2] == "3. Emotion ou ressenti plausible"
    assert "timings synthetiques" in " ".join(payload["run"]["modality_notes"])
    assert "selected_timestep_image_policy" in payload["run"]


def test_build_chat_system_prompt_for_image_comparison_mentions_explicit_comparison():
    preds = np.zeros((2, 20484), dtype=float)
    events = build_text_events_from_text("placeholder")
    run = ImageComparisonRun(
        runs=[
            PredictionRun(events=events, preds=preds, segments=[], input_kind="image"),
            PredictionRun(events=events, preds=preds, segments=[], input_kind="image"),
        ]
    )

    prompt = build_chat_system_prompt(run)

    assert "compare explicitement image 1 vs image 2" in prompt
    assert "predictions TRIBE v2 et non de mesures fMRI reelles" in prompt


def test_build_chat_system_prompt_for_text_comparison_mentions_texts():
    preds = np.zeros((2, 20484), dtype=float)
    events = build_text_events_from_text("placeholder")
    run = ImageComparisonRun(
        runs=[
            PredictionRun(events=events, preds=preds, segments=[], input_kind="text", raw_text="one"),
            PredictionRun(events=events, preds=preds, segments=[], input_kind="text", raw_text="two"),
        ],
        compare_kind="text",
    )

    prompt = build_chat_system_prompt(run)

    assert "compare explicitement text 1 vs text 2" in prompt
    assert "deux textes" in prompt


def test_apply_warning_filters_ignores_transformers_path_alias_warning():
    apply_warning_filters()

    with warnings.catch_warnings(record=True) as caught:
        warnings.warn(
            "Accessing `__path__` from `.models.vit.image_processing_vit`. Returning `__path__` instead. Behavior may be different and this alias will be removed in future versions.",
            FutureWarning,
        )

    assert caught == []


def test_configure_file_logging_writes_package_logs(tmp_path: Path):
    log_file = configure_file_logging(tmp_path / "logs" / "tribev2-dashboard.log")

    logger = logging.getLogger("tribev2.test")
    logger.info("log-line-for-test")
    for handler in logging.getLogger("tribev2").handlers:
        handler.flush()

    assert log_file.exists()
    assert "log-line-for-test" in log_file.read_text(encoding="utf-8")


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


def test_normalize_signal_for_display_supports_shared_reference():
    signal = np.array([-1.0, 0.0, 1.0], dtype=float)
    local = normalize_signal_for_display(signal, percentile=100)
    shared = normalize_signal_for_display(
        signal,
        percentile=100,
        reference_signal=np.array([-10.0, 0.0, 10.0], dtype=float),
    )

    assert np.allclose(local, np.array([0.0, 0.5, 1.0]))
    assert np.allclose(shared, np.array([0.45, 0.5, 0.55]))


def test_build_comparison_display_reference_flattens_both_runs():
    events = build_text_events_from_text("comparison reference")
    run = ImageComparisonRun(
        runs=[
            PredictionRun(events=events, preds=np.array([[1.0, 2.0], [3.0, 4.0]]), segments=[], input_kind="text"),
            PredictionRun(events=events, preds=np.array([[5.0, 6.0]]), segments=[], input_kind="text"),
        ],
        compare_kind="text",
    )

    reference = build_comparison_display_reference(run)

    assert reference is not None
    assert np.allclose(reference, np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))


def test_openai_context_bundle_for_comparison_mentions_shared_display_normalization(monkeypatch):
    monkeypatch.setattr(openai_chat_module, "render_run_panel_bytes", lambda *args, **kwargs: b"fake-jpeg")
    preds_a = np.zeros((2, 20484), dtype=float)
    preds_b = np.ones((2, 20484), dtype=float)
    run = ImageComparisonRun(
        runs=[
            PredictionRun(
                events=build_text_events_from_text("text one"),
                preds=preds_a,
                segments=[],
                input_kind="text",
                raw_text="text one",
            ),
            PredictionRun(
                events=build_text_events_from_text("text two"),
                preds=preds_b,
                segments=[],
                input_kind="text",
                raw_text="text two",
            ),
        ],
        compare_kind="text",
    )

    context_text, image_parts, labels = build_openai_context_bundle(run, max_images=4)

    assert "shared_percentile_99_reference_across_all_compared_runs" in context_text
    assert '"display_normalization": "shared_percentile_99_reference"' in context_text
    assert image_parts
    assert labels


def test_get_cached_prediction_gif_supports_variant_cache_tags(tmp_path: Path, monkeypatch):
    dashboard_app_module.st.session_state.clear()
    run = PredictionRun(
        events=build_text_events_from_text("comparison gif"),
        preds=np.zeros((2, 20484), dtype=float),
        segments=[],
        input_kind="text",
        raw_text="comparison gif",
    )

    monkeypatch.setattr(
        dashboard_app_module,
        "render_prediction_gif",
        lambda *args, **kwargs: b"GIF89a-variant",
    )

    cache_tag = "comparison:shared_norm:item_1"
    gif_bytes = dashboard_app_module.get_cached_prediction_gif(
        run,
        cache_folder=tmp_path,
        cache_tag=cache_tag,
        display_reference=np.array([0.0, 1.0], dtype=float),
    )

    suffix = dashboard_app_module._artifact_variant_suffix(cache_tag)
    artifact_path = dashboard_app_module.get_saved_run_artifacts_folder(tmp_path, run) / f"brainplayback_072f{suffix}.gif"

    dashboard_app_module.st.session_state.pop("brain_gifs", None)
    monkeypatch.setattr(
        dashboard_app_module,
        "render_prediction_gif",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("GIF should load from artifact")),
    )
    gif_bytes_second = dashboard_app_module.get_cached_prediction_gif(
        run,
        cache_folder=tmp_path,
        cache_tag=cache_tag,
        display_reference=np.array([0.0, 1.0], dtype=float),
    )

    assert artifact_path.exists()
    assert gif_bytes == b"GIF89a-variant"
    assert gif_bytes_second == gif_bytes


def test_whisperx_runtime_config_forces_cpu_for_uvx_on_cuda(monkeypatch):
    monkeypatch.delenv("TRIBEV2_WHISPERX_DEVICE", raising=False)
    monkeypatch.setattr(ExtractWordsFromAudio, "whisperx_command", classmethod(lambda cls: ["uvx", "whisperx"]))
    monkeypatch.setattr(eventstransforms_module.torch.cuda, "is_available", lambda: True)

    device, compute_type = ExtractWordsFromAudio.whisperx_runtime_config()

    assert device == "cpu"
    assert compute_type == "int8"


def test_whisperx_runtime_config_honors_explicit_override(monkeypatch):
    monkeypatch.setenv("TRIBEV2_WHISPERX_DEVICE", "cuda")
    monkeypatch.setattr(ExtractWordsFromAudio, "whisperx_command", classmethod(lambda cls: ["uvx", "whisperx"]))

    device, compute_type = ExtractWordsFromAudio.whisperx_runtime_config()

    assert device == "cuda"
    assert compute_type == "float16"


def test_whisperx_subprocess_env_includes_embedded_ffmpeg(monkeypatch, tmp_path: Path):
    ffmpeg_dir = tmp_path / "ffmpeg-bin"
    ffmpeg_dir.mkdir()
    ffmpeg_exe = ffmpeg_dir / "ffmpeg.exe"
    ffmpeg_exe.write_bytes(b"")
    monkeypatch.setitem(
        sys.modules,
        "imageio_ffmpeg",
        types.SimpleNamespace(get_ffmpeg_exe=lambda: str(ffmpeg_exe)),
    )
    monkeypatch.setenv("PATH", "C:\\base-path")

    env = ExtractWordsFromAudio._build_subprocess_env()

    assert env["IMAGEIO_FFMPEG_EXE"] == str(ffmpeg_exe)
    assert env["PATH"].split(";")[0] == str(ffmpeg_dir)
