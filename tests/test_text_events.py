from pathlib import Path

import numpy as np

from tribev2.demo_utils import build_text_events_from_text
from tribev2.easy import (
    DEFAULT_TEXT_MODEL,
    describe_timestep,
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
