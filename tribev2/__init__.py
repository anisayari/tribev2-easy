# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import warnings

warnings.filterwarnings(
    "ignore",
    message=r"LabelEncoder: event_types has not been set.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"The events dataframe contains an `Index` column.*",
    category=UserWarning,
)
logging.getLogger("neuralset.extractors.base").setLevel(logging.ERROR)

from tribev2.demo_utils import TribeModel, build_text_events_from_text
from tribev2.easy import (
    DEFAULT_TEXT_MODEL,
    ImageComparisonRun,
    PredictionRun,
    build_explainability_report,
    build_image_comparison_guide,
    build_timestep_report_frame,
    build_video_from_image,
    describe_timestep,
    export_prediction_video,
    load_model,
    prepare_events,
    render_animated_brain_3d_html,
    render_interactive_brain_html,
    render_prediction_gif,
    resolve_text_model_name,
)

__all__ = [
    "DEFAULT_TEXT_MODEL",
    "ImageComparisonRun",
    "PredictionRun",
    "TribeModel",
    "build_explainability_report",
    "build_image_comparison_guide",
    "build_timestep_report_frame",
    "build_video_from_image",
    "build_text_events_from_text",
    "describe_timestep",
    "export_prediction_video",
    "load_model",
    "prepare_events",
    "render_animated_brain_3d_html",
    "render_interactive_brain_html",
    "render_prediction_gif",
    "resolve_text_model_name",
]
