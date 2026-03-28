# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Study registrations.

This module is imported for its side effects by ``tribev2.main``. Streamlit can
re-run user code in a long-lived Python process while clearing only part of the
module graph, which means the global ``neuralset`` study registry may still
contain old class objects from a previous run. Re-importing the study modules
would then raise a name-collision error even though the code did not change.

To keep local dashboard reruns reliable, we clear only this package's known
study names before importing and then explicitly re-register the live class
objects.
"""

from neuralset.events import study as _study

_STUDY_EXPORTS = (
    "Algonauts2025",
    "Algonauts2025Bold",
    "Lahner2024Bold",
    "Lebel2023Bold",
    "Wen2017",
)


def _clear_local_study_registry() -> None:
    for name in _STUDY_EXPORTS:
        existing = _study.STUDIES.get(name)
        if existing is not None and getattr(existing, "__module__", "").startswith(
            "tribev2.studies."
        ):
            _study.STUDIES.pop(name, None)
        _study.STUDY_PATHS.pop(name, None)


def _register_local_studies(*classes) -> None:
    for cls in classes:
        _study.STUDIES[cls.__name__] = cls


_clear_local_study_registry()

from .algonauts2025 import Algonauts2025, Algonauts2025Bold
from .lahner2024bold import Lahner2024Bold
from .lebel2023bold import Lebel2023Bold
from .wen2017 import Wen2017

_register_local_studies(
    Algonauts2025,
    Algonauts2025Bold,
    Lahner2024Bold,
    Lebel2023Bold,
    Wen2017,
)

__all__ = list(_STUDY_EXPORTS)
