<div align="center">

# TRIBE v2 Easy

**A practical local interface for Meta's TRIBE v2 multimodal brain-encoding model**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/facebookresearch/tribev2/blob/main/tribe_demo.ipynb)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

[Paper](https://ai.meta.com/research/publications/a-foundation-model-of-vision-audition-and-language-for-in-silico-neuroscience/) |
[Official Demo](https://aidemos.atmeta.com/tribev2/) |
[Weights](https://huggingface.co/facebook/tribev2)

</div>

TRIBE v2 is a multimodal brain-encoding model that predicts fMRI activity from video, audio, and text. It combines large pretrained encoders for language, vision, and audition, then maps those features to cortical activity on the `fsaverage5` surface.

This fork keeps the original research code, adds Windows-friendly inference fixes, supports modern Blackwell GPUs through newer PyTorch builds, and includes a packaged local dashboard for easier day-to-day use.

## Quick start

Load the pretrained model and predict responses from a video:

```python
from tribev2 import TribeModel

model = TribeModel.from_pretrained("facebook/tribev2", cache_folder="./cache")
events = model.get_events_dataframe(video_path="path/to/video.mp4")
preds, segments = model.predict(events)
print(preds.shape)  # (n_timesteps, n_vertices)
```

You can also build text events directly, without the TTS + ASR path:

```python
from tribev2 import TribeModel

model = TribeModel.from_pretrained("facebook/tribev2", cache_folder="./cache")
events = model.get_events_dataframe(
    text_path="path/to/script.txt",
    direct_text=True,
    seconds_per_word=0.45,
)
preds, segments = model.predict(events)
```

Predictions are returned on the `fsaverage5` cortical mesh, around 20k vertices.

## Installation

Basic inference:

```bash
pip install -e .
```

With plotting:

```bash
pip install -e ".[plotting]"
```

With the dashboard:

```bash
pip install -e ".[plotting,dashboard]"
tribev2-dashboard
```

With training dependencies:

```bash
pip install -e ".[training]"
```

## Dashboard

Launch the packaged dashboard with:

```bash
tribev2-dashboard
```

The dashboard supports:

- Video upload
- Audio upload
- Raw text or `.txt` input
- One image, or comparison of up to two images side by side
- Local inference
- Event dataframe inspection
- Timestep-by-timestep cortical visualization
- Interactive 3D cortical view in the browser
- MP4 export of prediction dynamics
- Built-in explanation of what the brain maps show
- CSV / NPY export of predictions

## Practical notes

- The packaged dashboard defaults to the public [unsloth/Llama-3.2-3B](https://huggingface.co/unsloth/Llama-3.2-3B) text backbone, which avoids the gated Meta repo.
- Audio transcription still requires `uvx whisperx` if you enable ASR in the pipeline or dashboard.
- For NVIDIA Blackwell GPUs such as the RTX 5090, use a PyTorch build with CUDA 12.8+ and Blackwell support, such as `torch 2.7.x` or newer.

## Training

Set the original environment variables before running the research training scripts:

```bash
export DATAPATH="/path/to/studies"
export SAVEPATH="/path/to/output"
export SLURM_PARTITION="your_partition"
```

Then run either:

```bash
python -m tribev2.grids.test_run
python -m tribev2.grids.run_cortical
python -m tribev2.grids.run_subcortical
```

## Project structure

```text
tribev2/
|-- main.py
|-- model.py
|-- pl_module.py
|-- demo_utils.py
|-- easy.py
|-- dashboard_app.py
|-- cli.py
|-- eventstransforms.py
|-- plotting/
|-- studies/
`-- grids/
```

## Citation

```bibtex
@article{dAscoli2026TribeV2,
  title={A foundation model of vision, audition, and language for in-silico neuroscience},
  author={d'Ascoli, Stephane and Rapin, Jeremy and Benchetrit, Yohann and Brookes, Teon and Begany, Katelyn and Raugel, Josephine and Banville, Hubert and King, Jean-Remi},
  year={2026}
}
```

## License

This project remains licensed under CC-BY-NC-4.0. See [LICENSE](LICENSE).
