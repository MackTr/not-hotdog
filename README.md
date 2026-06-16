# Not Hotdog 🌭

A binary image classifier that decides: **Hotdog** or **Not Hotdog**.
Inspired by HBO's *Silicon Valley*. This is a learning project — the goal is to
*understand* how a CNN works, not just to get a number.

## The learning path

Each phase is one sit-down session. Do them in order — the early phases build the
intuition that makes the later ones make sense.

| Phase | File | What you'll learn |
|-------|------|-------------------|
| 0 | (this README + setup) | Project + environment setup |
| 1 | `notebooks/01_images_as_tensors.ipynb` | An image is just a grid of numbers; what a convolution does |
| 2 | `notebooks/02_data.ipynb` | Loading data, train/val/test splits, transforms & augmentation |
| 3 | `notebooks/03_train_from_scratch.ipynb` | Build a small CNN by hand; write the training loop; *see overfitting* |
| 4 | `notebooks/04_transfer_learning.ipynb` | Fine-tune a pretrained model; watch accuracy jump |
| 5 | `notebooks/05_evaluate.ipynb` | Confusion matrix, worst mistakes, Grad-CAM heatmaps |
| 6 | `src/app.py` | Ship a Gradio demo: drag in a photo, get a verdict |

## Setup

```bash
# 1. Create the environment (one time)
conda create -n nothotdog python=3.11 -y
conda activate nothotdog

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Jupyter and open the Phase 1 notebook
jupyter lab
```

Every time you come back to the project:

```bash
conda activate nothotdog
jupyter lab
```

## Project layout

```
not-hotdog/
├── data/         # datasets live here (git-ignored — too big to commit)
├── notebooks/    # one notebook per learning phase
├── src/          # reusable code + the final Gradio app
├── models/       # saved model weights (git-ignored)
├── requirements.txt
└── README.md
```

## Hardware note

You're on a Mac. PyTorch can use the Apple GPU via the **MPS** backend, so training
is fast — no cloud GPU needed. The code checks for MPS automatically.
