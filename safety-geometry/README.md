# Safety Geometry: Do Multimodal LLMs Geometrically Perceive Visual Harm?

**Hypothesis**: Text-based safety training creates a safe/harmful direction in the LLM's hidden state space. Harmful images, when processed by the same LLM, land on the *safe side* of this direction — explaining why multimodal models often fail to refuse visually harmful inputs.

---

## Repository Structure

```
safety-geometry/
├── config.yaml            # All hyperparameters
├── run_pipeline.py        # Entry point — runs all steps
├── data/
│   ├── pku_saferlhf.py    # PKU-SafeRLHF loader (text, safe/harmful pairs)
│   └── visual_harm.py     # HOD / ToViLaG loader (images + labels)
├── extract/
│   └── hidden_states.py   # Hidden state extraction from any HF multimodal LM
├── analysis/
│   ├── layer_locator.py   # Identify safety-active layers via linear probing
│   └── pls_direction.py   # PLS direction fitting + projection
└── visualize/
    └── plots.py           # All figures
```

---

## Experimental Steps

### Step 1 — Extract Hidden States
For each input (text or image+prompt), run a forward pass through the LLM and record the hidden state at each layer.

- **Text inputs**: PKU-SafeRLHF prompt+response pairs, labeled safe (1) or harmful (0).
- **Image inputs**: HOD/ToViLaG images, each paired with a neutral prompt.
- **Pooling**: `last` (last non-padding token) or `mean` (all tokens). Both are supported via `config.yaml`.
- **Output shape**: `(N_samples, d_hidden)` per layer.

### Step 2 — Locate Safety-Active Layers
Fit a logistic regression probe (safe vs harmful) on each layer's hidden states using cross-validation. The layers with the highest accuracy are the ones where the model geometrically distinguishes safe from harmful *text* content. Only these layers are used downstream.

**Why not all layers?** Most layers carry no safety-relevant signal; using all of them adds noise and computation with no benefit.

### Step 3 — Find the Safe/Harmful Direction with PLS
On the safety-active layers, fit a **Partial Least Squares (PLS)** regression between hidden states (X) and safety labels (Y). The first PLS component is the *safety direction* — the axis of maximum covariance between the representation space and the safe/harmful distinction.

PLS is preferred over PCA because PCA maximizes variance in X regardless of Y, while PLS directly optimizes for the safety-relevant axis.

**Critical design choice**: The direction is fit *only on text data*. Image hidden states are projected onto this text-derived direction — this is intentional. We want to measure where images land in the *text safety space*, not build a separate image classifier.

### Step 4 — Project Image Hidden States
Project harmful and safe image hidden states onto the text-derived PLS direction. The score sign tells us which side of the safety boundary each image falls on.

**Expected finding** (if the hypothesis holds): Harmful images score *positive* (safe side), meaning the model's internal safety mechanism does not "see" the visual harm. This is stronger evidence than cosine similarity because it operates directly in the safety decision space.

**Boundary analysis**: Samples with `|score| < threshold` are near the decision boundary — these are the most ambiguous cases and provide the clearest evidence of model uncertainty at the text/image interface.

---

## Data Setup

```
data/
├── pku_saferlhf/        ← HF snapshot: PKU-Alignment/PKU-SafeRLHF
│   └── train.parquet
└── visual_harm/
    ├── labels.csv       ← columns: image_path, label (0=safe, 1=harmful)
    └── images/
        └── *.jpg / *.png
```

Download PKU-SafeRLHF:
```bash
huggingface-cli download PKU-Alignment/PKU-SafeRLHF --local-dir data/pku_saferlhf
```

---

## Usage

```bash
pip install -r requirements.txt

# Full pipeline (both text and image)
python run_pipeline.py

# Text only (no image data needed)
python run_pipeline.py --text-only

# Use cached hidden states (skip re-running the model)
python run_pipeline.py --skip-extract

# Override pooling strategy or target specific layers
python run_pipeline.py --pooling mean
python run_pipeline.py --layers 14 15 16 17

# Custom config
python run_pipeline.py --config my_config.yaml
```

---

## Outputs

All outputs are saved to `outputs/`:

| File | Description |
|------|-------------|
| `plots/layer_accuracy.png` | Linear probe accuracy per layer; red = selected safety layers |
| `plots/projection_layer{N}.png` | KDE of PLS scores: text safe/harmful and image safe/harmful |
| `plots/boundary_layer{N}.png` | Samples near the decision boundary (most ambiguous) |
| `plots/summary_heatmap.png` | Mean PLS score per group across all safety layers |
| `cache/text_layer{N}.npy` | Cached text hidden states |
| `cache/image_layer{N}.npy` | Cached image hidden states |

**Interpreting the key plot** (`projection_layer{N}.png`):
- Green curve: text-safe samples cluster on the right (+).
- Red curve: text-harmful samples cluster on the left (−).
- Orange distribution: harmful images. If this overlaps with the green (safe) region, the safety mechanism is geometrically blind to the visual harm.

---

## Configuration

All parameters are in `config.yaml`. Key fields:

| Field | Default | Meaning |
|-------|---------|---------|
| `model.name` | `llava-hf/llava-1.5-7b-hf` | Any HF multimodal LM |
| `extraction.pooling` | `last` | `last` or `mean` token pooling |
| `extraction.layers` | `null` | null = all layers |
| `analysis.n_safety_layers` | `5` | Top-k layers by probe accuracy |
| `analysis.n_pls_components` | `5` | PLS components (first used as direction) |
| `analysis.boundary_threshold` | `0.35` | `|score| < threshold` = boundary sample |
