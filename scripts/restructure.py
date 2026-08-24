"""Rebuild cnn-project.ipynb into a narrative, portfolio-ready notebook.

Code cells are kept verbatim (with their saved outputs) — only markdown is
added/replaced, plus the Kaggle boilerplate cell is dropped.
"""
import base64
import json
import os


SRC = "cnn-project.ipynb"
DST = "notebooks/histopathologic-cancer-detection.ipynb"

os.makedirs("notebooks", exist_ok=True)
os.makedirs("figures", exist_ok=True)
nb = json.load(open(SRC))
old = nb["cells"]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


cells = []

cells.append(md("""# Histopathologic Cancer Detection with a CNN

**Goal:** identify metastatic cancer in small image patches taken from digital pathology scans of lymph node sections.

This notebook works through the [Kaggle Histopathologic Cancer Detection](https://www.kaggle.com/competitions/histopathologic-cancer-detection) dataset (a cleaned subset of the [PatchCamelyon](https://github.com/basveeling/pcam) benchmark). Each sample is a 96×96 RGB `.tif` patch; a patch is labeled **positive** if the *center 32×32 region* contains at least one pixel of tumor tissue.

**Contents**

1. [Setup](#1.-Setup)
2. [Exploratory Data Analysis](#2.-Exploratory-Data-Analysis) — label balance, structural validation, visual inspection, artifact detection, and pixel-intensity analysis
3. [Modeling](#3.-Modeling) — a small CNN whose architecture *and* optimizer settings are selected by Bayesian hyperparameter search (KerasTuner)
4. [Results & Discussion](#4.-Results-&-Discussion)

> Written to run in a Kaggle notebook environment with the competition dataset mounted at `/kaggle/input/histopathologic-cancer-detection`. To run locally, download the data (see the repository's `data/README.md`) and update the path in the first cell of section 2.
"""))

cells.append(md("""## 1. Setup

Standard scientific-Python stack for the EDA; TensorFlow/Keras and KerasTuner are imported later, in the modeling section, since the EDA can be run without them.
"""))
cells.append(old[1])  # imports

cells.append(md("""## 2. Exploratory Data Analysis

Before modeling, I verify the dataset is what the documentation claims it is: the advertised sample counts, a consistent image format, no corrupted files, and no surprises in the images themselves. Findings from each check are summarized as they appear.
"""))

cells.append(md("""### 2.1 Initial Data Assessment

Load the label table, resolve each image ID to a file path, and confirm the advertised dataset sizes (220,025 labeled training patches, 57,458 test patches).
"""))
cells.extend(old[5:10])  # paths, load+assert, head, gen paths, class balance

cells.append(md("""**Observation — class balance.** 89,117 positive vs. 130,908 negative patches (≈40.5% / 59.5%). The imbalance is mild, so I keep plain binary cross-entropy and rely on AUC — which is insensitive to class priors — as the selection metric, rather than reweighting or resampling.
"""))

cells.append(md("""### 2.2 Structure Verification

Every image should be a 96×96×3 array. Opening all 220k files is slow, so a random sample of 3,000 is checked for shape and readability.
"""))
cells.extend(old[11:13])

cells.append(md("""**Observation.** All 3,000 sampled images are readable and exactly 96×96×3 — no resizing or corruption handling is needed in the input pipeline.
"""))

cells.append(md("""### 2.3 Visual Inspection

A quick look at random patches from each class. Tumor tissue is only guaranteed to appear in the *center 32×32* of positive patches — the outer border exists to give context (and to allow rotation/translation augmentation without moving tumor out of frame). This motivates the `CenterCrop` layer tuned in section 3.
"""))
cells.extend(old[14:16])  # viz functions, sample grid

cells.append(md("""#### Artifact detection

Whole-slide scans often contain patches that are almost entirely slide background (white) or scanner border (black). A cheap screen — flagging images whose overall pixel standard deviation falls below a threshold — quantifies how common these degenerate patches are, on a 10,000-image sample.
"""))
cells.extend(old[16:20])  # detect fn, run, viz fn, run

cells.append(md("""#### Side-by-side class comparison

Interleaving positive and negative patches makes the qualitative differences easier to see: positive patches tend to be densely cellular (heavily stained, darker), while many negatives contain large regions of stroma, fat, or background.
"""))
cells.append(old[20])

cells.append(md("""**Observation — artifacts are rare and slightly informative.** Only 33 of 10,000 sampled patches (≈0.3%) are essentially monochromatic — 28 white (slide background) and 5 dark. 31 of the 33 are negatives, which makes sense: a blank patch cannot contain tumor in its center. At this prevalence they are left in the training data; a production pipeline might filter or downweight them.
"""))

cells.append(md("""### 2.4 Pixel Intensity Distributions

Do the classes differ in raw color statistics? Pixel intensities are aggregated per channel over a 5,000-image sample for each class.
"""))
cells.extend(old[22:26])  # analyze fn, run, viz fn, run

cells.append(md("""**Observation — staining density separates the classes at the pixel level.** Positive patches are systematically darker and less variable than negatives, with the largest gap in the **green channel** (mean difference ≈ −19 intensity levels). Under H&E staining, hematoxylin-stained nuclei absorb strongly in green, so dense tumor cellularity shows up exactly there. The negative class also has a large spike at intensity ≈255 — the white background patches found above. The distributions overlap heavily, so color statistics alone won't classify patches, but this confirms there is real, learnable signal and suggests even simple models should beat chance comfortably.
"""))

cells.append(md("""## 3. Modeling

Rather than hand-picking one architecture, I define a *family* of small VGG-style CNNs and let **Bayesian optimization (KerasTuner)** choose among them. The search space covers both architecture and optimization:

| Group | Hyperparameters |
|---|---|
| Input | center-crop size (32–96 px) |
| Conv stack | number of blocks (1–3), convs per block (1–3), initial filters (16–64), filter growth rate (1.25–2×), padding, batch-norm on/off |
| Head | dense units (128–256), dropout (0.4–0.8) |
| Loss | label smoothing (1e-8–1e-1) |
| Optimizer | Adam vs. RMSprop, learning rate (1e-5–1e-3), β₁/β₂ or ρ/momentum |

The tunable center-crop directly exploits the labeling rule from section 2.3: since only the center 32×32 determines the label, the border may be more distraction than context.
"""))

cells.append(md("""### 3.1 Model Specification
"""))
cells.append(old[27])

cells.append(md("""The tuner runs 16 trials of Bayesian optimization (8 random warm-up points), with early stopping on validation AUC.
"""))
cells.append(old[28])

cells.append(md("""### 3.2 Data Pipeline

A stratified 67/33 train/validation split feeds two Keras `ImageDataGenerator`s. Training data gets augmentation appropriate for histopathology — patches have no canonical orientation, so free 90° rotations and both flips are label-preserving — plus mild brightness and zoom jitter. Validation data is only rescaled to [0, 1].
"""))
cells.extend(old[31:33])

cells.append(md("""### 3.3 Bayesian Hyperparameter Search

Each trial trains for up to 10 short epochs (100 steps ≈ 12,800 images per epoch) — enough to rank configurations without paying for full convergence.
"""))
cells.extend(old[34:36])

cells.append(md("""### 3.4 Final Training

The best configuration from the search is trained further with 10× more steps per epoch.
"""))
cells.append(old[36])

cells.append(md("""## 4. Results & Discussion

**Search results.** Over 16 trials (≈3 hours on a Kaggle GPU), the best configuration reached a **validation AUC of 0.869**. Training metrics during the final run continued to improve, reaching **training AUC 0.905 / accuracy 0.835** by epoch 7.

**Known issue — validation collapse during final training.** In the extended training run the validation metrics are pathological: `val_AUC` is pinned at exactly 0.5, validation accuracy sits at the majority-class rate (0.587), and validation loss explodes (37 → 255) while *training* metrics steadily improve. Metrics this degenerate usually mean the evaluation path is broken rather than the model failing to learn. The leading suspects:

1. **Batch-normalization inference statistics.** `batch` is a tuned hyperparameter; if the winning model uses BatchNorm, its moving mean/variance can be badly mis-estimated after short-epoch tuning followed by continued training, making inference-mode (validation) outputs saturate — consistent with confident wrong predictions and a huge BCE loss.
2. **Retraining a tuner-returned model.** `get_best_models()[0]` returns the trial's trained weights; continuing `fit()` on it with a fresh optimizer state (and the tuner's checkpointing behavior) is fragile. The standard remedy is to rebuild the model from the best *hyperparameters* (`tuner.get_best_hyperparameters()`) and retrain from scratch.
3. `validation_steps=100` evaluates only ~12,800 of the 72,609 validation images per epoch — not the cause of a 0.5 AUC, but worth fixing for stable estimates.

**Next steps**

- Rebuild from best hyperparameters and retrain from scratch with full validation passes; confirm validation AUC recovers to the ~0.87 seen in the search.
- Replace `ImageDataGenerator` (deprecated) with a `tf.data` pipeline for throughput.
- Benchmark against transfer learning (e.g. an ImageNet-pretrained ResNet/EfficientNet backbone), the usual stronger baseline for PCam.
- Generate predictions on the 57,458-image test set and submit for a leaderboard score.
"""))

nb["cells"] = cells
json.dump(nb, open(DST, "w"), indent=1)
print("cells:", len(cells))

# Export key EDA figures embedded in the original notebook's outputs.
FIGS = {
    15: "figures/eda-sample-images.png",
    19: "figures/monochromatic-artifacts.png",
    20: "figures/class-comparison.png",
    25: "figures/pixel-intensity-distributions.png",
}
for idx, dest in FIGS.items():
    pngs = [o["data"]["image/png"] for o in old[idx].get("outputs", []) if "image/png" in o.get("data", {})]
    assert pngs, f"no image output in source cell {idx}"
    with open(dest, "wb") as f:
        f.write(base64.b64decode("".join(pngs[-1])))
    print("wrote", dest)
