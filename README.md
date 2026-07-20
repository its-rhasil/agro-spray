# Plant Disease Detection & Automated Pesticide Spraying System

A multi-plant disease detection system using deep learning and transfer learning, deployed on a Raspberry Pi 5-powered robotic car for automated, targeted pesticide spraying. Built as a final-year engineering project, supporting a faculty advisor's broader research work.

## Project Overview

The system uses a camera-equipped robotic car (Raspberry Pi 5 + ESP32) to identify diseased leaves in real time and trigger automated spraying — entirely on-device, no cloud dependency. Current development uses betel, plumeria, and hibiscus as working plants (**final plant selection not yet confirmed** — these reflect the data collected so far, not a locked-in list), with plans to expand to 8–10 plants overall.

**Current status:** binary healthy/diseased classification for betel is working and validated. Multi-disease classification (identifying *which* disease, not just healthy/diseased) and the full multi-plant pipeline are in progress — see [Roadmap](#roadmap).

## Hardware

- Raspberry Pi 5 (4GB RAM) — on-device inference, no cloud
- ESP32 — controls the robotic car and spraying mechanism
- Camera module — captures leaf images in the field

## Dataset

| Plant | Healthy | Diseased |
|---|---|---|
| Betel | 1080 | 956 |
| Plumeria | 100 | 92 |
| Hibiscus | 160 | 92 |

- Split 70/15/15 (train/val/test) via `prepare.py`
- Known issues: betel images are lower resolution than plumeria/hibiscus; plumeria and hibiscus are severely under-represented (target: 300–400 images/class before full multi-plant integration); betel and hibiscus are visually similar, flagged as a confusion risk for plant-ID stages

## Model & Approach

- **Backbone:** ResNet18 (pretrained on ImageNet), used for experimentation — small, fast, well documented
- **Deployment target:** MobileNetV3-Small / MobileNetV2 (edge-optimized, faster CPU inference on the Pi) — not yet benchmarked
- VGG16 and ResNet50 were evaluated and ruled out — too heavy for on-device inference on the Pi

### Training strategy: staged transfer learning

1. **Frozen backbone** — only the final FC layer trained, all other layers frozen. Establishes a baseline with minimal compute.
2. **Layer4 fine-tuning** — `layer4` unfrozen and trained at a small learning rate (to avoid large deviation from pretrained ImageNet features), FC layer continues training at its original rate. This stage produced the current best model.

### Layer4 learning-rate search

A controlled 3-way experiment varied the FC layer's learning rate (`layer4` fixed at `1e-4`), all runs starting from the same checkpoint and random seed for a fair comparison:

| FC LR | Best val acc | Diseased recall | Missed diseased (FN) |
|---|---|---|---|
| **1e-3** | **0.954** (epoch 9) | **0.937** | **9** |
| 5e-4 | 0.951 | 0.930 | 10 |
| 1e-4 | 0.948 | 0.909 | 13 |

`fc_lr = 1e-3` (unchanged from the frozen-backbone stage) performed best and is the current production checkpoint.

**Note on overfitting:** all three configs showed train accuracy climbing to ~99.5% by epoch 10 while validation accuracy plateaued around 94–95%, with validation loss spiking mid-run (epoch 8) even where validation accuracy held steady — a sign that loss is a more sensitive early overfitting signal than accuracy alone. Best-checkpoint saving (rather than last-epoch saving) protects the final model from this degradation. Addressing this properly (via data augmentation) is a near-term next step.

### Why recall on the diseased class matters most

For a spraying system, a false negative (diseased leaf classified as healthy) means real disease goes untreated and spreads — a worse outcome than a false positive (spraying a healthy leaf, which is merely wasteful). All evaluation in this project prioritizes diseased-class recall over aggregate accuracy for exactly this reason.

## Training Infrastructure

The training loop (see `notebooks/`) includes:

- **Best-checkpoint saving** — only saves when validation accuracy improves on the current best, not at every epoch or only at the end. Across every run in this project, the best epoch was never the final one — this was essential, not optional.
- **Early stopping with patience** — training stops after a set number of epochs (patience) without improvement. Patience was calibrated against real observed noise in the validation curve (max 3-epoch losing streak observed → patience set to 5) rather than an arbitrary guess.
- **Per-epoch history logging** — train/val loss and accuracy logged every epoch to CSV via `pandas`, independent of whether that epoch was a new best, for later plotting and diagnosis.
- **Confusion matrix evaluation** — computed on the validation set at each milestone checkpoint (not just at the end), tracking diseased-class recall specifically, since aggregate accuracy alone can mask asymmetric errors.

## Repository Structure

```
plant-disease-detection/
  prepare.py                        # plant-agnostic train/val/test split
  notebooks/
    01_betel_baseline.ipynb         # frozen-backbone training
    02_betel_layer4_lr_search.ipynb # layer4 fine-tuning + LR search
  .gitignore
  README.md
```

`.gitignore` excludes `data/`, `*.pth`, `*.pt`, `__pycache__/`.

## Roadmap

- [x] Data augmentation (rotation, flips, crop/zoom, mild brightness/contrast — conservative on color, since disease symptoms are color-dependent) to close the train/val gap observed in layer4 fine-tuning
- [ ] **Multi-disease classification** — extend beyond healthy/diseased to identify the *specific* disease per plant (new requirement; architecture and data labeling still being scoped)
- [ ] Collect additional plumeria and hibiscus images (target: 300–400/class)
- [ ] Repeat the frozen-backbone → layer4 pipeline for plumeria and hibiscus
- [ ] Plant identification model (3-way classifier), with specific attention to betel/hibiscus confusion
- [ ] Evaluate architecture options for the full multi-plant + multi-disease system with real experimental numbers: flat classifier, hierarchical (plant-first), disease-first, and multi-head — decision to be evidence-based, not theoretical
- [ ] MobileNetV3-Small comparison and on-device inference benchmarking on Raspberry Pi 5
- [ ] Multi-frame inference aggregation for the deployed system — averaging per-frame diseased-class probabilities across 2–3 frames per capture before making a spray decision, biasing the threshold toward catching disease (false negatives are the costly failure mode) rather than pure 50/50 confidence
- [ ] Final evaluation on the held-out test set (reserved untouched until the pipeline is finalized)

## Key Engineering Notes

- Test set is reserved strictly for a single, final evaluation once the whole pipeline is settled — all decisions during development use the validation set only.
- Checkpoints are always reloaded fresh immediately before evaluation, rather than trusting whatever is currently in memory, to guarantee the reported metrics match the actual saved artifact.
- A single noisy epoch (a spike immediately followed by decline) is not treated as evidence of real improvement — a result is trusted only if it's sustained across several epochs, not just present once.