# Tactile Material Classification from Deformation & Shear (C2b)

**Research Track II · COGAR Assignment C2b · University of Genoa (DIBRIS)**
Amr Magdy Mohamed Elsayed Abdalla (S8082888) · Gianluca Galvagni

Material classification from the **mechanical-response** modalities of a vision-based
tactile sensor — deformation and shear — without using vision.

## Research question

> Does fusing deformation and shear improve material-classification accuracy over
> either modality alone, and how robustly does that advantage hold as contact
> conditions vary?

## Headline results (10 materials, honest leakage-free evaluation)

| Finding | Result |
|---|---|
| Spatial footprint vs. raw magnitude | **65.5%** vs. 42.8% — identity is geometric |
| Fusion vs. best single modality | **+4 to +9 points** in every condition |
| Temporal attention vs. classical ceiling | 71% → **~80%** honest held-out |
| Best single configuration (label smoothing) | **82.8%** |
| Reported result | **~80% ± 5–6%** on 10 materials |
| Visually identical PLA cubes | **0 cross-confusions** — touch beats vision |

Accuracy is bounded by data quantity (hand-collected, ~20 trials/class), not by
model capacity — shown by a nine-configuration regularization sweep.

## Repository layout
