# Project Narrative — Tactile Material Classification from Deformation & Shear

*The complete arc of the project, told as research stages rather than by collection date. Each stage is framed by the question it answered and the result that motivated the next. This is the logical spine for the paper's Method, Results, and Discussion sections.*

---

## Stage 0 — Literature review and the research question

The project began by situating itself against four works spanning the field (full analysis in `literature_analysis.md`):

- **GelSight (Yuan et al., 2017)** established the physical foundation: a soft elastomer's vertical deformation encodes contact *geometry*, while lateral/marker motion is roughly proportional to *shear force*. This grounds the two modalities the project studies.
- **Amin, Gianoglio & Valle (2023)**, from the same lab (UniGe/DITEN) on the same robot (Baxter), provided the pipeline template — tactile sensing → low-cost frame features → classical + neural classifiers — but reduced each frame to mean/std of a *pressure* array, and classified *hardness*, not material identity.
- **Liao et al. (2025)** validated the Daimon sensor lineage and the value of *normal-force dynamics* (dFz/dt) — but used only normal force, discarding shear.
- **FG-CLTP (2025)** marked the state-of-the-art ceiling, arguing that *quantitative* contact states (force magnitude, contact geometry) carry more than qualitative texture descriptors.

**The gap these left open — and the project's question.** Amin averaged the spatial field away; Liao discarded shear; none ran a controlled test of whether *fusing deformation and shear* beats either alone. That became the research question:

> *Does fusing deformation and shear improve material-classification accuracy over either modality alone, and how robustly does that advantage hold as contact-condition variability increases?*

Three working hypotheses framed it: **H1** — fusion beats single-modality at low variability; **H2** — the fused advantage widens as variability grows; **H3** — each single modality fails on a predictable subset of materials.

---

## Stage 1 — Sensor setup and the collection-method pivot

The hardware is the **Daimon WS vision-based tactile sensor**, which returns per-frame deformation and shear fields, downsampled to a (frames, 30, 40, 2) tensor per modality. The intended platform was the **Baxter robot** for repeatable, force-controlled grasps.

**The constraint that shaped everything.** Baxter access was limited (shared with other students), so a reliable robotic constant-force protocol was not feasible within the timeline. The project pivoted to **hand collection** — pressing materials against the held sensor — using a release-gated collector that guaranteed one clean press per trial.

This pivot is the single most consequential decision in the project. It made data collection possible, but it introduced the **force confound**: because pressing force is applied by hand and not held constant, applied force is entangled with material identity. Every downstream result — and the project's main limitation — flows from this. The honest framing throughout is that results are *prototyping/validation grade*, not *force-controlled deployment grade*.

---

## Stage 2 — Pilot study: the magnitude trap and the spatial breakthrough

The first dataset was a 6-material pilot. The earliest models used **magnitude-only features** (peak force, mean, std — mirroring Amin's mean/std approach) and plateaued around **40%**. Several deep models that averaged across the spatial field (`train_dl_timeseries`) did even worse (20–33%).

**The diagnostic turn.** Exploratory analysis (Kruskal–Wallis tests, PCA, per-material contact imprints) revealed why: a magnitude-heavy PCA explained ~97% of variance along essentially one axis — "how hard the press was" — while the classes overlapped along it. The discriminative information was not in *how much* the sensor deformed but in the *spatial shape* of the deformation.

Re-engineering the feature set around **spatial-footprint geometry** (centroids, spreads, spatial moments of both fields) broke the ceiling: a 30-feature spatial+magnitude set reached ~**80% on 6 materials**, with spatial features dominating. This is the project's first real finding and the conceptual core of everything after: **material identity lives in the contact geometry, not the contact magnitude.** It directly extends Amin (whose mean/std pooling discards exactly this) and instantiates FG-CLTP's quantitative-contact-state thesis at small scale.

---

## Stage 3 — Scaling the dataset: 10 materials, two contact modes

To move from pilot to a real study, the dataset was rebuilt and scaled: **11 materials × two contact modes × 20 trials** (soft_bottle later excluded as too noisy → **10 materials**). The two modes were deliberate, and they *are* the variability axis for the research question:
- **press** — a controlled, consistent contact.
- **airhold** — the material squeezed against the sensor in the air, producing more variable contact but **stronger shear** (compliant/textured materials slip and deform tangentially).

This design lets the project test robustness to contact-condition variation (H2) directly, by comparing press, airhold, and pooled performance — turning the collection-method constraint into an experimental variable.

---

## Stage 4 — Classical analysis: the two ablations that answer the question

With the scaled dataset, classical models (SVM, Random Forest, XGBoost, 5-fold CV) established the baseline and ran the two ablations the literature never did.

**Accuracy baseline:** SVM and RF converged at **~71–75%** across conditions; XGBoost did not exceed SVM. Four model families landing near the same number suggested a ceiling.

**Feature-group ablation (the spatial finding, confirmed at scale):**
- magnitude-only ≈ **43%**, shape-only ≈ 26%, **spatial-only ≈ 65%**, all-features ≈ **71%**.
- Spatial features beat magnitude by ~1.5× and reach within ~5 points of the full set. The Stage-2 finding holds at 10 materials.

**Modality ablation (the direct answer to the research question — H1):**
- deformation-only ≈ 62–68%, shear-only ≈ 55–65%, **fused ≈ 71–74.5%**.
- **Fusion beats the best single modality by 4–9 points in every condition.** This is the headline answer: fusing deformation and shear *does* help — substantially more than in the rigid-only pilot, because the compliant/textured materials added at scale finally activate the shear channel.

**Why fusion now helps (H3 mechanism):** feature-importance analysis put `def_spread_y` and `shear_spread_y` as the top features — the *spatial spread of both fields*. Shear, negligible on rigid pilot materials, became informative once rubber and textured plastics were present (≈4× shear range in airhold). The single modalities fail on predictable subsets (deformation struggles where stiffness is similar; shear is weak on smooth rigid items), and fusion covers both — confirming H3.

---

## Stage 5 — Deep learning: locating the ceiling

The ~71% convergence raised the question: is this a *data* ceiling or an *architecture* ceiling? A controlled progression of deep models answered it.

1. **Baseline CNN-LSTM ≈ 71%** — matched the classical ceiling. A naive temporal model (uniform frame striding, simple pooling) bought nothing over hand-designed spatial features.
2. **Attention CNN-LSTM (BiLSTM + temporal self-attention) ≈ 85% (biased eval)** — a large jump. Letting the model *re-weight which moments of the loading sequence matter* broke the convergence.
3. **Dual-branch (separate deformation/shear branches) ≈ 84.5%** — **no gain** over single-stream, despite ~2× the parameters.

The dual-branch null result is a clean isolation experiment: the improvement came specifically from **temporal attention**, not from added capacity or from architecturally separating the modalities (a single joint network already fuses them). The ceiling was therefore *partly architectural* — specifically, the earlier models modeled time poorly — not purely a data wall.

---

## Stage 6 — The honesty correction: how the headline number was earned

The attention results carried a methodological flaw inherited from the early scripts: **early stopping selected the epoch with the best *test-fold* accuracy** — peeking at the data being scored. With high-capacity models and many epochs, this inflates the number.

The fix (`run_05_attention_honest.py`) carved a **validation split out of the training fold**, selected the epoch by validation accuracy, and scored the test fold **once**, never using it for any decision. Verified zero leakage.

**Effect:** the honest number landed at **~79–80%** across conditions (press 80.5%, airhold 79.5%, pooled 79.2%) — about 6 points below the biased 85.5%. This is the reviewer-proof headline. The *relative* gains over baseline remained valid (baseline used the same biased scheme), but the absolute number is now defensible. Reporting a solid 79% rather than a questionable 85% is the difference between a result that survives scrutiny and one that does not.

A persistent **+12–17% train-test gap** (train ~93%, test ~80%) emerged here as the signature of the remaining limitation.

---

## Stage 7 — The regularization sweep: diagnosing the gap

To test whether the gap could be closed — and what *kind* of limit it was — nine configurations were run on the honest-eval model:

| Intervention | Effect | Reading |
|---|---|---|
| **Label smoothing 0.1** | **82.8%** (gap +12.4%) | ✅ best observed; gentle output regularization helps |
| Weight decay 1e-3 | 79.8% | ✅ marginal |
| Baseline | 79.2% | reference |
| Label-preserving augmentation | 78.5% | ≈ neutral (can't manufacture data) |
| Combined (smooth + decay) | 80.5% | no better than smoothing alone; gap back to +14% |
| Capacity reduction (small model) | 75.5% | ❌ lowered ceiling, gap unchanged |
| ±1px translation | 75.5% | ❌ perturbs real position features |
| Loading-window crop | 74.8% | ❌ removed steady-state signal |
| Aggressive augmentation (rotate/scale) | 74.0% | ❌ corrupts the label |

**The diagnosis.** The gap stayed at +12–17% under *every* intervention. Capacity reduction lowered both train and test equally (gap unchanged) → the gap is **not capacity-driven**. Augmentation was neutral or harmful → it **cannot substitute for real trials**. Only gentle output regularization helped, and only by ~1.6 points. Two re-runs of the *identical* architecture scored 82.8% and 77.0% — a 6-point swing from random seed alone.

**Conclusion:** the gap is **data-quantity-driven** (~20 trials/class), not fixable by architecture or data manipulation. This converts "we need more data" from a hand-wave into an evidence-backed claim.

**A second finding fell out of this stage:** tactile data admits an *unusually narrow set of valid augmentations*. Because the label is encoded in spatial geometry and force magnitude, rotation, magnitude-scaling, cropping, and even ±1px translation all corrupt the label and hurt — only temporal jitter and small sensor-scale noise are label-preserving. This is itself a methodological contribution about the modality.

---

## Stage 8 — Consolidation: the honest result

The final reported result is framed as a **band, not a peak**:

> The attention CNN-LSTM achieves **~80% ± 5–6% honest held-out accuracy** on 10 materials (5-fold CV, validation-based early stopping). The best single configuration — **label smoothing 0.1 — reached 82.8%**. No regularization or architectural variant reliably exceeded this band; the persistent train-test gap is data-quantity-driven.

Supporting findings, all consistent across models:
- **Spatial footprint ≫ magnitude** (~65% vs ~43%) — material identity is geometric.
- **Fusion beats single modality by 4–9 points** — the research question answered affirmatively, with a physical mechanism (shear activation by compliant/textured materials).
- **Temporal attention broke the classical ceiling** (71% → ~80%); dual-branch confirmed attention, not capacity, was the cause.
- **"Touch beats vision" is airtight:** the visually-identical PLA cubes (pla_hard vs pla_soft) show ≤2 cross-confusions in every honest confusion matrix — essentially never mistaken for one another by touch, though a camera cannot tell them apart.
- **Residual errors track genuine physical similarity** — the rigid-flat cluster (hardwood/carton_box/pla_hard) and the two cups (steel/ceramic) — not model artifacts.

---

## Stage 9 — What remains

The **modeling phase is complete**: the sweep shows further tuning resamples noise rather than revealing signal. Remaining work is deliberately scoped:

- **Cross-mode transfer (Task 4 sensitivity):** train on press, test on airhold and vice-versa — the script and data exist, unrun. This is the cleanest remaining quantification of robustness to contact-condition variation (H2).
- **Baxter deployment/transfer test:** train-on-hand, deploy-on-robot — the only genuinely *new* result still available, and the proper completion of the repeatability requirement. Pending robot access.
- **RT2 IEEE paper:** the results sections are written (`day2_results_v2.md`, `day3_results.md`); intro, related work (`literature_analysis.md`), method, discussion, conclusion remain.
- **Jupyter notebook deliverable.**
- **Citation re-verification** against the source PDFs before submission.

---

## The arc in one paragraph

A limited-access robot forced hand collection, which introduced a force confound but made a controlled study possible. Magnitude-only features failed (~43%); exploratory analysis traced the failure to discarded spatial structure, and spatial-footprint features recovered the signal (~71%). Scaling to 10 materials and two contact modes let the project run the ablations the prior literature never did: spatial geometry dominates magnitude, and **fusing deformation with shear beats either alone by 4–9 points** because compliant and textured materials activate the shear channel. A baseline CNN-LSTM matched the classical ceiling, but **temporal self-attention broke it to ~80%** under honest, leakage-free evaluation — and a dual-branch null result proved the gain came from attention, not capacity. A nine-configuration regularization sweep then showed the residual ~14% train-test gap is **data-quantity-driven**, immovable by architecture or augmentation, and that tactile data permits only a narrow set of label-preserving augmentations. Throughout, visually-identical PLA cubes were separated by touch with near-zero error — the clearest demonstration that this modality sees what vision cannot. The honest result is **~80% ± 5–6% on 10 materials**, a complete and self-aware answer to the research question, bounded by data scale rather than method.
