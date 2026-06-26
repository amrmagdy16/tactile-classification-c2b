# Literature Analysis — Tactile Material/Hardness Classification from Deformation & Shear

*This section positions the project against four works that span the field from foundational sensing principles to the current state of the art. Each is summarized as it actually reports, then connected to a specific decision or claim in this project. Page/▪claim-level facts below were checked against the source PDFs; where a paper does **not** support a convenient framing, that is noted, so the relationship is honest rather than rhetorical.*

---

## 1. GelSight — the physical foundation (Yuan, Dong & Adelson, *Sensors*, 2017)

**What it is.** A foundational vision-based optical tactile sensor. A camera images a soft elastomer pad; the elastomer's surface deformation is reconstructed as high-resolution 3D geometry. The paper's central reframing is that *geometry sensing is as important as force sensing*: GelSight "basically measures geometry," and "contact force and slip can be inferred from the sensor's deformation." When the surface is printed with markers, **the magnitude of marker motion is roughly proportional to the applied force or torque**, and the marker-motion field encodes normal force, shear force, and slip state.

**Relation to this project.** GelSight grounds the project's terminology and its core modeling assumption. It is the canonical justification for treating *deformation* (vertical/normal surface displacement → contact geometry) and *shear* (lateral/tangential displacement, proportional to tangential force) as the two physically meaningful channels — exactly the two modalities the Daimon WS sensor in this work provides. Two specific project decisions trace directly to GelSight:
- **Why spatial footprint geometry carries material identity.** GelSight establishes that the elastomer's deformation *is* a geometric imprint of the contact. This underpins the project's strongest empirical finding — that spatial features (e.g. `def_spread_y`) dominate raw magnitude features (~65% vs ~43% classical accuracy).
- **Why loading dynamics matter.** GelSight's treatment of the elastomer as a soft, viscoelastic medium motivates modeling the *temporal* loading sequence rather than a single peak frame — the basis for the temporal attention model in this work.

**Honest scope note.** GelSight is a *sensor-principles* paper, not a classification benchmark; it is cited for physical grounding, not as a performance comparator.

---

## 2. Amin, Gianoglio & Valle — the direct lab/method precedent (*Future Generation Computer Systems*, 2023)

**What it actually is.** The closest methodological precedent and, conveniently, the **same institution (DITEN, University of Genoa) on the Baxter robot**. The task is **5-class (and 3-class) object *hardness* classification** for resource-constrained robotic grippers. Three machine-learning algorithms are compared: **SVM, single-layer feed-forward neural networks (SLFNN, including FC and ELM), and a CNN**. Headline results: **CNN achieves best accuracy (>98%)**; SVM gives the lowest memory (1576 bytes), inference time (<0.077 ms), and energy (<5.74 µJ); SLFNNs reached ~96.3% on the 3-class problem. Features are the **mean and standard deviation of each tactile frame** (per 8×8 pressure image), a deliberately low-cost pre-processing choice for embedded deployment.

**Important modality distinction (do not overstate the match).** Amin's sensor is **P(VDF-TrFE) piezoelectric pressure arrays** — patches of 8 sensors producing pressure-over-time signals (grasp/release transients), **not a deformation+shear field like the Daimon sensor used here**. Ground-truth hardness was established separately with a materials testing machine (compression curves; 170 compression trials per object — these 170 are the *mechanical ground-truthing*, not the tactile-dataset trials). So Amin is a precedent for the **pipeline and lab context**, not for the sensing modality.

**Relation to this project.** Amin is the template this project both *imitates and extends*:
- **Pipeline imitation:** Baxter-mounted tactile sensing → low-cost frame features (mean/std) → classical + neural classifiers, compared head-to-head. This project mirrors that structure (SVM/RF/XGBoost + CNN-LSTM/attention) on the same robot and lab.
- **The extension that defines this project's contribution:** Amin reduces each tactile frame to scalar *mean/std* (magnitude statistics). This project shows directly — via feature-group ablation — that such magnitude statistics are the *weak* features (~43%), and that the **spatial distribution of the contact** (discarded by mean/std pooling) is what carries material identity (~65%). In other words, this project quantifies what Amin's frame-averaging leaves on the table.
- **Task framing:** Amin classifies *hardness*; this project classifies *material identity* across 10 materials (including visually-identical PLA cubes of different stiffness), a finer-grained and arguably harder discrimination.

---

## 3. Liao et al. — the sensor-specific reference (*Quantitative Hardness Assessment with Vision-based Tactile Sensing*, 2025)

**What it is.** The reference tied to the exact sensor family used here: the **DM-Tac vision-based tactile sensor from Daimon Robotics**, which "can capture high-resolution surface deformation" and decomposes contact into **normal and shear force components**. Liao estimates **fruit hardness from the *dynamics of the normal force*** during a single contact: hardness `H = g(dFz/dδ, dFz/dt, δmax)` — i.e. the slope/rate of the normal-force field, not its static peak. A universal criterion based on *average normal-force dynamics* is shown to generalize across fruit types.

**Relation to this project.**
- **Sensor lineage:** Liao validates the Daimon deformation+shear decomposition this project relies on, on the same vendor's hardware — direct evidence the modality is real and usable.
- **A concrete, citable feature idea:** Liao's use of the **normal-force rate (dFz/dt)** as the discriminative quantity supports this project's emphasis on *temporal loading dynamics* over a single peak frame. The attention model's value here — re-weighting the loading phase — is the learned analogue of Liao's hand-designed rate feature.
- **A contrast worth stating, not just a precedent:** Liao *decomposes* shear but then **uses only the normal force** for hardness. This project's central question is the complement — does *adding* shear to deformation help? The project's modality-ablation finding (fusion beats deformation-alone by 4–9 points once compliant/textured materials are present) is precisely the experiment Liao's normal-force-only method does not run. Liao is therefore both a sensor reference and a foil that motivates the fusion question.

**Honest scope note.** Liao is single-contact hardness *regression* on fruit, not multi-class material *classification* — so it is a methods/feature reference, not an accuracy comparator.

---

## 4. FG-CLTP — the state-of-the-art ceiling (Ma et al., 2025)

**What it is.** A current state-of-the-art framework: **Fine-Grained Contrastive Language–Tactile Pretraining**. It aligns **3D tactile point clouds** with quantitative, contact-state-aware language descriptions (a CLIP-style contrastive objective), using a 100k-pair dataset and a discretized numerical tokenization that injects explicit physical metrics (force magnitude, contact geometry, principal-axis orientation) into a multimodal space. It reports **95.9% classification accuracy**, a 52.6% MAE reduction over prior SOTA, and a 3.5% sim-to-real gap, and feeds a 3D tactile-language-action (3D-TLA) policy for manipulation.

**Relation to this project.**
- **The argument this project leans on:** FG-CLTP explicitly criticizes tactile representations that "rely on qualitative descriptors (e.g. texture), neglecting quantitative contact states such as force magnitude, contact geometry, and principal-axis orientation." It argues **3D point clouds capture spatial deformation better than image-only representations**. This directly endorses *this project's* premise — that the *quantitative spatial structure* of deformation and shear (not a scalar or a texture label) is the information-bearing signal. The project's spatial-feature dominance result is a small-scale empirical instance of FG-CLTP's thesis.
- **The ceiling and the gap:** FG-CLTP's 95.9% (with 100k pairs, contrastive pretraining, and 3D point clouds) marks the resource/scale frontier. This project's ~80% honest accuracy on ~20 trials/class with a single sensor and no pretraining sits far below that frontier *by design*, and FG-CLTP quantifies what the missing ingredients are — data scale and richer 3D representation — which is exactly what this project's *future-work* section gestures toward.

**Honest scope note (do not over-attribute).** FG-CLTP's primary contribution is *contrastive language-tactile pretraining for VLA manipulation*, not an argument that "shear beats vision" per se. Framing deformation+shear as a standalone modality group worth studying is **this project's positioning**, supported by FG-CLTP's quantitative-contact-state argument — it should be cited as *consistent with*, not *claimed by*, FG-CLTP.

---

## 5. Synthesis — the arc and where this project sits

The four works form a ladder, and this project occupies a deliberate rung on it:

| Work | Role | Sensor / data | Modality used | Task | Headline |
|---|---|---|---|---|---|
| GelSight 2017 | Physical foundation | Optical elastomer | Deformation + (marker) shear — *principles* | Sensing/geometry | qualitative |
| Amin 2023 | Lab/pipeline precedent | PVDF pressure arrays, Baxter, UniGe | Pressure transients (mean/std) | 5-class hardness | CNN >98% |
| Liao 2025 | Sensor reference | Daimon DM-Tac | Normal-force dynamics (shear discarded) | Fruit hardness (regression) | universal criterion |
| FG-CLTP 2025 | SOTA ceiling | 3D point clouds, 100k pairs | Quantitative contact state | Material/contact + VLA | 95.9% |
| **This project** | — | **Daimon WS, hand-collected, ~20/class** | **Deformation + shear (fused, spatial+temporal)** | **10-material classification** | **~80% honest** |

**The project's distinct position.** It is the only one of the set that (a) uses the **deformation+shear field as its explicit, fused modality**, (b) runs the **controlled modality-ablation** (deformation-only vs shear-only vs fused) that none of the four perform, and (c) reports under a **deliberately honest, reproducibility-bounded evaluation** (~80% ± 5–6%, validation-based early stopping) rather than a single peak number. Where Amin averages the spatial field away and Liao discards shear, this project's contribution is to show *what is recovered by keeping both the spatial structure and the shear channel* — grounded physically by GelSight and pointed, as future work, toward the data-scale and 3D-representation frontier that FG-CLTP defines.

---

## 6. Citation-integrity notes (for the author)

- **Amin's "170 trials"** refers to materials-machine compression ground-truthing, not the tactile-dataset size — do not cite it as "170 tactile trials per object."
- **Amin's modality is PVDF pressure, not deformation+shear** — cite it for *pipeline/lab precedent*, and explicitly distinguish the sensing modality so a reviewer does not catch an overstated parallel.
- **Liao uses only normal force** for hardness despite decomposing shear — cite the dFz/dt feature idea and the Daimon lineage, and use the shear-discarding as a *motivation* for your fusion question, not as a fusion precedent.
- **FG-CLTP is contrastive language-tactile pretraining**, not a "shear vs vision" paper — cite its quantitative-contact-state argument as *support for* your premise, not as its thesis.
- All four full texts are in the project files (`document_pdf_3` = GelSight, `document_pdf` = Amin, `document_pdf_1` = Liao, `document_pdf_2` = FG-CLTP); re-verify any direct quote against them before submission.
