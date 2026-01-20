"""
Structure-Centric-FG-SBIR
=========================

Structure-Centric Fine-Grained Sketch-Based Image Retrieval
-----------------------------------------------------------

Official PyTorch implementation of the paper:
"Structure-Centric Fine-Grained Sketch-Based Image Retrieval"



-----------------------------------------------------------
1. Overview
-----------------------------------------------------------

Fine-Grained Sketch-Based Image Retrieval (FG-SBIR) aims to retrieve the
exact photo instance corresponding to a free-hand sketch query.
This task is challenging due to:
- Severe abstraction and sparsity in sketches
- Structural ambiguity and missing details
- Cross-modal gap between sketches and photos

We propose a structure-centric FG-SBIR framework that explicitly models
geometric structure, cross-modal alignment, and sketch uncertainty.

-----------------------------------------------------------
2. Method Summary (Section 3 in Paper)
-----------------------------------------------------------

The framework consists of four stages:

1) Backbone Feature Extraction
2) Structural Graph Construction
3) Structure-to-Structure Alignment
4) Uncertainty-Aware Structural Weighting

-----------------------------------------------------------
3. Code Structure
-----------------------------------------------------------

Structure-Centric-FG-SBIR/
│
├── configs/                 # Dataset-specific YAML configs
├── datasets/                # Dataset loaders (Sketchy, ShoeV2, ChairV2)
├── models/
│   ├── backbone.py          # CNN / ViT feature extractor
│   ├── graph_encoder.py     # Structural graph encoding (Eq. 3–4)
│   ├── alignment_module.py  # Cross-modal alignment (Eq. 5)
│   ├── uncertainty_weighting.py  # Uncertainty modeling (Eq. 7)
│   └── full_model.py        # End-to-end model
│
├── losses/
│   ├── triplet_loss.py      # Metric learning (Eq. 8)
│   └── structural_consistency_loss.py  # Structural loss (Eq. 9)
│
├── utils/
│   ├── graph_utils.py
│   ├── metrics.py
│   └── visualization.py
│
├── train.py
├── test.py
└── evaluate.py

-----------------------------------------------------------
4. Reproducibility
-----------------------------------------------------------

- Fixed random seeds
- Deterministic data splits
- Config-driven training
- Clear mapping between equations and code

-----------------------------------------------------------
📂 Datasets
-----------------------------------------------------------
ShoeV2 / ChairV2
Sketchy Official Website
Google Drive Download

Sketchy
Sketchy Official Website
Google Drive Download

TU-Berlin
TU-Berlin Official Website
Google Drive Download

-----------------------------------------------------------
Citation: If you use this code, please cite:
-----------------------------------------------------------

-----------------------------------------------------------
Citation: If you use this code, please cite:

title = {Structure-Centric Representation Learning for Fine-Grained Sketch-Based Image Retrieval},

author = {Mohammed A. S. Al-Mohamadi and Prabhakar C. J.},

journal = {.............}, year = {2026} }

Contact: almohmdy30@gmail.com GitHub: https://github.com/mohammedalmohmdy

-----------------------------------------------------------

"""


Citation: If you use this code, please cite:

title = {FREQDIFFFORMER: FREQUENCY-GUIDED TRANSFORMER–DIFFUSION FRAMEWORK FOR FINE-GRAINED SKETCH-BASED IMAGE RETRIEVAL},

author = {Mohammed A. S. Al-Mohamadi and Prabhakar C. J.},

journal = {Multimedia Tools and Applications}, year = {2025} }

Contact: almohmdy30@gmail.com GitHub: https://github.com/mohammedalmohmdy
