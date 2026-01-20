# Structure-Centric-FG-SBIR

**Structure-Centric Fine-Grained Sketch-Based Image Retrieval**

Official PyTorch implementation of the paper:

> **Structure-Centric Representation Learning for Fine-Grained Sketch-Based Image Retrieval**

---

## 1. Overview

Fine-Grained Sketch-Based Image Retrieval (FG-SBIR) aims to retrieve the **exact photo instance**
corresponding to a free-hand sketch query.
This task is particularly challenging due to:

- Severe abstraction and sparsity in sketches  
- Structural ambiguity and missing parts  
- Large cross-modal gap between sketches and photos  

To address these challenges, we propose a **structure-centric FG-SBIR framework**
that explicitly models **geometric structure**, **cross-modal alignment**, and **sketch uncertainty**.

---

## 2. Method Summary (Section 3 in Paper)

Our framework consists of four main stages:

1. **Backbone Feature Extraction**  
2. **Structural Graph Construction**  
3. **Structure-to-Structure Alignment**  
4. **Uncertainty-Aware Structural Weighting**

Each module directly corresponds to a component described in **Section 3** of the paper,
with explicit mapping between equations and code.

---

## 3. Code Structure

```text
Structure-Centric-FG-SBIR/
│
├── configs/
├── datasets/
├── models/
├── losses/
├── utils/
├── train.py
├── test.py
└── evaluate.py
```
### 📂 Datasets

- **ShoeV2 / ChairV2**  
  [Sketchy Official Website](https://sketchx.eecs.qmul.ac.uk/downloads/)  
  [Google Drive Download](https://drive.google.com/file/d/1frltfiEd9ymnODZFHYrbg741kfys1rq1/view)

- **Sketchy**  
  [Sketchy Official Website](https://sketchx.eecs.qmul.ac.uk/downloads/)  
  [Google Drive Download](https://drive.google.com/file/d/11GAr0jrtowTnR3otyQbNMSLPeHyvecdP/view)

- **TU-Berlin**  
  [TU-Berlin Official Website](https://www.tu-berlin.de/)  
  [Google Drive Download](https://drive.google.com/file/d/12VV40j5Nf4hNBfFy0AhYEtql1OjwXAUC/view)
---

## Citation

```bibtex
@article{AlMohamadi2026StructureCentricFGSBIR,
  title   = {Structure-Centric Representation Learning for Fine-Grained Sketch-Based Image Retrieval},
  author  = {Mohammed A. S. Al-Mohamadi and Prabhakar C. J.},
  year    = {2026}
}
```
