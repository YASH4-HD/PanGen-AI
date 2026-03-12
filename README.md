# 🧬 PanGen-AI
### An Integrated Deep Learning and Multi-Track Genome Visualization Framework for Pangenomic Data Analysis

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://cd40-immunosome-tool-yash.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Version](https://img.shields.io/badge/version-1.0.0-blue)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18850205.svg)](https://doi.org/10.5281/zenodo.18850205)
---

## Overview
<p align="center">
  <img src="dashboard.png" width="900">
</p>
**PanGen-AI Suite** is an interactive computational genomics framework that integrates classical bioinformatics algorithms with modern deep learning approaches to explore genomic variation, pangenome structures, and CRISPR genome engineering strategies.

The platform provides a unified interface for:

- Graph-based pangenome exploration
- Deep learning–based variant impact prediction 
- Genome compression and FM-index search algorithms
- CRISPR guide RNA design
- Interactive genome track visualization

  
PanGen-AI aims to serve both as a research prototyping platform and an educational computational genomics toolkit.


This repository accompanies the preprint:

> **Nama, Y. (2026).**  
> *An Integrated Deep Learning and Multi-Track Genome Visualization Framework for Pangenomic Data Analysis*

---

## Biological Motivation

Understanding genomic variation requires integration of multiple computational approaches:

- Comparative genomics to analyze variation across genomes
- Machine learning to predict functional variant impact
- Genome indexing algorithms for efficient sequence search
- Genome editing design tools for experimental validation

PanGen-AI provides a modular environment where these analyses can be performed within a single computational framework.


The platform addresses questions such as:

1. How can graph-based models represent variation across genomes?
2. Which genomic positions are predicted to have high functional impact?
3. How can computational predictions guide CRISPR editing strategies?
4. How can compressed genome indexing enable rapid sequence search?

---

## System Architecture

PanGen-AI follows a modular architecture integrating multiple computational genomics components.

DNA Input
   ↓
Pangenome Graph Construction
   ↓
Deep Learning Variant Prediction
   ↓
Genome Compression & FM-Index Search
   ↓
CRISPR Guide RNA Design
   ↓
Multi-Track Genome Visualization

Each module can operate independently or as part of an integrated analysis workflow.

---


## Core Modules

### 1️⃣ Pangenome Explorer

Graph-based representation of genomic variation.
Features:

- k-mer based pangenome graph construction
- visualization of sequence relationships
- conservation analysis across sequences
- FASTA dataset input support


Applications:

- comparative genomics
- microbial genome analysis
- structural variation visualization



---

### 2️⃣ DeepNCV – AI Variant Impact Prediction

Deep learning model for functional variant prediction.

Capabilities:

- CNN-based DNA sequence analysis
- mutation impact heatmap generation
- gradient-based saliency visualization
- batch variant prediction


Applications:

- regulatory variant discovery
- functional genomics analysis
- mutation hotspot detection
---

## 3️⃣ Geno-Compressor (BWT + FM-Index)

Genome compression and sequence search module.

Implements:

- Burrows-Wheeler Transform (BWT)
- FM-index construction
- backward search algorithm
- compressed genome pattern matching


Applications:

- genome indexing
- sequence alignment preprocessing
- bioinformatics algorithm education

---

## 4️⃣ CRISPR Guide Designer

Identification of candidate CRISPR-Cas9 guide RNAs.

Features:

- PAM-aware NGG scanning
- GC content filtering
- off-target similarity estimation
- candidate guide ranking


Applications:

- genome editing experiments
- functional genomics perturbation studies

---

## Interactive Dashboard

The Streamlit interface provides real-time interaction with genomic datasets.

Capabilities include:

- dynamic parameter tuning
- real-time visualization of genomic analysis
- mutation heatmaps and genome tracks
- exportable CSV and figure outputs

The interface enables rapid exploration of genomic hypotheses without requiring extensive programming.  

**Live Web App:**  
https://pangen-ai-yash.streamlit.app/

---

## Example Datasets

PanGen-AI includes curated example datasets for demonstration:

| Dataset                    | Purpose                       |
| -------------------------- | ----------------------------- |
| Bacterial genome fragments | Pangenome graph construction  |
| BRCA1 regulatory sequence  | Variant impact prediction     |
| SARS-CoV-2 genome fragment | FM-index search demonstration |
| Human gene exon region     | CRISPR guide design           |


---

## 📂 Repository Structure

```text
PanGen-AI-Suite/
│
├── app.py
├── requirements.txt
├── README.md
├── LICENSE
├── CITATION.cff
├── dashboard.png
└── assets/
```
---
## 🛠 Installation
**1️⃣ Clone the repository**
```
git clone https://github.com/YASH4-HD/PanGen-AI-Suite.git
cd PanGen-AI-Suite
```
**2️⃣ Install dependencies**
```
pip install -r requirements.txt
```
**3️⃣ Launch the dashboard**
```
streamlit run app.py
```
---
## 🔁 Reproducibility
All analyses are reproducible using:

- deterministic model initialization
- defined dataset inputs
- explicit algorithm implementations
- open-source Python libraries

The platform does not require proprietary datasets.

---

## 📜 Citation
If you use this suite in your research, please cite it as:
> **Nama, Y. (2026).** *PanGen-AI Suite: An Integrated Platform for Pangenome Analysis, AI Variant Prediction, and Genome Engineering. (Version 1.0.0)
Zenodo. https://doi.org/10.5281/zenodo.18850205.* GitHub. [https://github.com/YASH4-HD/PanGen-AI](https://github.com/YASH4-HD/PanGen-AI)

---

## Author

**Yashwant Nama**  
*Independent Researcher | Systems Immunology & Computational Modeling*

**Focus:** Systems Immunology, Mechanobiology, Computational Modeling and Reproducible Bioinformatics.

🔗 **Connect & Verify:**
*   **ORCID:** [0009-0003-3443-4413](https://orcid.org/0009-0003-3443-4413)
*   **LinkedIn:** [Yashwant Nama](https://www.linkedin.com/in/yashwant-nama-232b2437b/)
*   **Project Website:** [Streamlit Dashboard](https://pangen-ai-yash.streamlit.app/)

---

💡 **PanGen-AI combines classical bioinformatics algorithms with modern AI approaches to create an integrated genomic analysis environment.**

