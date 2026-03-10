# WSI 360° — Whole Slide Image Comprehensive Classifier

A end-to-end Jupyter notebook pipeline (`Classifier.ipynb`) for **deep computational pathology analysis** of H&E-stained Whole Slide Images (WSI). The pipeline implements **44 single-slide analysis modules** plus a full **multi-slide cohort analysis** block, covering classical image processing, deep learning, graph topology, spatial statistics, and prognostic scoring.

---

## Table of Contents

- [Overview](#overview)
- [Features by Module](#features-by-module)
  - [Module I — Core Single-Slide Analysis (Cells 1–21)](#module-i--core-single-slide-analysis-cells-121)
  - [Module II — Extended Analysis (Cells 22–44)](#module-ii--extended-analysis-cells-2244)
  - [Cohort Analysis — Multi-Slide Extensions](#cohort-analysis--multi-slide-extensions)
- [Output Files](#output-files)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Architecture & Key Libraries](#architecture--key-libraries)
- [References](#references)

---

## Overview

The pipeline accepts a single `.svs` (Aperio) whole slide image and produces:

- Tissue segmentation masks and stain-normalised patches
- Tile-level classical, deep, and transformer feature vectors
- Nucleus detection and cell morphometry tables
- Spatial heatmaps, Grad-CAM saliency, and ABMIL attention maps
- Unsupervised clustering, graph topology, and ITH metrics
- Multi-biomarker risk score with pseudo-Kaplan-Meier curves
- Cross-slide cohort-level QC, feature analysis, and population risk stratification
- A full HTML report exported via `nbconvert`

```
wsi_analysis_output/
├── 00_summary.json               ← single-slide summary
├── 00_extended_summary.json      ← 44-module extended summary
├── 06_classical_features.csv
├── 07_deep_features.csv
├── 09_cell_morphometry.csv
├── 12_slide_features.csv
├── 18_tme_features.csv
├── 22_haralick_glcm.csv
├── 24_vit_features.csv
├── 28_all_nuclei.csv
├── 29_tile_risk_scores.csv
├── 32_wavelet_features.csv
├── 33_gabor_features.csv
├── 34_lab_hsd_features.csv
├── 36_lumen_features.csv
├── 38_fourier_shape_feats.csv
├── 41_laws_features.csv
├── cohort_qc.csv
├── cohort_risk_scores.csv
├── cohort_tile_features.csv
└── *.png                         ← all visualisation figures
```

---

## Features by Module

### Module I — Core Single-Slide Analysis (Cells 1–21)

| Cell | Title | Description |
|------|-------|-------------|
| 1 | **Environment Setup** | Imports OpenSlide, PyTorch, scikit-image, scikit-learn, XGBoost, UMAP, staintools, HistomicsTK, and more. Configures GPU device. |
| 2 | **Slide Metadata & QC** | Reads vendor, MPP, objective power, scan date. Computes Laplacian variance (focus score), mean saturation, and tissue fraction. Plots thumbnail, tissue mask, and RGB histogram. |
| 3 | **Multi-Resolution Pyramid** | Displays all pyramid levels side-by-side. Selects optimal working level (level 2, ~2048 px) for downstream processing. |
| 4 | **Tissue Segmentation** | HSV-based Otsu thresholding with morphological closing/opening and connected-component filtering to produce a clean binary tissue mask. |
| 5 | **Stain Normalisation** | Compares three methods — **Macenko**, **Vahadane**, and **Reinhard** (LAB-based) — using the `staintools` library. Macenko-normalised image is used for all downstream steps. |
| 6 | **Tissue-Aware Tile Extraction** | Extracts 256×256 px tiles at level 1 with a minimum tissue fraction threshold (40%). Supports up to 200 tiles per slide (configurable). |
| 7 | **Classical Feature Extraction** | Per-tile: RGB/HSV histograms (8 bins/channel), LBP (P=24, R=3, uniform), GLCM Haralick (6 properties, 2 distances, 4 angles), HOG summary statistics, Canny edge density, Shannon entropy. Produces ~80-d feature vector. |
| 8 | **Deep CNN Feature Extraction** | ResNet-50 (ImageNet-pretrained, classification head removed) extracts **2048-d** feature vectors per tile via `torchvision`. Saved to CSV. |
| 9 | **Nucleus Detection** | Watershed-based instance segmentation: invert-gray → Gaussian blur → Otsu threshold → distance transform → peak local max seeds → watershed. Outputs labelled regions and region properties. |
| 10 | **Cell Morphometry Analysis** | Histograms and statistics for 8 morphometric features: area, perimeter, eccentricity, solidity, major/minor axis, circularity, mean intensity. Includes inter-feature correlation heatmap. |
| 11 | **H&E Colour Deconvolution** | Ruifrok & Johnston matrix inversion to separate Haematoxylin and Eosin channels. K-means (k=4) on (H, E, saturation) to label tissue components: Nucleus, Cytoplasm, Stroma, Background. |
| 12 | **Spatial Heatmaps** | 64×64 px grid-level maps for: tissue occupancy, Shannon entropy (grayscale), haematoxylin proxy, and nucleus density (Laplacian energy). Overlaid on slide thumbnail. |
| 13 | **MIL-Style Feature Aggregation** | Aggregates tile features into a single slide descriptor using mean, std, and percentiles (10/25/50/75/90). Produces slide-level classical and deep feature vectors. |
| 14 | **PCA + UMAP Dimensionality Reduction** | PCA scree plot and cumulative variance. UMAP embedding (using top-20 PCs) of ResNet-50 tile features, coloured by tile index. |
| 15 | **K-Means Tile Clustering** | MiniBatchKMeans (k=4) on PCA-reduced features generates pseudo-labels (Type-A/B/C/D) for supervised classifier training in the absence of ground-truth annotations. |
| 16 | **Classical ML Classifiers** | Stratified 3-fold CV comparison of **Random Forest**, **SVM (RBF)**, and **XGBoost** on classical features with pseudo-labels. Confusion matrix and top-20 RF feature importances plotted. |
| 17 | **Attention-Based MIL (ABMIL)** | Implements Ilse et al. (ICML 2018) dual-attention network: 2-layer encoder → gated attention (V·U·w) → weighted mean pooling → classifier. Trained for 60 epochs; outputs slide-level prediction and per-tile attention weights. |
| 18 | **Grad-CAM & Attention Heatmap** | Grad-CAM on ResNet-50 `layer4` for the highest-attention tile. ABMIL attention weights projected back onto slide coordinate space and overlaid as a heatmap. |
| 19 | **Tumour Microenvironment (TME) Analysis** | 128×128 px grid analysis of nuclear density (H-channel mean), stromal ratio (E/(H+E)), and textural activity (Laplacian variance). Pivot heatmaps + scatter plots. |
| 20 | **Mitotic Figure Detection Proxy** | OpenCV `SimpleBlobDetector` on inverted grayscale with area/circularity/convexity filters as a proxy for mitotic figure detection. Per-tile count histogram. |
| 21 | **Comprehensive Summary Report** | Writes `00_summary.json` with all key metrics. Generates 2×4 overview figure grid. Lists all output files with sizes. |

---

### Module II — Extended Analysis (Cells 22–44)

| Cell | Title | Description |
|------|-------|-------------|
| 22 | **Cell-Graph Construction** | Builds a patch graph using **Delaunay triangulation** (SciPy) on tile centroids. Computes networkx graph metrics: node degree distribution, local clustering coefficient, graph density, average edge length. Visualises graph overlay on slide and clustering coefficient map. |
| 23 | **Fractal Dimension & Lacunarity** | Box-counting **fractal dimension** for tissue mask and nucleus mask. Lacunarity curves at multiple box sizes. Log-log plots with regression fit. |
| 24 | **Multi-Scale Haralick GLCM** | Extended GLCM at 4 distances × 4 angles for 6 Haralick properties (contrast, dissimilarity, homogeneity, energy, correlation, ASM). Per-cluster distribution histograms and full correlation matrix. |
| 25 | **Ripley's K/L Spatial Point Process** | Monte Carlo envelope (99 simulations, CSR) for Ripley's L-function on tile centroids. Reports clustering vs. regularity fractions and peak L-radius. Nucleus density heatmap. |
| 26 | **ViT / DINO Patch Features** | **ViT-B/16** (ImageNet, proxy for HIPT/UNI) extracts 768-d [CLS] token features. PCA → t-SNE and UMAP embeddings compared with ResNet-50 UMAP. Saved to `24_vit_features.csv`. |
| 27 | **SHAP Explainability** | `shap.TreeExplainer` on Random Forest trained on classical features. Beeswarm plot (top-10 features) and horizontal bar chart of mean |SHAP| for top-20 features. |
| 28 | **Intra-Tumoral Heterogeneity (ITH)** | Computes **phenotypic entropy** (Shannon), **Pielou's J evenness**, **Jensen-Shannon divergence** matrix across clusters, **UMAP dispersion score**, and coefficient of variation of features. |
| 29 | **Louvain Community Detection** | cosine-similarity k-NN tile graph (k=8) → Louvain community detection (python-louvain fallback: SpectralClustering). Reports NMI and ARI vs. K-means. Tile similarity matrix heatmap. |
| 30 | **Nuclear Pleomorphism** | WHO pleomorphism grading (Grades 1–3) based on nuclear area coefficient of variation. Shape metrics: form factor, elongation, compactness. Scatter and bar visualisations. |
| 31 | **Multi-Biomarker Risk Score** | Fuses Classical, ResNet-50, ViT, and Haralick PC features. **Risk PCA** → composite risk score (0–1) → Low/Intermediate/High tiers. Pseudo-Kaplan-Meier curves. Risk score UMAP overlay. Biomarker correlation waterfall plot. Saves `29_tile_risk_scores.csv`. |
| 32 | **Wavelet Texture Features** | Discrete wavelet transform (DWT) with **Haar, db4, sym4** at 3 decomposition levels. Energy and entropy per subband (H/V/D). Saves 54+ feature columns to `32_wavelet_features.csv`. |
| 33 | **Gabor Filter Bank** | 4 spatial frequencies × 6 orientations (0°–150° in 30° steps). Mean and std of response magnitude per filter (48 features). Frequency-orientation response heatmap and per-cluster box plots. |
| 34 | **CIE L\*a\*b\* & HSD Color Features** | CIE L\*a\*b\* (mean, std, skew, kurtosis, IQR per channel), **HSD** (Hue-Saturation-Density optical density decomposition for H&E), and YCrCb colour statistics. 39-feature vector. |
| 35 | **Moran's I Spatial Autocorrelation** | Row-normalised k-NN spatial weight matrix (k=8) for 5 tile-level signals: haematoxylin proxy, entropy, edge density, ABMIL attention, PCA-PC1. Values > 0 indicate positive spatial clustering. |
| 36 | **Gland Lumen Detection** | Otsu thresholding + morphological closing + size/eccentricity filtering for bright lumen regions. GlaS-challenge–style approach. Per-tile: lumen count, lumen area fraction, mean circularity/eccentricity/solidity. |
| 37 | **TransMIL** | Implements Shao et al. NeurIPS 2021: lightweight TransMIL with **2D sinusoidal positional encoding**, Transformer encoder (2 layers, 4 heads), and [CLS] token classification head. Trained 80 epochs on deep tile features with tile grid coordinates. |
| 38 | **Nuclear Fourier Shape Descriptors** | Zahn & Roskies (1972) complex Fourier descriptors on nuclear boundary contours (14 harmonics). Mean and std per harmonic across nuclei in each tile. Feature space scatter (h1 vs h2). |
| 39 | **Elastic Net Feature Selection** | Multinomial logistic regression with elastic net penalty (α=0.5, C=0.1, SAGA solver) on 400+ combined features (Classical + Wavelet + Gabor + Lab/HSD). Reports selected feature count and top coefficients by modality. |
| 40 | **Spatial Tile Co-occurrence Matrix** | cKD-tree adjacency (r < 1.5× tile size) for all tile pairs. Raw co-occurrence counts and row-normalised **transition probability matrix** between tissue-type clusters. |
| 41 | **Laws' Texture Energy Masks** | Laws (1979) separable 1-D filters (L3, E3, S3) → 9 filter pairs × macro+micro energy (18 features). Cluster-mean energy heatmap and PCA scatter. |
| 42 | **Nuclear Nearest-Neighbour Spatial Statistics** | scikit-learn kNN for all detected nuclei. **Clark-Evans R-index** (R<1 = clustered, R>1 = dispersed). Ripley's L-function approximation. NN distance histogram with CSR expected line. |
| 43 | **Multi-Scale Feature Fusion & Spectral Clustering** | Concatenates Classical + Wavelet + Gabor + Laws features → PCA(30) → SpectralClustering (k=4, nearest-neighbors affinity). Silhouette comparison vs. K-means. UMAP + modality-contribution bar chart from elastic net weights. |
| 44 | **Extended Prognostic Index & Summary** | Composite 14-biomarker prognostic index from nuclear pleomorphism, tissue entropy, wavelet HF energy, Gabor disorder, Lab stain variability, glandular complexity, Laws E3E3 energy, nuclear clustering, ABMIL attention entropy, TransMIL uncertainty, Moran's I, ITH entropy, spectral NMI, and Fourier shape complexity. **Radar chart** visualisation. Writes `00_extended_summary.json`. |

---

### Cohort Analysis — Multi-Slide Extensions

Cells prefixed **COHORT** process a configurable list of `.svs` slides (up to 10 by default) from a cohort directory. Runs in **demo/fallback mode** if slides are not present.

| Cell | Title | Description |
|------|-------|-------------|
| Config | **Cohort Configuration** | Sets `SLIDE_DIR`, `SLIDE_NAMES`, `SLIDE_COHORT`, and `PRIMARY_SLIDE_IDX`. Checks slide availability. |
| COHORT-1 | **Thumbnail Grid** | 3-column thumbnail grid for all cohort slides with dimensions, MPP, and objective power. Primary slide highlighted with red border. |
| COHORT-2 | **Batch QC** | Per-slide: focus (Laplacian variance), tissue fraction, mean saturation, QC PASS/CHECK flag. Bar charts and saves `cohort_qc.csv`. |
| COHORT-3 | **Batch Classical Feature Extraction** | Extracts 60 random tissue tiles per slide. Computes RGB moments, LBP histogram, and GLCM features. Aggregates to `cohort_tile_features.csv`. |
| COHORT-4 | **Cross-Slide PCA + UMAP** | PCA scree + PC1/PC2 scatter; UMAP of all tiles coloured by slide index. Reveals inter-slide feature overlap and batch effects. |
| COHORT-5 | **Slide-Level ML (RF + LOO-CV)** | Aggregates tile features (mean + std) to slide-level vectors. Random Forest with Leave-One-Out cross-validation. Plots top-15 slide-level feature importances. |
| COHORT-6 | **Cross-Slide Statistical Comparison** | Box plots for 8 key features per slide with Kruskal-Wallis test (p-values + significance stars). Slide-slide feature correlation heatmap (seaborn). |
| COHORT-7 | **Population Risk Stratification** | Linear weighted risk score from GLCM contrast/homogeneity, RGB std, and energy. Normalises to 0–100. Low/Mid/High tier assignment. Feature profile radar + pie chart. Saves `cohort_risk_scores.csv`. |
| COHORT-8 | **Multi-Slide ABMIL Training** | Per-slide bag training of a 2-class ABMIL with LayerNorm, GELU, and dropout. Plots training loss/accuracy and sorted per-slide attention weight curves. Saves `cohort_abmil_training.png`. |
| Export | **HTML Report** | Exports the full executed notebook to `results/Classifier.html` using `nbconvert.HTMLExporter`. |

---

## Output Files

| File | Description |
|------|-------------|
| `wsi_analysis_output/00_summary.json` | Core 21-module summary (metadata, QC, tile counts, classifier results) |
| `wsi_analysis_output/00_extended_summary.json` | Full 44-module summary (all feature dims, prognostic index, spatial stats) |
| `wsi_analysis_output/06_classical_features.csv` | Per-tile classical features (~80 dims) |
| `wsi_analysis_output/07_deep_features.csv` | Per-tile ResNet-50 features (2048 dims) |
| `wsi_analysis_output/09_cell_morphometry.csv` | Detected nucleus morphometric properties |
| `wsi_analysis_output/12_slide_features.csv` | MIL-aggregated slide-level feature vector |
| `wsi_analysis_output/18_tme_features.csv` | TME grid-level nuclear density / stromal ratio |
| `wsi_analysis_output/22_haralick_glcm.csv` | Extended Haralick GLCM features |
| `wsi_analysis_output/24_vit_features.csv` | ViT-B/16 [CLS] patch features (768 dims) |
| `wsi_analysis_output/28_all_nuclei.csv` | All nucleus instances with shape metrics |
| `wsi_analysis_output/29_tile_risk_scores.csv` | Per-tile risk score + tier + cluster label |
| `wsi_analysis_output/32_wavelet_features.csv` | DWT Haar/db4/sym4 subband features |
| `wsi_analysis_output/33_gabor_features.csv` | Gabor filter bank response features |
| `wsi_analysis_output/34_lab_hsd_features.csv` | L\*a\*b\*, HSD, YCrCb colour features |
| `wsi_analysis_output/36_lumen_features.csv` | Gland lumen detection statistics |
| `wsi_analysis_output/38_fourier_shape_feats.csv` | Nuclear Fourier shape descriptors |
| `wsi_analysis_output/41_laws_features.csv` | Laws' texture energy mask features |
| `wsi_analysis_output/cohort_qc.csv` | Per-slide QC metrics for cohort |
| `wsi_analysis_output/cohort_tile_features.csv` | Batch-extracted tile features for cohort |
| `wsi_analysis_output/cohort_risk_scores.csv` | Per-slide population risk scores |
| `results/Classifier.html` | Full notebook HTML export |

---

## Requirements

**Python** ≥ 3.9, **CUDA** optional (CPU fallback supported).

Key dependencies:

```
openslide-python
numpy, pandas, scipy
scikit-image, scikit-learn
opencv-python
torch, torchvision
staintools
histomicstk          # optional — graceful fallback if unavailable
umap-learn
xgboost
shap
pywt                 # PyWavelets
networkx
plotly
seaborn
tqdm
nbconvert, nbformat  # for HTML export
```

> See `requirements.txt` for the full pinned dependency list.

---

## Installation

```bash
# 1. Clone / navigate to the project directory
cd /path/to/mlwork

# 2. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install OpenSlide system library (Ubuntu/Debian)
sudo apt-get install openslide-tools

# 5. (Optional) GPU driver + CUDA toolkit for GPU acceleration
```

---

## Usage

```bash
# Activate environment
source .venv/bin/activate

# Launch Jupyter
jupyter notebook Classifier.ipynb
```

1. **Set `SVS_PATH`** in Cell 1 to point to your `.svs` file.
2. **Set `SLIDE_DIR` / `SLIDE_NAMES`** in the Cohort Configuration cell for multi-slide analysis.
3. Run all cells sequentially (`Kernel → Restart & Run All`).
4. All outputs are written to `wsi_analysis_output/`. The HTML report is saved to `results/Classifier.html`.

---

## Architecture & Key Libraries

```
Input SVS
   │
   ├─ OpenSlide ──────────────── Pyramid access, metadata, tile reading
   │
   ├─ Cell Segmentation
   │     ├─ OpenCV / scikit-image ─ Tissue mask, HSV thresholding
   │     └─ Watershed ───────────── Nucleus instance segmentation
   │
   ├─ Feature Extraction
   │     ├─ Classical ─────────── RGB/HSV hist, LBP, GLCM, HOG, entropy
   │     ├─ Wavelet ───────────── PyWavelets DWT (Haar/db4/sym4)
   │     ├─ Gabor ─────────────── scikit-image GaborFilter
   │     ├─ Laws' Masks ─────────── scipy.ndimage convolve1d
   │     ├─ Lab/HSD ───────────── OpenCV cvtColor + custom OD decomposition
   │     ├─ Haralick GLCM ──────── scikit-image graycomatrix
   │     ├─ Fourier Shapes ──────── numpy.fft on nuclear boundary contours
   │     ├─ ResNet-50 ──────────── torchvision (2048-d, ImageNet weights)
   │     └─ ViT-B/16 ───────────── torchvision ViT [CLS] token (768-d)
   │
   ├─ Classification / Clustering
   │     ├─ K-Means / MiniBatchKMeans (scikit-learn)
   │     ├─ Spectral Clustering
   │     ├─ Random Forest, SVM, XGBoost
   │     ├─ Elastic Net (sklearn LogisticRegression saga)
   │     ├─ ABMIL (custom PyTorch — gated attention)
   │     └─ TransMIL (custom PyTorch — Transformer + 2D PE)
   │
   ├─ Spatial / Graph Analysis
   │     ├─ Delaunay Patch Graph (scipy.spatial + networkx)
   │     ├─ Louvain Community Detection (python-louvain)
   │     ├─ Ripley's K / L (custom Monte Carlo)
   │     ├─ Moran's I (custom row-normalised weight matrix)
   │     ├─ Clark-Evans R-index (sklearn NearestNeighbors)
   │     └─ Spatial Co-occurrence Matrix (scipy cKDTree)
   │
   ├─ Explainability
   │     ├─ Grad-CAM (torch backward hooks on ResNet-50 layer4)
   │     └─ SHAP TreeExplainer (shap library)
   │
   └─ Output
         ├─ CSV feature tables
         ├─ PNG visualisation figures (44+)
         ├─ JSON summary files
         └─ HTML report (nbconvert)
```

---

## References

The pipeline implements methods from the following publications:

| Method | Reference |
|--------|-----------|
| Macenko stain normalisation | Macenko et al., ISBI 2009 |
| Vahadane stain normalisation | Vahadane et al., IEEE TMI 2016 |
| Colour deconvolution | Ruifrok & Johnston, Anal Quant Cytol Histol 2001 |
| Haralick GLCM texture | Haralick et al., IEEE TSMC 1973 |
| Local Binary Patterns | Ojala et al., IEEE TPAMI 2002 |
| ABMIL | Ilse et al., ICML 2018 |
| TransMIL | Shao et al., NeurIPS 2021 |
| Grad-CAM | Selvaraju et al., ICCV 2017 |
| HIPT / ViT patch features | Chen et al., CVPR 2022 |
| UNI foundation model | Chen et al., Nature Medicine 2024 |
| PatchGCN cell graphs | Chen et al., MICCAI 2021 |
| Histocartography | Jaume et al., ICCV 2021 |
| Fractal dimension | Landini, IEEE TBME / Losa & Nonnenmacher, Birkhäuser |
| Ripley's K / L function | Baddeley & Turner, J Stat Software 2005 |
| Spatial TIL analysis | Kather et al., JCO Precis Oncol 2018 |
| Nuclear pleomorphism | Elmore et al., JAMA 2015; Veta et al., IEEE TBME 2014 |
| ITH assessment | Andor et al., Nature Genetics 2016; Gerlinger et al., NEJM 2012 |
| Louvain communities | Pati et al., Medical Image Analysis 2022 |
| DSMIL / CLAM | Lu et al., Lancet Digital Health 2021 |
| Risk score synthesis | Mobadersany et al., PNAS 2018; Chen et al., Cancer Cell 2022 |
| Pan-cancer risk | Kather et al., Nature Cancer 2020 |
| SHAP | Lundberg & Lee, NeurIPS 2017 |
| Elastic net | Zou & Hastie, JRSS 2005 |
| GlaS gland challenge | Sirinukunwattana et al., Medical Image Analysis 2017 |
| Fourier shape descriptors | Zahn & Roskies, IEEE TSMC 1972 |
| Laws texture energy | Laws, DARPA/USCIPI 1979 |
| Gabor filter bank | Linder et al., Bioinformatics 2012 |
| HSD colour space | van Eyndhoven et al., EMBC 2019 |
| Moran's I | Anselin, Geographical Analysis 1995 |
| Clark-Evans index | Clark & Evans, Ecology 1954 |
| Spatial co-occurrence | Diao et al., Cell Systems 2021 |
| Multi-scale feature fusion | Lu et al., Lancet Digital Health 2021 |

---

## License

This project is for research and educational purposes. Please ensure compliance with your institution's data governance policies when using patient-derived slide images.
