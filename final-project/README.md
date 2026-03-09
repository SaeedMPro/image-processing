# Low-Light Image Enhancement

Computer Vision project: analysis, enhancement, and evaluation of low-light images using classical methods and a lightweight autoencoder.

## Structure

```
final-project/
├── lowlight/                 # Main package
│   ├── config.py             # Paths, image size, dataset limits
│   ├── data/                 # Dataset discovery and preprocessing
│   ├── features/             # Handcrafted features (mean, std, entropy, skewness)
│   ├── enhancement/          # Classical methods + autoencoder
│   ├── evaluation/           # PSNR and SSIM
│   └── utils/                # Visualization helpers
├── scripts/
│   └── check_data.py         # Validate dataset (pair count, shapes)
├── data/
│   ├── low/                  # Low-light images
│   └── normal/               # Normal-light references
├── figures/                  # Saved plots for the report
├── models/                   # Saved classifier and autoencoder
├── enhancement_notebook.ipynb
├── requirements.txt
├── REPORT.md                 # Technical report
└── README.md
```

## Setup

1. Create and activate a virtual environment:

   ```bash
   cd final-project
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   ```

2. Install dependencies (and optionally the package in editable mode):

   ```bash
   pip install -r requirements.txt
   # Or: pip install -e .   (if pyproject.toml exists)
   ```

3. Validate the dataset:

   ```bash
   python scripts/check_data.py
   ```

   Expected: pair count between 80 and 120, sample image shapes printed.

## Running the notebook

- Open `enhancement_notebook.ipynb` with Jupyter or VS Code. Use **final-project** as the working directory.
- The first code cell adds the project root to `sys.path` if needed.
- Run all cells (Kernel → Restart & Run All) to reproduce the full pipeline and figures.

## Dataset

- **Source:** LOL-v2 (Low-Light) dataset, real-world subset.
- **Layout:** `data/low/*.png` and `data/normal/*.png` with matching numeric IDs (e.g. `low00001.png` ↔ `normal00001.png`).
- **Usage:** 80–120 paired images; images are resized to 256×256 and normalized to [0, 1].

## Deliverables

- **Notebook:** `enhancement_notebook.ipynb` — runs all phases and saves figures.
- **Report:** `REPORT.md` — technical report (Introduction, Methodology, Experiments, Results, Discussion, Conclusion).
- **Figures:** `figures/` — confusion matrix, enhancement comparison, PSNR/SSIM bar charts.
