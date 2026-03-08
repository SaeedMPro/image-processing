# Low-Light Image Enhancement – BSc Project

Computer Vision Course, Bu-Ali Sina University, Winter 2026.

## Project structure (best practice)

```
final-project/
├── lowlight/                 # Main package – all logic lives here
│   ├── __init__.py
│   ├── config.py             # Paths, IMAGE_SIZE, MIN/MAX_PAIRS
│   ├── data/                 # Dataset discovery and preprocessing
│   │   ├── discovery.py      # discover_pairs()
│   │   └── preprocess.py     # load_image, load_dataset, resize, normalize
│   ├── features/             # Phase 1 handcrafted features
│   │   └── handcrafted.py    # mean, std, entropy, skewness
│   ├── enhancement/          # Phase 2 enhancement methods
│   │   ├── classical.py      # HE, CLAHE, Gamma, SSR
│   │   └── autoencoder.py    # Lightweight conv AE
│   ├── evaluation/           # Phase 3 metrics
│   │   └── metrics.py       # PSNR, SSIM
│   └── utils/
│       └── viz.py            # plot_pair, plot_enhancement_comparison
├── scripts/
│   └── check_data.py         # Step 1 validation (uses lowlight.data)
├── data/
│   ├── low/                  # Low-light images
│   └── normal/               # Normal-light references
├── figures/                  # Saved plots for report
├── models/                   # Saved classifier and autoencoder
├── enhancement_notebook.ipynb   # Main deliverable – runs all phases
├── requirements.txt
├── pyproject.toml            # pip install -e .
├── PROJECT_PLAN.md
└── README.md
```

**Design principles:**
- **Single package (`lowlight`)**: All code is in the package; the notebook only imports and calls.
- **Configuration in one place** (`lowlight.config`): Paths and constants so experiments are reproducible.
- **Clear layers**: data → features → enhancement → evaluation; utils for shared helpers.
- **Proposal alignment**: Each module references the proposal section it implements.

## Setup

1. **Create and activate a virtual environment** (from repo root or final-project):
   ```bash
   cd final-project
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   ```

2. **Install the project in editable mode** (so `import lowlight` works from anywhere):
   ```bash
   pip install -e .
   ```
   Or install dependencies only (run notebook from `final-project` so `lowlight` is on path):
   ```bash
   pip install -r requirements.txt
   ```
   If you don’t use `pip install -e .`, add the project root to `sys.path` in the notebook (first cell already handles this when run from `final-project`).

3. **Validate dataset (Step 1)**:
   ```bash
   python scripts/check_data.py
   ```
   Expected: pair count 80–120, sample image shapes printed.

## Running the notebook

- Open `enhancement_notebook.ipynb` with Jupyter (or VS Code). Set kernel to your venv.
- Preferred: use **final-project** as the current working directory when starting the kernel (e.g. “Open Folder” → final-project, then open the notebook).
- The first code cell adds the project root to `sys.path` if needed; with `pip install -e .` no path hack is required.

## Dataset

- **Source:** LOL-v2 (Low-Light) – Real part only.
- **Constraint:** 80–120 paired images (Proposal §2).
- **Layout:** `data/low/*.png`, `data/normal/*.png` with matching numeric IDs (e.g. `low00001.png` ↔ `normal00001.png`).

## Deliverables

- **One executable Jupyter Notebook** (`enhancement_notebook.ipynb`).
- **Technical report** (8–10 pages): Introduction, Methodology, Experiments, Results, Discussion and Conclusion.
- **Figures** in `figures/` (visual comparisons, bar charts).

## References

- Project proposal: `Low-light Image Enhancement Project (bsc Version)WM.pdf`
- Plan and milestones: `PROJECT_PLAN.md`
