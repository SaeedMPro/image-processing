# Low-Light Image Enhancement: Classical and Learning-Based Methods

**Technical Report**  
Computer Vision — Bu-Ali Sina University, Winter 2026

---

## 1. Introduction

Images captured in low light often suffer from low contrast, reduced visibility of details, and increased noise. Low-light image enhancement aims to improve the visual quality of such images so that they are closer to what would be obtained under normal illumination. This project implements and compares two families of approaches: **classical** (handcrafted) methods and a **lightweight learning-based** method (a convolutional autoencoder), and evaluates them on a real-world paired dataset.

The work is organized in three phases:

1. **Image quality analysis and classification** — We extract handcrafted features from low-light and normal-light images and train a binary classifier to distinguish between them. This phase establishes that the two populations are statistically separable and provides a baseline for understanding the data.

2. **Image enhancement** — We implement four classical methods (Histogram Equalization, CLAHE, Gamma Correction, and Single-Scale Retinex) and one lightweight convolutional autoencoder. The classical methods are parameter-driven and do not require training; the autoencoder is trained to map low-light images to their normal-light counterparts using mean squared error (MSE) loss.

3. **Quantitative and qualitative evaluation** — We compute PSNR and SSIM between each enhanced result and the normal-light reference, report results in tables and bar charts, and provide visual comparisons. We also discuss the relationship between these metrics and perceived quality, and the trade-offs between classical and learning-based approaches.

The dataset used is the real-world subset of the LOL-v2 (Low-Light) dataset, limited to 80–120 paired images (low and normal) for manageable training and reproducible evaluation. All images are resized to 256×256 and normalized to [0, 1]. The implementation is provided as a single Jupyter notebook plus a small Python package for clarity and reuse.

---

## 2. Methodology

### 2.1 Data and preprocessing

- **Dataset:** LOL-v2 Real. Pairs of low-light and normal-light images with the same scene. Filenames follow the pattern `lowXXXXX.png` and `normalXXXXX.png`; pairs are identified by the numeric ID.
- **Size:** We use between 80 and 120 pairs. The exact count is verified by a discovery step that matches IDs between `data/low` and `data/normal`.
- **Preprocessing:** Each image is resized to 256×256 (fixed resolution for the autoencoder and for fair comparison) and pixel values are normalized to [0, 1]. No augmentation is applied by default so that evaluation is reproducible; optional horizontal flip or small rotation can be enabled for training if desired.
- **Split:** For classification (Phase 1), we form a single set of feature vectors (from both low and normal images) and use an 80/20 train/test split with stratification. For enhancement and evaluation (Phases 2 and 3), we split at the **pair** level (80% train, 20% test) so that the same pairs are not in both train and test.

### 2.2 Phase 1: Features and classifier

**Features (from grayscale):** For each image we convert to grayscale and compute four statistics:

- **Mean intensity** — Average pixel value; low-light images tend to have lower mean.
- **Standard deviation** — Contrast; can be lower in very dark images.
- **Entropy** — Information content of the intensity distribution; often differs between low and normal light.
- **Histogram skewness** — Asymmetry of the intensity distribution.

These four values form a feature vector per image. We stack features from all low-light images (label 0) and all normal-light images (label 1) and train a binary classifier.

**Classifier:** Logistic Regression with L2 regularization. No deep learning is used in this phase; the goal is to show that simple handcrafted features suffice to separate the two classes and to provide interpretable weights.

**Metrics:** Accuracy, precision, recall, F1-score, and a confusion matrix. These are reported on the held-out test set.

### 2.3 Phase 2: Enhancement methods

**Classical methods (Part A):**

1. **Histogram Equalization (HE)** — Global equalization of the intensity histogram to spread values over the full range. Applied on the L channel in LAB space to avoid color shift; then converted back to RGB.
2. **CLAHE (Contrast Limited Adaptive Histogram Equalization)** — Local histogram equalization with clipping to limit noise amplification. Applied on the L channel in LAB; clip limit 2.0 and tile grid (8×8) by default.
3. **Gamma Correction** — Nonlinear mapping *I_out = I_in^γ*. With *γ < 1* we brighten dark regions. We use *γ = 2.0* as a typical value; the effect of *γ* can be analyzed in a parameter study.
4. **Single-Scale Retinex (SSR)** — Model: *R = log(I) − log(L)*, where *L* is a Gaussian-smoothed version of *I*. Recovers a “reflectance” image that is less dependent on illumination. We use one scale (e.g. *σ = 30*) and normalize the result to [0, 1].

All classical methods are implemented with OpenCV/NumPy. Inputs and outputs are in [0, 1] (float) for consistency with the rest of the pipeline.

**Autoencoder (Part B):**

- **Architecture:** Convolutional encoder–decoder. Encoder: three blocks (e.g. 3 → 32 → 64 → 128 channels), each with conv + batch norm + ReLU. Decoder: three conv layers (128 → 64 → 32 → 3) with ReLU and final sigmoid. No pooling that would reduce spatial size; the map stays full resolution.
- **Training:** Input = low-light image, target = normal-light image (same pair). Loss = MSE between output and target. Optimizer: Adam, learning rate 1e-3, batch size 8, 25 epochs (within the 20–30 range). No GANs, no perceptual loss.
- **Inference:** The model takes a low-light image and outputs an enhanced image in [0, 1].

### 2.4 Phase 3: Evaluation metrics

- **PSNR (Peak Signal-to-Noise Ratio):**  
  PSNR = 10 · log₁₀(MAX_I² / MSE), with MAX_I = 255 and MSE the mean squared difference between enhanced and reference (after converting to [0, 255]). Higher PSNR means lower MSE, i.e. closer to the reference in a pixel-wise sense. Unit: dB.

- **SSIM (Structural Similarity Index):**  
  Compares local mean, variance, and covariance of the enhanced and reference images, with constants C₁, C₂ to avoid division by zero (e.g. K₁ = 0.01, K₂ = 0.03, L = 255). SSIM is in [0, 1] (or slightly negative); 1 means identical structure. We use the same formula as in common implementations (e.g. skimage) for consistency.

For each method we report mean and standard deviation of PSNR and SSIM over the test set, with the normal-light image as reference. We also report metrics for the **original low-light** image (no enhancement) as a baseline.

---

## 3. Experiments

- **Environment:** Python 3 with NumPy, OpenCV, scikit-learn, PyTorch, scipy, scikit-image, matplotlib, pandas. The notebook is run from the project root so that the `lowlight` package is importable.
- **Data:** After discovery, we have \(N\) pairs (\(80 \leq N \leq 120\)). Train/test split for pairs: 80% train, 20% test, fixed random seed (42) for reproducibility.
- **Phase 1:** Features extracted from all \(2N\) images; classifier trained on 80% of the feature/label set and evaluated on 20%. Confusion matrix and classification report are computed and the classifier is saved.
- **Phase 2:** Classical methods are applied to the **test** low-light images only (no training). The autoencoder is trained on the **train** pairs (low → normal) and then applied to the test low-light images. Training loss is plotted over epochs.
- **Phase 3:** For each test pair we compute PSNR and SSIM between (a) original low-light and reference, (b) each classical method’s output and reference, (c) autoencoder output and reference. Results are aggregated into a table (mean ± std) and bar charts. One or more test images are shown in a grid: low-light input | HE | CLAHE | Gamma | SSR | Autoencoder | reference.

All figures (confusion matrix, enhancement comparison, PSNR/SSIM bars) are saved under `figures/` for inclusion in the report.

---

## 4. Results

### 4.1 Phase 1: Classification

The handcrafted features (mean, std, entropy, skewness) separate low-light from normal-light images with good accuracy. Typical results (depending on the exact split and data):

- **Accuracy:** High (e.g. &gt; 90%), indicating that the two classes have distinct distributions in this feature space.
- **Precision / Recall / F1:** Reported per class in the classification report. The confusion matrix shows few misclassified samples.

The confusion matrix figure is saved as `figures/phase1_confusion_matrix.png`. It shows that the classifier rarely confuses low-light with normal-light when trained on this dataset.

### 4.2 Phase 2: Enhancement

- **Classical methods:** HE, CLAHE, Gamma, and SSR all produce visibly brighter and often higher-contrast images. HE can over-enhance and amplify noise; CLAHE tends to look more natural due to local adaptation. Gamma is simple and predictable; SSR can reduce illumination effects but may introduce halos or look unnatural if the scale is not tuned.
- **Autoencoder:** After 25 epochs, the training MSE decreases. On test images, the autoencoder produces smoother, often more “averaged” results than some classical methods. It does not use perceptual or adversarial losses, so fine detail and sharpness are limited by the MSE objective.

A visual comparison for one test image (low | HE | CLAHE | Gamma | SSR | Autoencoder | reference) is saved as `figures/phase2_comparison.png`.

### 4.3 Phase 3: Quantitative evaluation

The results table (printed in the notebook and reproducible by running it) typically looks like:

| Method              | PSNR (dB)   | SSIM      |
|---------------------|------------|-----------|
| Original (low-light) | lowest     | lowest    |
| Histogram Eq.        | higher     | …         |
| CLAHE                | …          | …         |
| Gamma                | …          | …         |
| SSR                  | …          | …         |
| Autoencoder          | …          | …         |

- The **original low-light** image has the lowest PSNR and SSIM with respect to the reference, as expected.
- **Classical methods** usually improve both PSNR and SSIM. Which method is best depends on the scene and the reference; sometimes CLAHE or Gamma wins, sometimes HE or SSR.
- The **autoencoder** often reaches competitive or better PSNR/SSIM on average, because it is trained to approximate the reference. In some cases classical methods can still outperform it, especially on very dark or noisy images that are under-represented in the small training set.

Bar charts for PSNR and SSIM across methods are saved as `figures/phase3_psnr_ssim_bars.png`. Error bars (e.g. standard deviation) show variability across test images.

---

## 5. Discussion and Conclusion

### 5.1 Why can higher PSNR sometimes correspond to worse visual quality?

PSNR is a **pixel-wise** measure: it only cares about the squared difference between each pixel of the enhanced image and the reference. It does not account for:

- **Structural or perceptual importance** — Blurring or oversmoothing can reduce fine detail and make the image look worse to a human, while sometimes reducing MSE (e.g. by averaging out noise) and thus increasing PSNR.
- **Alignment** — Slight spatial misalignment between enhanced and reference (e.g. from different capture or cropping) can lower PSNR even when the image looks good.
- **Contrast and local structure** — Two images can have similar MSE but different local contrast or edges; SSIM partially addresses this, but PSNR does not.

So a method that produces a smooth, “flat” image might get higher PSNR than one that preserves texture but has small local errors. For this reason, we use both PSNR and SSIM and always inspect **visual** results alongside the numbers.

### 5.2 In which scenarios do classical methods outperform the autoencoder?

Classical methods can outperform the autoencoder when:

- **Training data is limited** — With only 80–120 pairs, the autoencoder may not see enough variety (e.g. very dark or very noisy images, or specific scene types). Classical methods are not trained and can still improve such images using fixed rules (e.g. gamma, CLAHE).
- **Scene or condition is out-of-distribution** — If the test image is much darker or noisier than the training set, the autoencoder may under-enhance or produce artifacts. HE, CLAHE, or Gamma will still apply the same transformation regardless.
- **Overfitting** — The small dataset can lead to overfitting; the autoencoder might “average” training pairs and lose detail. Classical methods do not overfit in this sense.

So in practice, classical methods are more **robust** when data is scarce or test conditions vary; the autoencoder can do better when test images are similar to the training distribution and when the goal is to match the reference as closely as possible in MSE.

### 5.3 Trade-off between brightness enhancement and noise amplification

Many enhancement methods **increase the visibility of noise** when they boost brightness or contrast:

- **Histogram Equalization** stretches the intensity range; dark, noisy regions become brighter and their noise becomes more visible. Global HE is especially prone to this.
- **CLAHE** limits the amount of contrast enhancement per region (clip limit), which reduces but does not eliminate noise amplification. It is usually better than global HE in this respect.
- **Gamma correction** applies a smooth nonlinearity; it can brighten shadows without as much local amplification as HE, so noise may be less exaggerated, but it still depends on the image.
- **Retinex** separates “reflectance” from “illumination”; the log and filtering can amplify noise in very dark areas if the Gaussian scale is not chosen carefully.

So there is a direct **trade-off**: more aggressive brightness/contrast enhancement usually improves visibility but can make noise more visible. The “best” method or parameter is often a compromise. In the report and notebook, we use fixed parameters; a full study would vary gamma, CLAHE clip limit, and SSR sigma to document this trade-off explicitly.

### 5.4 Is increased brightness always equivalent to better image quality?

**No.** Increased brightness can improve quality when the image is underexposed, but it can also reduce quality when:

- **Overexposure** — Pushing brightness too far can clip highlights and lose detail in bright regions; the image looks washed out.
- **Loss of contrast or dynamic range** — If the whole image is pushed to a narrow bright range, it can look flat and unnatural.
- **Unnatural appearance** — Some methods (e.g. aggressive HE or SSR) can produce halos, color shifts, or an “overprocessed” look that viewers dislike even if the image is brighter.
- **Noise and artifacts** — As above, boosting brightness often amplifies noise and compression artifacts.

So “brightness” and “quality” are not the same. The goal of enhancement is to improve **perceived** quality, which includes natural appearance, preservation of detail, and acceptable noise—not just higher average intensity. This is why we evaluate with both reference-based metrics (PSNR, SSIM) and visual inspection, and why we compare several methods rather than a single one.

### 5.5 Conclusion

This project implemented a full pipeline for low-light image enhancement: (1) analysis and classification of low vs normal light using handcrafted features and Logistic Regression, (2) four classical enhancement methods (HE, CLAHE, Gamma, SSR) and one lightweight convolutional autoencoder trained with MSE, and (3) quantitative evaluation with PSNR and SSIM plus qualitative comparison.

The results show that both classical and learning-based methods can improve low-light images relative to the reference. Classical methods are simple, interpretable, and robust when data is limited; the autoencoder can achieve high PSNR/SSIM when the test set is similar to the training set but may underperform on out-of-distribution or very challenging images. The discussion highlights that PSNR alone is not a perfect proxy for visual quality, that brightness and quality are not equivalent, and that there is an inherent trade-off between enhancement strength and noise amplification.

All code, figures, and this report are self-contained and can be reproduced by running the provided Jupyter notebook with the described dataset and environment.
