# Olivetti faces: pixel MLP baseline vs HOG-style features + logistic regression

Question: can a regularised linear model on HOG-style gradient-orientation features beat a
pixel-based MLP baseline (stand-in for the paper's ResNet50) on Olivetti faces, using a
within-subject stratified train/test split?

Method:
- Data: `fetch_olivetti_faces` (400 images, 40 subjects), one stratified 70/30 split (seed 42).
- Baseline: standardised pixel MLPClassifier, tuned over alpha and hidden layer size with 3-fold CV on the training split.
- Improved: standardised LogisticRegression (class_weight='balanced') on HOG-style cell histograms, tuned over C with the same 3-fold CV on the training split.
- Test set is scored once per model seed (42, 43, 44).

Run: `pip install -r requirements.txt && python analysis.py` (CPU only, a few minutes).
Outputs: `results.json` and `figures/accuracy_comparison.png`.

Caveats: the test split has about 120 images, so small accuracy differences are within noise; the
MLP is not ResNet50 and the task is 40-way identity recognition, not sweetness regression.

## Results (latest run)

On the Olivetti faces dataset (a 120-image test split, not the paper's orange data), a standardised pixel MLP baseline reached 95.3% test accuracy, while a class-balanced logistic regression on HOG-style features reached 100%, a gain within noise given the small test set.

| Metric | Value |
|---|---|
| baseline_score | 0.9528 |
| improved_score | 1 |
| improvement | 0.04722 |
| baseline_score_std | 0.01964 |
| improved_score_std | 0 |
| baseline_f1 | 0.95 |
| improved_f1 | 1 |
| baseline_logloss | 0.2942 |
| improved_logloss | 0.07817 |

Run on 2026-10-19. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Olivetti faces is a stand-in for the paper's orange images (no public orange-Brix dataset with a download link); faces are not oranges and the task is 40-way identity recognition, not sweetness regression. The MLP is not ResNet50. The test split has only about 120 images, so accuracy differences of a few points are within noise; std is over three model seeds only, with one data split. HOG-style features are a simplified numpy implementation, not skimage HOG. The previous LightGBM improved model was replaced because its CV accuracy was far below the baseline's (boosted trees overfit with ~7 images per subject).

## Reference

This project is motivated by the following paper:

> C Pornpanomchai (2025). *Image-based orange (Citrus Reticulata and Citrus Reticulata Blanco) sweetness measurement by ResNet50*. https://so01.tci-thaijo.org/index.php/APST/article/download/275389/180797/1166249

Paper author: Prof. Assoc. Prof. Dr. Chomtip Pornpanomchai, Mahidol University (chomtip.por@mahidol.ac.th)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
