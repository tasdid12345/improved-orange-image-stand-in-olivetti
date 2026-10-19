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

## Reference

This project is motivated by the following paper:

> C Pornpanomchai (2025). *Image-based orange (Citrus Reticulata and Citrus Reticulata Blanco) sweetness measurement by ResNet50*. https://so01.tci-thaijo.org/index.php/APST/article/download/275389/180797/1166249

Paper author: Prof. Assoc. Prof. Dr. Chomtip Pornpanomchai, Mahidol University (chomtip.por@mahidol.ac.th)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
