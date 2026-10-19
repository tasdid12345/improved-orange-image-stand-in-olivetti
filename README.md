# Gradient boosting on compact image features vs a pixel-network baseline (face-image stand-in)

**Motivation.** The paper "Image-based orange (Citrus Reticulata and Citrus Reticulata Blanco) sweetness measurement by ResNet50" (2025) estimates fruit sweetness from video frames with a ResNet50 CNN, using frames from a small number of clips, with a random frame split that leaves the same fruit on both sides. Its reported accuracy is high, but it relies on a deep network and a dataset that is not publicly available. This experiment asks whether a cheaper, more interpretable model can do as well on a public image set with repeated subjects.

**Question.** Can a gradient-boosting model on compact image features beat a pixel-based neural-network stand-in for the paper's ResNet50 on a public image set with repeated subjects, when the test split is kept within the same subjects?

**Dataset.** The Olivetti faces set from scikit-learn (`fetch_olivetti_faces`), a public grayscale face image set in which each subject appears many times. It is a stand-in for the repeated-object setting, not the paper's own orange dataset (`same_as_paper: false`). The split is stratified within each subject, so the same subjects appear in training and test.

**Baseline vs improved.**
- Baseline: standardised pixel MLP (`MLPClassifier`), standing in for the paper's ResNet50.
- Improved: LightGBM with balanced class weights on HOG-style gradient-orientation histograms.
- Both models are tuned with 3-fold stratified cross-validation on the training split only. The test split is scored once per model seed.

**How to run**
```
pip install -r requirements.txt
python analysis.py
```

**Results** are written to `results.json` (and `figures/accuracy_comparison.png`).

## Reference

This project is motivated by the following paper:

> C Pornpanomchai (2025). *Image-based orange (Citrus Reticulata and Citrus Reticulata Blanco) sweetness measurement by ResNet50*. https://so01.tci-thaijo.org/index.php/APST/article/download/275389/180797/1166249

Paper author: Prof. Assoc. Prof. Dr. Chomtip Pornpanomchai, Mahidol University (chomtip.por@mahidol.ac.th)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
