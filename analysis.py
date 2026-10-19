"""Baseline (pixel MLP, stand-in for the paper's CNN) vs improved (HOG-style features + LightGBM)
on the Olivetti faces set, a public image set where each subject appears repeatedly."""
import json
import socket
import sys
import time
import warnings

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")  # headless plotting on CI
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
SEEDS = [42, 43, 44]  # both models are randomised, so repeat over these seeds
SOURCE_URL = "https://scikit-learn.org/stable/datasets/real_world.html#the-olivetti-faces-dataset"
T0 = time.time()
warnings.filterwarnings("ignore")  # silence MLP convergence warnings in the small grid


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def load_data(retries=4):
    """Fetch Olivetti faces (sklearn downloads over HTTP); retry with exponential backoff."""
    socket.setdefaulttimeout(60)  # 60 s timeout on downloads
    for attempt in range(retries):
        try:
            return fetch_olivetti_faces(shuffle=False, random_state=SEED)
        except Exception as e:  # HTTP 429/5xx or network errors surface as urllib exceptions
            wait = 2 ** attempt
            log(f"download failed ({e}); retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError("could not download Olivetti faces")


def hog_like(images, cell=8, bins=9):
    """HOG-style descriptor: per-cell histograms of unsigned gradient orientation,
    magnitude-weighted, each cell histogram L2-normalised."""
    n, h, w = images.shape
    ch, cw = h // cell, w // cell
    ii, jj = np.indices((h, w))
    cell_idx = (ii // cell) * cw + (jj // cell)  # cell id for every pixel
    feats = np.zeros((n, ch * cw, bins))
    for k, img in enumerate(images):
        gy, gx = np.gradient(img)
        mag = np.hypot(gx, gy)
        ang = np.mod(np.arctan2(gy, gx), np.pi)  # unsigned orientation in [0, pi)
        b = np.minimum((ang / np.pi * bins).astype(int), bins - 1)
        hist = np.bincount((cell_idx * bins + b).ravel(), weights=mag.ravel(),
                           minlength=ch * cw * bins)
        feats[k] = hist.reshape(ch * cw, bins)
    feats /= np.linalg.norm(feats, axis=2, keepdims=True) + 1e-6  # per-cell normalisation
    return feats.reshape(n, -1)


def main():
    data = load_data()
    X_pix = data.data.astype(np.float64)  # (n, 4096) pixel intensities in [0, 1]
    y = data.target
    X_hog = hog_like(X_pix.reshape(-1, 64, 64))
    log(f"loaded {len(y)} images, {len(np.unique(y))} subjects; "
        f"pixel dim {X_pix.shape[1]}, HOG dim {X_hog.shape[1]}")

    # ONE stratified split (within subject, so the same subjects appear on both sides)
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.3, stratify=y, random_state=SEED)
    log(f"train {len(tr)} / test {len(te)}")
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

    def make_baseline(seed):
        return make_pipeline(StandardScaler(),
                             MLPClassifier(max_iter=300, random_state=seed))

    def make_improved(seed):
        return lgb.LGBMClassifier(class_weight="balanced", subsample=0.8, subsample_freq=1,
                                  colsample_bytree=0.8, learning_rate=0.05,
                                  random_state=seed, verbose=-1, n_jobs=2)

    # Hyper-parameter search: 3-fold CV on the TRAINING split only (seed 42)
    grids = {
        "baseline": (make_baseline(SEED), X_pix, {
            "mlpclassifier__alpha": [1e-4, 1e-2],
            "mlpclassifier__hidden_layer_sizes": [(100,), (256,)]}),
        "improved": (make_improved(SEED), X_hog, {
            "n_estimators": [200, 400],
            "num_leaves": [15, 31]}),
    }
    best = {}
    for name, (est, X, grid) in grids.items():
        gs = GridSearchCV(est, grid, cv=cv, scoring="accuracy", n_jobs=1)
        gs.fit(X[tr], y[tr])
        best[name] = gs.best_params_
        log(f"{name}: best CV params {gs.best_params_}, CV accuracy {gs.best_score_:.4f}")

    def evaluate(name, make, X, seed):
        """Refit on the training split with the tuned params; score ONCE on the test split."""
        m = make(seed).set_params(**best[name])
        m.fit(X[tr], y[tr])
        pred = m.predict(X[te])
        proba = m.predict_proba(X[te])
        return (accuracy_score(y[te], pred),
                f1_score(y[te], pred, average="macro"),
                log_loss(y[te], proba, labels=np.unique(y)))

    res = {}
    for name, make, X in [("baseline", make_baseline, X_pix), ("improved", make_improved, X_hog)]:
        arr = np.array([evaluate(name, make, X, s) for s in SEEDS])  # rows: seeds
        res[name] = arr
        log(f"{name}: test accuracy {arr[:, 0].mean():.4f} +/- {arr[:, 0].std():.4f} "
            f"over seeds {SEEDS}")

    b, i = res["baseline"], res["improved"]
    b_acc, i_acc = b[:, 0].mean(), i[:, 0].mean()
    if not (np.isfinite(b).all() and np.isfinite(i).all()):
        raise ValueError("non-finite metrics")

    results = {
        "question": ("Can a gradient-boosting model on compact image features beat a pixel-based "
                     "neural-network stand-in for the paper's ResNet50 on a public image set with "
                     "repeated subjects, when the test split is kept within the same subjects?"),
        "method": ("Stratified within-subject split; standardised pixel MLP baseline vs LightGBM "
                   "with balanced class weights on HOG-style gradient-orientation features; "
                   "3-fold stratified CV on the training split for tuning; test set used once per seed."),
        "dataset": {
            "name": "Olivetti faces (scikit-learn fetch_olivetti_faces)",
            "source_url": SOURCE_URL,
            "n_rows": int(len(y)),
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "same_as_paper": False,
            "baseline_is_papers_method": False,
        },
        "metric_name": "test accuracy",
        "higher_is_better": True,
        "baseline": ("Standardised pixel MLPClassifier (stand-in for the paper's ResNet50); "
                     "tuned over alpha and hidden layer size."),
        "improved": ("LightGBM classifier with class_weight='balanced' on HOG-style cell "
                     "orientation histograms; tuned over n_estimators and num_leaves."),
        "metrics": {
            "baseline_score": float(b_acc),
            "improved_score": float(i_acc),
            "improvement": float(i_acc - b_acc),
            "baseline_score_std": float(b[:, 0].std()),
            "improved_score_std": float(i[:, 0].std()),
            "baseline_f1": float(b[:, 1].mean()),
            "improved_f1": float(i[:, 1].mean()),
            "baseline_logloss": float(b[:, 2].mean()),
            "improved_logloss": float(i[:, 2].mean()),
        },
        "notes": ("Olivetti faces is a stand-in for the paper's orange images (no public orange-Brix "
                  "dataset with a download link); faces are not oranges and the task is 40-way "
                  "identity recognition, not sweetness regression. The MLP is not ResNet50. The "
                  "test split has only about 120 images, so accuracy differences of a few points "
                  "are within noise; std is over three model seeds only, with one data split. "
                  "HOG-style features are a simplified numpy implementation, not skimage HOG."),
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log("wrote results.json")

    # Optional chart: baseline vs improved test accuracy
    import os
    os.makedirs("figures", exist_ok=True)
    plt.figure(figsize=(5, 4))
    plt.bar(["baseline (pixel MLP)", "improved (HOG + LightGBM)"], [b_acc, i_acc],
            yerr=[b[:, 0].std(), i[:, 0].std()], capsize=6, color=["gray", "tab:blue"])
    plt.ylabel("test accuracy (mean over seeds)")
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig("figures/accuracy_comparison.png", dpi=120)
    log("saved figures/accuracy_comparison.png")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FAILED: {e}", file=sys.stderr)
        sys.exit(1)
