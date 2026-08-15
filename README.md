# Dry Bean Variety Classifier

Six classification models trained on the UCI Dry Bean dataset, served through
an interactive Streamlit app.

- **Live app:** <streamlit URL>
- **Repository:** https://github.com/avishek-seal/ml-assignment-02

## a. Problem statement

Classify dried bean grains into one of seven varieties from sixteen geometric
features extracted from grain images. Manual varietal sorting is slow and
inconsistent; an image-derived classifier automates seed-quality grading.

This is a **multiclass** problem (7 classes) on tabular numeric data.

## b. Dataset description

**Source:** UCI Machine Learning Repository, Dry Bean Dataset (ID 602) —
https://archive.ics.uci.edu/dataset/602/dry+bean+dataset

| Property | Value |
|---|---|
| Instances (raw) | 13,611 |
| Instances (after removing 68 exact duplicates) | 13,543 |
| Features | 16, all continuous numeric |
| Target | `Class` — 7 bean varieties |
| Missing values | None |
| Train / test split | 10,834 / 2,709 (80/20, stratified) |

**Features:** Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRation,
Eccentricity, ConvexArea, EquivDiameter, Extent, Solidity, roundness,
Compactness, ShapeFactor1, ShapeFactor2, ShapeFactor3, ShapeFactor4.

**Class distribution:** DERMASON 3,546 · SIRA 2,636 · SEKER 2,027 ·
HOROZ 1,928 · CALI 1,630 · BARBUNYA 1,322 · BOMBAY 522.

The largest class is ~6.8× the smallest. **All averaged metrics are therefore
macro-averaged**, weighting each variety equally; weighted averaging would let
DERMASON dominate and overstate performance on the rare varieties.

**Preprocessing:** 68 exact duplicate rows removed (identical values across
all 16 continuous features indicate a data-preparation artifact, and left in
place they straddle the split and inflate scores). Features standardised with
`StandardScaler` fitted on the training split only, bundled inside each model
pipeline so the same transform is applied at inference.

## c. GitHub repository link

https://github.com/avishek-seal/ml-assignment-02

## d. Models used

| Model | Notes |
|---|---|
| Logistic Regression | multinomial, `max_iter=2000` |
| Decision Tree | `max_depth` selected by 3-fold CV |
| k-Nearest Neighbors | `k` selected by 3-fold CV |
| Naive Bayes | Gaussian (standardised features are negative, ruling out Multinomial) |
| Random Forest | 200 trees, `min_samples_leaf=2` |
| Gradient Boosting | histogram-based, the sixth ensemble |

All six are scikit-learn `Pipeline` objects bundling their own scaler.
Random seed `18082026` throughout.

### Evaluation metrics

Measured on the 2,709-row held-out test set. AUC is macro one-vs-rest;
precision, recall and F1 are macro-averaged.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9247 | 0.9955 | 0.9376 | 0.9354 | 0.9365 | 0.9089 |
| Decision Tree | 0.9125 | 0.9744 | 0.9255 | 0.9235 | 0.9245 | 0.8941 |
| k-Nearest Neighbors | 0.9214 | 0.9893 | 0.9380 | 0.9332 | 0.9353 | 0.9049 |
| Naive Bayes | 0.8970 | 0.9917 | 0.9054 | 0.9062 | 0.9056 | 0.8757 |
| Random Forest | 0.9273 | 0.9948 | 0.9393 | 0.9372 | 0.9381 | 0.9121 |
| Gradient Boosting | 0.9240 | 0.9954 | 0.9361 | 0.9359 | 0.9359 | 0.9080 |

Best by MCC: Random Forest

### Observations

Tuning selected `k=11` for k-Nearest Neighbors and `max_depth=10` for the
Decision Tree via 3-fold CV.

| Model | Observation |
|---|---|
| Logistic Regression | Posts the **highest AUC of all six models (0.9955)** — beating both ensembles — while ranking only 3rd on accuracy (0.9247). AUC measures whether the model's per-class scores rank a true instance above a false one across every threshold; accuracy measures a single argmax decision. A gap this direction (near-best ranking, mid-pack hard labels) is the signature of classes that are close to **linearly separable** in the standardised 16-D space — the softmax probabilities order instances correctly almost everywhere — but a few varieties sit close enough to each other's linear boundary (bean shape descriptors vary continuously between similar cultivars) that small boundary shifts flip the occasional argmax, costing accuracy without denting the ranking. |
| Decision Tree | Worst model on every metric except precision/recall parity with itself — lowest AUC (0.9744), lowest accuracy (0.9125), lowest MCC (0.8941). A single depth-10 tree assigns every test point in a leaf the *same* class-frequency probability regardless of how close that point sits to the split boundary; this coarse, piecewise-constant scoring is exactly what depresses AUC, since AUC needs fine-grained relative ranking, not just a correct majority vote per leaf. It also explains why bagging/boosting over trees (Random Forest, Gradient Boosting) recover so much AUC and MCC: averaging many decorrelated trees smooths those step-function probability estimates into something closer to continuous — a **variance fix**, not a bias fix, since the underlying axis-aligned split logic is unchanged. |
| k-Nearest Neighbors | Mid-table on every metric (accuracy 0.9214, AUC 0.9893, MCC 0.9049) — worse than Logistic Regression and both ensembles, better than the Decision Tree and Naive Bayes. kNN's Euclidean distance in the standardised feature space treats all 16 dimensions as equally informative, but four of them (Area, Perimeter, ConvexArea, EquivDiameter) are near-duplicates of each other (pairwise correlations 0.9669–0.9999, verified below) — so "size" is effectively counted several times over in the distance metric, diluting the contribution of the less-correlated shape descriptors (Eccentricity, roundness, ShapeFactor*) that actually separate visually similar varieties. The CV-selected `k=11` smooths over local label noise but also blurs fine boundaries between adjacent classes, consistent with a model that is solid but not competitive with the top three. |
| Naive Bayes | Last on accuracy (0.8970) and MCC (0.8757), yet 3rd on AUC (0.9917) — the largest rank-vs-accuracy gap in the table. We checked the mechanism directly: the correlation matrix of `Area`, `Perimeter`, `ConvexArea`, `EquivDiameter` on the (deduplicated) training data shows correlations from **0.9669 to 0.9999**, with `Area`–`ConvexArea` at **0.9999** — these four "size" features are almost one feature repeated four times. Gaussian NB's core assumption is that features are conditionally independent given the class, so it multiplies four essentially-identical per-feature likelihoods into the posterior, over-weighting size relative to the genuinely independent shape descriptors and distorting the decision boundary — which is why accuracy and MCC suffer most of any model. AUC survives much better because the inflated size likelihood still increases *monotonically* with the true class-conditional density: the ranking of scores across instances is roughly preserved even though the absolute posterior is miscalibrated, so threshold-independent ranking (AUC) degrades far less than the single argmax decision (accuracy). |
| Random Forest | **Best model on accuracy (0.9273), MCC (0.9121) and macro-F1 (0.9381)**, and 3rd on AUC (0.9948, within 0.001 of the top two). 200 bootstrap-sampled, feature-subsampled trees average out exactly the coarse leaf-probability problem that hurts the single Decision Tree (0.9125 accuracy, 0.8941 MCC): each tree sees a different bootstrap sample and a different random feature subset per split, so their leaf-frequency errors are decorrelated and cancel under averaging. The jump from one tree to two hundred is a **variance reduction**, not a change in inductive bias — the trees are still axis-aligned learners — which is consistent with Random Forest and Gradient Boosting (also tree-ensembles) occupying the top of the table together. |
| Gradient Boosting | 2nd on AUC (0.9954, essentially tied with Logistic Regression's 0.9955) and 3rd on accuracy (0.9240), just behind Random Forest and Logistic Regression. Sequential boosting fits each new tree to the residual errors of the ensemble so far under a log-loss objective, which directly optimises probability *calibration and ranking* rather than a single hard split — that objective match is why its AUC sits at the very top alongside Logistic Regression's naturally probabilistic softmax output, while its accuracy, though strong, trails Random Forest's fully-averaged bootstrap ensemble slightly. |

**Overall winner:** **Random Forest** — it is the best model on **MCC (0.9121)**
and **macro-F1 (0.9381)**, not just accuracy (0.9273, also the highest). MCC
in particular is the right tie-breaker here because it is computed from the
whole confusion matrix and is not inflated by the ~6.8:1 class imbalance the
way plain accuracy can be; Random Forest leading on MCC as well as accuracy
means its advantage holds up on the rare varieties (e.g. BOMBAY, 522 rows),
not only on the dominant DERMASON class.

## Streamlit app

Three tabs:

1. **Evaluate** — upload a CSV (or load the bundled test set), pick a model,
   see all six metrics, a confusion matrix and a per-class report.
2. **Compare** — all six models scored live on the loaded data, tabulated and
   charted.
3. **Dataset** — class distribution, feature summary and correlation heatmap.

The app handles partial uploads: missing feature columns produce a named
error, a missing `Class` column falls back to predictions only, and data
covering only some varieties still renders a full 7×7 confusion matrix with
AUC flagged as partial.

## Project structure

```
app.py                  Streamlit application
train.py                Trains all six models, writes artifacts
requirements.txt        Runtime dependencies
test_data.csv           Held-out test split (2,709 rows)
src/
  config.py             Constants
  data.py               Loading, deduplication, splitting
  pipeline.py           Model definitions and tuning grids
  evaluate.py           Metric computation
scripts/
  fetch_data.py         One-off dataset download
  render_readme_table.py
model/                  Six pipeline pickles + metrics.json
data/dry_bean.csv       Full dataset
tests/                  pytest suite
```

## Running locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python train.py                  # retrain (artifacts are committed)
streamlit run app.py
```

Tests: `pip install -r requirements-dev.txt && pytest`
