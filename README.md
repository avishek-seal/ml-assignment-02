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
| Logistic Regression | Posts the **highest AUC of all six models (0.9955)** — beating both ensembles — while ranking 2nd on accuracy (0.9247), just 0.0026 behind Random Forest's 0.9273. AUC measures whether the model's per-class scores rank a true instance above a false one across every threshold; accuracy measures a single argmax decision. LR having the best ranking quality of all six models is the signature of classes that are close to **linearly separable** in the standardised 16-D space — the softmax probabilities order instances correctly almost everywhere. The small residual gap between its (best) AUC rank and its (second-best) accuracy rank suggests a handful of borderline points near a linear boundary still cross the argmax threshold the wrong way, without materially denting the overall ranking. |
| Decision Tree | Worst of all six models on **AUC alone (0.9744, last place)** — but 5th, not last, on accuracy (0.9125), precision, recall, F1 and MCC (0.8941), ahead of Naive Bayes on every one of those. A single depth-10 tree assigns every test point in a leaf the *same* class-frequency probability regardless of how close that point sits to the split boundary; this coarse, piecewise-constant scoring is exactly what depresses AUC, since AUC needs fine-grained relative ranking, not just a correct majority vote per leaf — the argmax decision itself is still reasonably accurate, which is why the tree doesn't fall to last on the threshold metrics too. This also explains why bagging/boosting over trees (Random Forest, Gradient Boosting) recover so much AUC: averaging many decorrelated trees smooths those step-function probability estimates into something closer to continuous — a **variance fix**, not a bias fix, since the underlying axis-aligned split logic is unchanged. |
| k-Nearest Neighbors | 4th of six on accuracy (0.9214), recall, F1 and MCC (0.9049), but 2nd on precision (0.9380, behind only Random Forest) and 5th on AUC (0.9893) — ahead of only the Decision Tree on ranking quality. kNN's Euclidean distance in the standardised feature space treats all 16 dimensions as equally informative, but four of them (Area, Perimeter, ConvexArea, EquivDiameter) are near-duplicates of each other (pairwise correlations 0.9669–0.9999, verified below) — so "size" is effectively counted several times over in the distance metric, diluting the contribution of the less-correlated shape descriptors (Eccentricity, roundness, ShapeFactor*) that would otherwise sharpen the ranking between visually similar varieties. The CV-selected `k=11` smooths over local label noise but also blurs fine boundaries between adjacent classes, consistent with a model that is solid on hard-label metrics but comparatively weak on ranking. |
| Naive Bayes | Last on accuracy (0.8970), precision, recall, F1 and MCC (0.8757), yet 4th on AUC (0.9917) — ahead of both kNN (0.9893) and the Decision Tree (0.9744) — the largest rank-vs-accuracy gap in the table. We checked the mechanism directly: the correlation matrix of `Area`, `Perimeter`, `ConvexArea`, `EquivDiameter` on the deduplicated dataset (13,543 rows) shows correlations from **0.9669 to 0.9999**, with `Area`–`ConvexArea` at **0.9999** — these four "size" features are almost one feature repeated four times. Gaussian NB's core assumption is that features are conditionally independent given the class, so it multiplies four essentially-identical per-feature likelihoods into the posterior, over-weighting size relative to the genuinely independent shape descriptors and distorting the decision boundary — which is why accuracy and MCC suffer most of any model. AUC survives much better because the inflated size likelihood still increases *monotonically* with the true class-conditional density: the ranking of scores across instances is roughly preserved even though the absolute posterior is miscalibrated, so threshold-independent ranking (AUC) degrades far less than the single argmax decision (accuracy). |
| Random Forest | **Best model on accuracy (0.9273), MCC (0.9121) and macro-F1 (0.9381)**, and 3rd on AUC (0.9948, within 0.001 of the top two). 200 bootstrap-sampled, feature-subsampled trees average out exactly the coarse leaf-probability problem that hurts the single Decision Tree (0.9125 accuracy, 0.8941 MCC): each tree sees a different bootstrap sample and a different random feature subset per split, so their leaf-frequency errors are decorrelated and cancel under averaging. The jump from one tree to two hundred is a **variance reduction**, not a change in inductive bias — the trees are still axis-aligned learners — which is consistent with Random Forest and Gradient Boosting (also tree-ensembles) occupying the top of the table together. |
| Gradient Boosting | 2nd on AUC (0.9954, essentially tied with Logistic Regression's 0.9955) and 3rd on accuracy (0.9240), just behind Random Forest and Logistic Regression. Sequential boosting fits each new tree to the residual errors of the ensemble so far under a log-loss objective, which directly optimises probability *calibration and ranking* rather than a single hard split — that objective match is why its AUC sits at the very top alongside Logistic Regression's naturally probabilistic softmax output, while its accuracy, though strong, trails Random Forest's fully-averaged bootstrap ensemble slightly. |

### Per-class behaviour: rarity and difficulty are not the same thing

Recall per variety on the 2,709-row test split:

| Model | BARBUNYA (265) | BOMBAY (104) | CALI (326) | DERMASON (709) | HOROZ (372) | SEKER (406) | SIRA (527) |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.909 | **1.000** | 0.929 | 0.917 | 0.957 | 0.961 | *0.875* |
| Decision Tree | 0.898 | **1.000** | 0.902 | 0.911 | 0.944 | 0.948 | *0.861* |
| k-Nearest Neighbors | 0.898 | **1.000** | 0.957 | 0.911 | 0.944 | 0.953 | *0.869* |
| Naive Bayes | *0.804* | **1.000** | 0.893 | 0.889 | 0.949 | 0.948 | 0.861 |
| Random Forest | 0.925 | **1.000** | 0.923 | 0.935 | 0.952 | 0.968 | *0.858* |
| Gradient Boosting | 0.932 | **1.000** | 0.926 | 0.922 | 0.949 | 0.966 | *0.856* |

Bold marks the rarest class; italics mark each model's worst class.

The imbalance intuition does not survive contact with the data. **BOMBAY is the
rarest variety (104 test instances, 522 overall) and every one of the six models
classifies it perfectly.** The hardest variety is **SIRA (0.856–0.875), the
second-largest class**. Difficulty here is driven by *feature overlap*, not by
frequency, and both halves of that are checkable:

- **BOMBAY is linearly separable on a single feature.** Its smallest test-set
  `Area` is 131,488, while the largest `Area` across all six other varieties
  combined is 106,806 — the two ranges do not touch. Any model that can threshold
  one feature gets BOMBAY right, which is exactly what all six do.
- **SIRA overlaps DERMASON.** Of Random Forest's 75 SIRA misclassifications, 55
  (73%) are predicted as DERMASON; the remainder scatter thinly across HOROZ (7),
  SEKER (6), BARBUNYA (6) and CALI (1). SIRA's median `Area` (45,000) sits above
  DERMASON's (31,663) but their distributions overlap heavily, and no single
  geometric feature cleanly divides them.

Two consequences worth stating. First, macro averaging remains the correct
reporting choice — it is the honest default under a 6.8:1 imbalance and does not
depend on the imbalance turning out to be benign — but on *this* dataset it
happens to cost the models little, because the rare class is the easy one.
Second, this is why no class re-weighting or resampling was applied: there is no
rare-class failure to correct. Applying `class_weight="balanced"` here would
have optimised for a problem the data does not have.

### Overall winner

**Random Forest**, on the point estimates: best on accuracy (0.9273), MCC
(0.9121) and macro-F1 (0.9381). MCC is the right primary criterion because it is
computed from the whole confusion matrix and is not inflated by class imbalance
the way plain accuracy can be.

**But the margin is inside the noise, and the honest conclusion is a four-way
tie.** With 2,709 test instances at roughly 92.7% accuracy, the binomial
standard error is 0.0050, so a 95% interval spans about ±0.010 — while Random
Forest leads Logistic Regression by only 0.0026. McNemar's exact test on the
paired predictions confirms this:

| Random Forest vs | RF right, other wrong | RF wrong, other right | p-value | Verdict |
|---|---|---|---|---|
| Logistic Regression | 57 | 50 | 0.562 | not significant |
| k-Nearest Neighbors | 61 | 45 | 0.145 | not significant |
| Gradient Boosting | 37 | 28 | 0.321 | not significant |
| Decision Tree | 89 | 49 | **0.0008** | significant |
| Naive Bayes | 131 | 49 | **<0.0001** | significant |

So the defensible reading is: **Random Forest, Logistic Regression, k-Nearest
Neighbors and Gradient Boosting are statistically indistinguishable on this test
set**, and Random Forest is chosen as the winner on its point estimates and on
being best across all three threshold-based summary metrics simultaneously. The
Decision Tree and Naive Bayes are genuinely, measurably worse — those two gaps
are real.

If a single model had to be deployed, Random Forest is the reasonable pick; but
Logistic Regression deserves note as the cheapest and most interpretable member
of that tied group, and it has the best AUC (0.9955) of all six.

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

Tests: `pip install -r requirements-dev.txt && python -m pytest`
