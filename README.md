# Dry Bean Variety Classifier

Six classification models trained on the UCI Dry Bean dataset, served through a Streamlit app.

- **Live app:** https://2025ac05071.streamlit.app/
- **Repository:** https://github.com/avishek-seal/ml-assignment-02

## a. Problem statement

Dry beans are sorted by variety before they are sold as seed. Doing this by hand is slow, and two workers often disagree on the same bean. A camera can photograph each grain instead. Software then measures the shape of the grain and predicts which variety it is.

This project does the prediction step. The input is 16 shape measurements taken from a grain image. The output is one of 7 bean varieties.

So this is a **multiclass classification** problem on tabular numeric data.

## b. Dataset description

**Source:** UCI Machine Learning Repository, Dry Bean Dataset (ID 602)
https://archive.ics.uci.edu/dataset/602/dry+bean+dataset

| Property | Value |
| --- | --- |
| Instances (raw) | 13,611 |
| Instances (after removing 68 duplicate rows) | 13,543 |
| Features | 16, all continuous numeric |
| Target | `Class`, 7 bean varieties |
| Missing values | None |
| Train / test split | 10,834 / 2,709 (80/20, stratified) |

**Features:** Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRation, Eccentricity, ConvexArea, EquivDiameter, Extent, Solidity, roundness, Compactness, ShapeFactor1, ShapeFactor2, ShapeFactor3, ShapeFactor4.

**Class distribution:** DERMASON 3,546 · SIRA 2,636 · SEKER 2,027 · HOROZ 1,928 · CALI 1,630 · BARBUNYA 1,322 · BOMBAY 522.

The biggest class has about 6.8 times more rows than the smallest one. That changes how the scores should be averaged. This project uses **macro averaging**, which gives every variety equal weight. The other option, weighted averaging, would let DERMASON decide most of the score and hide weak results on the rare varieties.

**Preprocessing:** two steps.

1. Removed 68 duplicate rows. Each of these had the exact same value in all 16 columns, which points to a copy made during data preparation rather than two real beans. Left in place, the same row can land in both the train and the test half, and the scores come out higher than they should.
2. Scaled the features with `StandardScaler`. Area runs into the hundreds of thousands while Solidity sits close to 1. Without scaling, kNN and Logistic Regression treat Area as the only feature that counts. The scaler is fitted on the training rows only and is packed inside each model pipeline, so the same scaling is applied again at prediction time.

## c. GitHub repository link

https://github.com/avishek-seal/ml-assignment-02

## d. Models used

| Model | Notes |
| --- | --- |
| Logistic Regression | multinomial, `max_iter=2000` |
| Decision Tree | `max_depth` picked by 3-fold CV |
| k-Nearest Neighbors | `k` picked by 3-fold CV |
| Naive Bayes | Gaussian (scaled features go negative, so Multinomial is ruled out) |
| Random Forest | 200 trees, `min_samples_leaf=2` |
| Gradient Boosting | histogram-based, the sixth model |

All six are scikit-learn `Pipeline` objects, and each one carries its own scaler. Random seed `18082026` is used everywhere, so the numbers below can be reproduced.

### Evaluation metrics

Measured on the 2,709-row test set, which no model saw during training.

What the six metrics mean in plain terms:

- **Accuracy:** out of 100 beans, how many the model labelled correctly.
- **AUC:** how well the model ranks. Pick one bean that really is SIRA and one that is not. AUC is the chance the model gives the real SIRA the higher score. Reported here as macro one-vs-rest.
- **Precision:** when the model says SIRA, how often it is right.
- **Recall:** out of all the real SIRA beans, how many the model found.
- **F1:** one number that balances precision and recall.
- **MCC (Matthews Correlation Coefficient):** a score from -1 to +1 built from the whole confusion matrix, where 0 means random guessing. It stays honest when the classes are uneven, so it is used here as the deciding score.

Precision, recall and F1 are macro averaged.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.9247 | 0.9955 | 0.9376 | 0.9354 | 0.9365 | 0.9089 |
| Decision Tree | 0.9125 | 0.9744 | 0.9255 | 0.9235 | 0.9245 | 0.8941 |
| k-Nearest Neighbors | 0.9214 | 0.9893 | 0.9380 | 0.9332 | 0.9353 | 0.9049 |
| Naive Bayes | 0.8970 | 0.9917 | 0.9054 | 0.9062 | 0.9056 | 0.8757 |
| Random Forest | 0.9273 | 0.9948 | 0.9393 | 0.9372 | 0.9381 | 0.9121 |
| Gradient Boosting | 0.9240 | 0.9954 | 0.9361 | 0.9359 | 0.9359 | 0.9080 |

Best by MCC: Random Forest.

### Observations

Tuning picked `k=11` for k-Nearest Neighbors and `max_depth=10` for the Decision Tree, both by 3-fold CV.

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | **Highest AUC of all six models at 0.9955**, which beats both tree ensembles. On accuracy it comes second at 0.9247, only 0.0026 behind Random Forest. The two scores measure different things. AUC checks ranking across every threshold. Accuracy checks one final pick, the top-scoring class. Best-in-table ranking means the model puts the right variety near the top almost every time, and that is what happens when the classes can be split by **straight lines** in the scaled 16-feature space. A few beans sit right on a boundary and fall on the wrong side of the final pick, which costs a little accuracy without denting the ranking. |
| Decision Tree | **Lowest AUC of the six at 0.9744**, but fifth rather than last on accuracy (0.9125), precision, recall, F1 and MCC (0.8941), beating Naive Bayes on every one of those. A single depth-10 tree hands every bean in the same leaf the same probability. A bean sitting right next to the split line scores the same as one deep inside the leaf. Those coarse, step-shaped scores are what pull the AUC down, because AUC needs fine ranking rather than a correct majority vote per leaf. The vote itself is still usually right, so accuracy holds up better than AUC does. This is also why Random Forest and Gradient Boosting win the AUC back. Averaging many different trees smooths those steps into something closer to a continuous score. It is a **variance fix**, not a change in how trees think, since they still cut the space with axis-aligned lines. |
| k-Nearest Neighbors | Fourth of six on accuracy (0.9214), recall, F1 and MCC (0.9049). Second on precision (0.9380), behind only Random Forest. Fifth on AUC (0.9893), ahead of only the Decision Tree. kNN measures straight-line distance and treats all 16 features as equally important, but four of them (Area, Perimeter, ConvexArea, EquivDiameter) are close to the same measurement under different names, with pairwise correlations of 0.9669 to 0.9999. Size therefore gets counted four times over inside the distance, and the shape features that actually separate lookalike varieties (Eccentricity, roundness, ShapeFactor1 to ShapeFactor4) get drowned out. CV picked `k=11`. Looking at 11 neighbours cancels noisy labels, but it also blurs the border between classes that sit next to each other. The result is a model that is solid at picking a label and weaker at ranking. |
| Naive Bayes | Last on accuracy (0.8970), precision, recall, F1 and MCC (0.8757), yet fourth on AUC (0.9917), ahead of both kNN (0.9893) and the Decision Tree (0.9744). That is the widest gap between ranking and accuracy in the table, and the cause is checkable. On the 13,543 clean rows, `Area`, `Perimeter`, `ConvexArea` and `EquivDiameter` correlate from **0.9669 to 0.9999**, with `Area` and `ConvexArea` at **0.9999**. These four are close to one feature repeated four times. Gaussian NB assumes the features are independent once the class is known, so it multiplies each feature likelihood together. Multiplying four near-copies lets size overpower the genuinely separate shape features, the decision boundary shifts, and accuracy and MCC take the biggest hit of any model. AUC survives because the inflated size term still moves in the same direction as the truth: bigger bean, bigger score. The order across beans stays roughly right even though the probability values themselves are off, so ranking holds up while the single final pick does not. |
| Random Forest | **Best model on accuracy (0.9273), MCC (0.9121) and macro F1 (0.9381)**, and third on AUC (0.9948), within 0.001 of the top two. It fixes the exact weakness of the single Decision Tree (0.9125 accuracy, 0.8941 MCC). Two hundred trees are grown, each on a different bootstrap sample and each picking from a random subset of features at every split, so their leaf errors point in different directions and cancel out under averaging. Going from 1 tree to 200 cuts **variance** without changing how a tree thinks, since the splits are still axis-aligned. That is why the two tree ensembles sit at the top of the table together. |
| Gradient Boosting | Second on AUC (0.9954), effectively tied with Logistic Regression at 0.9955, and third on accuracy (0.9240). Boosting adds trees one at a time, each new tree correcting the mistakes the ensemble has made so far, using a log-loss objective. Log loss rewards good probabilities rather than just a correct final pick, which is why the AUC lands at the top next to Logistic Regression, a model that outputs probabilities by design. Accuracy is strong but stays a little behind Random Forest, whose full averaging is better suited to the single hard decision. |

### Per-class results: rare does not mean hard

Recall per variety on the 2,709-row test split:

| Model | BARBUNYA (265) | BOMBAY (104) | CALI (326) | DERMASON (709) | HOROZ (372) | SEKER (406) | SIRA (527) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.909 | **1.000** | 0.929 | 0.917 | 0.957 | 0.961 | *0.875* |
| Decision Tree | 0.898 | **1.000** | 0.902 | 0.911 | 0.944 | 0.948 | *0.861* |
| k-Nearest Neighbors | 0.898 | **1.000** | 0.957 | 0.911 | 0.944 | 0.953 | *0.869* |
| Naive Bayes | *0.804* | **1.000** | 0.893 | 0.889 | 0.949 | 0.948 | 0.861 |
| Random Forest | 0.925 | **1.000** | 0.923 | 0.935 | 0.952 | 0.968 | *0.858* |
| Gradient Boosting | 0.932 | **1.000** | 0.926 | 0.922 | 0.949 | 0.966 | *0.856* |

Bold marks the rarest class. Italics mark each model's weakest class.

The usual worry with uneven classes is that the rare one will suffer. It does not happen here. **BOMBAY is the rarest variety, 104 rows in the test set and 522 overall, and all six models get it right every single time.** The hardest variety is **SIRA at 0.856 to 0.875 recall, and SIRA is the second largest class.** Difficulty comes from *features overlapping*, not from how often a class turns up, and both halves of that can be checked:

- **BOMBAY can be split off using one feature.** Its smallest test-set `Area` is 131,488, while the largest `Area` across all six other varieties put together is 106,806. The two ranges never touch. Any model that can draw a single threshold gets BOMBAY right, and all six do.
- **SIRA overlaps DERMASON.** Random Forest gets 75 SIRA beans wrong, and 55 of them (73%) are labelled DERMASON. The rest scatter thinly: HOROZ 7, SEKER 6, BARBUNYA 6, CALI 1. SIRA has a median `Area` of 45,000 against DERMASON's 31,663, but the two spreads overlap heavily and no single shape feature cuts cleanly between them.

Two things follow. First, macro averaging is still the right way to report. It is the safe default when the classes are 6.8 to 1 uneven, and it does not depend on the imbalance turning out to be harmless. On this dataset it costs the models very little, because the rare class is the easy one. Second, this is why no class weighting or resampling was applied. There is no rare-class failure to fix, and setting `class_weight="balanced"` would have solved a problem this data does not have.

### Overall winner

**Random Forest**, on the raw numbers. It is best on accuracy (0.9273), MCC (0.9121) and macro F1 (0.9381). MCC is used as the deciding score because it is built from the whole confusion matrix and does not get inflated by uneven classes the way plain accuracy can.

**But the gap is small enough to be noise, and the honest answer is a four-way tie.** With 2,709 test rows at roughly 92.7% accuracy, the standard error is 0.0050, so a 95% range covers about plus or minus 0.010. Random Forest beats Logistic Regression by only 0.0026, which sits well inside that range. McNemar's exact test on the paired predictions agrees:

| Random Forest vs | RF right, other wrong | RF wrong, other right | p-value | Verdict |
| --- | --- | --- | --- | --- |
| Logistic Regression | 57 | 50 | 0.562 | not significant |
| k-Nearest Neighbors | 61 | 45 | 0.145 | not significant |
| Gradient Boosting | 37 | 28 | 0.321 | not significant |
| Decision Tree | 89 | 49 | **0.0008** | significant |
| Naive Bayes | 131 | 49 | **<0.0001** | significant |

Read it this way. **Random Forest, Logistic Regression, k-Nearest Neighbors and Gradient Boosting cannot be told apart on this test set.** Random Forest is named the winner because it has the best raw numbers and because it tops all three threshold-based scores at the same time. The gaps down to Decision Tree and Naive Bayes are real, and both show up as significant.

If only one model can be deployed, Random Forest is the sensible pick. Logistic Regression still deserves a mention: it is the cheapest to run, the easiest to explain, and it has the best AUC of all six at 0.9955.

## Streamlit app

Three tabs:

1. **Evaluate:** upload a CSV or load the bundled test set, pick a model, then see all six metrics, a confusion matrix and a per-class report.
2. **Compare:** all six models scored on the same data, shown as a table and a chart.
3. **Dataset:** class counts, feature summary and a correlation heatmap.

The app does not fall over on odd files. A missing feature column gives a clear error naming the column. A file with no `Class` column still returns predictions, just without metrics. Data covering only some varieties still draws the full 7 by 7 confusion matrix, with AUC marked as partial.

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

Requires Python 3.11 or newer, because `scikit-learn==1.8.0` needs it. If your default `python` or `python3` points to something older, name the interpreter directly, for example `python3.12`.

Linux:

```
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python train.py                  # retrain (artifacts are committed)
streamlit run app.py
```

Windows:

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python train.py
streamlit run app.py
```

Tests:

```
pip install -r requirements-dev.txt
python -m pytest
```
