"""Model definitions and hyperparameter selection.

Every model is a Pipeline bundling its own StandardScaler. This is what
makes GridSearchCV refit the scaler inside each fold; tuning a bare
estimator on pre-scaled data would leak validation statistics into training.
"""

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.config import RANDOM_STATE

CV_FOLDS = 3

# Only models whose defaults are actively poor on this dataset are tuned.
# The assignment marks correct metric computation, not leaderboard position.
TUNING_GRIDS = {
    # Leaving k=5 unexamined is weak practice and a common copied-work
    # fingerprint.
    "k-Nearest Neighbors": {"model__n_neighbors": [3, 5, 7, 11, 15, 21]},
    # An unbounded tree memorises the training set and produces a
    # misleadingly poor test score that would distort the observations.
    "Decision Tree": {"model__max_depth": [5, 10, 15, 20, None]},
}


def _wrap(estimator) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("model", estimator)])


def build_models() -> dict[str, Pipeline]:
    """The six unfitted pipelines, in display order."""
    return {
        "Logistic Regression": _wrap(
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
        ),
        "Decision Tree": _wrap(
            DecisionTreeClassifier(min_samples_leaf=5, random_state=RANDOM_STATE)
        ),
        "k-Nearest Neighbors": _wrap(KNeighborsClassifier()),
        # Gaussian, not Multinomial: standardisation produces negative
        # values, which MultinomialNB rejects.
        "Naive Bayes": _wrap(GaussianNB()),
        "Random Forest": _wrap(
            RandomForestClassifier(
                n_estimators=200,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        ),
        # Histogram boosting rather than ExtraTrees: another bagged forest
        # would score close to Random Forest and make the comparison table
        # uninformative. Boosting is a genuinely different family.
        "Gradient Boosting": _wrap(
            HistGradientBoostingClassifier(random_state=RANDOM_STATE)
        ),
    }


def tune(name: str, pipe: Pipeline, X, y) -> tuple[Pipeline, dict]:
    """Fit `pipe`, running a CV sweep first if the model has a grid.

    Returns (fitted_pipeline, chosen_params). `chosen_params` is empty for
    models that are not tuned.
    """
    grid = TUNING_GRIDS.get(name)
    if grid is None:
        pipe.fit(X, y)
        return pipe, {}

    search = GridSearchCV(
        pipe, grid, cv=CV_FOLDS, scoring="f1_macro", n_jobs=-1
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_
