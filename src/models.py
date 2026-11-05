"""Model zoo: sklearn pipelines (TF-IDF -> classifier) and their tuning grids.

All supervised models use class_weight='balanced' because the Low and Critical
classes are minorities and the primary metric is macro-F1.
"""
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from . import config
from .features import word_tfidf


def build_models():
    """Return {name: (pipeline, param_grid)}. Grids are small to keep runtime low."""
    seed = config.SEED
    return {
        "Majority baseline": (
            Pipeline([("tfidf", word_tfidf()), ("clf", DummyClassifier(strategy="most_frequent"))]),
            {},
        ),
        "Complement NB": (
            Pipeline([("tfidf", word_tfidf()), ("clf", ComplementNB())]),
            {"clf__alpha": [0.1, 0.3, 1.0]},
        ),
        "Logistic Regression": (
            Pipeline([("tfidf", word_tfidf()),
                      ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed))]),
            {"clf__C": [1.0, 4.0, 16.0]},
        ),
        "Linear SVM": (
            Pipeline([("tfidf", word_tfidf()),
                      ("clf", LinearSVC(class_weight="balanced", random_state=seed))]),
            {"clf__C": [0.1, 0.3, 1.0]},
        ),
        "Random Forest": (
            Pipeline([("tfidf", word_tfidf()),
                      ("clf", RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                                     n_jobs=1, random_state=seed))]),
            {"clf__min_samples_leaf": [1, 2]},
        ),
    }
