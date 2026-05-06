# =============================================================================
# model.py — submission entry point for Track B (Fox vs NBC headlines).
# =============================================================================
# Purpose:
#   - Define the final feature + classifier pipeline (Hybrid_v2_style features
#     feeding a stacked LR + LinearSVC + SGD + RF ensemble with an LR meta-learner).
#   - Expose `Model` and `get_model()` per the course's submission contract:
#     the grading backend imports this module, calls `get_model()`, and uses
#     `Model.predict(headlines)` to score a hidden test set.
#   - Provide a `__main__` block that re-fits the pipeline on
#     data/expanded_headlines.csv and writes the artifact to model.pt.
#
# Artifact format:
#   model.pt is a torch.save({"pipeline_bytes": <joblib bytes>}) container.
#   The backend only accepts `.pt` and runs torch.load before handing control
#   to Model.__init__, which then joblib-loads the real sklearn pipeline.
#
# Companion files:
#   preprocess.py — provides prepare_data() and the style_branch transformer.
#   analysis/analysis.py — dev notebook where the pipeline + hyperparameters
#                          were selected (not submitted).
# =============================================================================

# Libraries
import os
import sys

# put this file's directory on sys.path so `from preprocess import ...`
# resolves regardless of where the grader invokes us from
_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
if _MODEL_DIR not in sys.path:
    sys.path.insert(0, _MODEL_DIR)

import io

import joblib
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import LinearSVC

from preprocess import style_branch, prepare_data

RANDOM_STATE = 42
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pt")


def _build_pipeline():
    # word-level branch -- captures vocabulary differences between Fox and NBC
    word_branch = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"\[?[\w]{2,}\]?",
        ngram_range=(1, 2),
        max_features=5000,
    )

    # character n-gram branch -- captures stylistic/morphological patterns
    char_branch = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(4, 6),
        max_features=2500,
    )

    # merge word + char text features
    combined_branches = FeatureUnion(
        [
            ("words", word_branch),
            ("chars", char_branch),
        ]
    )

    # merge text features with handcrafted style features (caps, "U.S.", etc.)
    word_char_style_branch = FeatureUnion(
        [
            ("text", combined_branches),
            ("style", style_branch),
        ]
    )

    # tuned base learners from Section 5 grid searches in analysis.py
    best_lr = LogisticRegression(
        C=1, solver="lbfgs", max_iter=1000, random_state=RANDOM_STATE
    )
    best_lsvc = LinearSVC(C=0.1, loss="hinge", max_iter=2000, random_state=RANDOM_STATE)
    best_sgd = SGDClassifier(
        alpha=0.001,
        l1_ratio=0.15,
        learning_rate="adaptive",
        eta0=0.01,
        loss="modified_huber",
        penalty="elasticnet",
        random_state=RANDOM_STATE,
    )
    best_rf = RandomForestClassifier(
        max_depth=None,
        max_features="sqrt",
        min_samples_leaf=1,
        n_estimators=300,
        random_state=RANDOM_STATE,
    )

    # stack the 4 base learners under an LR meta-learner
    # XGBoost excluded
    clf = StackingClassifier(
        estimators=[
            ("lr", best_lr),
            ("lsvc", best_lsvc),
            ("sgd", best_sgd),
            ("rf", best_rf),
        ],
        final_estimator=LogisticRegression(
            C=1, max_iter=1000, random_state=RANDOM_STATE
        ),
        cv=5,
        stack_method="auto",
        n_jobs=-1,
        passthrough=False,
    )

    return Pipeline(
        [
            ("branches", word_char_style_branch),
            ("scaler", MaxAbsScaler()),
            ("clf", clf),
        ]
    )


class Model:
    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Could not find fitted pipeline at {MODEL_PATH}. "
                "Run `python model.py` to train and produce model.pt."
            )
        # backend submission only accepts .pt, so we wrap the joblib-pickled
        # sklearn pipeline inside a torch.save container -- backend's torch.load
        # step in eval flow succeeds, and we extract the real pipeline here.
        state = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        self.pipeline = joblib.load(io.BytesIO(state["pipeline_bytes"]))
        # sklearn version mismatch: newer sklearn checks self.clip in MaxAbsScaler.transform
        # but pickles from older sklearn don't have the attribute -- add the default
        for _, step in self.pipeline.steps:
            if isinstance(step, MaxAbsScaler) and not hasattr(step, "clip"):
                step.clip = False

    def eval(self):
        return None

    def load_state_dict(self, state_dict, strict=True):
        # no-op -- backend calls this after torch.load(model.pt); our pipeline
        # is already loaded in __init__, and an sklearn Pipeline has no state
        # dict to apply. accepting and ignoring keeps the backend flow alive.
        return None

    def predict(self, batch):
        return self.pipeline.predict(pd.Series(list(batch))).tolist()


def get_model():
    return Model()


if __name__ == "__main__":
    DATA_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data",
        "expanded_headlines.csv",
    )

    # (Spanish/length/page-artifact filters + dedupe + URL-or-source label inference)
    X, y = prepare_data(DATA_PATH, verbose=True)

    pipeline = _build_pipeline()
    pipeline.fit(X, y)

    # serialize the fitted pipeline for Model.__init__ to load at grading time.
    # joblib-pickle into a buffer, then wrap in torch.save so the artifact is a
    # real torch zip (backend's torch.load step in the eval flow won't crash).
    buf = io.BytesIO()
    joblib.dump(pipeline, buf)
    torch.save({"pipeline_bytes": buf.getvalue()}, MODEL_PATH)
    print(f"Wrote {MODEL_PATH}")
