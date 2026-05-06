# =============================================================================
# 0. IMPORTS
# =============================================================================

import os
import sys

# Resolve the repo root whether this file is run as a script from analysis/
# or executed cell-by-cell in a notebook (where __file__ is undefined).
_here = (
    os.path.dirname(os.path.abspath(__file__))
    if "__file__" in globals()
    else os.getcwd()
)
# If we're inside analysis/, step up one level so imports and relative
# data paths (e.g. "data/...") resolve against the repo root.
REPO_ROOT = os.path.dirname(_here) if os.path.basename(_here) == "analysis" else _here
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

# Import shared preprocessing helpers from the repo-root preprocess file
from preprocess import prepare_data, clean_hed, style_branch, STYLE_FEATURE_NAMES

import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    VotingClassifier,
    StackingClassifier,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.base import clone
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, MaxAbsScaler

# from xgboost import XGBClassifier  # leaderboard env lacks xgboost
import joblib
import matplotlib.pyplot as plt

# =============================================================================
# 1. BASELINE MODEL
# =============================================================================
# SUMMARY: Loads the scraped headlines from data/expanded_headlines.csv (or scrapes
# from URLs if not yet cached), drops missing rows, and does an 80/20 split. Trains
# a Logistic Regression on TF-IDF (100 features, English stopwords). No filtering or
# feature engineering applied here; that's all in Section 2 onward.
# Result on the expanded dataset: 0.6442 accuracy — class 0 (NBC) recall 0.80 but
# precision 0.60, class 1 (Fox) precision 0.72 but recall 0.50 (macro F1 0.64).
# This is the baseline we aim to beat from Section 2 onward.

# Data collection
base_url_df = pd.read_csv("data/url_only_data.csv")

if os.path.exists("data/expanded_headlines.csv"):
    news_df_base = pd.read_csv("data/expanded_headlines.csv")
else:
    headlines = []
    for i, url in enumerate(base_url_df["url"]):
        print(f"Scraping {i + 1}/{len(base_url_df)}: {url}")
        response = requests.get(url)
        if response.status_code != 200:
            headlines.append({"headline": None, "source": None})
            continue
        soup = BeautifulSoup(response.text, "html.parser")
        if "foxnews.com" in url:
            title = soup.find("h1", class_="headline speakable")
            source = "FoxNews"
        else:
            title = soup.find("h1")
            source = "NBC"
        headlines.append(
            {"headline": title.get_text() if title else None, "source": source}
        )
    news_df_base = pd.DataFrame(headlines)
    news_df_base.to_csv("data/base_scraped_headlines.csv", index=False)


# Data splitting

# (80% train, 20% test)
news_df_base = news_df_base.dropna(subset=["headline", "source"])

RANDOM_STATE = 42

X_train_base, X_test_base, y_train_base, y_test_base = train_test_split(
    news_df_base["headline"],
    news_df_base["source"],
    test_size=0.2,
    random_state=RANDOM_STATE,
)


# Basic cleaning/pre-processing
y_train_base = y_train_base.apply(lambda x: 1 if x == "FoxNews" else 0)
y_test_base = y_test_base.apply(lambda x: 1 if x == "FoxNews" else 0)

vectorizer = TfidfVectorizer(stop_words="english", max_features=100)
X_train_tfidf = vectorizer.fit_transform(X_train_base)
X_test_tfidf = vectorizer.transform(X_test_base)


# Train base model
model = LogisticRegression(max_iter=100, random_state=RANDOM_STATE)
model.fit(X_train_tfidf, y_train_base)

y_pred = model.predict(X_test_tfidf)


# Evaluate base model
accuracy = accuracy_score(y_test_base, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:\n", classification_report(y_test_base, y_pred))

# Accuracy: 0.6442
# Classification Report:
#                precision    recall  f1-score   support

#            0       0.60      0.80      0.69      2894
#            1       0.72      0.50      0.59      3022

#     accuracy                           0.64      5916
#    macro avg       0.66      0.65      0.64      5916
# weighted avg       0.66      0.64      0.64      5916


# =============================================================================
# 2. FEATURE ENGINEERING
# =============================================================================
# SUMMARY: Loads + filters data via prepare_data() from preprocess.py (drops Spanish
# headlines, enforces 25–140 char and ≥4-word bounds, removes scrape artifacts like
# "- Page N" and "- FOX News Radio", dedupes), then stratified-splits 80/20.
# Builds three feature branches merged via FeatureUnion (horizontal concat):
#   word_branch:  TF-IDF on words/bigrams (cleaned text) — captures vocabulary differences
#   char_branch:  TF-IDF on 3–5 char n-grams (cleaned text) — captures morphological/stylistic patterns
#   style_branch: 12 handcrafted numeric features on raw text (imported from preprocess.py)
#                 — captures formatting conventions (caps, punctuation, "U.S." vs "US",
#                 title case, end-period, hyphenation, etc.)
#   combined_branches = word + char (5000 features)
#   word_char_style_branch = word + char + style (5012 features)

# Load + clean via prepare_data() (Spanish filter, length bounds, page-artifact removal, dedupe)
X, y = prepare_data("data/expanded_headlines.csv", verbose=True)

# Stratified split to preserve class balance after filtering
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)


# All-in one transformer for data cleaning
def apply_clean_hed(x):
    if isinstance(x, list):
        x = pd.Series(x)
    return x.apply(clean_hed)


clean_transform = FunctionTransformer(apply_clean_hed)

# word-level branch -- will capture tokens
word_branch = TfidfVectorizer(
    analyzer="word",
    token_pattern=r"\[?[\w]{2,}\]?",
    ngram_range=(1, 2),
    max_features=2500,  # limit features to speed up our testing
)

# will see 3, 4, 5 character combos for non-tokens
char_branch = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    max_features=2500,  # limit features to speed up our testing
)

# merge for hybrid pipeline later on
combined_branches = FeatureUnion(
    [
        ("words", word_branch),
        ("chars", char_branch),
    ]
)

# merge text and style branches -- used in Hybrid_v2_style
# (style_branch imported from preprocess.py — 12 handcrafted features)
word_char_style_branch = FeatureUnion(
    [
        ("text", combined_branches),
        ("style", style_branch),
    ]
)


# =============================================================================
# 3. PIPELINE DEVELOPMENT & EVALUATION
# =============================================================================
# SUMMARY: Loops through cleaning/feature engineering pipeline variants, evaluating
# each with a fixed baseline classifier (Logistic Regression) via cross-validation.
# Goal: isolate the best feature representation before varying the classifier.
# Section 3b then runs a grid search on the winning pipeline to tune its hyperparameters.

# --- Helper Functions --------------------------------------------------------


# Run k-fold CV on a single pipeline and return a one-row DataFrame with
# macro-F1 and accuracy (mean + std). Macro-F1 is our primary metric since
# the classes are close to balanced but we still want both classes weighted
# equally; accuracy is reported alongside for continuity with the baseline.
def quick_eval(name, pipeline, X, y, cv=5):
    f1 = cross_val_score(pipeline, X, y, cv=cv, scoring="f1_macro")
    acc = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
    return pd.DataFrame(
        [
            {
                "pipeline": name,
                "f1_mean": round(f1.mean(), 4),
                "f1_std": round(f1.std(), 4),
                "acc_mean": round(acc.mean(), 4),
                "acc_std": round(acc.std(), 4),
            }
        ]
    )


# Run quick_eval over a {name: pipeline} dict and return a single DataFrame
# sorted by macro-F1 descending — used to compare candidate pipelines side-by-side.
def eval_results(pipeline_dictionary, X, y, cv=5):
    results = [
        quick_eval(name, pipe, X, y, cv) for name, pipe in pipeline_dictionary.items()
    ]
    return pd.concat(results, ignore_index=True).sort_values("f1_mean", ascending=False)


# Disk-cached GridSearchCV: reuse the pickled fitted searcher if it exists,
# otherwise run the search and persist it.
def cached_grid_search(cache_path, grid_search, X, y):
    if os.path.exists(cache_path):
        return joblib.load(cache_path)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    grid_search.fit(X, y)
    joblib.dump(grid_search, cache_path)
    return grid_search


# Disk-cached version of eval_results: save the comparison DataFrame to CSV
# so re-running the notebook reads the prior sweep results instead of re-running CV.
def cached_eval(cache_path, pipeline_dictionary, X, y, cv=5):
    if os.path.exists(cache_path):
        return pd.read_csv(cache_path)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    results = eval_results(pipeline_dictionary, X, y, cv)
    results.to_csv(cache_path, index=False)
    return results


# Side-by-side bar chart of macro-F1 vs accuracy (with std error bars) across
# pipelines, sorted by F1 — visual companion to the eval_results DataFrame.
def plot_f1_acc_comparison(results):
    df = pd.DataFrame(results).sort_values("f1_mean", ascending=False)
    x = np.arange(len(df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(
        x - width / 2,
        df["f1_mean"],
        width,
        yerr=df["f1_std"],
        label="Macro F1",
        capsize=4,
    )
    ax.bar(
        x + width / 2,
        df["acc_mean"],
        width,
        yerr=df["acc_std"],
        label="Accuracy",
        capsize=4,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(df["pipeline"], rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Pipeline Comparison")
    ax.legend()
    plt.tight_layout()
    plt.show()


# Fit each pipeline on the training split and overlay ROC curves on the held-out
# test set, falling back to decision_function for classifiers without predict_proba
# (e.g. LinearSVC).
def plot_roc_curves(pipelines_dict, X_train, y_train, X_test, y_test):
    fig, ax = plt.subplots(figsize=(10, 7))

    for name, pipe in pipelines_dict.items():
        fitted = clone(pipe).fit(X_train, y_train)

        if hasattr(fitted, "predict_proba"):
            scores = fitted.predict_proba(X_test)[:, 1]
        else:
            scores = fitted.decision_function(X_test)

        fpr, tpr, _ = roc_curve(y_test, scores)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


# --- Pipeline Definitions ----------------------------------------------------

# Different permutations of pipelines to iterate thru
pipelines = {
    # Baseline: replicates Section 1 (TF-IDF 100 features + LR) on the
    # filtered dataset so we have an apples-to-apples comparison
    "Baseline": Pipeline(
        [
            ("tfidf", TfidfVectorizer(stop_words="english", max_features=100)),
            ("clf", LogisticRegression(max_iter=100, random_state=RANDOM_STATE)),
        ]
    ),
    # Cleaned: same Baseline capacity but routes text through clean_hed first
    # and uses a custom token pattern (allows [bracketed], 2+ char tokens),
    # tests whether normalization alone moves the needle at low capacity
    "Cleaned": Pipeline(
        [
            ("cleaning", clean_transform),
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words="english",
                    max_features=100,
                    # optional [] wrappers for tokens
                    # keep hyphenated words together
                    # allow for quotes, colons, hyphens
                    # 2 or more chars
                    token_pattern=r"\[?[\w]{2,}\]?",
                ),
            ),
            ("clf", LogisticRegression(max_iter=100, random_state=RANDOM_STATE)),
        ]
    ),
    # Optimized_v2: scales vocab to 5k, adds bigrams and sublinear TF,
    # the standard "good defaults" TF-IDF + LR setup, no custom cleaning.
    "Optimized_v2": Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words="english",
                    max_features=5000,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                ),
            ),
            ("clf", LogisticRegression(max_iter=100, random_state=RANDOM_STATE)),
        ]
    ),
    # Optimized_v2_tokenized: Optimized_v2 + clean_hed + custom token pattern
    # + min_df=2. Isolates the effect of cleaning + tokenization
    # on top of the strong word-ngram baseline.
    "Optimized_v2_tokenized": Pipeline(
        [
            ("cleaning", clean_transform),
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words="english",
                    max_features=5000,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    min_df=2,
                    token_pattern=r"\[?[\w]{2,}\]?",
                ),
            ),
            ("clf", LogisticRegression(max_iter=100, random_state=RANDOM_STATE)),
        ]
    ),
    # V3_char_grams: drops words entirely and uses 3–5 char_wb n-grams,
    # tests whether sub-word morphology (suffixes, casing patterns, "U.S.")
    # alone is enough signal to outperform pure word features.
    "V3_char_grams": Pipeline(
        [
            ("cleaning", clean_transform),
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=5000,
                    sublinear_tf=True,
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                ),
            ),
            ("clf", LogisticRegression(max_iter=100, random_state=RANDOM_STATE)),
        ]
    ),
    # Hybrid: word + char branches concatenated,
    # both at 2500 features. First model that combines vocabulary and
    # sub-word signals, expected to dominate the single-branch variants.
    "Hybrid": Pipeline(
        [
            ("cleaning", clean_transform),
            ("branches", combined_branches),
            ("clf", LogisticRegression(max_iter=100, random_state=RANDOM_STATE)),
        ]
    ),
    # Hybrid_v1: same shape as Hybrid but with tuned hyperparameters,
    # 5k features per branch and wider char n-grams (4–6) to capture
    # longer morphological patterns.
    "Hybrid_v1": Pipeline(
        [
            ("cleaning", clean_transform),
            (
                "branches",
                FeatureUnion(
                    [
                        (
                            "words",
                            TfidfVectorizer(
                                analyzer="word",
                                token_pattern=r"\[?[\w]{2,}\]?",
                                ngram_range=(1, 2),
                                max_features=5000,
                                stop_words="english",
                            ),
                        ),
                        (
                            "chars",
                            TfidfVectorizer(
                                analyzer="char_wb",
                                ngram_range=(4, 6),
                                max_features=5000,
                            ),
                        ),
                    ]
                ),
            ),
            ("clf", LogisticRegression(max_iter=100, random_state=RANDOM_STATE)),
        ]
    ),
    # Hybrid_v2_style: word + char + 12 style features
    # Skips clean_transform because the style branch needs raw caps
    # and punctuation; MaxAbsScaler keeps style magnitudes from dominating
    # the sparse TF-IDF columns. This is the top performer and the basis
    # for the submission pipeline.
    "Hybrid_v2_style": Pipeline(
        [
            ("branches", word_char_style_branch),
            ("scaler", MaxAbsScaler()),
            ("clf", LogisticRegression(max_iter=100, random_state=RANDOM_STATE)),
        ]
    ),
}

# Compare pipelines
pipeline_results = cached_eval(
    "cache/new_filters_features/pipeline_results.csv", pipelines, X_train, y_train
)
print(pipeline_results.to_string(index=False))
#               pipeline  f1_mean  f1_std  acc_mean  acc_std
#        Hybrid_v2_style   0.7899  0.0061    0.7919   0.0060
#                 Hybrid   0.7766  0.0058    0.7791   0.0058
#              Hybrid_v1   0.7751  0.0051    0.7777   0.0051
#          V3_char_grams   0.7583  0.0032    0.7611   0.0030
# Optimized_v2_tokenized   0.7407  0.0048    0.7450   0.0046
#           Optimized_v2   0.7389  0.0062    0.7435   0.0060
#               Baseline   0.5805  0.0074    0.6101   0.0061
#                Cleaned   0.5792  0.0053    0.6100   0.0047

# Visualize comparisons
plot_f1_acc_comparison(pipeline_results)

# --- 4b. Top Features ---------------------------------------------------------

# Extract most predictive features from different pipelines
top_pipelines = ["Hybrid_v2_style", "Hybrid_v1", "Optimized_v2_tokenized", "Hybrid"]
tables = []

# For each top pipeline, fit on the training set and pull the LR coefficients
# alongside the corresponding feature names so we can inspect the most
# predictive tokens/n-grams/style features. The "branches" step (FeatureUnion)
# vs "tfidf" step (single vectorizer) lookup handles both pipeline shapes.
# Sign of the coefficient indicates the direction: positive = Fox (class 1),
# negative = NBC (class 0); we keep the top 10 by absolute weight for each.
for name in top_pipelines:
    p = pipelines[name]
    p.fit(X_train, y_train)

    # Hybrid pipelines expose features under "branches"; single-vectorizer
    # pipelines expose them under "tfidf".
    if "branches" in p.named_steps:
        transformer = p.named_steps["branches"]
    else:
        transformer = p.named_steps["tfidf"]
    feats = transformer.get_feature_names_out()

    # LR coef_ is shape (1, n_features) for binary classification — index [0]
    # gives the per-feature weights aligned with `feats`.
    coefs = p.named_steps["clf"].coef_[0]

    feat_df = (
        pd.DataFrame(
            {
                f"{name}_Feature": feats,
                f"{name}_Weight": coefs,
                "Absolute_wt": np.abs(coefs),
            }
        )
        .sort_values(by="Absolute_wt", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )

    tables.append(feat_df[[f"{name}_Feature", f"{name}_Weight"]])

horiz = pd.concat(tables, axis=1)  # side-by-side
print(horiz.to_string())
#   Hybrid_v2_style_Feature  Hybrid_v2_style_Weight Hybrid_v1_Feature  Hybrid_v1_Weight Optimized_v2_tokenized_Feature  Optimized_v2_tokenized_Weight Hybrid_Feature  Hybrid_Weight
# 0              style__len                6.007983       chars__ us          10.650751                    [endperiod]                      -5.663634      words__us       6.025619
# 1           style__n_caps                4.599154        chars__ a          -4.886196                            dem                       5.121357     chars__ a       -5.270436
# 2        style__n_allcaps                4.549666         words__dc          4.040245                            fox                       4.618976     words__fox       4.819008
# 3        style__n_periods               -4.431419       chars__ u.s         -3.906650                           dies                      -4.305280     words__dem       4.209657
# 4        text__words__fox                3.481718        words__dem          3.856442                           dems                       4.138604     chars__s:        4.160395
# 5         text__words__dc                3.235553       chars__u.s.         -3.776005                             dc                       4.119991      words__dc       4.096565
# 6   text__words__dateline               -2.923759      chars__ u.s.         -3.738751                           desk                      -4.002695     chars__ u.      -3.920106
# 7   text__words__kornacki               -2.900211       chars__.s.          -3.542829                            jan                      -3.731810     chars__n.       -3.522511
# 8           style__has_US                2.860763      chars__u.s.          -3.514705                        illegal                       3.633353    chars__ us        3.342435
# 9   text__words__iran war               -2.787516     chars__ u.s.          -3.476931                        outkick                       3.497859     chars__e:        3.206480

# --- 4c. Hyperparameter Tuning on Best Pipeline --------------------------------

# Grid search on Hybrid_v2_style
hybrid_v2_param_grid = {
    # how many words to keep
    "branches__text__words__max_features": [1000, 2500, 5000],
    # number of words to group
    "branches__text__words__ngram_range": [(1, 1), (1, 2)],
    # character fragment lengths
    "branches__text__chars__ngram_range": [(3, 5), (4, 6)],
}

hybrid_v2_grid_search = GridSearchCV(
    pipelines["Hybrid_v2_style"],
    hybrid_v2_param_grid,
    cv=5,
    scoring="f1_macro",
    n_jobs=-1,
)
hybrid_v2_grid_search = cached_grid_search(
    "cache/new_filters_features/hybrid_v2_grid_search.pkl",
    hybrid_v2_grid_search,
    X_train,
    y_train,
)
print(f"Hybrid_v2_style - Best F1: {hybrid_v2_grid_search.best_score_:.4f}")
print(f"Hybrid_v2_style - Best Parameters: {hybrid_v2_grid_search.best_params_}")
# Hybrid_v2_style - Best F1: 0.7939
# Hybrid_v2_style - Best Parameters: {'branches__text__chars__ngram_range': (4, 6), 'branches__text__words__max_features': 5000, 'branches__text__words__ngram_range': (1, 2)}

# =============================================================================
# 4. CLASSIFIER SWEEP ON BEST PIPELINE
# =============================================================================
#
# SECTION SUMMARY:
#   Swap out the classifier on the best-performing feature pipeline (Hybrid_v2_style
#   with tuned params from 3b) across a range of models to find the best classifier.
#

# Rebuild Hybrid_v2_style with optimal params from 3b grid search
BEST_PIPELINE = "Hybrid_v2_style"
pipelines[BEST_PIPELINE] = clone(pipelines[BEST_PIPELINE]).set_params(
    **hybrid_v2_grid_search.best_params_
)
print(
    "Active params on best pipeline:",
    {k: pipelines[BEST_PIPELINE].get_params()[k] for k in hybrid_v2_param_grid},
)

# List of models to start out with
models = {
    "logistic_regression": LogisticRegression(random_state=RANDOM_STATE),
    "linear_svc": LinearSVC(random_state=RANDOM_STATE),
    "sgd": SGDClassifier(random_state=RANDOM_STATE),
    "knn": KNeighborsClassifier(),
    "decision_tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "random_forest": RandomForestClassifier(random_state=RANDOM_STATE),
    "adaboost": AdaBoostClassifier(random_state=RANDOM_STATE),
    "multinomial_nb": MultinomialNB(),
    "complement_nb": ComplementNB(),
    # "xgboost": XGBClassifier(random_state=RANDOM_STATE),
}

# Iterate thru models, train and predict
classifier_pipelines = {
    name: clone(pipelines[BEST_PIPELINE]).set_params(clf=clf)
    for name, clf in models.items()
}

# Compare results
classifier_results = cached_eval(
    "cache/new_filters_features/classifier_results.csv",
    classifier_pipelines,
    X_train,
    y_train,
)
print(classifier_results.to_string(index=False))

#            pipeline  f1_mean  f1_std  acc_mean  acc_std
# logistic_regression   0.7939  0.0045    0.7959   0.0045
#                 sgd   0.7875  0.0053    0.7898   0.0052
#          linear_svc   0.7707  0.0061    0.7725   0.0061
#       random_forest   0.7677  0.0057    0.7716   0.0055
#      multinomial_nb   0.7327  0.0090    0.7335   0.0091
#       complement_nb   0.7322  0.0093    0.7328   0.0093
#            adaboost   0.6838  0.0037    0.6911   0.0039
#       decision_tree   0.6798  0.0096    0.6823   0.0094
#                 knn   0.6523  0.0074    0.6594   0.0066

# Visualize comparisons
plot_f1_acc_comparison(classifier_results)
plot_roc_curves(classifier_pipelines, X_train, y_train, X_test, y_test)


# =============================================================================
# 5. HYPERPARAMETER TUNING — TOP 4 CLASSIFIERS
# =============================================================================
#
# SECTION SUMMARY:
#   Grid-search the classifier hyperparameters for the four best models from
#   Section 4 (LR, LinearSVC, RF, SGD — XGBoost excluded, not in backend env),
#   all on the Hybrid_v2_style
#   feature pipeline with optimal feature params fixed from Section 3b.
#   Best estimators are stored as best_lr / best_lsvc / best_rf / best_sgd
#   for use in Section 6 ensembles.

# Base pipeline to clone for each classifier (features already tuned in 3b)
base_pipeline = clone(pipelines[BEST_PIPELINE])
print(
    "Base pipeline params:",
    {k: v for k, v in base_pipeline.get_params().items() if not k.startswith("clf")},
)

# ── 5a. Logistic Regression ──────────────────────────────────────────────────
# Tuning: C (regularization strength) and solver (lbfgs vs saga).
# max_iter fixed at 1000.

lr_tune_pipeline = clone(base_pipeline).set_params(
    clf=LogisticRegression(random_state=RANDOM_STATE)
)

lr_param_grid = {
    "clf__C": [0.01, 0.1, 1, 10, 100],
    "clf__solver": ["lbfgs", "saga"],
    "clf__max_iter": [1000],
}

lr_grid_search = GridSearchCV(
    lr_tune_pipeline, lr_param_grid, cv=5, scoring="f1_macro", n_jobs=-1, verbose=2
)
lr_grid_search = cached_grid_search(
    "cache/new_filters_features/lr_grid_search.pkl", lr_grid_search, X_train, y_train
)
print(f"Best LR F1:     {lr_grid_search.best_score_:.4f}")
print(f"Best LR Params: {lr_grid_search.best_params_}")

# ── 5b. Linear SVC ───────────────────────────────────────────────────────────
# Tuning: C (regularization strength) and loss (hinge vs squared_hinge).
# max_iter fixed at 2000.

lsvc_tune_pipeline = clone(base_pipeline).set_params(
    clf=LinearSVC(random_state=RANDOM_STATE)
)

lsvc_param_grid = {
    "clf__C": [0.01, 0.1, 1, 10, 100],
    "clf__loss": ["hinge", "squared_hinge"],
    "clf__max_iter": [2000],
}

lsvc_grid_search = GridSearchCV(
    lsvc_tune_pipeline, lsvc_param_grid, cv=5, scoring="f1_macro", n_jobs=-1, verbose=2
)
lsvc_grid_search = cached_grid_search(
    "cache/new_filters_features/lsvc_grid_search.pkl",
    lsvc_grid_search,
    X_train,
    y_train,
)
print(f"Best LinearSVC F1:     {lsvc_grid_search.best_score_:.4f}")
print(f"Best LinearSVC Params: {lsvc_grid_search.best_params_}")

# ── 5c. Random Forest ────────────────────────────────────────────────────────
# Tuning: n_estimators, max_depth, min_samples_leaf, and max_features
# (sqrt vs log2).

rf_tune_pipeline = clone(base_pipeline).set_params(
    clf=RandomForestClassifier(random_state=RANDOM_STATE)
)

rf_param_grid = {
    "clf__n_estimators": [100, 200, 300],
    "clf__max_depth": [None, 10, 20],
    "clf__min_samples_leaf": [1, 2, 4],
    "clf__max_features": ["sqrt", "log2"],
}

rf_grid_search = GridSearchCV(
    rf_tune_pipeline, rf_param_grid, cv=5, scoring="f1_macro", n_jobs=-1, verbose=2
)
rf_grid_search = cached_grid_search(
    "cache/new_filters_features/rf_grid_search.pkl", rf_grid_search, X_train, y_train
)
print(f"Best RF F1:     {rf_grid_search.best_score_:.4f}")
print(f"Best RF Params: {rf_grid_search.best_params_}")

# ── 5d. XGBoost ──────────────────────────────────────────────────────────────
# Skipped: leaderboard env doesn't ship xgboost, so the model can't be submitted.
#
# xgb_tune_pipeline = clone(base_pipeline).set_params(
#     clf=XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE)
# )
#
# xgb_param_grid = {
#     "clf__n_estimators": [100, 200, 300],
#     "clf__learning_rate": [0.05, 0.1, 0.2],
#     "clf__max_depth": [3, 5, 7],
#     "clf__subsample": [0.8, 1.0],
#     "clf__colsample_bytree": [0.8, 1.0],
# }
#
# xgb_grid_search = GridSearchCV(
#     xgb_tune_pipeline, xgb_param_grid, cv=5, scoring="f1_macro", n_jobs=-1, verbose=2
# )
# xgb_grid_search = cached_grid_search(
#     "cache/new_filters_features/xgb_grid_search.pkl", xgb_grid_search, X_train, y_train
# )
# print(f"Best XGB F1:     {xgb_grid_search.best_score_:.4f}")
# print(f"Best XGB Params: {xgb_grid_search.best_params_}")

# ── 5e. SGD ──────────────────────────────────────────────────────────────────
# Tuning: loss (hinge / modified_huber / log_loss), alpha (regularization
# strength), penalty (l2 / l1 / elasticnet), l1_ratio (only used when
# penalty=elasticnet), and learning_rate schedule (optimal vs adaptive).

sgd_tune_pipeline = clone(base_pipeline).set_params(
    clf=SGDClassifier(random_state=RANDOM_STATE)
)

sgd_param_grid = {
    "clf__loss": ["hinge", "modified_huber", "log_loss"],
    "clf__alpha": [1e-5, 1e-4, 1e-3, 1e-2],
    "clf__penalty": ["l2", "l1", "elasticnet"],
    "clf__l1_ratio": [0.15, 0.5],
    "clf__learning_rate": ["optimal", "adaptive"],
}

sgd_grid_search = GridSearchCV(
    sgd_tune_pipeline, sgd_param_grid, cv=5, scoring="f1_macro", n_jobs=-1, verbose=2
)
sgd_grid_search = cached_grid_search(
    "cache/new_filters_features/sgd_grid_search.pkl", sgd_grid_search, X_train, y_train
)
print(f"Best SGD F1:     {sgd_grid_search.best_score_:.4f}")
print(f"Best SGD Params: {sgd_grid_search.best_params_}")

# ── 5f. Tuning results summary ───────────────────────────────────────────────

tuning_results = pd.DataFrame(
    [
        {
            "model": "logistic_regression",
            "best_f1": lr_grid_search.best_score_,
            "best_params": lr_grid_search.best_params_,
        },
        {
            "model": "linear_svc",
            "best_f1": lsvc_grid_search.best_score_,
            "best_params": lsvc_grid_search.best_params_,
        },
        {
            "model": "random_forest",
            "best_f1": rf_grid_search.best_score_,
            "best_params": rf_grid_search.best_params_,
        },
        # {
        #     "model": "xgboost",
        #     "best_f1": xgb_grid_search.best_score_,
        #     "best_params": xgb_grid_search.best_params_,
        # },
        {
            "model": "sgd",
            "best_f1": sgd_grid_search.best_score_,
            "best_params": sgd_grid_search.best_params_,
        },
    ]
)
# Pull each model's pre-tuning macro-F1 from the Section 4 classifier sweep
# and join it onto tuning_results so we can show before/after side-by-side.
baseline_f1 = classifier_results.set_index("pipeline")["f1_mean"]
tuning_results["before_f1"] = tuning_results["model"].map(baseline_f1)
# Delta = how much grid search improved over the default-hyperparameter run.
tuning_results["delta"] = tuning_results["best_f1"] - tuning_results["before_f1"]
# Reorder columns and sort by tuned score so the best model is on top.
tuning_results = tuning_results[
    ["model", "before_f1", "best_f1", "delta", "best_params"]
].sort_values("best_f1", ascending=False)
print(tuning_results.to_string(index=False))

#               model  before_f1  best_f1     delta                                                                                                                                 best_params
#                 sgd     0.7875 0.797150  0.009650 {'clf__alpha': 0.001, 'clf__l1_ratio': 0.15, 'clf__learning_rate': 'adaptive', 'clf__loss': 'modified_huber', 'clf__penalty': 'elasticnet'}
# logistic_regression     0.7939 0.793878 -0.000022                                                                                {'clf__C': 1, 'clf__max_iter': 1000, 'clf__solver': 'lbfgs'}
#          linear_svc     0.7707 0.792912  0.022212                                                                                {'clf__C': 0.1, 'clf__loss': 'hinge', 'clf__max_iter': 2000}
#       random_forest     0.7677 0.773705  0.006005                                 {'clf__max_depth': None, 'clf__max_features': 'sqrt', 'clf__min_samples_leaf': 1, 'clf__n_estimators': 300}

best_lr = lr_grid_search.best_estimator_.named_steps["clf"]
best_lsvc = lsvc_grid_search.best_estimator_.named_steps["clf"]
best_rf = rf_grid_search.best_estimator_.named_steps["clf"]
# best_xgb = xgb_grid_search.best_estimator_.named_steps["clf"]
best_sgd = sgd_grid_search.best_estimator_.named_steps["clf"]


# =============================================================================
# 6. ENSEMBLE METHODS
# =============================================================================
#
# SECTION SUMMARY:
#   Three ensemble strategies built on top of the 4 tuned classifiers from
#   Section 5 (LR, LinearSVC, SGD, RF — XGBoost excluded since it's not in the
#   leaderboard env), all using the Hybrid_v2_style feature pipeline with optimal
#   params fixed from Section 3b (max_features=5000, words (1,2), chars (4,6)).
#   All pipelines are cloned from pipelines[BEST_PIPELINE] to ensure correct
#   feature params — NOT built from word_char_style_branch directly.
#
#   Ensemble_Soft_3:  soft vote over the 3 proba models (LR + SGD + RF);
#                     LinearSVC excluded since it lacks predict_proba.
#   Ensemble_Soft_4:  soft vote over all 4, with LinearSVC wrapped in
#                     CalibratedClassifierCV to give it calibrated probs.
#   Ensemble_Stack:   stacking with all 4 base learners feeding an LR
#                     meta-learner (stack_method="auto" → predict_proba for
#                     LR/SGD/RF, decision_function for LinearSVC).

# ── 6a. Soft voting — 3 proba models (LR + SGD + RF; XGB removed) ───────────
# LinearSVC excluded here since it lacks predict_proba

pipelines["Ensemble_Soft_3"] = clone(pipelines[BEST_PIPELINE]).set_params(
    clf=VotingClassifier(
        estimators=[
            ("lr", best_lr),
            ("sgd", best_sgd),
            # ("xgb", best_xgb),  # leaderboard env lacks xgboost
            ("rf", best_rf),
        ],
        voting="soft",
        n_jobs=-1,
    )
)

# ── 6b. Soft voting — all 4 (CalibratedClassifierCV wraps LinearSVC; XGB removed) ─

cal_lsvc = CalibratedClassifierCV(best_lsvc, cv=5)
pipelines["Ensemble_Soft_4"] = clone(pipelines[BEST_PIPELINE]).set_params(
    clf=VotingClassifier(
        estimators=[
            ("lr", best_lr),
            ("sgd", best_sgd),
            # ("xgb", best_xgb),  # leaderboard env lacks xgboost
            ("rf", best_rf),
            ("lsvc", cal_lsvc),
        ],
        voting="soft",
        n_jobs=-1,
    )
)

# ── 6c. Stacking — 4 base models --> LR meta-learner (XGB removed) ───────────
# stack_method="auto": predict_proba for LR/SGD/RF, decision_function for LinearSVC

pipelines["Ensemble_Stack"] = clone(pipelines[BEST_PIPELINE]).set_params(
    clf=StackingClassifier(
        estimators=[
            ("lr", best_lr),
            ("lsvc", best_lsvc),
            ("sgd", best_sgd),
            # ("xgb", best_xgb),  # leaderboard env lacks xgboost
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
)

# ── 6d. Stacking — submission ensemble (no XGBoost — not in backend env) ───────
# Now identical to Ensemble_Stack since XGB has been removed everywhere; commented out.
#
# pipelines["Ensemble_Stack_NoXGB"] = clone(pipelines[BEST_PIPELINE]).set_params(
#     clf=StackingClassifier(
#         estimators=[
#             ("lr", best_lr),
#             ("lsvc", best_lsvc),
#             ("sgd", best_sgd),
#             ("rf", best_rf),
#         ],
#         final_estimator=LogisticRegression(
#             C=1, max_iter=1000, random_state=RANDOM_STATE
#         ),
#         cv=5,
#         stack_method="auto",
#         n_jobs=-1,
#         passthrough=False,
#     )
# )

# ── 6e. Evaluate and cache all ensembles ─────────────────────────────────────

ensemble_results = cached_eval(
    "cache/new_filters_features/ensemble_results.csv",
    {k: v for k, v in pipelines.items() if k.startswith("Ensemble_")},
    X_train,
    y_train,
)
print(ensemble_results.to_string(index=False))
#        pipeline  f1_mean  f1_std  acc_mean  acc_std
#  Ensemble_Stack   0.8030  0.0063    0.8051   0.0062
# Ensemble_Soft_3   0.8024  0.0053    0.8046   0.0051
# Ensemble_Soft_4   0.8013  0.0063    0.8036   0.0062

# ── 6f. Visual comparison: tuned models vs. ensembles ────────────────────────

tuned_pipelines = {
    "logistic_regression": lr_grid_search.best_estimator_,
    "linear_svc": lsvc_grid_search.best_estimator_,
    "sgd": sgd_grid_search.best_estimator_,
    # "xgboost": xgb_grid_search.best_estimator_,
    "random_forest": rf_grid_search.best_estimator_,
}

tuned_eval = cached_eval(
    "cache/new_filters_features/tuned_eval.csv", tuned_pipelines, X_train, y_train
)

comparison_results = pd.concat([tuned_eval, ensemble_results], ignore_index=True)
plot_f1_acc_comparison(comparison_results)

comparison_pipelines = {
    **tuned_pipelines,
    **{k: v for k, v in pipelines.items() if k.startswith("Ensemble_")},
}
plot_roc_curves(comparison_pipelines, X_train, y_train, X_test, y_test)


# =============================================================================
# 7. FINAL EVALUATION ON HELD-OUT TEST SET
# =============================================================================

FINAL_PIPELINE = "Ensemble_Stack"

final_pipeline = clone(pipelines[FINAL_PIPELINE])
final_pipeline.fit(X_train, y_train)

y_pred_final = final_pipeline.predict(X_test)

final_accuracy = accuracy_score(y_test, y_pred_final)
print(f"Final pipeline: {FINAL_PIPELINE}")
print(f"Test accuracy: {final_accuracy:.4f}")
print(
    "Classification Report:\n",
    classification_report(y_test, y_pred_final, target_names=["NBC", "FoxNews"]),
)

# Final pipeline: Ensemble_Stack
# Test accuracy: 0.8119
# Classification Report:
#                precision    recall  f1-score   support

#          NBC       0.82      0.84      0.83      2911
#      FoxNews       0.81      0.78      0.79      2468

#     accuracy                           0.81      5379
#    macro avg       0.81      0.81      0.81      5379
# weighted avg       0.81      0.81      0.81      5379
