# =============================================================================
# 0. IMPORTS
# =============================================================================

import os
import re
import unicodedata
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
)
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.base import clone
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, MaxAbsScaler
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# =============================================================================
# 1. BASELINE MODEL
# =============================================================================
# SUMMARY: Scrapes headlines from URLs (cached to CSV after first run), splits 80/20,
# and trains a Logistic Regression on TF-IDF (100 features, English stopwords).
# FINDINGS: Baseline accuracy ~66.49% — sets the floor to beat in later sections.

# Data collection
base_url_df = pd.read_csv("data/url_only_data.csv")

if os.path.exists("data/base_scraped_headlines.csv"):
    news_df_base = pd.read_csv("data/base_scraped_headlines.csv")
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

X_train_base, X_test_base, y_train_base, y_test_base = train_test_split(
    news_df_base["headline"], news_df_base["source"], test_size=0.2, random_state=42
)


# Basic cleaning/pre-processing
y_train_base = y_train_base.apply(lambda x: 1 if x == "FoxNews" else 0)
y_test_base = y_test_base.apply(lambda x: 1 if x == "FoxNews" else 0)

vectorizer = TfidfVectorizer(stop_words="english", max_features=100)
X_train_tfidf = vectorizer.fit_transform(X_train_base)
X_test_tfidf = vectorizer.transform(X_test_base)


# Train base model
model = LogisticRegression(max_iter=100)
model.fit(X_train_tfidf, y_train_base)

y_pred = model.predict(X_test_tfidf)


# Evaluate base model
accuracy = accuracy_score(y_test_base, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:\n", classification_report(y_test_base, y_pred))


# =============================================================================
# 2. EDA & CLEANING
# =============================================================================
# SUMMARY: Analyzes punctuation/formatting patterns across Fox vs. NBC headlines,
# then builds a cleaning pipeline (encoding normalization, punctuation standardization,
# special tokens for money and end-periods, lowercasing).
# FINDINGS: Key style differences — Fox uses "US" and colons; NBC uses "U.S.", periods,
# em dashes. These inform both the style_branch features and cleaning decisions.


# Checking for possible identifiers between sources
def pattern_summary(df, text_col, group_col, patterns):
    df = df.copy()
    # indicate if pattern exists
    for name, pattern in patterns.items():
        df[name] = df[text_col].astype(str).str.contains(pattern, na=False)
    feature_cols = list(patterns.keys())
    # count Trues
    counts = df.groupby(group_col)[feature_cols].sum().T.astype(int)

    # sort by diff in occurrences between nbc/fox
    if counts.shape[1] == 2:
        diff = (counts.iloc[:, 0] - counts.iloc[:, 1]).abs()
        counts = counts.loc[diff.sort_values(ascending=False).index]

    # add n for each source group as ref
    totals = df.groupby(group_col).size().to_frame().T
    totals.index = ["total_n"]

    # combine to one table
    return pd.concat([counts, totals])


patterns = {
    "US": "US",
    "U.S.": "U.S.",
    "period": r"\.",
    "emdash": r"\u2014",
    "colon": r":",
    "semicolon": r";",
    "single_quote": r"\'",
    "starts_num": r"^\d+",
}

# table
pattern_summary(
    news_df_base, text_col="headline", group_col="source", patterns=patterns
)

"""Noted:

NBC style guide
- U.S.
- May be more likely to:
	- Use periods
	- Use em dashes
	- Start heds w/ a number

Fox style guide
- US
- May be more likely to:
	- Use colons
	- Use quotes
"""


# Data cleaning, tokenizing for models
def repl_punct(text):
    """
    Returns text w/ standardized punctuation marks
    """
    punct_remap = {
        "\u2013": "-",  # en dash -> hyphen
        "\u2014": "-",  # em dash -> hyphen
        "\u2018": "'",  # left single quote -> single
        "\u2019": "'",  # right single quote -> single
        "\u201c": "'",  # left double quote -> single
        "\u201d": '"',  # right double quote -> single
    }
    # apply to text
    return text.translate(str.maketrans(punct_remap))


def repl_money(text):
    """
    Replaces references to monetary values with '[MONEY]',
    allowing models to treat monetary values as a single token.
    """
    money_ref = r"([$£€¥]\d+(?:\.\d+)?(?:[,\d+]+)?(?:\s?(?:hundred|thousand|million|billion|trillion|k|m|b|t))?)"
    return re.sub(money_ref, "[MONEY]", text, flags=re.IGNORECASE)


def clean_hed(text):
    """
    Cleans news headlines
    Standardizes encoding, punctuation, converts to lowercase.
    Creates tokens for references to money, periods at end of heds.
    Converts to lowercase and cleans up whitespace
    """
    if not isinstance(text, str):
        return ""

    # standardize encoding - adjust diacritics
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")

    # standardize dashes, quotes
    text = repl_punct(text)

    # replace refs to money with '[MONEY]' token
    text = repl_money(text)

    # replace periods at end of hed with '[END_PERIOD]' token
    # we only see NBC use this format in our initial sample
    text = re.sub(r"\.\s*$", " [ENDPERIOD]", text)

    # everything to lower
    text = str(text).lower()

    # keep letters, numbers, spaces, apostrophes, hyphens, periods, brackets, colons
    text = re.sub(r"[^a-z0-9\s'\-\.\[\]:]", " ", text)

    # fix whitespace
    text = " ".join(text.split())
    return text


# =============================================================================
# 3. FEATURE ENGINEERING
# =============================================================================
# SUMMARY: Builds three feature branches merged via FeatureUnion (horizontal concat):
#   word_branch:  TF-IDF on words/bigrams (cleaned text) — captures vocabulary differences
#   char_branch:  TF-IDF on 3–5 char n-grams (cleaned text) — captures morphological/stylistic patterns
#   style_branch: handcrafted numeric features (raw text) — captures formatting conventions (caps, punctuation, "U.S." vs "US")
#   combined_branches = word + char (5000 features)
#   word_char_style_branch = word + char + style (5006 features)

# Recreate news_df to avoid potential issues caused in initial EDA
news_df = pd.read_csv("data/base_scraped_headlines.csv")
news_df = news_df.dropna(subset=["headline", "source"]).drop_duplicates()

news_df["label"] = news_df["source"].apply(lambda x: 1 if x == "FoxNews" else 0)

# Splitting
X_train, X_test, y_train, y_test = train_test_split(
    news_df["headline"],
    news_df["label"],
    test_size=0.2,
    random_state=42,
)

# All-in one transformer for data cleaning
clean_transform = FunctionTransformer(lambda x: x.apply(clean_hed))

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


# style feature extractor -- operates on raw (uncleaned) text
# cleaning would strip the capitalization and punctuation these features depend on
def extract_style(series):
    s = series.astype(str)
    features = pd.DataFrame(
        {
            "len": s.str.len(),
            "n_caps": s.str.count(r"[A-Z]"),
            "has_colons": s.str.contains(":").astype(int),
            "has_period": s.str.contains(r"\.").astype(int),
            "has_U_S": s.str.contains(r"U\.S").astype(int),
            "has_US": s.str.contains(r"\bUS\b").astype(int),
        }
    )
    return features.values


style_branch = FunctionTransformer(
    extract_style,
    feature_names_out=lambda self, input_features: [
        "len",
        "n_caps",
        "colon",
        "period",
        "U.S.",
        "US",
    ],
)

# merge text and style branches -- used in Hybrid_v2_style
word_char_style_branch = FeatureUnion(
    [
        ("text", combined_branches),
        ("style", style_branch),
    ]
)


# =============================================================================
# 4. PIPELINE DEVELOPMENT & EVALUATION
# =============================================================================
# SUMMARY: Loops through cleaning/feature engineering pipeline variants, evaluating
# each with a fixed baseline classifier (Logistic Regression) via cross-validation.
# Goal: isolate the best feature representation before varying the classifier.
# Section 4b then runs a grid search on the winning pipeline to tune its hyperparameters.
# FINDINGS: Hybrid_v2_style (word + char TF-IDF + style features) performed best
# at 0.7901 F1 / 0.7920 accuracy, selected as BEST_PIPELINE for Section 5.
# Grid search confirmed optimal params: max_features=5000, words (1,2), chars (4,6) --> F1 = 0.7935.

# --- Helper Functions --------------------------------------------------------


# Helper function to log results as we iterate on new pipelines/models
# Using F1 score as primary metric, accuracy on the side
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


def eval_results(pipeline_dictionary, X, y, cv=5):
    results = [
        quick_eval(name, pipe, X, y, cv) for name, pipe in pipeline_dictionary.items()
    ]
    return pd.concat(results, ignore_index=True).sort_values("f1_mean", ascending=False)


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
    "Baseline": Pipeline(
        [
            ("tfidf", TfidfVectorizer(stop_words="english", max_features=100)),
            ("clf", LogisticRegression(max_iter=100)),
        ]
    ),
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
            ("clf", LogisticRegression(max_iter=100)),
        ]
    ),
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
            ("clf", LogisticRegression(max_iter=100)),
        ]
    ),
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
            ("clf", LogisticRegression(max_iter=100)),
        ]
    ),
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
            ("clf", LogisticRegression(max_iter=100)),
        ]
    ),
    "Hybrid": Pipeline(
        [
            ("cleaning", clean_transform),
            ("branches", combined_branches),
            ("clf", LogisticRegression(max_iter=100)),
        ]
    ),
    # Hybrid with tuned hyperparameters (larger vocab, wider char n-grams)
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
            ("clf", LogisticRegression(max_iter=100, random_state=42)),
        ]
    ),
    # Hybrid with word, char, AND style features -- no clean_transform
    # (style branch needs raw caps and punctuation)
    "Hybrid_v2_style": Pipeline(
        [
            ("branches", word_char_style_branch),
            ("scaler", MaxAbsScaler()),
            ("clf", LogisticRegression(max_iter=100, random_state=42, C=1)),
        ]
    ),
}

# Compare pipelines
results_df = eval_results(pipelines, X_train, y_train)
print(results_df.to_string(index=False))

#                pipeline  f1_mean  f1_std  acc_mean  acc_std
#         Hybrid_v2_style   0.7901  0.0203    0.7920   0.0200
#               Hybrid_v1   0.7884  0.0200    0.7905   0.0198
# Optimized_v2_tokenized   0.7837  0.0259    0.7875   0.0255
#                  Hybrid   0.7806  0.0238    0.7826   0.0242
#            Optimized_v2   0.7782  0.0222    0.7822   0.0222
#           V3_char_grams   0.7632  0.0212    0.7657   0.0210
#                 Cleaned   0.6894  0.0250    0.6917   0.0220
#                Baseline   0.6804  0.0194    0.6891   0.0171

# Visualize comparisons
plot_f1_acc_comparison(results_df)

# --- 4b. Hyperparameter Tuning on Best Pipeline --------------------------------

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
hybrid_v2_grid_search.fit(X_train, y_train)
print(f"Hybrid_v2_style - Best F1: {hybrid_v2_grid_search.best_score_:.4f}")
print(f"Hybrid_v2_style - Best Parameters: {hybrid_v2_grid_search.best_params_}")
# Result: F1 = 0.7935 | max_features=5000, words (1,2), chars (4,6)


# =============================================================================
# 5. CLASSIFIER SWEEP ON BEST PIPELINE
# =============================================================================
#
# SECTION SUMMARY:
#   Swap out the classifier on the best-performing feature pipeline (Hybrid_v2_style
#   with tuned params from 4b) across a range of models to find the best classifier.
#
# FINDINGS:
#   #________

# Rebuild Hybrid_v2_style with optimal params from 4b grid search
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
    "logistic_regression": LogisticRegression(),
    "linear_svc": LinearSVC(),
    "sgd": SGDClassifier(),
    "knn": KNeighborsClassifier(),
    "decision_tree": DecisionTreeClassifier(),
    "random_forest": RandomForestClassifier(),
    "adaboost": AdaBoostClassifier(),
    "multinomial_nb": MultinomialNB(),
    "complement_nb": ComplementNB(),
    "xgboost": XGBClassifier(),
}

# Iterate thru models, train and predict
classifier_pipelines = {
    name: clone(pipelines[BEST_PIPELINE]).set_params(clf=clf)
    for name, clf in models.items()
}

# Compare results
results_df2 = eval_results(classifier_pipelines, X_train, y_train)
print(results_df2.to_string(index=False))

# Visualize comparisons
# Will take longer for roc curves because we need to fit/predict across
# classification thresholds for each model
plot_f1_acc_comparison(results_df2)
plot_roc_curves(classifier_pipelines, X_train, y_train, X_test, y_test)


# =============================================================================
# 6. TOP FEATURES
# =============================================================================

# Extract most predictive features from different pipelines
top_models = ["Hybrid_v1", "Hybrid_v2_style", "Optimized_v2_tokenized", "V3_char_grams"]
tables = []

for name in top_models:
    p = pipelines[name]  # pull this particular pipeline
    p.fit(X_train, y_train)  # fit to data

    if "branches" in p.named_steps:
        transformer = p.named_steps["branches"]
    else:
        transformer = p.named_steps["tfidf"]
    feats = transformer.get_feature_names_out()

    coefs = p.named_steps["clf"].coef_[0]  # pull coefs for features

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


# =============================================================================
# 7. ENSEMBLE METHODS
# =============================================================================

# XGBoost pipeline on Hybrid_v2_style features
pipelines["Hybrid_XGB"] = Pipeline(
    [
        ("branches", word_char_style_branch),
        ("scaler", MaxAbsScaler()),
        (
            "clf",
            XGBClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=42,
            ),
        ),
    ]
)
xgb_eval = quick_eval("Hybrid_XGB", pipelines["Hybrid_XGB"], X_train, y_train)
print(xgb_eval)
# f1 = 0.7807, acc = 0.7833

xgb_param_grid = {
    "clf__n_estimators": [100, 200],
    "clf__learning_rate": [0.05, 0.1],
    "clf__max_depth": [3, 5],
}

xgb_grid_search = GridSearchCV(
    pipelines["Hybrid_XGB"],
    xgb_param_grid,
    cv=5,
    scoring="f1_macro",
    n_jobs=-1,
    verbose=1,
)
xgb_grid_search.fit(X_train, y_train)
print(f"Best XGB Score: {xgb_grid_search.best_score_:.4f}")
print(f"Best XGB Params: {xgb_grid_search.best_params_}")
# Result: F1 = 0.7920, lr=0.1, max_depth=5, n_estimators=200

# Extract best classifiers from grid searches above
best_lr = grid_search.best_estimator_.named_steps["clf"]
best_xgb = xgb_grid_search.best_estimator_.named_steps["clf"]

# Soft-voting ensemble: trust LR 2x more than XGB (it outperforms individually)
pipelines["Ensemble_LR_XGB"] = Pipeline(
    [
        ("branches", word_char_style_branch),
        ("scaler", MaxAbsScaler()),
        (
            "clf",
            VotingClassifier(
                estimators=[("lr", best_lr), ("xgb", best_xgb)],
                voting="soft",
                weights=[2, 1],
                n_jobs=-1,
            ),
        ),
    ]
)

ens_eval = quick_eval("Ensemble_LR_XGB", pipelines["Ensemble_LR_XGB"], X_train, y_train)
print(ens_eval)
# Result: F1 = 0.8004, acc = 0.8021

# *** note: this takes forever to run!
# =============================================================================
# 8. HYPERPARAMETER TUNING 2 - GRID SEARCHES ON RF, XGB COMBINATIONS
# =============================================================================

# Commented out because it underperforms XGB
# # Random Forest ------------------------------------------
# rf_param_grid = {
#     'clf__n_estimators': [100, 200],           # n trees
#     'clf__max_depth': [10, 20, None],
#     'clf__min_samples_leaf': [2, 5, 10],
#     'clf__max_features': ['sqrt', 'log2'],     # how many features to sample each split
#     'clf__bootstrap': [True]
# }

# rf_grid_search = GridSearchCV(
#     pipelines["Hybrid_RF"],
#     rf_param_grid,
#     cv = 5,
#     scoring = 'f1_macro',
#     n_jobs = -1,
#     verbose = 1
# )

# # run grid, print best results
# print("Random forest grid search --")
# rf_grid_search.fit(X_train, y_train)
# print(f"Best RF Score: {rf_grid_search.best_score_:.4f}")
# print(f"Best RF Params: {rf_grid_search.best_params_}")

# # RF RESULT: best F1: 0.7771
# # best params:  bootstrap: True / max_depth: none /
# #               max_features: sqrt / min_samples = 2 / n_est = 200


# XGBoost ----------------------------------------------------------
# smaller grid to try to cut down on time this takes to run
# xgb_param_grid = {
#     'clf__n_estimators': [100, 200],
#     'clf__learning_rate': [0.05, 0.1],
#     'clf__max_depth': [3, 5]
# }

# xgb_grid_search = GridSearchCV(
#     pipelines["Hybrid_XGB"],
#     xgb_param_grid,
#     cv=5,
#     scoring='f1_macro',
#     n_jobs=-1,
#     verbose=1
# )

# run grid, print best results
# print("XGBoost grid search --")
# xgb_grid_search.fit(X_train, y_train)

# print(f"Best XGB Score: {xgb_grid_search.best_score_:.4f}")
# print(f"Best XGB Params: {xgb_grid_search.best_params_}")
# XGB RESULT: best F1: 0.7920
# best: {'clf__learning_rate': 0.1,
#        'clf__max_depth': 5,
#        'clf__n_estimators': 200}
# ** ADJUSTED ACCORDINGLY IN PT 8 PIPELINE BUILD
