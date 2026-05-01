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
# SUMMARY: Scrapes headlines from URLs (cached to CSV after first run), splits 80/20,
# and trains a Logistic Regression on TF-IDF (100 features, English stopwords).

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
news_df = pd.read_csv("data/expanded_headlines.csv")
news_df = news_df.dropna(subset=["headline", "source"]).drop_duplicates()

news_df["label"] = news_df["source"].apply(lambda x: 1 if x == "FoxNews" else 0)

# Splitting
X_train, X_test, y_train, y_test = train_test_split(
    news_df["headline"],
    news_df["label"],
    test_size=0.2,
    random_state=RANDOM_STATE,
)


# All-in one transformer for data cleaning
def apply_clean_hed(x):
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


def style_feature_names(self, input_features):
    return ["len", "n_caps", "colon", "period", "U.S.", "US"]


style_branch = FunctionTransformer(
    extract_style,
    feature_names_out=style_feature_names,
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


def cached_grid_search(cache_path, grid_search, X, y):
    if os.path.exists(cache_path):
        return joblib.load(cache_path)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    grid_search.fit(X, y)
    joblib.dump(grid_search, cache_path)
    return grid_search


def cached_eval(cache_path, pipeline_dictionary, X, y, cv=5):
    if os.path.exists(cache_path):
        return pd.read_csv(cache_path)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    results = eval_results(pipeline_dictionary, X, y, cv)
    results.to_csv(cache_path, index=False)
    return results


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
            ("clf", LogisticRegression(max_iter=100, random_state=RANDOM_STATE)),
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
            ("clf", LogisticRegression(max_iter=100, random_state=RANDOM_STATE)),
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
            ("clf", LogisticRegression(max_iter=100, random_state=RANDOM_STATE)),
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
            ("clf", LogisticRegression(max_iter=100, random_state=RANDOM_STATE)),
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
            ("clf", LogisticRegression(max_iter=100, random_state=RANDOM_STATE)),
        ]
    ),
    "Hybrid": Pipeline(
        [
            ("cleaning", clean_transform),
            ("branches", combined_branches),
            ("clf", LogisticRegression(max_iter=100, random_state=RANDOM_STATE)),
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
            ("clf", LogisticRegression(max_iter=100, random_state=RANDOM_STATE)),
        ]
    ),
    # Hybrid with word, char, AND style features -- no clean_transform
    # (style branch needs raw caps and punctuation)
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
    "cache/expanded_data/pipeline_results.csv", pipelines, X_train, y_train
)
print(pipeline_results.to_string(index=False))
#               pipeline  f1_mean  f1_std  acc_mean  acc_std
#        Hybrid_v2_style   0.8052  0.0064    0.8052   0.0064
#                 Hybrid   0.7983  0.0025    0.7983   0.0026
#              Hybrid_v1   0.7931  0.0018    0.7931   0.0018
#          V3_char_grams   0.7764  0.0051    0.7764   0.0051
# Optimized_v2_tokenized   0.7672  0.0037    0.7673   0.0036
#           Optimized_v2   0.7650  0.0045    0.7652   0.0045
#                Cleaned   0.6387  0.0056    0.6467   0.0048
#               Baseline   0.6380  0.0067    0.6464   0.0055

# Visualize comparisons
plot_f1_acc_comparison(pipeline_results)

# --- 4b. Top Features ---------------------------------------------------------

# Extract most predictive features from different pipelines
top_pipelines = ["Hybrid_v2_style", "Hybrid_v1", "Optimized_v2_tokenized", "Hybrid"]
tables = []

for name in top_pipelines:
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
#  Hybrid_v2_style_Feature  Hybrid_v2_style_Weight Hybrid_v1_Feature  Hybrid_v1_Weight Optimized_v2_tokenized_Feature  Optimized_v2_tokenized_Weight Hybrid_Feature  Hybrid_Weight
# 0              style__len                8.072314       chars__ us          10.849496                           page                       7.336800      words__us       6.171184
# 1           style__n_caps                6.314248        words__fox          4.291501                    [endperiod]                      -6.242857     chars__ a       -4.384548
# 2         text__words__dc                3.721698         words__dc          4.282381                            fox                       6.075642    words__page       4.379179
# 3   text__words__dateline               -3.525635        chars__ a          -3.976460                            dem                       5.066266      words__dc       4.371533
# 4        text__chars__ -                 3.002103       chars__ u.s         -3.915901                             la                       4.634403     chars__n.       -4.065691
# 5        text__chars__ |                 2.932598        words__dem          3.868698                       fox news                       4.383548     words__dem       4.045900
# 6   text__words__iran war               -2.916701       words__page          3.791145                             dc                       4.305521     words__fox       4.030966
# 7    text__words__rundown               -2.864868       chars__u.s.         -3.775597                           dies                      -4.121655     chars__ u.      -3.835230
# 8         text__words__uk                2.792536      chars__ u.s.         -3.713847                             en                       4.044690     chars__s:        3.673565
# 9               style__US                2.783728       words__herd          3.623850                            jan                      -4.028994     chars__e:        3.643648

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
    "cache/expanded_data/hybrid_v2_grid_search.pkl",
    hybrid_v2_grid_search,
    X_train,
    y_train,
)
print(f"Hybrid_v2_style - Best F1: {hybrid_v2_grid_search.best_score_:.4f}")
print(f"Hybrid_v2_style - Best Parameters: {hybrid_v2_grid_search.best_params_}")
# Hybrid_v2_style - Best F1: 0.8127
# Hybrid_v2_style - Best Parameters: {'branches__text__chars__ngram_range': (4, 6), 'branches__text__words__max_features': 5000, 'branches__text__words__ngram_range': (1, 2)}


# =============================================================================
# 5. CLASSIFIER SWEEP ON BEST PIPELINE
# =============================================================================
#
# SECTION SUMMARY:
#   Swap out the classifier on the best-performing feature pipeline (Hybrid_v2_style
#   with tuned params from 4b) across a range of models to find the best classifier.
#

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
    "cache/expanded_data/classifier_results.csv", classifier_pipelines, X_train, y_train
)
print(classifier_results.to_string(index=False))

#            pipeline  f1_mean  f1_std  acc_mean  acc_std
# logistic_regression   0.8127  0.0041    0.8127   0.0041
#                 sgd   0.8059  0.0050    0.8060   0.0050
#       random_forest   0.7912  0.0019    0.7915   0.0018
#          linear_svc   0.7893  0.0048    0.7893   0.0048
#      multinomial_nb   0.7515  0.0052    0.7516   0.0051
#       complement_nb   0.7514  0.0054    0.7515   0.0054
#       decision_tree   0.6960  0.0057    0.6961   0.0057
#            adaboost   0.6868  0.0107    0.6919   0.0089
#                 knn   0.5641  0.0097    0.5989   0.0052

# Visualize comparisons
plot_f1_acc_comparison(classifier_results)
plot_roc_curves(classifier_pipelines, X_train, y_train, X_test, y_test)


# =============================================================================
# 6. HYPERPARAMETER TUNING — TOP 5 CLASSIFIERS
# =============================================================================
#
# SECTION SUMMARY:
#   Grid-search the classifier hyperparameters for the five best models from
#   Section 5 (LR, LinearSVC, RF, XGBoost, SGD), all on the Hybrid_v2_style
#   feature pipeline with optimal feature params fixed from Section 4b.
#   Best estimators are stored as best_lr / best_lsvc / best_rf / best_xgb /
#   best_sgd for use in Section 7 ensembles.

# Base pipeline to clone for each classifier (features already tuned in 4b)
base_pipeline = clone(pipelines[BEST_PIPELINE])
print(
    "Base pipeline params:",
    {k: v for k, v in base_pipeline.get_params().items() if not k.startswith("clf")},
)

# ── 6a. Logistic Regression ──────────────────────────────────────────────────

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
    "cache/expanded_data/lr_grid_search.pkl", lr_grid_search, X_train, y_train
)
print(f"Best LR F1:     {lr_grid_search.best_score_:.4f}")
print(f"Best LR Params: {lr_grid_search.best_params_}")

# ── 6b. Linear SVC ───────────────────────────────────────────────────────────

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
    "cache/expanded_data/lsvc_grid_search.pkl", lsvc_grid_search, X_train, y_train
)
print(f"Best LinearSVC F1:     {lsvc_grid_search.best_score_:.4f}")
print(f"Best LinearSVC Params: {lsvc_grid_search.best_params_}")

# ── 6c. Random Forest ────────────────────────────────────────────────────────

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
    "cache/expanded_data/rf_grid_search.pkl", rf_grid_search, X_train, y_train
)
print(f"Best RF F1:     {rf_grid_search.best_score_:.4f}")
print(f"Best RF Params: {rf_grid_search.best_params_}")

# ── 6d. XGBoost ──────────────────────────────────────────────────────────────
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
#     "cache/expanded_data/xgb_grid_search.pkl", xgb_grid_search, X_train, y_train
# )
# print(f"Best XGB F1:     {xgb_grid_search.best_score_:.4f}")
# print(f"Best XGB Params: {xgb_grid_search.best_params_}")

# ── 6e. SGD ──────────────────────────────────────────────────────────────────

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
    "cache/expanded_data/sgd_grid_search.pkl", sgd_grid_search, X_train, y_train
)
print(f"Best SGD F1:     {sgd_grid_search.best_score_:.4f}")
print(f"Best SGD Params: {sgd_grid_search.best_params_}")

# ── 6f. Tuning results summary ───────────────────────────────────────────────

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
baseline_f1 = classifier_results.set_index("pipeline")["f1_mean"]
tuning_results["before_f1"] = tuning_results["model"].map(baseline_f1)
tuning_results["delta"] = tuning_results["best_f1"] - tuning_results["before_f1"]
tuning_results = tuning_results[
    ["model", "before_f1", "best_f1", "delta", "best_params"]
].sort_values("best_f1", ascending=False)
print(tuning_results.to_string(index=False))

# model                 before_f1  best_f1    delta                                                                                                                                best_params
#                 sgd     0.8059 0.814979 0.009079 {'clf__alpha': 0.001, 'clf__l1_ratio': 0.15, 'clf__learning_rate': 'optimal', 'clf__loss': 'modified_huber', 'clf__penalty': 'elasticnet'}
# logistic_regression     0.8127 0.812713 0.000013                                                                                {'clf__C': 1, 'clf__max_iter': 1000, 'clf__solver': 'saga'}
#          linear_svc     0.7893 0.811567 0.022267                                                                       {'clf__C': 0.1, 'clf__loss': 'squared_hinge', 'clf__max_iter': 2000}
#       random_forest     0.7912 0.797668 0.006468                                {'clf__max_depth': None, 'clf__max_features': 'sqrt', 'clf__min_samples_leaf': 1, 'clf__n_estimators': 300}

best_lr = lr_grid_search.best_estimator_.named_steps["clf"]
best_lsvc = lsvc_grid_search.best_estimator_.named_steps["clf"]
best_rf = rf_grid_search.best_estimator_.named_steps["clf"]
# best_xgb = xgb_grid_search.best_estimator_.named_steps["clf"]
best_sgd = sgd_grid_search.best_estimator_.named_steps["clf"]


# =============================================================================
# 7. ENSEMBLE METHODS
# =============================================================================
#
# SECTION SUMMARY:
#   Three ensemble strategies built on top of the 5 tuned classifiers from
#   Section 6, all using the Hybrid_v2_style feature pipeline with optimal
#   params fixed from Section 4c (max_features=5000, words (1,2), chars (4,6)).
#   All pipelines are cloned from pipelines[BEST_PIPELINE] to ensure correct
#   feature params — NOT built from word_char_style_branch directly.
#
# FINDINGS:
#   Stacking edges out soft voting (F1=0.8032 vs 0.8028), but the margin is
#   negligible. Adding calibrated LinearSVC (Soft_5) slightly hurts vs. the
#   4-model soft vote — calibration noise outweighs its contribution.

# ── 7a. Soft voting — 3 proba models (LR + SGD + RF; XGB removed) ───────────
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

# ── 7b. Soft voting — all 4 (CalibratedClassifierCV wraps LinearSVC; XGB removed) ─

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

# ── 7c. Stacking — 4 base models --> LR meta-learner (XGB removed) ───────────
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

# ── 7d. Stacking — submission ensemble (no XGBoost — not in backend env) ───────
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

# ── 7e. Evaluate and cache all ensembles ─────────────────────────────────────

ensemble_results = cached_eval(
    "cache/expanded_data/ensemble_results.csv",
    {k: v for k, v in pipelines.items() if k.startswith("Ensemble_")},
    X_train,
    y_train,
)
print(ensemble_results.to_string(index=False))
#        pipeline  f1_mean  f1_std  acc_mean  acc_std
#  Ensemble_Stack   0.8222  0.0037    0.8222   0.0037
# Ensemble_Soft_3   0.8218  0.0050    0.8219   0.0050
# Ensemble_Soft_4   0.8201  0.0052    0.8201   0.0052

# ── 7f. Visual comparison: tuned models vs. ensembles ────────────────────────

tuned_pipelines = {
    "logistic_regression": lr_grid_search.best_estimator_,
    "linear_svc": lsvc_grid_search.best_estimator_,
    "sgd": sgd_grid_search.best_estimator_,
    # "xgboost": xgb_grid_search.best_estimator_,
    "random_forest": rf_grid_search.best_estimator_,
}

tuned_eval = cached_eval(
    "cache/expanded_data/tuned_eval.csv", tuned_pipelines, X_train, y_train
)

comparison_results = pd.concat([tuned_eval, ensemble_results], ignore_index=True)
plot_f1_acc_comparison(comparison_results)

comparison_pipelines = {
    **tuned_pipelines,
    **{k: v for k, v in pipelines.items() if k.startswith("Ensemble_")},
}
plot_roc_curves(comparison_pipelines, X_train, y_train, X_test, y_test)


# =============================================================================
# 8. FINAL EVALUATION ON HELD-OUT TEST SET
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
# Test accuracy: 0.8259
# Classification Report:
#                precision    recall  f1-score   support

#          NBC       0.81      0.84      0.83      2907
#      FoxNews       0.84      0.82      0.83      3009

#     accuracy                           0.83      5916
#    macro avg       0.83      0.83      0.83      5916
# weighted avg       0.83      0.83      0.83      5916
