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
from xgboost import XGBClassifier
import joblib
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
news_df = pd.read_csv("data/base_scraped_headlines.csv")
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
    "cache/pipeline_results.csv", pipelines, X_train, y_train
)
print(pipeline_results.to_string(index=False))

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

# Hybrid_v2_style_Feature  Hybrid_v2_style_Weight Hybrid_v1_Feature  Hybrid_v1_Weight Optimized_v2_tokenized_Feature  Optimized_v2_tokenized_Weight Hybrid_Feature  Hybrid_Weight
# 0              style__len                3.350485        words__jan         -2.025015                            jan                      -3.424734      words__us       2.389302
# 1           style__period               -1.752561        chars__ a          -1.683278                           best                      -2.976003     words__jan      -1.869728
# 2               style__US                1.708950        words__cnn          1.502579                    [endperiod]                      -2.694165     chars__ a       -1.825240
# 3            style__colon                1.675516       chars__ us           1.454715                           gaza                      -2.432246     words__cnn       1.658027
# 4        text__chars__ a                -1.649641      chars__ and          -1.375990                          biden                       2.381995     chars__s'        1.593260
# 5       text__chars__ dem                1.440520       chars__ and         -1.370195                         israel                      -2.093440     words__and      -1.511618
# 6    text__words__and the                1.305231       chars__the          -1.332073                            cnn                       2.051903     chars__e'        1.405264
# 7        text__words__dem                1.218429      chars__ the          -1.291582                       election                      -1.940925     chars__n.       -1.394808
# 8        text__words__cnn                1.201619       words__gaza         -1.268110                       american                       1.714240     chars__ u.      -1.341437
# 9           style__n_caps                1.171434       chars__ the         -1.240809                      netanyahu                      -1.661681     chars__s:        1.322135

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
    "cache/hybrid_v2_grid_search.pkl", hybrid_v2_grid_search, X_train, y_train
)
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
#   Logistic Regression wins outright at F1=0.7935/acc=0.7954, matching the grid
#   search ceiling. Linear SVC and XGBoost follow. KNN and Decision Tree lag.

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
    "xgboost": XGBClassifier(random_state=RANDOM_STATE),
}

# Iterate thru models, train and predict
classifier_pipelines = {
    name: clone(pipelines[BEST_PIPELINE]).set_params(clf=clf)
    for name, clf in models.items()
}

# Compare results
classifier_results = cached_eval(
    "cache/classifier_results.csv", classifier_pipelines, X_train, y_train
)
print(classifier_results.to_string(index=False))

#            pipeline  f1_mean  f1_std  acc_mean  acc_std
# logistic_regression   0.7935  0.0165    0.7954   0.0170
#          linear_svc   0.7850  0.0092    0.7871   0.0097
#       random_forest   0.7847  0.0057    0.7871   0.0059
#             xgboost   0.7815  0.0172    0.7833   0.0176
#                 sgd   0.7765  0.0198    0.7777   0.0195
#       complement_nb   0.7703  0.0260    0.7706   0.0262
#      multinomial_nb   0.7702  0.0228    0.7706   0.0230
#            adaboost   0.7236  0.0060    0.7262   0.0051
#       decision_tree   0.6874  0.0290    0.6902   0.0288
#                 knn   0.6259  0.0497    0.6617   0.0287

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
    "cache/lr_grid_search.pkl", lr_grid_search, X_train, y_train
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
    "cache/lsvc_grid_search.pkl", lsvc_grid_search, X_train, y_train
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
    "cache/rf_grid_search.pkl", rf_grid_search, X_train, y_train
)
print(f"Best RF F1:     {rf_grid_search.best_score_:.4f}")
print(f"Best RF Params: {rf_grid_search.best_params_}")

# ── 6d. XGBoost ──────────────────────────────────────────────────────────────

xgb_tune_pipeline = clone(base_pipeline).set_params(
    clf=XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE)
)

xgb_param_grid = {
    "clf__n_estimators": [100, 200, 300],
    "clf__learning_rate": [0.05, 0.1, 0.2],
    "clf__max_depth": [3, 5, 7],
    "clf__subsample": [0.8, 1.0],
    "clf__colsample_bytree": [0.8, 1.0],
}

xgb_grid_search = GridSearchCV(
    xgb_tune_pipeline, xgb_param_grid, cv=5, scoring="f1_macro", n_jobs=-1, verbose=2
)
xgb_grid_search = cached_grid_search(
    "cache/xgb_grid_search.pkl", xgb_grid_search, X_train, y_train
)
print(f"Best XGB F1:     {xgb_grid_search.best_score_:.4f}")
print(f"Best XGB Params: {xgb_grid_search.best_params_}")

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
    "cache/sgd_grid_search.pkl", sgd_grid_search, X_train, y_train
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
        {
            "model": "xgboost",
            "best_f1": xgb_grid_search.best_score_,
            "best_params": xgb_grid_search.best_params_,
        },
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

#  model                before_f1  best_f1    delta                                                                                                                                best_params
# logistic_regression     0.7935 0.793532 0.000032                                                                               {'clf__C': 1, 'clf__max_iter': 1000, 'clf__solver': 'lbfgs'}
#                 sgd     0.7765 0.791890 0.015390 {'clf__alpha': 0.01, 'clf__l1_ratio': 0.15, 'clf__learning_rate': 'adaptive', 'clf__loss': 'modified_huber', 'clf__penalty': 'elasticnet'}
#          linear_svc     0.7850 0.790932 0.005932                                                                      {'clf__C': 0.01, 'clf__loss': 'squared_hinge', 'clf__max_iter': 2000}
#             xgboost     0.7815 0.790383 0.008883            {'clf__colsample_bytree': 0.8, 'clf__learning_rate': 0.1, 'clf__max_depth': 7, 'clf__n_estimators': 200, 'clf__subsample': 1.0}
#       random_forest     0.7847 0.788651 0.003951                                {'clf__max_depth': None, 'clf__max_features': 'sqrt', 'clf__min_samples_leaf': 2, 'clf__n_estimators': 300}

best_lr = lr_grid_search.best_estimator_.named_steps["clf"]
best_lsvc = lsvc_grid_search.best_estimator_.named_steps["clf"]
best_rf = rf_grid_search.best_estimator_.named_steps["clf"]
best_xgb = xgb_grid_search.best_estimator_.named_steps["clf"]
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

# ── 7a. Soft voting — 4 proba models (LR + SGD + XGB + RF) ──────────────────
# LinearSVC excluded here since it lacks predict_proba

pipelines["Ensemble_Soft_4"] = clone(pipelines[BEST_PIPELINE]).set_params(
    clf=VotingClassifier(
        estimators=[
            ("lr", best_lr),
            ("sgd", best_sgd),
            ("xgb", best_xgb),
            ("rf", best_rf),
        ],
        voting="soft",
        n_jobs=-1,
    )
)

# ── 7b. Soft voting — all 5 (CalibratedClassifierCV wraps LinearSVC) ─────────

cal_lsvc = CalibratedClassifierCV(best_lsvc, cv=5)
pipelines["Ensemble_Soft_5"] = clone(pipelines[BEST_PIPELINE]).set_params(
    clf=VotingClassifier(
        estimators=[
            ("lr", best_lr),
            ("sgd", best_sgd),
            ("xgb", best_xgb),
            ("rf", best_rf),
            ("lsvc", cal_lsvc),
        ],
        voting="soft",
        n_jobs=-1,
    )
)

# ── 7c. Stacking — all 5 base models → LR meta-learner ───────────────────────
# stack_method="auto": predict_proba for LR/SGD/RF/XGB, decision_function for LinearSVC

pipelines["Ensemble_Stack"] = clone(pipelines[BEST_PIPELINE]).set_params(
    clf=StackingClassifier(
        estimators=[
            ("lr", best_lr),
            ("lsvc", best_lsvc),
            ("sgd", best_sgd),
            ("xgb", best_xgb),
            ("rf", best_rf),
        ],
        final_estimator=LogisticRegression(C=1, max_iter=1000, random_state=RANDOM_STATE),
        cv=5,
        stack_method="auto",
        n_jobs=-1,
        passthrough=False,
    )
)

# ── 7d. Evaluate and cache all ensembles ─────────────────────────────────────

ensemble_results = cached_eval(
    "cache/ensemble_results.csv",
    {k: v for k, v in pipelines.items() if k.startswith("Ensemble_")},
    X_train,
    y_train,
)
print(ensemble_results.to_string(index=False))
