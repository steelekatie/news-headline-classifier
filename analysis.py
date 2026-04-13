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
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.base import clone
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# =============================================================================
# 1. BASELINE MODEL
# =============================================================================

# Data collection
base_url_df = pd.read_csv("data/url_only_data.csv")

if os.path.exists("data/base_scraped_headlines.csv"):
    news_df = pd.read_csv("data/base_scraped_headlines.csv")
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
    news_df = pd.DataFrame(headlines)
    news_df.to_csv("data/base_scraped_headlines.csv", index=False)


# Data splitting

# (80% train, 20% test)
news_df = news_df.dropna(subset=["headline", "source"])

X_train, X_test, y_train, y_test = train_test_split(
    news_df["headline"], news_df["source"], test_size=0.2, random_state=42
)


# Basic cleaning/pre-processing
y_train = y_train.apply(lambda x: 1 if x == "FoxNews" else 0)
y_test = y_test.apply(lambda x: 1 if x == "FoxNews" else 0)

vectorizer = TfidfVectorizer(stop_words="english", max_features=100)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# Train base model
model = LogisticRegression(max_iter=100)
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)


# Evaluate base model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred))


# =============================================================================
# 2. CLEANING FUNCTIONS
# =============================================================================


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
pattern_summary(news_df, text_col="headline", group_col="source", patterns=patterns)

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
# 3. DATA PREP & FEATURE-ENGINEERING BUILDING BLOCKS
# =============================================================================

# Recreate news_df to avoid potential issues caused in initial EDA
news_df2 = pd.read_csv("data/base_scraped_headlines.csv")
news_df2 = news_df2.dropna(subset=["headline", "source"])

news_df2["label"] = news_df2["source"].apply(lambda x: 1 if x == "FoxNews" else 0)

# Splitting
X_train2, X_test2, y_train2, y_test2 = train_test_split(
    news_df2["headline"],
    news_df2["label"],
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


# =============================================================================
# 4. PIPELINE DEVELOPMENT & EVALUATION
# =============================================================================

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
}


# Helper function to log results as we iterate on new pipelines/models
# Using F1 score as primary metric, accuracy on the side
def eval_results(pipeline_dictionary, X, y, cv=5):
    results = []
    for name, pipe in pipeline_dictionary.items():
        f1 = cross_val_score(pipe, X, y, cv=cv, scoring="f1_macro")
        acc = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
        results.append(
            {
                "pipeline": name,
                "f1_mean": round(f1.mean(), 4),
                "f1_std": round(f1.std(), 4),
                "acc_mean": round(acc.mean(), 4),
                "acc_std": round(acc.std(), 4),
            }
        )
    return pd.DataFrame(results).sort_values("f1_mean", ascending=False)


def plot_model_comparison(results, title="Model Comparison"):
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
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_roc_curves(models_dict, X_train, y_train, X_test, y_test, tfidf=None):
    if tfidf is None:
        tfidf = TfidfVectorizer(
            stop_words="english",
            max_features=5000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
        )

    fig, ax = plt.subplots(figsize=(10, 7))

    for name, clf in models_dict.items():
        pipe = Pipeline([("tfidf", clone(tfidf)), ("clf", clone(clf))])
        pipe.fit(X_train, y_train)

        if hasattr(pipe, "predict_proba"):
            scores = pipe.predict_proba(X_test)[:, 1]
        else:
            scores = pipe.decision_function(X_test)

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


# Compare pipelines
results_df = eval_results(pipelines, X_train2, y_train2)
print(results_df.to_string(index=False))

# Visualize comparisons
plot_model_comparison(results_df, title="Pipeline Comparison")


# =============================================================================
# 5. MULTI-CLASSIFIER COMPARISON
# =============================================================================

# Updated based on results from last section
BEST_PIPELINE = "Hybrid"

# Pull the cleaning and feature extraction steps directly from the best pipeline,
# so each of the 10 models below runs through the exact same preprocessing.
best_cleaning = pipelines[BEST_PIPELINE].named_steps["cleaning"]
best_features = pipelines[BEST_PIPELINE].named_steps["branches"]

# List of models to start out with
models = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "linear_svc": LinearSVC(max_iter=1000),
    "sgd": SGDClassifier(),
    "knn": KNeighborsClassifier(),
    "decision_tree": DecisionTreeClassifier(),
    "random_forest": RandomForestClassifier(),
    "adaboost": AdaBoostClassifier(),
    "multinomial_nb": MultinomialNB(),
    "complement_nb": ComplementNB(),
    "xgboost": XGBClassifier(eval_metric="logloss"),
}

# Iterate thru models, train and predict
classifier_pipelines = {
    name: Pipeline(
        [("cleaning", best_cleaning), ("branches", best_features), ("clf", clf)]
    )
    for name, clf in models.items()
}

# Compare results
results_df2 = eval_results(classifier_pipelines, X_train2, y_train2)
print(results_df2.to_string(index=False))

# Visualize comparisons
# Will take longer for roc curves because we need to fit/predict across
# classification thresholds for each model
plot_model_comparison(results_df2, title="Classifier Comparison")
plot_roc_curves(models, X_train2, y_train2, X_test2, y_test2)


# =============================================================================
# 6. TOP FEATURES
# =============================================================================

# Extract most predictive features from different pipelines
top_models = ["Hybrid", "V3_char_grams", "Optimized_v2_tokenized"]
tables = []

for name in top_models:
    p = pipelines[name]  # pull this particular pipeline
    p.fit(X_train2, y_train2)  # fit to data

    if "branches" in p.named_steps:  # if branches exist (hybrid), diff command
        feats = p.named_steps["branches"].get_feature_names_out()
    else:
        feats = p.named_steps[
            "tfidf"
        ].get_feature_names_out()  # otherwise should be same for all

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
