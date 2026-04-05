# =============================================================================
# 0. IMPORTS
# =============================================================================

import os
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
from sklearn.pipeline import Pipeline
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
# 2. IMPROVED PIPELINE & FEATURE ENGINEERING
# =============================================================================

# Data prep, just replicate what we have above for now
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

# Build simple pipeline logic, we can iterate on this
pipeline = Pipeline(
    [
        (
            "tfidf",
            TfidfVectorizer(
                stop_words="english",
                max_features=100,
            ),
        ),
        ("clf", LogisticRegression(max_iter=100)),
    ]
)


# Test training/predicting using the pipline
pipeline.fit(X_train2, y_train2)
y_pred2 = pipeline.predict(X_test2)

print(f"Accuracy: {accuracy_score(y_test2, y_pred2):.4f}")
print("Classification Report:\n", classification_report(y_test2, y_pred2))

# Helper function to log results as we iterate on new pipelines/models
# Using F1 score as primary metric, accuracy on the side
results = []


def log_result(name, pipe, X, y, cv=5):
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


# Test logging function on baseline model
log_result("baseline", pipeline, X_train2, y_train2)

# New pipeline using improved TF-IDF params
pipeline_tfidf_v2 = Pipeline(
    [
        (
            "tfidf",
            TfidfVectorizer(
                stop_words="english",
                max_features=5000,
                ngram_range=(1, 2),
                sublinear_tf=True,
                min_df=2,
            ),
        ),
        ("clf", LogisticRegression(max_iter=100)),
    ]
)

log_result("tfidf_optimized", pipeline_tfidf_v2, X_train2, y_train2)


# Compare two pipelines so far
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))


# =============================================================================
# 3. MODEL COMPARISON
# =============================================================================


def plot_model_comparison(results):
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
    ax.set_title("Model Comparison")
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


results = []

# New tfidf object with improved params
tfidf_v2 = TfidfVectorizer(
    stop_words="english",
    max_features=5000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=2,
)

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
for name, clf in models.items():
    pipe = Pipeline([("tfidf", tfidf_v2), ("clf", clf)])
    log_result(name, pipe, X_train2, y_train2)

# Compare results
results_df = pd.DataFrame(results)
print(results_df.sort_values("f1_mean", ascending=False).to_string(index=False))

# Visualize comparisons
# Will take longer for roc curves because we need to fit/predict across
# classification thresholds for each model
plot_model_comparison(results)
plot_roc_curves(models, X_train2, y_train2, X_test2, y_test2)
