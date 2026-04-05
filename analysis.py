# =============================================================================
# 0. IMPORTS
# =============================================================================

import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

# =============================================================================
# 1. BASELINE MODEL
# =============================================================================

# --- Data Collection ---
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


# --- Data Splitting ---

# (80% train, 20% test)
news_df = news_df.dropna(subset=["headline", "source"])

X_train, X_test, y_train, y_test = train_test_split(
    news_df["headline"], news_df["source"], test_size=0.2, random_state=42
)


# --- Data Cleaning & Preprocessing ---
y_train = y_train.apply(lambda x: 1 if x == "FoxNews" else 0)
y_test = y_test.apply(lambda x: 1 if x == "FoxNews" else 0)

vectorizer = TfidfVectorizer(stop_words="english", max_features=100)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# --- Model Training ---
model = LogisticRegression(max_iter=100)
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)


# --- Model Evaluation ---
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred))


# =============================================================================
# 2. IMPROVED PIPELINE & MODELS
# =============================================================================

# --- Data Prep ---
news_df2 = pd.read_csv("data/base_scraped_headlines.csv")
news_df2 = news_df2.dropna(subset=["headline", "source"])

news_df2["label"] = news_df2["source"].apply(lambda x: 1 if x == "FoxNews" else 0)

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    news_df2["headline"],
    news_df2["label"],
    test_size=0.2,
    random_state=42,
)


# --- Pipeline ---
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


# --- Training & Evaluation ---
pipeline.fit(X_train2, y_train2)
y_pred2 = pipeline.predict(X_test2)

print(f"Accuracy: {accuracy_score(y_test2, y_pred2):.4f}")
print("Classification Report:\n", classification_report(y_test2, y_pred2))
