# Pipeline Planning

## Current Pipeline (simple-pipeline branch)

Two steps inside a single `sklearn.pipeline.Pipeline`, using the same params as the baseline:

```
Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=100)),
    ('clf',   LogisticRegression(max_iter=100)),
])
```

### Data prep (outside pipeline)

- Load `data/base_scraped_headlines.csv`
- `dropna` on `headline` and `source`
- Binary label encoding: FoxNews=1, NBC=0
- `train_test_split` (80/20, random_state=42)

---

## Future Work

### Data Cleaning

If raw headline noise (URLs, numbers, punctuation artifacts from scraping) hurts performance, add a custom `TextCleaner` step before TF-IDF:

```python
from sklearn.base import BaseEstimator, TransformerMixin
import re

class TextCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        #________
        #________
        #________
```

Possible cleaning operations to consider:

- Strip URLs (`https?://\S+`)
- Remove digits / standalone numbers
- Remove extra whitespace
- Remove punctuation (beyond what TfidfVectorizer's token_pattern already strips)

### Feature Engineering / TF-IDF

- Increase `max_features` (baseline used 100 — far too few for meaningful vocabulary coverage)
- Add bigrams via `ngram_range=(1, 2)` to capture phrases like "breaking news", "fake news"
- `sublinear_tf=True` — applies log(1 + tf) normalization, standard for text classification
- `min_df=2` — drops terms appearing in only one document (likely noise)
- Try character-level n-grams (`analyzer='char_wb'`) to catch stylistic differences

### Modeling

- Tune `LogisticRegression` hyperparameters (e.g. regularization strength `C`)
- Swap for `LinearSVC` or `SGDClassifier`
- Stratified `train_test_split` to ensure class balance
- Add `cross_val_score` for more reliable evaluation
- Hyperparameter search with `GridSearchCV` or `RandomizedSearchCV` wrapping the full pipeline
