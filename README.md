# CIS 4190/5190 — News Source Classifier

Final project for UPenn's Applied Machine Learning course (Track B), by
**Katie Steele** and **Isaac Herman**. A binary text classifier that labels
U.S. news headlines as **Fox News** (`1`) or **NBC News** (`0`).

The shipped model is a stacking ensemble (Logistic Regression + LinearSVC +
SGD + Random Forest, with an LR meta-learner) over a hybrid feature pipeline:
TF-IDF word n-grams + TF-IDF character n-grams + 12 handcrafted style
features. Held-out test accuracy is 81.2%.

## Repo layout

```
.
├── preprocess.py         # prepare_data(csv) -> (X, y); shared cleaning + style features
├── model.py              # Model / get_model() + retraining entry point
├── model.pt              # fitted sklearn pipeline (torch.save wrapper)
├── data/
│   ├── url_only_data.csv          # original URL list (course-supplied)
│   ├── url_with_headlines.csv     # initial scraped headlines
│   ├── expanded_headlines.csv     # full training set (used by model.py)
│   └── collect_rss_headlines.py   # Google News RSS scraper used to expand the dataset
├── analysis/
│   ├── analysis.py       # dev notebook: feature engineering, pipeline + classifier sweeps, tuning, ensembles
│   └── eda.py            # exploratory data analysis (informs the filters in preprocess.py)
├── cache/                # cached CV results / grid-search snapshots
└── instructions/         # course-supplied submission contract + local eval harness
```

## Submission contract

The grading backend imports `preprocess.py` and `model.py`:

- `preprocess.prepare_data(csv_path) -> (X, y)` reads a CSV with `url` and
  `headline` columns, infers labels from the URL substring, and applies the
  same EDA-driven filtering used at training time.
- `model.Model` / `model.get_model()` load the fitted pipeline from
  `model.pt` and expose `predict(headlines)`.

`model.pt` is a `torch.save({"pipeline_bytes": ...})` container wrapping the
joblib-pickled sklearn pipeline — the backend only accepts `.pt`.

## Usage

Retrain on the expanded dataset (overwrites `model.pt`):

```bash
python model.py
```
