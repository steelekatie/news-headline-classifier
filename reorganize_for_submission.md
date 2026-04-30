# Reorganizing for Submission

This doc explains how `preprocess.py`, `model.py`, and `model.joblib` were derived from `analysis.py`, and why each choice was made. It exists so that we (and our partner / future-us) can trace any line in the submission files back to the exploratory work in `analysis.py`.

`analysis.py` is the dev notebook — EDA, pipeline comparisons, classifier sweep, hyperparameter tuning, ensembles. It is **not** submitted. Only `preprocess.py`, `model.py`, and `model.joblib` go to the course backend (per `instructions/submission_instructions.md` §3.2).

---

## `preprocess.py`

### What this file is

One of the two files the course backend imports to grade us (`instructions/submission_instructions.md` §3.2). The contract:

```python
prepare_data(csv_path: str) -> (X, y)
```

The backend hands us a CSV "similar to `url_data_only.csv`" — as confirmed by `data/url_with_headlines.csv`, that means a CSV with `url` and `headline` columns. We return:

- `X`: list of raw headline strings (model expects raw text)
- `y`: list of integer labels (Fox = 1, NBC = 0)

### How this file is built

**1. Style feature extractor (`extract_style` + `style_feature_names` + `style_branch`)**

These are copied verbatim from `analysis.py` Section 3 (Feature Engineering), lines 289–311. They live in `preprocess.py` — not `model.py` — because the fitted pipeline serialized in `model.joblib` contains a `FunctionTransformer` that pickles a _reference_ to this function by qualified name. At grading time, `joblib.load` needs to import `preprocess.extract_style` to reconstitute the transformer. So these symbols **must** exist at module scope in `preprocess.py`.

What the features capture (Section 2 EDA findings, `analysis.py:154-168`):

- `len`, `n_caps` — Fox uses more all-caps formatting
- `has_colons` — Fox style ("Trump: I will...")
- `has_period` — NBC style (sentences as headlines)
- `has_U_S` vs `has_US` — NBC writes "U.S.", Fox writes "US" — strong signal

These are computed on **raw (uncleaned) text** since cleaning would strip caps and punctuation and destroy the signal.

**2. `prepare_data(csv_path)`**

Steps:

1. Read the CSV. Backend guarantees `url` and `headline` columns.
2. Drop rows with missing headlines (paywalled / dead URLs).
3. Infer label from URL substring: `"foxnews.com"` → 1, else 0. This is the same source-detection logic `analysis.py` uses when scraping (`analysis.py:55-60`) and the same label mapping (`analysis.py:84-85`).
4. Return `X` as a list of raw strings — the pipeline's `TfidfVectorizer`s and the `style_branch` all run on the raw text directly. We do **not** apply `clean_hed` here because the chosen pipeline (`Hybrid_v2_style`) skips the `clean_transform` step (`analysis.py:551-557`).

### Why no scraping in this file

`analysis.py` Section 1 scrapes URLs to build `data/base_scraped_headlines.csv`, but the backend supplies pre-scraped headlines (the `headline` column is already populated). So we don't need `requests` or `beautifulsoup4` at grading time — which is good because neither library is in the backend env (see `submission_instructions.md` §2).

---

## `model.py`

### What this file is

The second of the two files the backend imports (`instructions/submission_instructions.md` §3.2). The contract:

```python
Model() / get_model() -> instance with .predict(batch) and .eval()
```

`predict(batch)` returns a list of integer labels (0 = NBC, 1 = Fox). The backend calls `model.predict(batch)` directly with batches of 32.

### Why the pipeline looks this way (decision trail)

Every choice traces back to a numbered section in `analysis.py`.

**(1) Feature pipeline = `Hybrid_v2_style`** — `analysis.py` Section 4, lines 551–557

Cross-validated comparison of 8 feature pipelines (Section 4 results table, lines 566–574) showed `Hybrid_v2_style` won at F1 = 0.7901 / acc = 0.7920. It combines:

- word TF-IDF (1–2 grams)
- `char_wb` TF-IDF (n-gram range tuned below)
- handcrafted style features (caps count, "U.S." vs "US", etc.)

A 4c grid search (`analysis.py:630-651`) confirmed the best feature params: `max_features=5000`, words `ngram=(1,2)`, chars `ngram=(4,6)` → F1 = 0.7935.

**(2) Base learners = LR, LinearSVC, SGD, RF (NOT XGB)** — `analysis.py` Section 6, lines 737–897

Each was tuned via its own `GridSearchCV` on the `Hybrid_v2_style` feature pipeline. The hyperparameters hard-coded in `_build_pipeline()` are the `best_params_` output of each search (`analysis.py:886-891`):

| Model     | Tuned hyperparameters                                                                           |
| --------- | ----------------------------------------------------------------------------------------------- |
| LR        | `C=1, solver=lbfgs, max_iter=1000`                                                              |
| LinearSVC | `C=0.01, loss=squared_hinge, max_iter=2000`                                                     |
| SGD       | `alpha=0.01, l1_ratio=0.15, learning_rate=adaptive, loss=modified_huber, penalty=elasticnet` \* |
| RF        | `n_estimators=300, max_depth=None, max_features=sqrt, min_samples_leaf=2`                       |

\* `eta0=0.01` added because newer sklearn versions require `eta0>0` when `learning_rate='adaptive'`; the default of 0 errors out.

XGBoost was the original 5th base learner in `analysis.py`'s `Ensemble_Stack` (best F1 = 0.8032 in Section 7) but is **intentionally dropped** here because the backend env (`submission_instructions.md` §2) ships only `numpy / pandas / torch / torchvision / scikit-learn / opencv-python` — no `xgboost`. Held-out test accuracy with XGB removed: ~0.80 (vs ~0.80 with XGB), so the cost of dropping it is negligible.

**(3) Meta-learner = Stacking with LR final_estimator** — `analysis.py` Section 7c, lines 949–969

`StackingClassifier` with `cv=5` and `stack_method="auto"` (uses `predict_proba` where available, `decision_function` for LinearSVC). Section 7 found Stack edged out Soft-vote (0.8032 vs 0.8028); the margin is small but Stack is what we ship. `final_estimator=LogisticRegression(C=1, max_iter=1000)` matches `analysis.py` exactly.

### How this file works at grading time

1. Backend imports `model.py` via `importlib` (`eval_project_b.py:_dynamic_import`). The `sys.path` shim at the top makes `from preprocess import style_branch` resolve, regardless of the grader's CWD.
2. Backend instantiates `Model()` — no args. `__init__` loads `model.joblib` (the fitted pipeline from running `python model.py`) via `joblib.load`. We resolve the path relative to `model.py`'s own directory so it works no matter where the grader invokes us from.
3. Backend calls `model.eval()` (no-op for sklearn — present for compatibility with the `eval_project_b.py:_load_checkpoint` contract) and then `model.predict(batch)` per batch of 32 inputs.
4. `predict()` wraps the batch in `pd.Series` before passing to the pipeline. This is required because `extract_style` uses Series methods (`.astype`, `.str.count`, etc.). At training time the pipeline received a Series from `train_test_split`; the eval script gives us a list, so we re-wrap.

### Why no `model.pt`

`submission_instructions.md` says `model.pt` is "optional … required if your model needs them to evaluate". The backend's `_load_checkpoint` expects a torch state dict; a pickled sklearn pipeline isn't one. So we ship `model.joblib` alongside `model.py` + `preprocess.py` and load it ourselves inside `Model.__init__`. The backend never tries to `load_state_dict` because no `--weights` / `model.pt` is provided.

### Retrain

```bash
python model.py
```

Re-fits the chosen pipeline on `data/base_scraped_headlines.csv` and overwrites `model.joblib`. The `__main__` block uses the same train/test split (80/20, `random_state=42`) that `analysis.py` uses, so accuracy is comparable.

---

## End-to-end submission package

What gets uploaded to Hugging Face:

- `preprocess.py`
- `model.py`
- `model.joblib`

Local sanity check before submission:

```bash
python instructions/templates/eval_project_b.py \
    --model model.py \
    --preprocess preprocess.py \
    --csv data/url_with_headlines.csv
```
