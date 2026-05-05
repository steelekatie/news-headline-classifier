# Baseline Metrics — analysis.py on base dataset

Recorded for comparison against re-run on expanded dataset (`data/expanded_headlines.csv`).
Source: `data/base_scraped_headlines.csv`.

## Section 1. Baseline Model

TF-IDF (100 features, English stopwords) + Logistic Regression

- Accuracy: **0.6649**

## Section 4. Pipeline Development & Evaluation (5-fold CV on X_train)

| pipeline               | f1_mean | f1_std | acc_mean | acc_std |
| ---------------------- | ------- | ------ | -------- | ------- |
| Hybrid_v2_style        | 0.7901  | 0.0203 | 0.7920   | 0.0200  |
| Hybrid_v1              | 0.7884  | 0.0200 | 0.7905   | 0.0198  |
| Optimized_v2_tokenized | 0.7837  | 0.0259 | 0.7875   | 0.0255  |
| Hybrid                 | 0.7806  | 0.0238 | 0.7826   | 0.0242  |
| Optimized_v2           | 0.7782  | 0.0222 | 0.7822   | 0.0222  |
| V3_char_grams          | 0.7632  | 0.0212 | 0.7657   | 0.0210  |
| Cleaned                | 0.6894  | 0.0250 | 0.6917   | 0.0220  |
| Baseline               | 0.6804  | 0.0194 | 0.6891   | 0.0171  |

### 4b. Grid search on Hybrid_v2_style

- Best F1: **0.7935**
- Best params: `max_features=5000`, words `ngram_range=(1,2)`, chars `ngram_range=(4,6)`

## Section 5. Classifier Sweep on Best Pipeline

Hybrid_v2_style with tuned feature params.

| pipeline            | f1_mean | f1_std | acc_mean | acc_std |
| ------------------- | ------- | ------ | -------- | ------- |
| logistic_regression | 0.7935  | 0.0165 | 0.7954   | 0.0170  |
| linear_svc          | 0.7850  | 0.0092 | 0.7871   | 0.0097  |
| random_forest       | 0.7847  | 0.0057 | 0.7871   | 0.0059  |
| xgboost             | 0.7815  | 0.0172 | 0.7833   | 0.0176  |
| sgd                 | 0.7765  | 0.0198 | 0.7777   | 0.0195  |
| complement_nb       | 0.7703  | 0.0260 | 0.7706   | 0.0262  |
| multinomial_nb      | 0.7702  | 0.0228 | 0.7706   | 0.0230  |
| adaboost            | 0.7236  | 0.0060 | 0.7262   | 0.0051  |
| decision_tree       | 0.6874  | 0.0290 | 0.6902   | 0.0288  |
| knn                 | 0.6259  | 0.0497 | 0.6617   | 0.0287  |

## Section 6. Hyperparameter Tuning — Top 5 Classifiers

| model               | before_f1 | best_f1  | delta    | best_params                                                                                        |
| ------------------- | --------- | -------- | -------- | -------------------------------------------------------------------------------------------------- |
| logistic_regression | 0.7935    | 0.793532 | 0.000032 | `C=1, max_iter=1000, solver='lbfgs'`                                                               |
| sgd                 | 0.7765    | 0.791890 | 0.015390 | `alpha=0.01, l1_ratio=0.15, learning_rate='adaptive', loss='modified_huber', penalty='elasticnet'` |
| linear_svc          | 0.7850    | 0.790932 | 0.005932 | `C=0.01, loss='squared_hinge', max_iter=2000`                                                      |
| xgboost             | 0.7815    | 0.790383 | 0.008883 | `colsample_bytree=0.8, learning_rate=0.1, max_depth=7, n_estimators=200, subsample=1.0`            |
| random_forest       | 0.7847    | 0.788651 | 0.003951 | `max_depth=None, max_features='sqrt', min_samples_leaf=2, n_estimators=300`                        |

## Section 7. Ensemble Methods (5-fold CV)

| pipeline             | f1_mean | f1_std | acc_mean | acc_std |
| -------------------- | ------- | ------ | -------- | ------- |
| Ensemble_Stack       | 0.8032  | 0.0238 | 0.8048   | 0.0240  |
| Ensemble_Soft_4      | 0.8028  | 0.0237 | 0.8044   | 0.0240  |
| Ensemble_Soft_5      | 0.8002  | 0.0246 | 0.8017   | 0.0247  |
| Ensemble_Stack_NoXGB | 0.7975  | 0.0246 | 0.7991   | 0.0247  |

## Section 8. Final Evaluation on Held-Out Test Set

- Final pipeline: `Ensemble_Stack_NoXGB`
- Held-out test accuracy: **~0.80**

## External Reference

- Class leaderboard accuracy (hidden post-deadline test set): **~0.76**
- Local/leaderboard gap: ~4 pts — motivation for the distribution-shift hypothesis and expanded dataset.
