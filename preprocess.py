# See reorganize_for_submission.md for the full decision trail behind this file.

import pandas as pd
from sklearn.preprocessing import FunctionTransformer


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


def prepare_data(csv_path):
    # backend supplies a CSV with `url` and `headline` columns
    df = pd.read_csv(csv_path)

    # drop rows whose scraped headline is missing
    df = df.dropna(subset=["headline"]).reset_index(drop=True)

    # infer source label from URL substring -- Fox=1, NBC=0
    y = [1 if "foxnews.com" in str(url) else 0 for url in df["url"]]

    # raw headlines -- pipeline expects untouched text since the style
    # branch reads caps and punctuation (cleaning would strip them)
    X = list(df["headline"].astype(str))

    return X, y
