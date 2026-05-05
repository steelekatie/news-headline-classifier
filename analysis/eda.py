# =============================================================================
# EDA (separated from analysis script) - informed preprocess.py
# =============================================================================
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scipy.stats import chi2_contingency
import re
import unicodedata
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import random
import matplotlib.pyplot as plt
from preprocess import extract_style, prepare_data, STYLE_FEATURE_NAMES

news_df_raw = pd.read_csv("data/expanded_headlines.csv")
print(f"Raw dataset: {len(news_df_raw)} headlines")

# =============================================================================
# ── 1) SPANISH HEADLINE ANALYSIS
# =============================================================================
print("\n---------------------------------------------------------------")
print("2a) Removing Spanish headlines")
print("---------------------------------------------------------------")
# Our Fox News scraping returned Spanish headlines.
# To address this, we try a two-pass filter:
# First, we search for common Spanish words + characters in each headline.
# Second, we try to capture any remaining Spanish headlines by calculating
# a "ratio" of these common words/characters within each headline.
# We implement a systematic removal of these invalid headlines in prepare_data().

spanish_words = [
    "el",
    "la",
    "los",
    "las",
    "un",
    "una",
    "del",
    "por",
    "que",
    "con",
    "ante",
    "más",
    "también",
    "según",
    "está",
    "fue",
    "han",
    "era",
    "pero",
    "ser",
    "de",
    "en",
    "se",
    "su",
    "al",
    "le",
    "si",
    "ya",
    "hay",
    "uno",
    "dos",
    "tres",
    "y",
    "unos",
    "para",
    "como",
    "contra",
]
spanish_chars = r"[«»ñ¿¡áéíóúÁÉÍÓÚ]"
spanish_chars_in_word = r"\b[a-záéíóúñü]+\b"


# -----------------------------------------------------------------------------
# PASS 1: remove headlines with > 2 Spanish words or > 2 Spanish chars
# init vectorizer to look for these words, binary: exists or not
print("\nPASS 1: Spanish word and character counts")
vectorizer = CountVectorizer(
    vocabulary=spanish_words, token_pattern=r"(?u)\b\w+\b", binary=True, lowercase=True
)

# matrix matching headlines w/ these words
anchor_matrix = vectorizer.fit_transform(news_df_raw["headline"].astype(str))
# sum rows: if >0, we found at least one word
news_df_raw["spanish_word_count"] = anchor_matrix.toarray().sum(axis=1)

# create secondary mask for common Spanish character (uncommon English)
# * noted use of «» in Fox Spanish heds -- I think they use for quotes at times
news_df_raw["spanish_char_count"] = news_df_raw["headline"].str.count(spanish_chars)

# remove if more than 2 Spanish words or more than 2 Spanish characters
# We don't want to throw out refs to "résumé, Nicolás Maduro", or "Los Angeles"
# if the full hed is in English - so we're somewhat conservative here
is_spanish = (news_df_raw["spanish_word_count"] > 2) | (
    news_df_raw["spanish_char_count"] > 2
)

borderline_ex = news_df_raw[
    (
        (news_df_raw["spanish_word_count"] == 3)
        & (news_df_raw["spanish_char_count"] <= 1)
    )
    | (
        (news_df_raw["spanish_char_count"] == 3)
        & (news_df_raw["spanish_word_count"] <= 1)
    )
]

print("Example headlines near the word/char count cutoff:")
print(f"{'word_ct':<8} {'char_ct':<8} {'source':<10} headline")
print("-" * 50)
for _, row in borderline_ex.tail(5).iterrows():
    print(
        f"{row['spanish_word_count']:<8} {row['spanish_char_count']:<8} {row['source']:<10} {row['headline']}"
    )

print(f"\nPASS 1 - using word or char count > 2: Removes {is_spanish.sum()} headlines")
english_df = news_df_raw[~is_spanish].copy()  # drop these


# -----------------------------------------------------------------------------
# PASS 2: compute ratio for remaining headlines
# The above is imperfect, so we also take a second pass by computing (roughly)
# the percent of each headline in our dataset made up of these words/chars
print("\n\nPASS 2: Spanish word and character ratios")
english_df["spanish_ratio"] = english_df["headline"].apply(
    lambda text: len(
        [
            w
            for w in re.findall(spanish_chars_in_word, str(text).lower())
            if w in spanish_words
        ]
    )
    / max(len(re.findall(spanish_chars_in_word, str(text).lower())), 1)
)

# inspect borderline headlines to calibrate threshold -- commented to avoid long output
# pd.set_option("display.max_colwidth", None)
# borderline = clean_news_df[
#     (clean_news_df["spanish_ratio"] > 0.10) &
#     (clean_news_df["spanish_ratio"] < 0.20)
# ]
# Commented full print to avoid long output. Included smaller example below
# print(borderline.sort_values("spanish_ratio")[["headline", "source", "spanish_ratio"]].to_string())


borderline_ex = english_df[
    (english_df["spanish_ratio"] > 0.14) & (english_df["spanish_ratio"] < 0.18)
]
print("Example headlines w/ Spanish ratio at cutoff borderline:")
print(f"{'ratio':<8} {'source':<10} headline")
print("-" * 50)
for _, row in borderline_ex.sort_values("spanish_ratio").iterrows():
    print(f"{row['spanish_ratio']:.3f}    {row['source']:<10} {row['headline']}")


# Conclusion: Since we train our models on English, we filter these in prepare_data()
# English headlines with Spanish place names (Los Angeles, El Paso)
# cluster at ~0.11, while genuine Spanish headlines start at 0.15 - 0.16
# -- cutoff set at 0.16
print(
    f"\nPASS 2 - using word or char ratio > 0.16: Removes {(english_df['spanish_ratio'] > 0.16).sum()} additional headlines"
)

english_df = english_df[english_df["spanish_ratio"] < 0.16].copy()  # drop these


# =============================================================================
# -- 2) HEADLINE LENGTH ANALYSIS
# =============================================================================
print("\n\n---------------------------------------------------------------")
print("2b) Investigating scraped headlines for data issues")
print("---------------------------------------------------------------")
# After standardizing to English as much as possible, we look for any
# remaining oddities in our scraped data.
# We implement a systematic removal of these invalid headlines in prepare_data().

pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_rows", 100)

# -----------------------------------------------------------------------------
## LENGTH DISTRIBUTION BY SOURCE
english_df["hed_len"] = english_df["headline"].str.len()
print("\nLength distribution by source:")
print(
    english_df.assign(hed_len=english_df["headline"].str.len())
    .groupby("source")["hed_len"]
    .describe()
)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [5, 3]})

# distribution
for source, grp in english_df.groupby("source"):
    axes[0].hist(grp["hed_len"], bins=30, alpha=0.6, label=source)
axes[0].set_xlabel("Headline character length")
axes[0].set_ylabel("Count")
axes[0].set_title("Headline Length Distribution by Source")
axes[0].legend()

# characters boxplot
english_df.boxplot(column="hed_len", by="source", ax=axes[1], widths=0.18)
axes[1].set_title("Headline Length by Source")
axes[1].set_xlabel("Source")
axes[1].set_ylabel("Characters")

plt.suptitle("")  # suppress super title
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# INSPECT SHORT HEADLINES
# We see these are likely bylines or page/section titles
print("\n\nShortest headlines by source:")
for source in ["FoxNews", "NBC"]:
    print(f"   {source}:")
    for hed in (
        english_df[english_df["source"] == source]
        .nsmallest(3, "hed_len")["headline"]
        .tolist()
    ):
        print(f"     ({len(hed)}) {hed}")
    print()

# inspect more short headlines to locate patterns
english_df["hed_len"] = english_df["headline"].str.len()
short_heds = english_df[(english_df["hed_len"] < 25) & (english_df["hed_len"] > 15)][
    "headline"
].tolist()
sample_short = random.sample(short_heds, min(10, len(short_heds)))
print("\nShort headlines (<25 chars) — random sample:")
for hed in sample_short:
    print(f"   - {hed}")

# -----------------------------------------------------------------------------
# INSPECT LONG HEADLINES
english_df["hed_len"] = english_df["headline"].str.len()

print("\n\nLongest headlines by source:")
for source in ["FoxNews", "NBC"]:
    print(f"   {source}:")
    for hed in (
        english_df[english_df["source"] == source]
        .nlargest(3, "hed_len")["headline"]
        .tolist()
    ):
        print(f"     ({len(hed)}) {hed}")
    print()

english_df = english_df.drop(columns=["hed_len"])


# =============================================================================
# -- 3) PAGE STRUCTURE ARTIFACTS
# =============================================================================
print("\n---------------------------------------------------------------")
print("2c) Scraped page structure artifact investigation")
print("---------------------------------------------------------------")
# We see some problematic results above and investigate further.
# Running on "cleaned" data (no Spanish) for initial exploration of significance.
# We implement a systematic removal of these invalid headlines in prepare_data().

checks = {
    "'Page' references": r" - Page \d+$",
    "Archive pages": r"^Articles\s*[–-]",
    "Show previews": r"^Coming up on",
    "Site maps": r"^Video Site Map",
    "Fox News Radio episodes": r"- FOX News Radio$",
    "Fox Nation episodes": r"- Fox Nation$",
}

for label, pattern in checks.items():
    matches = english_df[
        english_df["headline"].str.contains(pattern, case=False, na=False)
    ]["headline"].tolist()
    sample = random.sample(matches, min(3, len(matches)))
    print(f"\n{label} ({len(matches)} total):")
    for hed in sample:
        print(f"   - {hed}")


# CONCLUSION:
# All of what we see above are site artifacts, not real headlines
# They point to source-specific branding or general page titles, not meaningful heds
# -- We implement their removal via filters in prepare_data() in preprocess.py
# -- At bottom of script, we can see the basic effects of these new filters.

# =============================================================================
# -- 4) STYLE DIFFERENCES BETWEEN OUTLETS
# =============================================================================
print("\n\n---------------------------------------------------------------")
print("2d) Style features - significance testing")
print("---------------------------------------------------------------")
# Based on one group member's experience as a newspaper copy editor, we also
# investigate specific stylistic differences between our two outlets (which are
# likely attributable to their organization's style guides).
# Below, we investigate the significance of several features.
# We will leverage this in preprocess.py and our modeling.


style_matrix = extract_style(english_df["headline"])
style_df = pd.DataFrame(
    style_matrix, columns=STYLE_FEATURE_NAMES, index=english_df.index
)
english_df = pd.concat([english_df, style_df], axis=1)

count_features = ["len", "n_periods", "n_caps", "n_allcaps"]
style_features = STYLE_FEATURE_NAMES

results = []
for feature in style_features:
    contingency = pd.crosstab(english_df["source"], english_df[feature])
    chi2, p, dof, expected = chi2_contingency(contingency)
    n = contingency.sum().sum()
    cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))
    means = english_df.groupby("source")[feature].mean()
    results.append((feature, chi2, p, cramers_v, means["NBC"], means["FoxNews"]))

# sort binary and count separately, both by chi^2 descending
binary_results = sorted(
    [r for r in results if r[0] not in count_features], key=lambda x: x[1], reverse=True
)
count_results = sorted(
    [r for r in results if r[0] in count_features], key=lambda x: x[1], reverse=True
)

print("\nStyle features on non-Spanish data:\n")
print(
    f"{'feature':16s}  {'chi2':>9}  {'p':>7}  {'cramers_v':>9}  {'NBC':>7}  {'Fox':>7}"
)
print("-" * 75)
for feature, chi2, p, cramers_v, nbc, fox in count_results:
    print(
        f"{feature:16s}  {chi2:>9.2f}  {p:>7.4f}  {cramers_v:>9.3f}  {nbc:>7.1f}  {fox:>7.1f}"
    )
print()
for feature, chi2, p, cramers_v, nbc, fox in binary_results:
    print(
        f"{feature:16s}  {chi2:>9.2f}  {p:>7.4f}  {cramers_v:>9.3f}  {nbc:>7.1%}  {fox:>7.1%}"
    )


english_df = english_df.drop(columns=style_features)


# Style features on cleaned data:

# feature                chi2        p  cramers_v      NBC      Fox
# ---------------------------------------------------------------------------
# len                 3493.60   0.0000      0.344     72.0     79.9
# n_caps              1701.86   0.0000      0.240      3.8      5.3
# n_periods           1418.72   0.0000      0.219      0.3      0.1
# n_allcaps            417.05   0.0000      0.119      0.3      0.4

# has_U_S              952.40   0.0000      0.179     7.3%     0.4%
# is_title_case        751.42   0.0000      0.159     0.7%     6.8%
# has_US               672.27   0.0000      0.151     0.2%     4.9%
# has_colon            507.95   0.0000      0.131    11.5%    21.2%
# has_endperiod        273.36   0.0000      0.096     2.3%     0.2%
# has_quote             99.44   0.0000      0.058    27.1%    32.4%
# has_singlequote       87.20   0.0000      0.054    27.0%    32.0%
# has_hyphen            27.88   0.0000      0.031     9.6%    11.5%


# =============================================================================
# -- 5) UPDATED*: CHECKING DISTRIBUTION AFTER PREPROCESS FILTERS
# =============================================================================
# After implementing the cleaning decisions from Sections 1-3 in preprocess.py's
# prepare_data(), we verify the results below:

X, y = prepare_data("data/expanded_headlines.csv", verbose=True)
processed_df = pd.DataFrame({"headline": X, "label": y})
processed_df["source"] = processed_df["label"].map({1: "FoxNews", 0: "NBC"})
processed_df["hed_len"] = processed_df["headline"].str.len()
print(processed_df.groupby("source")["hed_len"].describe())
#            count       mean        std   min   25%   50%   75%    max
# source
# FoxNews  12338.0  82.740639  17.546252  25.0  71.0  84.0  97.0  138.0
# NBC      14553.0  72.377517  17.990908  25.0  58.0  73.0  86.0  139.0

# Plot
fig, axes = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [5, 3]})

# distribution
for source, grp in processed_df.groupby("source"):
    axes[0].hist(grp["hed_len"], bins=30, alpha=0.6, label=source)
axes[0].set_xlabel("Headline character length")
axes[0].set_ylabel("Count")
axes[0].set_title("Cleaned Headline Length Distribution by Source")
axes[0].legend()

# characters boxplot
processed_df.boxplot(column="hed_len", by="source", ax=axes[1], widths=0.18)
axes[1].set_title("Cleaned Headline Length by Source")
axes[1].set_xlabel("Source")
axes[1].set_ylabel("Characters")

plt.suptitle("")  # suppress super title
plt.tight_layout()
plt.show()

# We see clearly that both sources now have roughly normal, right-skewed
# distributions within the bounded range we specified.
# We expect that headline length will still be a meaningful signal,
# as Fox headlines are slightly longer at the median despite a heavier left tail.
