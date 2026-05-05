import re
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}
TIMEOUT = 12
DELAY = 1.0

# Two outlets we're classifying; values are Google's `site:` operator
SOURCES = {
    "FoxNews": "site:foxnews.com",
    "NBC": "site:nbcnews.com",
}
SUFFIXES = [" - Fox News", " - NBC News", " | Fox News", " | NBC News"]

# Broad topics, each gets its own per-day query so we hit Google's ~100-result cap less
TOPICS = [
    "politics",
    "election",
    "congress",
    "immigration",
    "economy",
    "inflation",
    "jobs",
    "foreign policy",
    "ukraine",
    "israel",
    "china",
    "crime",
    "police",
    "supreme court",
    "healthcare",
    "education",
    "climate",
    "technology",
    "sports",
    "entertainment",
]

START = date(2025, 11, 1)
END = date(2026, 4, 30)

REPO_ROOT = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd()
BASE_CSV = REPO_ROOT / "data" / "url_with_headlines.csv"
NEW_CSV = REPO_ROOT / "data" / "new_scraped_headlines.csv"
EXPANDED_CSV = REPO_ROOT / "data" / "expanded_headlines.csv"


def day_ranges(start, end):
    # Yield (after, before) pairs covering one calendar day each
    days = []
    cur = start
    while cur <= end:
        days.append((cur, cur + timedelta(days=1)))
        cur += timedelta(days=1)
    return days


def strip_source_suffix(title):
    for suffix in SUFFIXES:
        if title.endswith(suffix):
            return title[: -len(suffix)].strip()
    return title.strip()


def fetch_google_news_headlines(site_query, topic, after, before):
    # Hit Google News RSS for one (site, topic, day) slice; return cleaned titles
    q = f"{site_query} {topic} after:{after} before:{before}"
    url = (
        "https://news.google.com/rss/search"
        f"?q={requests.utils.quote(q)}&hl=en-US&gl=US&ceid=US:en"
    )
    try:
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        if r.status_code != 200:
            return []
    except Exception:
        return []

    soup = BeautifulSoup(r.text, "xml")
    headlines = []
    for item in soup.find_all("item"):
        title_tag = item.find("title")
        if not title_tag:
            continue
        text = strip_source_suffix(title_tag.get_text())
        if text:
            headlines.append(text)
    return headlines


def normalize(text):
    # Lowercase + strip non-alphanumerics so near-duplicates collapse for dedup
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def main():
    days = day_ranges(START, END)
    total_queries = len(days) * len(TOPICS) * len(SOURCES)
    print(
        f"Collecting headlines: {len(days)} days x {len(TOPICS)} topics x "
        f"{len(SOURCES)} sources = {total_queries} queries\n"
    )

    rows = []
    queries_done = 0
    start_time = time.time()
    for source, site_query in SOURCES.items():
        print(f"{'='*60}\nSource: {source}\n{'='*60}")
        for topic in TOPICS:
            topic_count = 0
            for after, before in days:
                headlines = fetch_google_news_headlines(
                    site_query,
                    topic,
                    after.strftime("%Y-%m-%d"),
                    before.strftime("%Y-%m-%d"),
                )
                for h in headlines:
                    rows.append(
                        {
                            "headline": h,
                            "source": source,
                            "date": after.strftime("%Y-%m-%d"),
                            "topic": topic,
                        }
                    )
                topic_count += len(headlines)
                queries_done += 1
                # Heartbeat every 30 queries (~30s) with overall progress + ETA
                if queries_done % 30 == 0:
                    elapsed = time.time() - start_time
                    rate = queries_done / elapsed
                    eta_min = (total_queries - queries_done) / rate / 60
                    print(
                        f"    [{queries_done}/{total_queries} queries, "
                        f"{queries_done/total_queries*100:.1f}%, "
                        f"ETA ~{eta_min:.0f} min] {source} {topic} {after}"
                    )
                time.sleep(DELAY)
            print(f"  {topic}: {topic_count} headlines")
        print()

    # Exact dedup on (headline, source); write the raw new-only CSV unfiltered by balancing
    new_df = pd.DataFrame(rows)
    print(f"Raw rows scraped: {len(new_df)}")
    new_df = new_df.drop_duplicates(subset=["headline", "source"]).reset_index(drop=True)
    print(f"After exact dedup: {len(new_df)}")
    NEW_CSV.parent.mkdir(parents=True, exist_ok=True)
    new_df.to_csv(NEW_CSV, index=False)
    print(f"Wrote {NEW_CSV}")
    print(new_df["source"].value_counts().to_string(), "\n")

    # Downsample each source to the smaller class so combined Fox/NBC counts are balanced
    balanced_new = new_df.groupby("source", group_keys=False).sample(
        n=new_df["source"].value_counts().min(), random_state=42
    )
    print(f"Balanced new (per-source min): {len(balanced_new)}")

    # Filter only the new portion against base; base rows are appended untouched
    base_df = pd.read_csv(BASE_CSV).dropna(subset=["headline"])
    # Infer source from URL substring, matching preprocess.py's labeling convention
    base_df["source"] = base_df["url"].str.contains("foxnews.com", case=False, na=False).map(
        {True: "FoxNews", False: "NBC"}
    )
    base_keys = set(base_df["headline"].astype(str).map(normalize))
    balanced_new = balanced_new.assign(
        _key=balanced_new["headline"].astype(str).map(normalize)
    )
    balanced_new = (
        balanced_new[balanced_new["_key"] != ""]
        .drop_duplicates(subset=["_key"])
        .loc[lambda d: ~d["_key"].isin(base_keys)]
        .drop(columns="_key")
    )
    print(f"New after normalized dedup vs base: {len(balanced_new)}")

    combined = pd.concat(
        [base_df[["headline", "source"]], balanced_new[["headline", "source"]]],
        ignore_index=True,
    )
    combined.to_csv(EXPANDED_CSV, index=False)
    print(f"Wrote {EXPANDED_CSV}: {len(combined)} rows")
    print(combined["source"].value_counts().to_string())


if __name__ == "__main__":
    main()
