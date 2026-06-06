from __future__ import annotations

from pathlib import Path
import math
import sys
import time
from datetime import timedelta
from typing import Iterable

import pandas as pd
from requests.exceptions import HTTPError

try:
    from pytrends.request import TrendReq
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "pytrends is required for this script. Install it with: pip install pytrends"
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "merged_reddit_raw.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "google_trends_raw.csv"

BASE_SLEEP_SECONDS = 7
MIN_429_BACKOFF_SECONDS = 20
MAX_RETRIES_PER_KEYWORD = 3
LOOKBACK_WEEKS = 16
GEO = ""
HL = "en-US"
TZ = 360


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = (
        cleaned.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "", regex=True)
    )
    return cleaned


def detect_keyword_column(df: pd.DataFrame) -> str:
    candidates = ["keyword", "keywords", "searchkeyword", "query", "term", "topic"]
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    for column in df.columns:
        if "keyword" in column:
            return column
    raise KeyError("No keyword column found in the input dataset.")


def detect_date_column(df: pd.DataFrame) -> str:
    candidates = [
        "published_at",
        "publishedat",
        "created_at",
        "createdat",
        "date",
        "datetime",
        "timestamp",
        "createdutc",
    ]
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    for column in df.columns:
        if any(token in column for token in ("date", "time", "created", "timestamp")):
            return column
    raise KeyError("No date column found in the input dataset.")


def load_input_dataframe(input_file: Path) -> pd.DataFrame:
    if not input_file.exists():
        raise FileNotFoundError(f"Missing input file: {input_file}")
    df = pd.read_csv(input_file)
    df = clean_column_names(df)
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def build_7day_windows(lookback_weeks: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    end_dt = pd.Timestamp.utcnow().normalize()
    start_dt = end_dt - pd.Timedelta(weeks=lookback_weeks) + pd.Timedelta(days=1)

    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cursor = start_dt
    while cursor <= end_dt:
        window_end = min(cursor + pd.Timedelta(days=6), end_dt)
        windows.append((cursor, window_end))
        cursor = window_end + pd.Timedelta(days=1)
    return windows


def prepare_reddit_temporal_frame(df: pd.DataFrame, keyword_column: str, date_column: str) -> pd.DataFrame:
    working = df.copy()
    working[date_column] = pd.to_datetime(working[date_column], errors="coerce", utc=True)
    working[keyword_column] = working[keyword_column].astype(str).str.strip()
    working = working.dropna(subset=[date_column, keyword_column])
    working = working.loc[working[keyword_column] != ""]

    # keep only last 16 weeks to match Trends window
    end_dt = pd.Timestamp.utcnow().normalize()
    start_dt = end_dt - pd.Timedelta(weeks=LOOKBACK_WEEKS) + pd.Timedelta(days=1)
    working = working[(working[date_column] >= start_dt) & (working[date_column] <= end_dt + pd.Timedelta(days=1))]

    working["date"] = working[date_column].dt.date.astype(str)

    agg_map: dict[str, str] = {}
    for column in ["view_count", "like_count", "comment_count", "score", "upvote_ratio"]:
        if column in working.columns:
            working[column] = pd.to_numeric(working[column], errors="coerce")
            agg_map[column] = "sum"

    grouped = working.groupby([keyword_column, "date"], as_index=False).agg(agg_map) if agg_map else working.groupby(
        [keyword_column, "date"], as_index=False
    ).size().drop(columns=["size"])
    grouped = grouped.rename(columns={keyword_column: "keyword"})

    counts = (
        working.groupby([keyword_column, "date"])
        .size()
        .reset_index(name="reddit_post_count")
        .rename(columns={keyword_column: "keyword"})
    )
    grouped = grouped.merge(counts, on=["keyword", "date"], how="left")
    grouped["reddit_keyword"] = grouped["keyword"]

    return grouped


def is_rate_limited_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    if "429" in msg or "too many requests" in msg or "rate limit" in msg:
        return True
    if isinstance(exc, HTTPError):
        try:
            if exc.response is not None and exc.response.status_code == 429:
                return True
        except Exception:
            pass
    return False


def parse_interest_frame(keyword: str, interest: pd.DataFrame) -> pd.DataFrame:
    if interest.empty:
        return pd.DataFrame(columns=["keyword", "date", "google_interest"])

    interest = interest.reset_index()
    if "isPartial" in interest.columns:
        interest = interest.drop(columns=["isPartial"])
    if keyword not in interest.columns:
        return pd.DataFrame(columns=["keyword", "date", "google_interest"])

    interest = interest.rename(columns={interest.columns[0]: "date", keyword: "google_interest"})
    interest["keyword"] = keyword
    interest["date"] = pd.to_datetime(interest["date"], errors="coerce", utc=True).dt.date.astype(str)
    interest["google_interest"] = pd.to_numeric(interest["google_interest"], errors="coerce")
    interest = interest.dropna(subset=["date", "google_interest"])
    return interest[["keyword", "date", "google_interest"]]


def fetch_keyword_7d_chunks(pytrends: TrendReq, keyword: str, windows: Iterable[tuple[pd.Timestamp, pd.Timestamp]], sleep_seconds: int) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for start_dt, end_dt in windows:
        timeframe = f"{start_dt.date().isoformat()} {end_dt.date().isoformat()}"
        pytrends.build_payload([keyword], cat=0, timeframe=timeframe, geo=GEO, gprop="")
        interest = pytrends.interest_over_time()
        parsed = parse_interest_frame(keyword, interest)
        if not parsed.empty:
            parts.append(parsed)
        time.sleep(sleep_seconds)

    if not parts:
        return pd.DataFrame(columns=["keyword", "date", "google_interest"])

    out = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["keyword", "date"], keep="last")
    out = out.sort_values(["keyword", "date"]).reset_index(drop=True)
    return out


def build_fallback_vector(reddit_temporal: pd.DataFrame, keyword: str, fallback_value: float) -> pd.DataFrame:
    kw_dates = (
        reddit_temporal.loc[reddit_temporal["keyword"] == keyword, ["keyword", "date"]]
        .drop_duplicates()
        .copy()
    )
    if kw_dates.empty:
        return pd.DataFrame(columns=["keyword", "date", "google_interest"])

    kw_dates["google_interest"] = float(fallback_value)
    return kw_dates[["keyword", "date", "google_interest"]]


def main() -> None:
    try:
        df = load_input_dataframe(INPUT_FILE)
        keyword_column = detect_keyword_column(df)
        date_column = detect_date_column(df)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    reddit_temporal = prepare_reddit_temporal_frame(df, keyword_column, date_column)

    unique_keywords = (
        df[keyword_column]
        .dropna()
        .astype(str)
        .str.strip()
    )
    unique_keywords = unique_keywords[unique_keywords != ""].unique().tolist()

    if not unique_keywords:
        print("ERROR: No valid keywords found after cleaning.", file=sys.stderr)
        sys.exit(1)

    try:
        pytrends = TrendReq(hl=HL, tz=TZ, retries=0)
    except Exception as exc:
        print(f"ERROR: Failed to initialize TrendReq: {exc}", file=sys.stderr)
        sys.exit(1)

    windows = build_7day_windows(LOOKBACK_WEEKS)
    google_frames: list[pd.DataFrame] = []
    fallback_frames: list[pd.DataFrame] = []

    print(f"Input rows after deduplication: {len(df):,}")
    print(f"Unique keywords: {len(unique_keywords):,}")
    print(f"Window strategy: last {LOOKBACK_WEEKS} weeks in {len(windows)} weekly chunks")

    for i, keyword in enumerate(unique_keywords, start=1):
        backoff = BASE_SLEEP_SECONDS
        failures = 0
        success = False

        while failures < MAX_RETRIES_PER_KEYWORD and not success:
            try:
                kw_frame = fetch_keyword_7d_chunks(pytrends, keyword, windows, sleep_seconds=backoff)
                if kw_frame.empty:
                    raise RuntimeError("empty_interest_over_time")
                google_frames.append(kw_frame)
                success = True
                print(f"[{i}/{len(unique_keywords)}] OK: {keyword} ({len(kw_frame):,} rows)")
            except Exception as exc:
                failures += 1
                if is_rate_limited_error(exc):
                    backoff = max(MIN_429_BACKOFF_SECONDS, backoff * 2)
                    print(
                        f"[{i}/{len(unique_keywords)}] 429: {keyword} "
                        f"(attempt {failures}/{MAX_RETRIES_PER_KEYWORD}) -> backoff {backoff}s",
                        file=sys.stderr,
                    )
                    time.sleep(backoff)
                else:
                    print(
                        f"[{i}/{len(unique_keywords)}] WARN: {keyword} "
                        f"(attempt {failures}/{MAX_RETRIES_PER_KEYWORD}) -> {exc}",
                        file=sys.stderr,
                    )
                    time.sleep(backoff)

        if not success:
            collected = pd.concat(google_frames, ignore_index=True) if google_frames else pd.DataFrame(columns=["google_interest"])
            global_mean = pd.to_numeric(collected.get("google_interest"), errors="coerce").mean() if not collected.empty else float("nan")
            if pd.isna(global_mean):
                global_mean = 50.0
            fallback = build_fallback_vector(reddit_temporal, keyword, fallback_value=float(global_mean))
            fallback_frames.append(fallback)
            print(f"[{i}/{len(unique_keywords)}] FALLBACK: {keyword} ({len(fallback):,} rows, value={global_mean:.2f})")

    google_trends = pd.concat(google_frames + fallback_frames, ignore_index=True) if (google_frames or fallback_frames) else pd.DataFrame(
        columns=["keyword", "date", "google_interest"]
    )
    google_trends["google_interest"] = pd.to_numeric(google_trends["google_interest"], errors="coerce")
    google_trends = google_trends.drop_duplicates(subset=["keyword", "date"], keep="last")

    final_df = reddit_temporal.merge(google_trends, on=["keyword", "date"], how="left")

    # guarantee non-blank google_interest
    per_keyword_mean = final_df.groupby("keyword")["google_interest"].transform("mean")
    global_mean_final = final_df["google_interest"].mean()
    if pd.isna(global_mean_final):
        global_mean_final = 50.0
    final_df["google_interest"] = final_df["google_interest"].fillna(per_keyword_mean).fillna(global_mean_final).fillna(50.0)

    # keep numeric and stable
    final_df["google_interest"] = pd.to_numeric(final_df["google_interest"], errors="coerce").fillna(50.0)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved: {OUTPUT_FILE}")
    print(f"Rows: {len(final_df):,}")
    print(f"Blank google_interest cells: {int(final_df['google_interest'].isna().sum()):,}")


if __name__ == "__main__":
    main()
