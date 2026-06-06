"""
Reddit Extraction Pipeline
--------------------------
Collect recent Reddit posts and comments for keyword monitoring,
then export the extracted data to an Excel file.
"""

import getpass
import os
import time
from datetime import datetime, timedelta, timezone
from typing import List

import pandas as pd
import praw
import prawcore
from praw.models import Comment, MoreComments


# =========================
# Configuration
# =========================
REDDIT_TIME_FILTER = "month"
REDDIT_RETRIES = 3
REDDIT_BACKOFF_SECONDS = 30
REDDIT_SEARCH_PAUSE_SECONDS = 1.0
REDDIT_COMMENT_PAUSE_SECONDS = 0.25

MAX_REDDIT_COMMENTS = 100
MAX_POSTS_PER_KEYWORD = 100

KEYWORDS_FILE = "keywords_hiking.txt"
KEYWORDS_COLUMN = "cluster_keyword"
KEYWORD_LIMIT = None

OUTPUT_XLSX = "reddit_hiking_keywords_bigger_v4.xlsx"


# =========================
# Credentials and Client
# =========================
def get_reddit_client() -> praw.Reddit:
    client_id = os.getenv("REDDIT_CLIENT_ID") or getpass.getpass("Enter REDDIT_CLIENT_ID: ")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET") or getpass.getpass("Enter REDDIT_CLIENT_SECRET: ")
    user_agent = os.getenv("REDDIT_USER_AGENT") or input("Enter REDDIT_USER_AGENT: ").strip()

    return praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
    )


# =========================
# Data Loading
# =========================
def load_keywords(file_path: str, column_name: str, limit=None) -> List[str]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found in current working directory")

    keywords_df = pd.read_csv(file_path)
    if column_name not in keywords_df.columns:
        raise KeyError(
            f"'{column_name}' not found in {file_path}. Available: {keywords_df.columns.tolist()}"
        )

    keywords = keywords_df[column_name].dropna().astype(str).tolist()
    if limit is not None:
        keywords = keywords[:limit]

    print(f"✅ Loaded {len(keywords)} keywords from {file_path}")
    return keywords


# =========================
# Reddit Search Helpers
# =========================
def safe_reddit_search(reddit_client: praw.Reddit, query: str, retries: int = REDDIT_RETRIES):
    for _ in range(1, retries + 1):
        try:
            print(f"   [search] time_filter={REDDIT_TIME_FILTER}, cap={MAX_POSTS_PER_KEYWORD}")
            posts = list(
                reddit_client.subreddit("all").search(
                    f'"{query}"',
                    sort="new",
                    time_filter=REDDIT_TIME_FILTER,
                    limit=MAX_POSTS_PER_KEYWORD,
                )
            )
            print(f"   ✅ Search returned {len(posts)} posts")
            return posts
        except prawcore.exceptions.TooManyRequests as exc:
            wait = getattr(exc, "sleep_time", REDDIT_BACKOFF_SECONDS)
            print(f"⏳ 429 TooManyRequests for '{query}' — sleeping {wait}s")
            time.sleep(wait)
        except Exception as exc:
            message = str(exc)
            if "429" in message or "temporarily" in message.lower():
                print(f"⏳ Temporary Reddit error for '{query}' — sleeping {REDDIT_BACKOFF_SECONDS}s")
                time.sleep(REDDIT_BACKOFF_SECONDS)
            else:
                print(f"⚠️ Reddit error for '{query}': {exc}")
                return []
    return []


# =========================
# Scraper
# =========================
def reddit_scrape(reddit_client: praw.Reddit, keywords: List[str]) -> pd.DataFrame:
    rows = []
    seen = set()
    cutoff = datetime.now(timezone.utc) - timedelta(days=30)

    for index, keyword in enumerate(keywords, start=1):
        print(f"\n👽 [{index}/{len(keywords)}] Reddit keyword: {keyword}")
        posts = safe_reddit_search(reddit_client, keyword)

        processed_posts = 0
        kept_recent_posts = 0
        keyword_comments = 0

        for post in posts:
            time.sleep(REDDIT_SEARCH_PAUSE_SECONDS)

            created = datetime.fromtimestamp(post.created_utc, tz=timezone.utc)
            if created < cutoff:
                continue

            processed_posts += 1
            kept_recent_posts += 1

            if processed_posts % 25 == 0:
                print(
                    f"   [progress] {keyword}: processed={processed_posts} recent posts, "
                    f"comments={keyword_comments}"
                )

            base = {
                "id": post.id,
                "title": post.title,
                "description": post.selftext or None,
                "channel_title": str(post.author) if post.author else None,
                "published_at": created.isoformat(),
                "view_count": post.score,
                "like_count": post.ups,
                "comment_count": post.num_comments,
                "keyword": keyword,
                "source": "Reddit",
                "post_url": f"https://www.reddit.com{post.permalink}",
                "subreddit": str(post.subreddit.display_name),
            }

            try:
                post.comments.replace_more(limit=0)
            except Exception:
                continue

            extracted = 0
            for comment in post.comments.list():
                time.sleep(REDDIT_COMMENT_PAUSE_SECONDS)

                if isinstance(comment, MoreComments):
                    continue

                comment_obj: Comment = comment

                key = (post.id, comment_obj.id)
                if key in seen:
                    continue
                seen.add(key)

                rows.append(
                    {
                        **base,
                        "comment_id": comment_obj.id,
                        "comment": comment_obj.body,
                        "comment_author": str(comment_obj.author) if comment_obj.author else None,
                        "comment_published_at": datetime.fromtimestamp(
                            comment_obj.created_utc, tz=timezone.utc
                        ).isoformat(),
                    }
                )

                extracted += 1
                keyword_comments += 1

                if keyword_comments % 100 == 0:
                    print(f"   [comments] {keyword}: {keyword_comments} collected")

                if extracted >= MAX_REDDIT_COMMENTS:
                    break

            if kept_recent_posts >= MAX_POSTS_PER_KEYWORD:
                print(f"   [cap] Reached post cap ({MAX_POSTS_PER_KEYWORD}) for '{keyword}'")
                break

        print(f"   [done] {keyword}: recent_posts={kept_recent_posts}, comments={keyword_comments}")

    return pd.DataFrame(rows)


# =========================
# Main
# =========================
def main() -> None:
    reddit_client = get_reddit_client()
    keywords = load_keywords(KEYWORDS_FILE, KEYWORDS_COLUMN, KEYWORD_LIMIT)

    result_df = reddit_scrape(reddit_client, keywords)
    print(f"\n🎉 Scrape complete. Rows: {len(result_df)}")

    result_df.to_excel(OUTPUT_XLSX, index=False)
    print(f"💾 Excel saved: {OUTPUT_XLSX}")

    if not result_df.empty:
        print("\nPreview:")
        print(result_df.head(10).to_string(index=False))
        print("\nTop keywords by rows:")
        print(result_df["keyword"].value_counts().head(10).to_string())
    else:
        print("No rows returned for current filter/caps.")


if __name__ == "__main__":
    main()
