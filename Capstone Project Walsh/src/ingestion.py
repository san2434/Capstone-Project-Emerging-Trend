from pathlib import Path
import sys
import pandas as pd


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names:
    - strip leading/trailing spaces
    - lowercase
    - remove all whitespace characters
    """
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "", regex=True)
    )
    return df


def find_keyword_column(df: pd.DataFrame) -> str | None:
    """
    Try to identify a likely keyword column.
    """
    candidates = [
        "keyword",
        "keywords",
        "searchkeyword",
        "term",
        "query",
        "productkeyword",
    ]
    for col in candidates:
        if col in df.columns:
            return col

    # fallback: any column containing 'keyword'
    for col in df.columns:
        if "keyword" in col:
            return col

    return None


def find_date_columns(df: pd.DataFrame) -> list[str]:
    """
    Identify likely date columns by name heuristics.
    """
    likely_tokens = ("date", "time", "created", "timestamp")
    return [c for c in df.columns if any(token in c for token in likely_tokens)]


def summarize_dataframe(df: pd.DataFrame) -> None:
    """
    Print:
    - total row count
    - unique keywords list
    - min/max dates
    """
    print("\n=== INGESTION SUMMARY ===")
    print(f"Total row count: {len(df):,}")

    # Unique keywords
    keyword_col = find_keyword_column(df)
    if keyword_col:
        keywords = (
            df[keyword_col]
            .dropna()
            .astype(str)
            .str.strip()
            .loc[lambda s: s != ""]
            .unique()
            .tolist()
        )
        keywords_sorted = sorted(keywords, key=lambda x: x.lower())
        print(f"Keyword column detected: '{keyword_col}'")
        print(f"Unique keywords found ({len(keywords_sorted)}): {keywords_sorted}")
    else:
        print("Keyword column detected: None")
        print("Unique keywords found: []")

    # Min/max dates across all likely date columns
    date_cols = find_date_columns(df)
    min_date = None
    max_date = None

    for col in date_cols:
        parsed = pd.to_datetime(df[col], errors="coerce", utc=True)
        if parsed.notna().any():
            col_min = parsed.min()
            col_max = parsed.max()
            min_date = col_min if min_date is None else min(min_date, col_min)
            max_date = col_max if max_date is None else max(max_date, col_max)

    if min_date is not None and max_date is not None:
        print(f"Date range (min): {min_date}")
        print(f"Date range (max): {max_date}")
    else:
        print("Date range (min): Not available (no parseable date columns found)")
        print("Date range (max): Not available (no parseable date columns found)")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"
    output_file = processed_dir / "merged_reddit_raw.csv"

    # Validate raw directory
    if not raw_dir.exists():
        print(f"ERROR: Raw data directory not found: {raw_dir}", file=sys.stderr)
        sys.exit(1)

    # Discover Excel files
    excel_files = sorted(raw_dir.glob("*.xlsx"))

    # Must read all 4 Excel files in data/raw/
    if len(excel_files) == 0:
        print(f"ERROR: No .xlsx files found in {raw_dir}", file=sys.stderr)
        sys.exit(1)

    if len(excel_files) < 4:
        found = [p.name for p in excel_files]
        print(
            f"ERROR: Expected 4 Excel files in {raw_dir}, but found {len(excel_files)}: {found}",
            file=sys.stderr,
        )
        sys.exit(1)

    if len(excel_files) > 4:
        print(
            f"WARNING: Found {len(excel_files)} Excel files. "
            "All will be ingested and concatenated."
        )

    frames: list[pd.DataFrame] = []

    for file_path in excel_files:
        try:
            df = pd.read_excel(file_path)
            df["source_file"] = file_path.name  # traceability
            frames.append(df)
            print(f"Loaded: {file_path.name} ({len(df):,} rows)")
        except FileNotFoundError:
            print(f"ERROR: Missing file: {file_path}", file=sys.stderr)
            sys.exit(1)
        except Exception as exc:
            print(f"ERROR: Failed to read {file_path.name}: {exc}", file=sys.stderr)
            sys.exit(1)

    # Consolidate
    merged_df = pd.concat(frames, ignore_index=True)

    # Clean column names
    merged_df = clean_column_names(merged_df)

    # Print summary
    summarize_dataframe(merged_df)

    # Save merged output
    try:
        processed_dir.mkdir(parents=True, exist_ok=True)
        merged_df.to_csv(output_file, index=False)
        print(f"\nSaved merged file: {output_file}")
    except Exception as exc:
        print(f"ERROR: Failed to save output CSV: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()