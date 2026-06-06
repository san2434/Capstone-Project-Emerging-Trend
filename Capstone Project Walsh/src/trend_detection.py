from __future__ import annotations

from pathlib import Path
import math
import sys

import numpy as np
import pandas as pd
from scipy.stats import t


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "google_trends_raw.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "model_results.csv"

MAX_LAG_WEEKS = 8
MIN_OVERLAP_POINTS = 4
WEEKLY_FREQUENCY = "W-MON"


def load_input_data(input_file: Path) -> pd.DataFrame:
	if not input_file.exists():
		raise FileNotFoundError(f"Input file not found: {input_file}")

	df = pd.read_csv(input_file)
	df.columns = (
		df.columns.astype(str)
		.str.strip()
		.str.lower()
		.str.replace(r"\s+", "", regex=True)
	)

	required = {"date", "keyword", "reddit_post_count", "google_interest"}
	missing = required.difference(df.columns)
	if missing:
		missing_list = ", ".join(sorted(missing))
		raise KeyError(f"Missing required column(s): {missing_list}")

	df["date"] = pd.to_datetime(df["date"], errors="coerce")
	df["keyword"] = df["keyword"].astype(str).str.strip()
	df["reddit_post_count"] = pd.to_numeric(df["reddit_post_count"], errors="coerce")
	df["google_interest"] = pd.to_numeric(df["google_interest"], errors="coerce")

	df = df.dropna(subset=["date", "keyword"])
	df = df.loc[df["keyword"] != ""]
	return df


def build_weekly_keyword_series(keyword_df: pd.DataFrame) -> pd.DataFrame:
	working = keyword_df.copy()
	working = working.sort_values("date")
	working = working.set_index("date")

	weekly = (
		working.resample(WEEKLY_FREQUENCY)
		.agg({"reddit_post_count": "sum", "google_interest": "mean"})
		.rename(columns={"reddit_post_count": "reddit_weekly", "google_interest": "google_weekly"})
	)

	if weekly.empty:
		return weekly

	full_index = pd.date_range(weekly.index.min(), weekly.index.max(), freq=WEEKLY_FREQUENCY)
	weekly = weekly.reindex(full_index)
	weekly["reddit_weekly"] = weekly["reddit_weekly"].fillna(0.0)

	return weekly


def compute_lagged_correlations(
	reddit_series: pd.Series,
	google_series: pd.Series,
	max_lag_weeks: int,
	min_overlap_points: int,
) -> pd.DataFrame:
	rows: list[dict[str, object]] = []

	for lag in range(-max_lag_weeks, max_lag_weeks + 1):
		shifted_google = google_series.shift(-lag)
		pair = pd.concat([reddit_series, shifted_google], axis=1).dropna()
		pair.columns = ["reddit", "google"]

		n_points = len(pair)
		if n_points < min_overlap_points:
			rows.append(
				{
					"lag_weeks": lag,
					"correlation": np.nan,
					"p_value": np.nan,
					"n_overlap": n_points,
				}
			)
			continue

		try:
			correlation = pair["reddit"].corr(pair["google"], method="pearson")
			if pd.isna(correlation):
				p_value = np.nan
			elif abs(correlation) >= 1:
				p_value = 0.0
			else:
				t_stat = correlation * math.sqrt((n_points - 2) / (1 - correlation**2))
				p_value = 2 * t.sf(abs(t_stat), df=n_points - 2)
				p_value = float(np.asarray(p_value).item())
		except Exception:
			correlation, p_value = np.nan, np.nan

		rows.append(
			{
				"lag_weeks": lag,
				"correlation": correlation,
				"p_value": p_value,
				"n_overlap": n_points,
			}
		)

	return pd.DataFrame(rows)


def pick_optimal_lag(lag_df: pd.DataFrame) -> tuple[object, object, object, object]:
	valid = lag_df.dropna(subset=["correlation", "p_value"])
	if valid.empty:
		return math.nan, 0, math.nan, 0

	best_idx = valid["correlation"].abs().idxmax()
	best_row = valid.loc[best_idx]

	return (
		best_row["correlation"],
		best_row["lag_weeks"],
		best_row["p_value"],
		best_row["n_overlap"],
	)


def run_tlcc(df: pd.DataFrame) -> pd.DataFrame:
	results: list[dict[str, object]] = []

	for keyword, keyword_df in df.groupby("keyword", sort=True):
		weekly = build_weekly_keyword_series(keyword_df)

		if weekly.empty:
			results.append(
				{
					"keyword": keyword,
					"max_correlation": np.nan,
					"optimal_lag_weeks": np.nan,
					"p_value": np.nan,
					"n_overlap": 0,
					"status": "no_weekly_data",
				}
			)
			continue

		lag_df = compute_lagged_correlations(
			reddit_series=weekly["reddit_weekly"],
			google_series=weekly["google_weekly"],
			max_lag_weeks=MAX_LAG_WEEKS,
			min_overlap_points=MIN_OVERLAP_POINTS,
		)

		max_corr, optimal_lag, p_value, n_overlap = pick_optimal_lag(lag_df)

		status = "ok"
		if is_missing_numeric(max_corr):
			status = "insufficient_overlap_or_variance"

		results.append(
			{
				"keyword": keyword,
				"max_correlation": max_corr,
				"optimal_lag_weeks": optimal_lag,
				"p_value": p_value,
				"n_overlap": n_overlap,
				"status": status,
			}
		)

	return pd.DataFrame(results)


def save_results(result_df: pd.DataFrame, output_file: Path) -> None:
	output_file.parent.mkdir(parents=True, exist_ok=True)
	result_df.to_csv(output_file, index=False)


def is_missing_numeric(value: object) -> bool:
	if value is None:
		return True
	if isinstance(value, (float, np.floating)):
		return math.isnan(float(value))
	return False


def main() -> None:
	try:
		source_df = load_input_data(INPUT_FILE)
		model_results = run_tlcc(source_df)
		save_results(model_results, OUTPUT_FILE)
	except Exception as exc:
		print(f"ERROR: {exc}", file=sys.stderr)
		sys.exit(1)

	print(f"Loaded rows: {len(source_df):,}")
	print(f"Keywords modeled: {model_results['keyword'].nunique():,}")
	print(f"Saved TLCC results: {OUTPUT_FILE}")
	print("Output columns: keyword, max_correlation, optimal_lag_weeks, p_value, n_overlap, status")


if __name__ == "__main__":
	main()
