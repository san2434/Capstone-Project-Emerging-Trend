from __future__ import annotations

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, t as t_dist
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "google_trends_raw.csv"
TLCC_FILE = PROJECT_ROOT / "data" / "processed" / "model_results.csv"
OUTPUT_MODELING_FILE = PROJECT_ROOT / "data" / "processed" / "modeling_results.csv"
OUTPUT_HYPOTHESIS_FILE = PROJECT_ROOT / "data" / "processed" / "hypothesis_tests.csv"

TRAIN_TEST_SPLIT = 0.7
MIN_WEEKS_FOR_MODEL = 8
ARIMA_ORDER = (1, 1, 1)


def load_and_prepare_data(input_file: Path) -> pd.DataFrame:
	if not input_file.exists():
		raise FileNotFoundError(f"Missing input file: {input_file}")

	df = pd.read_csv(input_file)
	df.columns = (
		df.columns.astype(str)
		.str.strip()
		.str.lower()
		.str.replace(r"\s+", "", regex=True)
	)

	required = {"date", "keyword", "reddit_post_count", "google_interest", "view_count", "comment_count"}
	missing = required.difference(df.columns)
	if missing:
		missing_list = ", ".join(sorted(missing))
		raise KeyError(f"Missing required columns: {missing_list}")

	df["date"] = pd.to_datetime(df["date"], errors="coerce")
	df["keyword"] = df["keyword"].astype(str).str.strip()
	df["reddit_post_count"] = pd.to_numeric(df["reddit_post_count"], errors="coerce").fillna(0.0)
	df["google_interest"] = pd.to_numeric(df["google_interest"], errors="coerce").fillna(50.0)
	df["view_count"] = pd.to_numeric(df["view_count"], errors="coerce").fillna(0.0)
	df["comment_count"] = pd.to_numeric(df["comment_count"], errors="coerce").fillna(0.0)

	df = df.sort_values(["keyword", "date"]).reset_index(drop=True)
	return df


def engineer_reddit_features(keyword_df: pd.DataFrame) -> pd.DataFrame:
	working = keyword_df.copy()
	working = working.sort_values("date").reset_index(drop=True)

	# Engagement per post
	working["engagement_per_post"] = (
		(working["view_count"] + working["comment_count"]) / (working["reddit_post_count"] + 1)
	).fillna(0.0)

	# Velocity (week-over-week growth in post count)
	working["post_velocity"] = working["reddit_post_count"].diff().fillna(0.0)

	# Volatility (rolling std dev of engagement)
	working["engagement_volatility"] = (
		working["engagement_per_post"].rolling(window=3, min_periods=1).std().fillna(0.0)
	)

	# Comment/view ratio (sentiment proxy)
	working["comment_view_ratio"] = (
		working["comment_count"] / (working["view_count"] + 1)
	).fillna(0.0)

	# Lagged Reddit signals (for autoregressive features)
	working["reddit_post_count_lag1"] = working["reddit_post_count"].shift(1).fillna(0.0)
	working["reddit_post_count_lag2"] = working["reddit_post_count"].shift(2).fillna(0.0)

	return working


def split_train_test(
	keyword_df: pd.DataFrame, split_ratio: float = TRAIN_TEST_SPLIT
) -> tuple[pd.DataFrame, pd.DataFrame]:
	n = len(keyword_df)
	split_idx = int(n * split_ratio)
	train = keyword_df.iloc[:split_idx].copy()
	test = keyword_df.iloc[split_idx:].copy()
	return train, test


def train_arima_model(
	train_df: pd.DataFrame, test_df: pd.DataFrame, order: tuple[int, int, int] = ARIMA_ORDER
) -> dict[str, object]:
	try:
		model = ARIMA(train_df["google_interest"], order=order)
		fitted = model.fit()
		predictions = fitted.get_forecast(steps=len(test_df)).predicted_mean.values
		mae = mean_absolute_error(test_df["google_interest"], predictions)
		rmse = np.sqrt(mean_squared_error(test_df["google_interest"], predictions))
		r2 = r2_score(test_df["google_interest"], predictions)
		return {
			"model_type": "ARIMA",
			"mae": mae,
			"rmse": rmse,
			"r2": r2,
			"predictions": predictions,
		}
	except Exception as exc:
		return {
			"model_type": "ARIMA",
			"mae": np.nan,
			"rmse": np.nan,
			"r2": np.nan,
			"predictions": np.full(len(test_df), np.nan),
			"error": str(exc),
		}


def train_gradient_boosting_model(
	train_df: pd.DataFrame, test_df: pd.DataFrame
) -> dict[str, object]:
	try:
		feature_cols = [
			"reddit_post_count",
			"engagement_per_post",
			"post_velocity",
			"engagement_volatility",
			"comment_view_ratio",
			"reddit_post_count_lag1",
			"reddit_post_count_lag2",
		]
		X_train = train_df[feature_cols].fillna(0.0)
		y_train = train_df["google_interest"]
		X_test = test_df[feature_cols].fillna(0.0)
		y_test = test_df["google_interest"]

		scaler = StandardScaler()
		X_train_scaled = scaler.fit_transform(X_train)
		X_test_scaled = scaler.transform(X_test)

		model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
		model.fit(X_train_scaled, y_train)
		predictions = model.predict(X_test_scaled)

		mae = mean_absolute_error(y_test, predictions)
		rmse = np.sqrt(mean_squared_error(y_test, predictions))
		r2 = r2_score(y_test, predictions)

		feature_importance = dict(zip(feature_cols, model.feature_importances_))

		return {
			"model_type": "GradientBoosting",
			"mae": mae,
			"rmse": rmse,
			"r2": r2,
			"predictions": predictions,
			"feature_importance": feature_importance,
		}
	except Exception as exc:
		return {
			"model_type": "GradientBoosting",
			"mae": np.nan,
			"rmse": np.nan,
			"r2": np.nan,
			"predictions": np.full(len(test_df), np.nan),
			"error": str(exc),
		}


def compute_model_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
	mae = mean_absolute_error(actual, predicted)
	rmse = np.sqrt(mean_squared_error(actual, predicted))
	r2 = r2_score(actual, predicted)
	mape = np.mean(np.abs((actual - predicted) / (actual + 1))) * 100
	directional_accuracy = np.mean((np.diff(actual) * np.diff(predicted)) > 0) * 100

	return {
		"mae": mae,
		"rmse": rmse,
		"r2": r2,
		"mape": mape,
		"directional_accuracy": directional_accuracy,
	}


def run_hypothesis_tests(
	tlcc_df: pd.DataFrame, modeling_results: pd.DataFrame
) -> pd.DataFrame:
	hypothesis_results: list[dict[str, object]] = []

	# H1 (RQ1): Model R² > 0.3
	avg_r2 = modeling_results["gb_r2"].mean()
	h1_pass = avg_r2 > 0.3
	hypothesis_results.append(
		{
			"hypothesis": "H1 (RQ1): Reddit signals predictive",
			"metric": "avg_r2",
			"value": float(avg_r2),
			"threshold": 0.3,
			"pass": h1_pass,
			"interpretation": "Reddit features have predictive power for Google Trends" if h1_pass else "Weak predictive power",
		}
	)

	# H2 (RQ2): Average lag < 0 weeks
	valid_lags = tlcc_df[tlcc_df["status"] == "ok"]["optimal_lag_weeks"].dropna()
	if not valid_lags.empty:
		avg_lag = valid_lags.mean()
		lag_std_err = valid_lags.std() / np.sqrt(len(valid_lags))
		t_stat = avg_lag / (lag_std_err + 1e-8)
		lag_pvalue = 2 * (1 - t_dist.cdf(abs(t_stat), len(valid_lags) - 1))
		h2_pass = avg_lag < 0 and lag_pvalue < 0.05
	else:
		avg_lag, lag_std_err, lag_pvalue, h2_pass = np.nan, np.nan, np.nan, False

	hypothesis_results.append(
		{
			"hypothesis": "H2 (RQ2): Reddit leads Google (avg_lag < 0)",
			"metric": "avg_lag_weeks",
			"value": float(avg_lag),
			"threshold": 0.0,
			"p_value": float(lag_pvalue),
			"pass": h2_pass,
			"interpretation": "Reddit signals lead Google Trends" if h2_pass else "No significant lead-time advantage",
		}
	)

	# H3 (RQ3): Feature importance variance > 0.2
	feature_imp_values = []
	for feature_dict in modeling_results["feature_importance"]:
		if isinstance(feature_dict, dict):
			feature_imp_values.extend(feature_dict.values())

	if feature_imp_values:
		feat_importance_var = np.var(feature_imp_values)
		h3_pass = feat_importance_var > 0.02
	else:
		feat_importance_var, h3_pass = np.nan, False

	hypothesis_results.append(
		{
			"hypothesis": "H3 (RQ3): Features differ in importance",
			"metric": "feature_importance_variance",
			"value": float(feat_importance_var),
			"threshold": 0.02,
			"pass": h3_pass,
			"interpretation": "Feature contributions vary significantly" if h3_pass else "Homogeneous feature importance",
		}
	)

	# H4 (RQ4): Keyword type effect (niche vs broad)
	niche_keywords = [kw for kw in tlcc_df["keyword"] if any(term in kw.lower() for term in ["ultralight", "minimalist", "thru", "solo"])]
	broad_keywords = [kw for kw in tlcc_df["keyword"] if kw not in niche_keywords]

	niche_corrs = tlcc_df[tlcc_df["keyword"].isin(niche_keywords)]["max_correlation"].dropna()
	broad_corrs = tlcc_df[tlcc_df["keyword"].isin(broad_keywords)]["max_correlation"].dropna()

	if len(niche_corrs) > 1 and len(broad_corrs) > 1:
		t_stat_rq4, pvalue_rq4 = ttest_rel(niche_corrs.values, broad_corrs.values[:len(niche_corrs)])
		effect_size = abs(niche_corrs.mean() - broad_corrs.mean())
		h4_pass = effect_size > 0.15 and pvalue_rq4 < 0.05
	else:
		effect_size, pvalue_rq4, h4_pass = np.nan, np.nan, False

	hypothesis_results.append(
		{
			"hypothesis": "H4 (RQ4): Keyword type affects performance",
			"metric": "correlation_effect_size",
			"value": float(effect_size),
			"threshold": 0.15,
			"p_value": float(pvalue_rq4),
			"pass": h4_pass,
			"interpretation": "Niche keywords show different lead-lag patterns" if h4_pass else "No keyword-type effect",
		}
	)

	return pd.DataFrame(hypothesis_results)


def main() -> None:
	try:
		df = load_and_prepare_data(INPUT_FILE)
		tlcc_df = pd.read_csv(TLCC_FILE)
	except Exception as exc:
		print(f"ERROR: {exc}", file=sys.stderr)
		sys.exit(1)

	modeling_results_list: list[dict[str, object]] = []

	print("=" * 80)
	print("PHASE 4: ADVANCED MODELING & HYPOTHESIS TESTING")
	print("=" * 80)
	print(f"Total rows: {len(df):,}")
	print(f"Keywords to model: {df['keyword'].nunique()}")

	for keyword, keyword_df in df.groupby("keyword", sort=True):
		if len(keyword_df) < MIN_WEEKS_FOR_MODEL:
			print(f"⊘ {keyword}: insufficient data ({len(keyword_df)} < {MIN_WEEKS_FOR_MODEL} required)")
			continue

		# Engineer features
		featured = engineer_reddit_features(keyword_df)

		# Split data
		train, test = split_train_test(featured)

		if len(test) < 2:
			print(f"⊘ {keyword}: insufficient test set")
			continue

		# Train models
		arima_result = train_arima_model(train, test)
		gb_result = train_gradient_boosting_model(train, test)

		# Compute metrics
		arima_metrics = (
			compute_model_metrics(test["google_interest"].values, arima_result.get("predictions", []))
			if not np.isnan(arima_result.get("r2", np.nan))
			else {k: np.nan for k in ["mae", "rmse", "r2", "mape", "directional_accuracy"]}
		)
		gb_metrics = (
			compute_model_metrics(test["google_interest"].values, gb_result.get("predictions", []))
			if not np.isnan(gb_result.get("r2", np.nan))
			else {k: np.nan for k in ["mae", "rmse", "r2", "mape", "directional_accuracy"]}
		)

		modeling_results_list.append(
			{
				"keyword": keyword,
				"n_train": len(train),
				"n_test": len(test),
				"arima_mae": arima_metrics.get("mae", np.nan),
				"arima_rmse": arima_metrics.get("rmse", np.nan),
				"arima_r2": arima_metrics.get("r2", np.nan),
				"gb_mae": gb_metrics.get("mae", np.nan),
				"gb_rmse": gb_metrics.get("rmse", np.nan),
				"gb_r2": gb_metrics.get("r2", np.nan),
				"gb_directional_accuracy": gb_metrics.get("directional_accuracy", np.nan),
				"feature_importance": gb_result.get("feature_importance", {}),
			}
		)

		best_r2 = max(arima_metrics.get("r2", -np.inf), gb_metrics.get("r2", -np.inf))
		status = "✓" if best_r2 > 0.3 else "·"
		print(f"{status} {keyword}: ARIMA_R²={arima_metrics.get('r2', np.nan):.3f}, GB_R²={gb_metrics.get('r2', np.nan):.3f}")

	modeling_df = pd.DataFrame(modeling_results_list)

	# Run hypothesis tests
	hypothesis_df = run_hypothesis_tests(tlcc_df, modeling_df)

	# Save outputs
	(PROJECT_ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)
	modeling_df.to_csv(OUTPUT_MODELING_FILE, index=False)
	hypothesis_df.to_csv(OUTPUT_HYPOTHESIS_FILE, index=False)

	print("\n" + "=" * 80)
	print("HYPOTHESIS TEST SUMMARY")
	print("=" * 80)
	for idx, row in hypothesis_df.iterrows():
		status = "✓ PASS" if row["pass"] else "✗ FAIL"
		print(f"{status} | {row['hypothesis']}")
		print(f"       Value: {row['value']:.4f} | Threshold: {row['threshold']:.4f}")
		if "p_value" in row and not pd.isna(row["p_value"]):
			print(f"       P-value: {row['p_value']:.4f}")
		print(f"       → {row['interpretation']}\n")

	print(f"Saved modeling results: {OUTPUT_MODELING_FILE}")
	print(f"Saved hypothesis tests: {OUTPUT_HYPOTHESIS_FILE}")


if __name__ == "__main__":
	main()
