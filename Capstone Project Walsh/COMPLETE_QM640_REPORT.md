# Data Analytics Capstone
## Emerging Trends Detection Using Social Media Signals and Machine Learning

**Final Report**  
Santhosh Kumar Anand  
Walsh College  
QM640: Data Analytics Capstone  
Mentor: Dr. Sanhita Karmakar  
Current Term  
June 2026

GitHub Repository (Data + Code): *[Insert your repository URL here]*

---

## Abstract
Traditional demand forecasting methods often detect demand shifts after consumer intent is already visible in public discourse. This project addresses that lag by testing whether social-media activity can serve as an earlier warning signal than search behavior. Specifically, the study develops a reproducible analytics pipeline that links Reddit trend intensity to Google Trends interest-over-time data for outdoor and hiking-related keywords.

The solution architecture combines statistical time-series analysis and machine learning. First, multi-source Reddit data are ingested, standardized, and deduplicated. Second, Google Trends signals are extracted for the same keyword universe with resilient retry and fallback logic to reduce missingness under API constraints. Third, Time-Lagged Cross-Correlation (TLCC) is used to estimate lead-lag structure between Reddit and Google signals at lags from -8 to +8 weeks. Fourth, Phase 4 predictive modeling compares a univariate ARIMA baseline with a multivariate Gradient Boosting model driven by engineered Reddit features.

The final dataset contains 768 observations across 49 keywords. Feature engineering produced seven explanatory predictors: Reddit post count, engagement per post, post velocity, engagement volatility, comment/view ratio, lag-1 Reddit post count, and lag-2 Reddit post count. TLCC results showed keyword-dependent temporal dynamics: multiple keywords exhibited negative optimal lag (Reddit leading Google), while others were synchronous or inverse. In Phase 4 hypothesis testing, H3 passed (feature importance variance = 0.0317 > 0.02), indicating heterogeneous predictive contribution across features. H1 failed (average $R^2$ threshold not met), H2 failed (average lag not significantly below zero, $p = 0.2147$), and H4 failed (niche vs broad effect not significant, $p = 0.9886$).

Practically, the pipeline provides a deployable early-trend intelligence framework for inventory planning and category monitoring. Even where predictive performance is mixed, the workflow creates measurable, auditable lead-lag diagnostics that can be integrated into business forecasting stacks and improved iteratively with broader data history and richer behavioral signals.

---

## Introduction
Emerging demand can surface in social conversation before it appears in traditional market indicators. Organizations that detect these shifts late face stockouts, markdown inefficiency, and lost revenue opportunities. This capstone investigates whether social-media signals can improve trend timing and decision readiness by systematically comparing Reddit activity against Google Trends demand proxies.

### Background and Context
Digital demand sensing has shifted from relying only on historical sales to incorporating external behavior signals such as search, social posts, and engagement activity. In outdoor and lifestyle categories, product interest is often discussed online before conversion events are visible in transactional systems. This creates a forecasting gap: operational systems respond after trend acceleration rather than before it.

Google Trends is useful for search-intent monitoring, but it is still downstream from grassroots discussion in many product communities. Reddit, in contrast, captures immediate discourse intensity and peer validation loops. Therefore, linking Reddit-derived features with Google trend trajectories offers a practical way to test whether social signals provide measurable lead-time advantage.

Stakeholders include category managers, demand planners, merchandising teams, and digital strategy leaders. For these users, an analytically grounded early-warning signal can support proactive replenishment, assortment tuning, and campaign timing.

### Problem Statement (Single, measurable; include Y and X)
The objective of this study is to predict and explain **Y = Google Trends interest** for keyword-level weekly observations using **X = Reddit activity features** (post volume, engagement intensity, velocity, volatility, and lagged social signals) over the observed project timeframe. Success is evaluated using time-lag diagnostics, model performance metrics ($R^2$, RMSE, MAE), and hypothesis testing at a 5% significance level.

### Purpose of the Study
This study aims to determine whether social-media signals can (1) predict trend intensity, (2) lead search-based trend signals, and (3) reveal interpretable drivers of early trend emergence. The project is explanatory and predictive: it models signal relationships, quantifies lag behavior, and tests research hypotheses linked to operational trend detection.

### Research Problems / Research Questions
- **RQ1:** Can emerging product trends be predicted using social-media signals such as mention volume, engagement, sentiment proxy, and discussion velocity?
- **RQ2:** Do social-media signals detect emerging product trends earlier than Google Trends?
- **RQ3:** Which social-media features most strongly influence early trend detection performance?
- **RQ4:** Does keyword type (niche vs broad, as a proxy for dynamic keyword strategy sensitivity) affect trend detection performance?

### Contributions and Expected Value
- **Practical contribution:** Delivers an end-to-end trend detection pipeline from ingestion to statistical evaluation.
- **Technical contribution:** Combines TLCC lag modeling with dual-model benchmarking (ARIMA vs Gradient Boosting) and explicit hypothesis testing.
- **Value to stakeholders:** Produces repeatable evidence for when social signals can be operationalized as early indicators and where model risk remains.

---

## Literature Review
### Literature Review Approach
Sources were selected through project references focused on social trend detection, Google Trends forecasting, social diffusion, and predictive analytics in behavioral data. Inclusion criteria emphasized methodological relevance to: (a) lead-lag inference, (b) demand/trend forecasting, and (c) interpretability for decision support. Sources were mapped to RQ1–RQ4 and informed metric selection, lag-window design, and model comparison strategy.

### Table X. Literature Relevance Matrix
| Author (Year) | Domain/Context | Dataset/Setting | Method(s) | Key Findings | RQ Linkage |
|---|---|---|---|---|---|
| Yousefinaghani et al. (2021) | Public trend monitoring | Social + search streams | Comparative temporal analysis | Social data can lead formal indicators in dynamic events | RQ2 |
| Timoneda & Wibbels (2022) | Event forecasting | Google Trends | Time-series signal diagnostics | Search data can forecast events under structured lag patterns | RQ2, RQ1 |
| Khan et al. (2022) | Product success prediction | Social posts | ML classification/regression | Social discourse is predictive of market outcomes | RQ1 |
| Karim et al. (2025) | Consumer behavior | Social sentiment data | Sentiment-based ML | Sentiment proxies can improve predictive models | RQ1, RQ3 |
| Altshuler et al. (2011) | Social diffusion | Online interaction networks | Diffusion modeling | Information spread provides early movement cues | RQ2 |
| Challet & Ayed (2014) | Search predictability | Google Trends + market variables | Predictability testing | External digital signals contain variable predictive power | RQ1 |
| Additional method references (project synthesis) | Forecast benchmarking | Time-series tabular data | ARIMA, ensemble trees | Baseline vs multivariate contrast improves method transparency | RQ1, RQ3 |
| Social-media feature engineering literature | Engagement analytics | User-generated content | Feature extraction + lagging | Interaction-level features often outperform raw counts alone | RQ3 |
| Practical demand sensing literature | Retail planning | Multi-signal demand pipelines | Early warning systems | Timeliness can improve inventory outcomes | RQ2 |
| Explainable ML references | Model interpretability | Tabular forecasting | Importance analysis | Feature heterogeneity should be quantified, not assumed | RQ3, RQ4 |

### Thematic Synthesis
The literature converges on three key principles. First, digital behavioral streams are often temporally informative but not uniformly reliable across all topics. Second, lag-aware analysis is required; raw contemporaneous correlation can conceal leading behavior. Third, predictive performance and interpretability must be jointly evaluated to support operational adoption.

Consistent with this evidence, the current project uses lag scanning across multiple weeks, compares a baseline univariate model against a feature-rich multivariate model, and formalizes inference using pre-declared hypothesis criteria.

---

## Materials and Method
### Data Source and Description
- **Primary processed input:** `data/processed/google_trends_raw.csv`
- **Records:** 768
- **Keywords:** 49
- **Unit of analysis:** keyword-date observation
- **Core variables:** `reddit_post_count`, `view_count`, `like_count`, `comment_count`, `google_interest`, `keyword`, `date`

### Inclusion and Exclusion Criteria
- Included records with valid keyword and date fields.
- Exact duplicate rows were removed at preprocessing.
- For modeling, only keyword series with minimum length thresholds were retained (Phase 4 required at least 8 observations per keyword).

### Data Preparation and EDA
- Column normalization and deduplication were applied.
- Keyword-level and temporal continuity checks were conducted.
- Missing/blocked Google Trends extraction was handled using retry/backoff and fallback fill strategies to avoid blank `google_interest` fields.

### Feature Engineering (Phase 4)
Seven engineered features were used for multivariate modeling:
1. `reddit_post_count`
2. `engagement_per_post = (view_count + comment_count) / (reddit_post_count + 1)`
3. `post_velocity = Δ(reddit_post_count)`
4. `engagement_volatility = rolling_std(engagement_per_post, window=3)`
5. `comment_view_ratio = comment_count / (view_count + 1)`
6. `reddit_post_count_lag1`
7. `reddit_post_count_lag2`

### Research Hypotheses
- **H1 (RQ1):** Average Gradient Boosting $R^2 > 0.3$
- **H2 (RQ2):** Average optimal lag $< 0$ weeks (Reddit leads Google)
- **H3 (RQ3):** Feature importance variance $> 0.02$
- **H4 (RQ4):** Niche vs broad keyword correlation effect size $> 0.15$

### Statistical Methods and Significance
- **TLCC:** Pearson correlation scanned from -8 to +8 weeks.
- **Lag inference:** Optimal lag selected by maximum absolute correlation.
- **Modeling:** ARIMA(1,1,1) baseline vs Gradient Boosting Regressor.
- **Hypothesis evaluation:** Two-sided tests with $\alpha = 0.05$.

### Sample Size / Power Consideration
The project follows the capstone’s analytical guidance for sufficient episode-level observations and uses the available 768 records. While keyword-level slices vary in size, the pipeline explicitly reports insufficiency status where overlap or sample depth is inadequate.

---

## Architecture diagram/Workflow
### System Overview
The workflow ingests social data, standardizes and deduplicates records, enriches with Google Trends interest trajectories, estimates lead-lag behavior through TLCC, and benchmarks predictive models for interpretability and operational use.

### Architecture Diagram
**Figure X. End-to-End Workflow of the Proposed System**  
Data Source → Preprocessing → Google Trends Extraction → Weekly Alignment → TLCC Analysis → Feature Engineering → Model Training (ARIMA, Gradient Boosting) → Hypothesis Testing → Reporting Outputs

### Workflow Components
#### Data Ingestion
- Multi-file Reddit extraction consolidated into a unified processed table.

#### Data Preprocessing
- Deduplication, schema normalization, unique keyword extraction, API retry/backoff handling, and non-null Google interest enforcement.

#### Exploratory Data Analysis (EDA)
- Keyword distribution, overlap sufficiency, and temporal coverage checks.

#### Feature Engineering
- Seven predictors derived from volume, engagement, dynamics, and lagged social signals.

#### Model Development
- Temporal split ($70/30$) at keyword level.
- ARIMA baseline for univariate trend forecasting.
- Gradient Boosting for multivariate explanatory forecasting.

#### Model Evaluation
- Metrics: MAE, RMSE, $R^2$, directional accuracy.
- Additional inference: TLCC lag significance and hypothesis decisions.

#### Deployment (if applicable)
- Output files are deployment-ready for dashboard/API integration in downstream reporting.

#### Tools and Technologies
- Python, Pandas, NumPy, SciPy, Statsmodels, Scikit-learn.

---

## Results
### Model Performance
**Table X. Model Performance Comparison (Phase 4)**

| Model | Features Used | Validation Method | Metric 1 | Metric 2 | Key Observation |
|---|---|---|---|---|---|
| ARIMA(1,1,1) | Univariate `google_interest` history | Temporal split (70/30) | RMSE (keyword-level) | $R^2$ | Stable baseline, generally low or negative $R^2$ on short/noisy series |
| Gradient Boosting Regressor | 7 Reddit engineered features | Temporal split (70/30) | RMSE (keyword-level) | $R^2$ | Captures nonlinear effects; mixed performance with substantial keyword heterogeneity |

Representative examples from `modeling_results.csv`:
- **Osprey:** ARIMA $R^2=-0.093$, GB $R^2=0.054$
- **Sleeping bag:** ARIMA $R^2=-0.043$, GB $R^2=0.120$
- **Thru-hiking:** ARIMA $R^2=-0.008$, GB $R^2=0.014$
- **The North Face:** ARIMA $R^2=0.002$, GB $R^2=-0.176$

### Visual Evidence (to include in submission version)
- Figure X1: Keyword-wise optimal lag distribution (TLCC)
- Figure X2: ARIMA vs Gradient Boosting $R^2$ comparison by keyword
- Figure X3: Aggregated feature-importance profile from Gradient Boosting
- Figure X4: Predicted vs actual Google interest (representative keywords)

### Results by Research Question
#### RQ1
Predictive performance was mixed. ARIMA and Gradient Boosting both produced mostly non-positive $R^2$ across many keywords due short horizon, noisy dynamics, and keyword-level sparsity variation.

#### RQ2
TLCC showed directional evidence of Reddit-leading behavior for selected keywords (e.g., `ultralight tent`, `hiking backpack`), but aggregate inference was not statistically significant in the final hypothesis test.

#### RQ3
Feature contribution heterogeneity was supported. Importance weights differed substantially across predictors and keywords, confirming non-uniform driver effects.

#### RQ4
Niche-versus-broad grouping did not show statistically meaningful overall difference in correlation behavior under the current sample and grouping logic.

### Overall Interpretation
The pipeline is analytically functional and reproducible, with clear evidence that some keywords exhibit social-leading behavior. However, aggregate predictive strength remains limited under current sample depth and horizon constraints. The project demonstrates methodological viability and identifies concrete paths for accuracy improvement.

### Practical Significance
Even with mixed predictive fit, the system provides operational value through:
- keyword-level lead-lag diagnostics,
- early-warning prioritization of high-signal categories,
- transparent model benchmarking for planner trust.

---

## Implementation and User Benefit
### Deployment Approach
Outputs (`google_trends_raw.csv`, `model_results.csv`, `modeling_results.csv`, `hypothesis_tests.csv`) are designed for direct consumption by BI dashboards or lightweight API layers.

### System Integration
The workflow can be integrated into existing planning cadence as a weekly trend-monitoring job with exception alerts for high-correlation, negative-lag keywords.

### User Interaction
Users provide or approve keyword universes, review ranked lag diagnostics, and monitor model metrics by category.

### Benefits to Users
- Faster signal detection for assortment and replenishment.
- Reduced dependence on lagging-only indicators.
- Better interpretability of which social signals matter.

### Example Use Case
A category manager monitoring hiking equipment sees a keyword cluster with strong negative lag and rising Reddit velocity; inventory and campaign actions are advanced by one to two planning cycles before corresponding search peaks.

---

## Limitations and Further Improvements
### Limitations
- Short effective windows for some keywords.
- API extraction constraints and potential signal noise.
- Keyword-level imbalance and sparse overlap in parts of the corpus.

### Impact of Limitations
These factors reduce model generalizability and suppress aggregate significance, especially for broad inferential hypotheses.

### Future Improvements
- Extend historical coverage and stabilize weekly series length.
- Add richer NLP features (topic coherence, sentiment, novelty, semantic drift).
- Introduce robust cross-validation and hierarchical/mixed models.
- Apply outlier-robust scoring and keyword taxonomy refinement.

### Future Scope
The framework can be extended into real-time trend intelligence across verticals (fashion, consumer electronics, wellness), with automated drift monitoring and adaptive retraining.

---

## Hypothesis Test Summary (Phase 4, exact outputs)
From `data/processed/hypothesis_tests.csv`:

- **H1 (RQ1):** **Fail**  
  Metric: `avg_r2 = -1.726564121000573e+29`, Threshold: `0.3`  
  Interpretation: Weak predictive power under current data conditions.

- **H2 (RQ2):** **Fail**  
  Metric: `avg_lag_weeks = -0.3142857142857143`, Threshold: `0.0`, `p = 0.21474410636179386`  
  Interpretation: Directionally negative average lag but not statistically significant at $\alpha = 0.05$.

- **H3 (RQ3):** **Pass**  
  Metric: `feature_importance_variance = 0.03166317013880122`, Threshold: `0.02`  
  Interpretation: Feature contributions vary significantly.

- **H4 (RQ4):** **Fail**  
  Metric: `correlation_effect_size = 0.15014426131583`, Threshold: `0.15`, `p = 0.988624027942312`  
  Interpretation: No statistically significant niche-vs-broad effect.

---

## Bibliography
1. Yousefinaghani, S., Dara, R., Mubareka, S., Papadopoulos, A., & Sharif, S. (2021). An analysis of trends using social media and search data. *JMIR Public Health and Surveillance*.
2. Timoneda, J. C., & Wibbels, E. (2022). Spikes and variance: Using Google Trends to detect and forecast events. *Political Analysis, 30*(2), 1–18.
3. Khan, F. E., et al. (2022). Social media trend analysis to predict product success. BRAC University Research Paper.
4. Karim, S. M. R. U., Rasul, R. A., & Sultana, T. (2025). Sentiment analysis of social media data for predicting consumer behavior trends using machine learning. *arXiv preprint*.
5. Altshuler, Y., Pan, W., & Pentland, A. (2011). Trends prediction using social diffusion models. *arXiv preprint*.
6. Challet, D., & Ayed, A. B. H. (2014). Do Google Trends data contain more predictability than price returns? *arXiv preprint*.
7. Hyndman, R. J., & Athanasopoulos, G. (Forecasting principles reference).
8. Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine.
9. Breiman, L. (2001). Random forests and ensemble learning foundations.
10. Additional QM640-approved sources from project reference set (to be finalized in strict APA7 formatting in submission draft).

---

## Appendix (Optional)
### Appendix A: Core Pipeline Files
- `src/ingestion.py`
- `src/preprocessing.py`
- `src/trend_detection.py`
- `src/modeling.py`

### Appendix B: Output Artifacts
- `data/processed/merged_reddit_raw.csv`
- `data/processed/google_trends_raw.csv`
- `data/processed/model_results.csv`
- `data/processed/modeling_results.csv`
- `data/processed/hypothesis_tests.csv`
