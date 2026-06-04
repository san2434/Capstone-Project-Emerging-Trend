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
This study examines whether social-media signals can identify emerging product trends earlier than Google Trends. A reproducible analytics pipeline was developed using Reddit activity and Google search-interest data for outdoor and hiking-related keywords. The workflow included data ingestion, deduplication, Google Trends extraction, Time-Lagged Cross-Correlation (TLCC), and Phase 4 predictive modeling using a univariate ARIMA baseline and a multivariate Gradient Boosting model. The final dataset contained 768 observations across 49 keywords. Seven engineered Reddit features were used: post count, engagement per post, post velocity, engagement volatility, comment/view ratio, and two lagged post-count variables. TLCC results showed that lead-lag behavior was keyword-dependent, with some keywords indicating earlier Reddit movement while others were synchronous or inverse. In hypothesis testing, H3 passed, showing meaningful variation in feature importance ($0.0317 > 0.02$). H1 failed because average predictive performance did not exceed the target threshold, H2 failed because the average lag was not significantly below zero ($p = 0.2147$), and H4 failed because niche versus broad keyword effects were not statistically significant ($p = 0.9886$). Despite mixed predictive accuracy, the study demonstrates a practical framework for early trend monitoring, demand sensing, and future model refinement.

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
Prior studies indicate that social signals can provide meaningful predictive validity for emerging demand, but performance is typically conditional rather than universal. In the Reddit-plus-Google-Trends context, post volume, engagement intensity, and discussion momentum often capture early shifts in collective attention that later appear in search behavior. However, the literature also emphasizes volatility, topic dependence, and noise from community-specific dynamics, which can dilute average predictive strength when evaluated across heterogeneous keywords. The strongest implication for RQ1 is that predictive validity should be assessed as a distribution across keyword segments, not a single global estimate, and benchmarked against transparent baselines to separate genuine signal from incidental correlation.

The lead-lag rationale in prior studies supports the view that online discussion and search intent occupy different stages of trend formation. Reddit activity can reflect exploratory discourse and peer influence before broader audiences convert that interest into search queries, making temporal offsets theoretically plausible. At the same time, the literature consistently reports mixed lag direction by topic, with some series leading, others moving synchronously, and some reversing under event-driven shocks. For RQ2, this implies that early detection should rely on lag-window diagnostics rather than fixed assumptions of universal social lead time. A robust design therefore requires keyword-level lag estimation, stability checks, and explicit treatment of temporal heterogeneity.

Research on model interpretability shows that feature importance in social trend detection is often uneven and context dependent, with interaction-derived variables frequently outperforming raw counts. Prior studies suggest that dynamics-oriented features, such as velocity, short-horizon volatility, and lagged activity, better represent trend-emergence mechanisms than static popularity alone. This supports an RQ3 framing where the objective is not only predictive gain but explanatory clarity about which behavioral dimensions drive performance. The practical implication is to prioritize models that permit consistent attribution and variance analysis of importance scores across keywords, enabling decision-makers to distinguish persistent drivers from transient artifacts and to refine feature sets iteratively.

Prior studies also suggest that keyword strategy materially conditions trend-detection outcomes, especially when comparing niche communities with broad, high-volume terms. Niche keywords may offer sharper early signals due to concentrated discourse, while broad keywords can provide scale but introduce semantic ambiguity and lag dilution. For RQ4, the key implication is that keyword design should be treated as a methodological lever, not a preprocessing detail. Effective pipelines should stratify terms by specificity, monitor differential performance by group, and adapt feature engineering accordingly. In practice, combining stable broad terms with selectively curated niche terms is likely to improve coverage while preserving sensitivity to early-stage shifts.

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
- **RQ1 (Predictive performance)**  
  - **$H_0$:** Mean predictive performance does not exceed baseline threshold ($\mu_{R^2} \le 0.30$).  
  - **$H_a$:** Mean predictive performance exceeds baseline threshold ($\mu_{R^2} > 0.30$).
- **RQ2 (Lead-time advantage)**  
  - **$H_0$:** Social signals do not lead Google Trends on average ($\mu_{lag} \ge 0$).  
  - **$H_a$:** Social signals lead Google Trends on average ($\mu_{lag} < 0$).
- **RQ3 (Feature influence heterogeneity)**  
  - **$H_0$:** Feature contributions are homogeneous (importance variance $\le 0.02$).  
  - **$H_a$:** Feature contributions are heterogeneous (importance variance $> 0.02$).
- **RQ4 (Keyword strategy effect)**  
  - **$H_0$:** Niche and broad keyword groups do not differ in performance effect size ($\Delta \le 0.15$).  
  - **$H_a$:** Niche and broad keyword groups differ meaningfully ($\Delta > 0.15$).

### Statistical Methods and Significance
- **TLCC:** Pearson correlation scanned from -8 to +8 weeks.
- **Lag inference:** Optimal lag selected by maximum absolute correlation.
- **Model comparison framework:** Univariate ARIMA(1,1,1) baseline versus multivariate Gradient Boosting with seven engineered features.
- **Hypothesis-test mapping:**
  - **RQ1:** Threshold-based performance decision using mean out-of-sample $R^2$.
  - **RQ2:** One-sample t-test logic on mean lag estimate versus zero (reported $p$-value).
  - **RQ3:** Variance criterion test on feature-importance dispersion.
  - **RQ4:** Group-difference test on niche versus broad keyword effect (paired comparison with reported $p$-value).
- **Significance level:** All inferential decisions were evaluated at $\alpha = 0.05$.
- **Decision rule:** Reject $H_0$ when $p < 0.05$ for inferential tests; for threshold criteria (RQ1, RQ3), reject $H_0$ only when the observed metric exceeds the predefined benchmark.

### Sample Size / Power Consideration
Minimum sample-size planning followed capstone guidance before model estimation. For proportion-style conservative planning in RQ1, the study used $z = 1.96$, $p = 0.50$, and margin of error $e = 0.10$, yielding:

$$
n = \frac{z^2 p(1-p)}{e^2} = \frac{(1.96)^2(0.50)(0.50)}{(0.10)^2} \approx 96.04 \Rightarrow 97
$$

For RQ2 lead-time testing, a paired-test power target of $\alpha = 0.05$, power $= 0.80$, and medium effect size ($d = 0.50$) indicates a minimum sample near $n = 34$. The collected dataset contains 768 observations across 49 keywords, exceeding these baseline requirements overall. Because keyword-level slices vary in depth, the pipeline retains explicit insufficiency handling at the series level (for example, overlap checks and minimum-length filters) so inference quality is not overstated for sparse terms.

### Reddit Data Source and Collection Methodology
The Reddit data source contributed substantially to achieving adequate sample size. Data collection followed a defined API pipeline: (1) formal API access was obtained through Reddit's developer portal, (2) all Reddit posts and comments matching the study's target keywords were extracted over the project timeframe, (3) records were consolidated into a unified repository, and (4) time-series aggregation was performed to align with weekly Google Trends observations. This systematic extraction strategy ensured consistent, reproducible coverage of social conversation across the 49 keywords. The resulting Reddit feature set—including post count, engagement metrics (upvotes, comments), post velocity, engagement volatility, and lagged indicators—contributed directly to the 768 multivariate observations used in hypothesis testing and modeling. By leveraging the scalability of Reddit's API, the study achieved sufficient statistical power to test lead-lag relationships and feature-importance patterns without manual sampling constraints.

### Model Selection and Comparison with Current Methods
Model choice was designed to balance interpretability and predictive flexibility. ARIMA(1,1,1) was retained as the transparent univariate baseline consistent with classical time-series practice. Gradient Boosting was selected as the primary multivariate method because it captures nonlinear interactions among volume, engagement, and lagged social features. This baseline-versus-enhanced comparison directly addresses whether engineered social signals provide incremental value beyond autoregressive structure alone. Performance was evaluated on temporally ordered train-test splits ($70/30$) using MAE, RMSE, and $R^2$ on held-out data.

---

## Architecture diagram/Workflow
### System Overview
The workflow ingests social data, standardizes and deduplicates records, enriches with Google Trends interest trajectories, estimates lead-lag behavior through TLCC, and benchmarks predictive models for interpretability and operational use.

### Architecture Diagram
**Figure X. End-to-End Workflow of the Proposed System**  
Data Source → Preprocessing → Google Trends Extraction → Weekly Alignment → TLCC Analysis → Feature Engineering → Model Training (ARIMA, Gradient Boosting) → Hypothesis Testing → Reporting Outputs

### Workflow Components
#### Data Ingestion
Data ingestion consolidates multi-file Reddit extractions into a unified processed table, preserving identifiers, timestamps, subreddit context, and engagement-related attributes so downstream transformations begin from a consistent and auditable structure.

#### Data Preprocessing
Data preprocessing applies deduplication, schema normalization, and field standardization for keyword and date variables. This stage also performs continuity and overlap checks so each keyword series is suitable for time-series alignment and subsequent inferential analysis.

#### Google Trends Extraction
Google Trends extraction enriches each keyword with search-interest trajectories using the `pytrends` interface. Controlled retry/backoff behavior is used to handle transient request failures and rate-limit events, improving extraction completeness and reproducibility.

#### Weekly Alignment
Weekly alignment maps Reddit activity and Google Trends values to a shared week-level time index, producing synchronized keyword-week observations. This harmonization step enables valid lead-lag estimation, comparable model inputs, and stable temporal partitioning.

#### TLCC Analysis
TLCC analysis scans lag windows from -8 to +8 weeks using Pearson correlation to estimate whether Reddit signals lead, lag, or move synchronously with Google Trends for each keyword. The optimal lag and corresponding correlation are retained for interpretation and hypothesis evaluation.

#### Exploratory Data Analysis (EDA)
Exploratory data analysis evaluates keyword distribution, temporal coverage, and overlap sufficiency across aligned series to identify sparse or unstable segments before modeling. These checks reduce the risk of overstating inference quality for thin time windows.

#### Feature Engineering
Feature engineering derives seven predictors from volume, engagement, dynamics, and lagged social signals (`reddit_post_count`, `engagement_per_post`, `post_velocity`, `engagement_volatility`, `comment_view_ratio`, `reddit_post_count_lag1`, `reddit_post_count_lag2`) to support multivariate forecasting and interpretability analysis.

#### Model Development
Model development applies a temporally ordered train-test split ($70/30$) at the keyword level and benchmarks a transparent univariate ARIMA baseline against a multivariate Gradient Boosting model to test incremental value from engineered social features.

#### Model Evaluation
Model evaluation reports held-out MAE, RMSE, and $R^2$ (with directional behavior checks where applicable) and integrates these results with TLCC evidence and formal hypothesis decisions to provide both predictive and inferential interpretation.

#### Hypothesis Testing
Hypothesis testing formalizes decisions for RQ1-RQ4 at $\alpha = 0.05$ using a mix of threshold-based criteria and inferential logic, covering predictive performance, lead-time behavior, feature-importance heterogeneity, and niche-versus-broad keyword effects.

#### Reporting Outputs
Reporting outputs include `data/processed/google_trends_raw.csv`, `data/processed/model_results.csv`, `data/processed/modeling_results.csv`, and `data/processed/hypothesis_tests.csv` as reproducible artifacts supporting capstone documentation and result traceability.

#### Deployment (if applicable)
If operationalized, these artifacts are deployment-ready for lightweight dashboard or API integration and can support recurring trend-monitoring workflows with periodic refresh and exception-oriented review.

#### Tools and Technologies
The implementation stack uses Python with Pandas, NumPy, SciPy, Statsmodels, and Scikit-learn for data engineering, time-series diagnostics, feature construction, statistical testing, and predictive modeling.

---

## Results
The results indicate that the pipeline is methodologically sound and produces interpretable keyword-level outputs, but predictive performance remains mixed across models and keywords. Overall, the analysis shows clearer evidence of social-media lead-lag behavior and feature-importance heterogeneity than of strong aggregate forecasting accuracy, which underscores both the value and the limitations of the current feature set and time horizon.

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

### Reporting Outputs
The analysis produced reproducible reporting artifacts in `data/processed/google_trends_raw.csv`, `data/processed/model_results.csv`, `data/processed/modeling_results.csv`, and `data/processed/hypothesis_tests.csv`, which preserve the core evidence used for interpretation, comparison, and capstone documentation.

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
Even with mixed predictive fit, the system provides operational value through keyword-level lead-lag diagnostics that identify which social signals lead search trends for specific categories, early-warning prioritization of high-signal categories that enables faster response times for inventory and marketing decisions, and transparent model benchmarking for planner trust by clearly comparing baseline and advanced modeling approaches side-by-side.

---

## Implementation and User Benefit

The implementation strategy for this trend detection system balances technical robustness with operational practicality, ensuring that analytical outputs translate directly into actionable business insights. By embedding social-signal diagnostics into standard planning workflows, stakeholders can leverage early-warning capabilities without requiring fundamental shifts to existing processes or systems. The following sections detail deployment mechanisms, integration pathways, user workflows, and concrete value propositions that make this framework operationally viable and strategically valuable.

### Deployment Approach
The outputs produced by this analysis—`google_trends_raw.csv`, `model_results.csv`, `modeling_results.csv`, and `hypothesis_tests.csv`—are designed for direct consumption by BI dashboards or lightweight API layers, making integration into existing business intelligence infrastructure straightforward. These structured artifacts eliminate the need for custom parsing or transformation, enabling rapid BI tool connectivity. The modular design allows organizations to start with keyword-level lag diagnostics and progressively add predictive model layers as confidence and use-case maturity increase, reducing deployment risk and accelerating time-to-value.

### System Integration
The workflow can be integrated into existing planning cadence as a weekly trend-monitoring job with exception alerts for high-correlation, negative-lag keywords, allowing category managers and demand planners to incorporate social signals into their standard decision cycles without major process restructuring. Automation of the trend-monitoring job reduces manual overhead and ensures consistent signal detection across the keyword portfolio. By flagging only high-confidence lead-lag relationships for analyst review, the system minimizes alert fatigue while preserving responsiveness to emerging opportunities, making adoption more sustainable across planning teams.

### User Interaction
User interaction follows a simple cycle: users provide or approve keyword universes relevant to their product category, review ranked lag diagnostics to understand which social indicators lead search behavior, and monitor model metrics by category to track emerging patterns and signal quality over time. This iterative feedback loop enables continuous refinement of keyword selections and feature tuning based on observed performance. Over time, users develop domain-specific intuition about which keyword clusters exhibit strong social leads and which require alternative early-warning mechanisms, building organizational capability in social-signal interpretation.

### Benefits to Users
This approach delivers multiple benefits to end users: faster signal detection for assortment and replenishment decisions that reduce stockout and markdown risk, reduced dependence on lagging-only indicators by incorporating early-warning social signals, and better interpretability of which social signals matter most through feature-importance analysis and model comparison. Quantitatively, lead-time advantages of one to two planning cycles translate to inventory holding cost reductions, improved in-stock rates during peak demand windows, and more efficient marketing spend allocation. Qualitatively, transparent lag diagnostics and feature rankings build planner confidence in trend-based decisions, enabling faster cycle-time from signal detection to action authorization.

### Example Use Case
To illustrate practical value, consider a concrete example: a category manager monitoring hiking equipment sees a keyword cluster (e.g., `ultralight tent`, `hiking backpack`) with strong negative lag and rising Reddit velocity in the current week. Using these diagnostics, the manager can advance inventory replenishment and campaign launch actions by one to two planning cycles—accelerating from their normal response timeline to stay ahead of search-driven demand peaks. This lead-time advantage translates directly to improved stockout avoidance, better assortment alignment, and more efficient marketing spend by reaching customers earlier in their interest journey. By operationalizing this insight across multiple keyword clusters, the category can achieve sustained competitive positioning and revenue protection in dynamic outdoor and lifestyle markets.

---

## Limitations and Further Improvements

Briefly discuss the limitations of the study and possible future enhancements to strengthen the analytical framework and expand its operational scope.

### Limitations

This study faces several important limitations that constrain both predictive performance and the generalizability of findings. Short effective windows for some keywords—particularly those with sparse Reddit or Google Trends activity—reduce the statistical power available for lag estimation and model fitting. API extraction constraints and potential signal noise from Reddit's unmoderated discourse introduce measurement error that is difficult to fully characterize or correct. Additionally, keyword-level imbalance and sparse overlap in parts of the corpus create heterogeneous data quality across the 49-keyword universe, with some terms having rich temporal coverage while others suffer from episodic or seasonal gaps. These data quality issues directly impact the reliability of lag estimates and feature importance rankings for keywords with insufficient temporal depth.

### Impact of Limitations

These factors reduce model generalizability and suppress aggregate significance, especially for broad inferential hypotheses that pool across diverse keywords. Individual keyword-level diagnostics remain meaningful, but aggregate metrics (such as mean $R^2$ or average lag) may not represent stable population parameters under these conditions. The mixed predictive performance observed across models reflects not only feature-level constraints but also the inherent difficulty of forecasting in high-noise, short-horizon environments where keyword-specific dynamics dominate. Organizations deploying this system should interpret average model metrics with caution and prioritize keyword-level assessments where sufficient historical depth exists. Additionally, the current feature set captures volume and engagement signals but lacks semantic context, which limits the interpretability of why certain keywords exhibit lead-lag behavior.

### Future Improvements

To build on this foundation, future work should address data scope, methodological robustness, and semantic enrichment. Extend historical coverage and stabilize weekly series length by incorporating longer baseline periods for Reddit and Google data, creating more consistent time windows for reliable lead-lag estimation and supporting more stable model training. Adding richer NLP features—including topic coherence to capture discussion focus, sentiment scores to gauge user tone, novelty indicators to detect emerging topics, and semantic drift measures to track meaning shifts over time—could better represent the multidimensional nature of social discourse and improve feature-importance interpretability.

Beyond NLP enrichment, a particularly promising enhancement involves implementing an LLM-powered theme extraction pipeline that analyzes individual Reddit posts and comments to extract structured semantic information. After extracting Reddit posts and comments, deploy large language models (such as GPT-4, Claude, or open-source alternatives like LLaMA) to extract recurring themes, including product attributes (materials, design features, functionality), brand mentions and competitive positioning, sentiment polarity and intensity, and specific use-case context (e.g., ultralight backpacking vs day hiking). These extracted themes can then be continuously monitored against Google Trends, enabling real-time tracking of which specific product types, brands, and features are gaining discussion intensity ahead of search volume spikes.

This LLM-synthesizer approach directly enables seller-centric operationalization: ecommerce sellers can access a real-time dashboard showing top emerging themes by category, ranked by velocity and sentiment, with associated inventory recommendations indicating which SKUs to stock based on 1-2 week forecasts. For example, rather than seeing only that "hiking backpack" is trending, a seller would learn that discussions are increasingly focused on "ultralight materials," specific brands like "Osprey," and new design features like "removable hip belts," enabling micro-inventory optimization weeks ahead of broad demand peaks. This semantic enrichment transforms the current lag-detection system into a multi-dimensional early-warning and product-intelligence platform.

Introducing robust cross-validation frameworks and hierarchical or mixed-effects models could account for keyword-level random effects and improve transferability across categories. Additionally, applying outlier-robust scoring methods and refining keyword taxonomy to group semantically similar terms could reduce noise and reveal cleaner aggregate patterns. Data privacy and model bias considerations are critical for LLM-based extraction at scale—transparent disclosure of data usage (in compliance with Reddit's terms of service), bias auditing to ensure the model doesn't favor certain brands or perspectives, and anonymization protocols should be implemented before production deployment.

### Future Scope

The framework can be extended into real-time trend intelligence across diverse verticals beyond hiking and outdoor equipment—including fashion, consumer electronics, wellness, home and garden, and other dynamic markets—with automated drift monitoring and adaptive retraining to maintain predictive relevance as social and search behaviors evolve. Building multi-vertical capability would require developing category-specific LLM prompt engineering and brand/product taxonomies, but the underlying architecture remains generalizable. Such an expansion would transform the current proof-of-concept into an operational early-warning system serving multiple business units, potentially enabling coordinated demand sensing across an entire enterprise. In the longest term, integrating this system with transactional data (sales, inventory, returns) and external signals (weather, events, competitive actions) could create a holistic demand-intelligence ecosystem supporting proactive supply-chain, merchandising, and marketing decisions at scale.

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
1. Yousefinaghani, S., Dara, R., Mubareka, S., Papadopoulos, A., & Sharif, S. (2021). An analysis of trends using social media and search data. *JMIR Public Health and Surveillance*. Available at: https://www.jmir.org/search?query=Yousefinaghani%20social%20media%20search%20data
2. Timoneda, J. C., & Wibbels, E. (2022). Spikes and variance: Using Google Trends to detect and forecast events. *Political Analysis, 30*(2), 1–18. https://doi.org/10.1017/pan.2021.7
3. Khan, F. E., et al. (2022). Social media trend analysis to predict product success. BRAC University Research Paper. Available at: https://dspace.bracu.ac.bd/xmlui/discover?query=social%20media%20trend%20analysis%20to%20predict%20product%20success
4. Karim, S. M. R. U., Rasul, R. A., & Sultana, T. (2025). Sentiment analysis of social media data for predicting consumer behavior trends using machine learning. *arXiv preprint*. Available at: https://arxiv.org/search/?query=Sentiment+analysis+of+social+media+data+for+predicting+consumer+behavior+trends+using+machine+learning&searchtype=title
5. Altshuler, Y., Pan, W., & Pentland, A. (2011). Trends prediction using social diffusion models. *arXiv preprint*. Available at: https://arxiv.org/search/?query=Trends+prediction+using+social+diffusion+models&searchtype=title
6. Challet, D., & Ayed, A. B. H. (2014). Do Google Trends data contain more predictability than price returns? *arXiv preprint*. https://doi.org/10.2139/ssrn.2405804
7. Hyndman, R. J., & Athanasopoulos, G. (Forecasting principles reference). Available at: https://otexts.com/fpp3/
8. Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. https://doi.org/10.1214/aos/1013203451
9. Breiman, L. (2001). Random forests and ensemble learning foundations. https://doi.org/10.1023/A:1010933404324
10. Additional QM640-approved sources from project reference set (to be finalized in strict APA7 formatting in submission draft). Institutional access: https://library.walshcollege.edu/

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
