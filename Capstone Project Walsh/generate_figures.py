"""
generate_figures.py
Generates all 4 report visualizations + APA-style architecture diagram
and saves them to 'results vizz/'
"""

import os
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# ── APA Theme (white background, muted professional palette) ──────────────
BG      = '#FFFFFF'
SURFACE = '#FFFFFF'
BORDER  = '#D0D7DE'
TEXT    = '#111111'
MUTED   = '#4A6278'
NAVY    = '#1B3A5C'
SLATE   = '#4A6278'
GRAY    = '#8B949E'

def style_fig(fig):
    fig.patch.set_facecolor(BG)

def style_ax(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=TEXT, labelsize=8, direction='out', length=3)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
        spine.set_linewidth(0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if title:  ax.set_title(title, fontsize=10, fontweight='bold', pad=10, color=TEXT)
    if xlabel: ax.set_xlabel(xlabel, fontsize=8)
    if ylabel: ax.set_ylabel(ylabel, fontsize=8)

BASE   = '/Users/macpromacpro/Documents/Capstone Project Walsh'
OUTDIR = os.path.join(BASE, 'results vizz')
os.makedirs(OUTDIR, exist_ok=True)

# ── Load CSVs ──────────────────────────────────────────────────────────────
tlcc = pd.read_csv(f'{BASE}/data/processed/model_results.csv')
modl = pd.read_csv(f'{BASE}/data/processed/modeling_results.csv')
gtrends = pd.read_csv(f'{BASE}/data/processed/google_trends_raw.csv')

print("TLCC columns:", tlcc.columns.tolist())
print("MODL columns:", modl.columns.tolist())
print("GTRENDS columns:", gtrends.columns.tolist())

# Normalise column names to lowercase/underscored
tlcc.columns  = [c.strip().lower().replace(' ', '_') for c in tlcc.columns]
modl.columns  = [c.strip().lower().replace(' ', '_') for c in modl.columns]
gtrends.columns = [c.strip().lower().replace(' ', '_') for c in gtrends.columns]

print("\nAfter normalisation:")
print("TLCC cols:", tlcc.columns.tolist())
print("MODL cols:", modl.columns.tolist())

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE X1 — TLCC Optimal Lag Distribution
# ══════════════════════════════════════════════════════════════════════════════
lag_col  = [c for c in tlcc.columns if 'lag' in c][0]
kw_col   = [c for c in tlcc.columns if 'keyword' in c][0]
stat_col = [c for c in tlcc.columns if 'status' in c] 

df1 = tlcc.copy()
if stat_col:
    df1 = df1[df1[stat_col[0]] == 'ok']

df1[lag_col] = pd.to_numeric(df1[lag_col], errors='coerce')
df1 = df1.dropna(subset=[lag_col])
lags = df1[lag_col].astype(int)

fig, ax = plt.subplots(figsize=(10, 5))
style_fig(fig)
style_ax(ax,
         title='Figure X1. Keyword-Wise Optimal Lag Distribution (TLCC)\nNegative = Reddit leads Google Trends',
         xlabel='Optimal Lag (weeks)',
         ylabel='Number of Keywords')

counts = lags.value_counts().sort_index()
colors_bar = [SLATE if v < 0 else (GRAY if v > 0 else '#C9D1D9') for v in counts.index]
bars = ax.bar(counts.index, counts.values, color=colors_bar, edgecolor='#AEB6BF', width=0.7)

ax.axvline(0, color=GRAY, linestyle='--', linewidth=1)
ax.set_xticks(range(int(lags.min()), int(lags.max()) + 1))

legend_items = [
    mpatches.Patch(color=SLATE,    label='Reddit leads (lag < 0)'),
    mpatches.Patch(color='#C9D1D9', label='Synchronous (lag = 0)'),
    mpatches.Patch(color=GRAY,     label='Reddit lags (lag > 0)'),
]
ax.legend(handles=legend_items, facecolor='white', edgecolor=BORDER,
          labelcolor=TEXT, fontsize=8, framealpha=1)

for bar in bars:
    h = bar.get_height()
    if h > 0:
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.05, str(int(h)),
                ha='center', va='bottom', color=TEXT, fontsize=8)

fig.tight_layout(pad=1.5)
out1 = os.path.join(OUTDIR, 'figure_x1_tlcc_lag_distribution.png')
fig.savefig(out1, dpi=300, bbox_inches='tight', facecolor=BG, edgecolor='none')
plt.close(fig)
print(f"Saved: {out1}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE X2 — ARIMA vs Gradient Boosting R² by Keyword
# ══════════════════════════════════════════════════════════════════════════════
kw2  = [c for c in modl.columns if 'keyword' in c][0]
ar2  = [c for c in modl.columns if 'arima' in c and 'r2' in c][0]
gb2  = [c for c in modl.columns if ('gb' in c or 'gradient' in c or 'boost' in c) and 'r2' in c][0]

df2 = modl[[kw2, ar2, gb2]].copy()
df2[ar2] = pd.to_numeric(df2[ar2], errors='coerce')
df2[gb2] = pd.to_numeric(df2[gb2], errors='coerce')
df2 = df2.dropna()

# Clamp extreme outliers for display
CLAMP = 1.5
df2[ar2] = df2[ar2].clip(-CLAMP, CLAMP)
df2[gb2] = df2[gb2].clip(-CLAMP, CLAMP)

df2 = df2.sort_values(gb2, ascending=True)

fig, ax = plt.subplots(figsize=(12, 7))
style_fig(fig)
style_ax(ax,
         title='Figure X2. ARIMA vs Gradient Boosting R² by Keyword (clamped at ±1.5)',
         xlabel='R² (held-out test set)',
         ylabel='Keyword')

y = np.arange(len(df2))
h = 0.35
ax.barh(y + h/2, df2[ar2].to_numpy(), height=h, color=SLATE, label='ARIMA(1,1,1)', edgecolor=BORDER, linewidth=0.6)
ax.barh(y - h/2, df2[gb2].to_numpy(), height=h, color=NAVY, label='Gradient Boosting', edgecolor=BORDER, linewidth=0.6)
ax.axvline(0, color=GRAY, linestyle='--', linewidth=1)

ax.set_yticks(y)
ax.set_yticklabels(df2[kw2].to_numpy(), fontsize=7, color=TEXT)
ax.legend(facecolor='white', edgecolor=BORDER, labelcolor=TEXT, fontsize=8, framealpha=1)

fig.tight_layout(pad=1.5)
out2 = os.path.join(OUTDIR, 'figure_x2_arima_vs_gb_r2.png')
fig.savefig(out2, dpi=300, bbox_inches='tight', facecolor=BG, edgecolor='none')
plt.close(fig)
print(f"Saved: {out2}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE X3 — Aggregated Feature Importance Profile
# ══════════════════════════════════════════════════════════════════════════════
# Parse "feature_name (score)" from the top-feature column
feat_col = [c for c in modl.columns if 'feature' in c or 'top' in c]
print("Feature col candidates:", feat_col)

# Build importance tally from serialized per-keyword importance dictionaries
feature_totals = {}
feature_occurrences = {}
top_feature_counts = {}

if feat_col:
    fc = feat_col[0]
    for raw_val in modl[fc].dropna():
        raw_text = str(raw_val)
        pair_matches = re.findall(r"'([^']+)':\s*np\.float64\(([0-9eE+\-.]+)\)", raw_text)
        if not pair_matches:
            pair_matches = re.findall(r"'([^']+)':\s*([0-9eE+\-.]+)", raw_text)

        if not pair_matches:
            continue

        local_items = []
        for feat_name, score_text in pair_matches:
            score = float(score_text)
            feature_totals[feat_name] = feature_totals.get(feat_name, 0.0) + score
            feature_occurrences[feat_name] = feature_occurrences.get(feat_name, 0) + 1
            local_items.append((feat_name, score))

        local_top_feature, _ = max(local_items, key=lambda x: x[1])
        top_feature_counts[local_top_feature] = top_feature_counts.get(local_top_feature, 0) + 1

if feature_totals:
    avg_imp = {
        feat_name: feature_totals[feat_name] / feature_occurrences[feat_name]
        for feat_name in feature_totals
    }
    df3 = pd.DataFrame({
        'feature': list(avg_imp.keys()),
        'avg_importance': list(avg_imp.values())
    })
    df3['top_count'] = df3['feature'].map(lambda f: top_feature_counts.get(f, 0))
    df3 = df3.sort_values('avg_importance', ascending=True)
else:
    fallback_features = [
        'reddit_post_count', 'engagement_per_post', 'post_velocity',
        'engagement_volatility', 'comment_view_ratio',
        'reddit_post_count_lag1', 'reddit_post_count_lag2'
    ]
    df3 = pd.DataFrame({
        'feature': fallback_features,
        'avg_importance': [0.0] * len(fallback_features),
        'top_count': [0] * len(fallback_features)
    })

feat_colors = [NAVY, SLATE, '#6E859C', '#8599AD', '#9CADBE', '#B6C2CD', '#CDD5DD']

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
style_fig(fig)

# Left: avg importance
ax = axes[0]
style_ax(ax, title='Avg Importance Score\n(when ranked #1 for keyword)',
         xlabel='Average Importance Score', ylabel='Feature')
colors_used = [feat_colors[i % len(feat_colors)] for i in range(len(df3))]
ax.barh(df3['feature'], df3['avg_importance'], color=colors_used, edgecolor=BORDER)

# Right: count of keywords where this feature ranked #1
df3b = df3.sort_values('top_count', ascending=True)
ax2 = axes[1]
style_ax(ax2, title='# Keywords Where Feature\nRanked as Top Predictor',
         xlabel='Count of Keywords', ylabel='')
ax2.barh(df3b['feature'], df3b['top_count'], color=colors_used, edgecolor=BORDER)
ax2.tick_params(axis='y', labelsize=8, colors=TEXT)
axes[0].tick_params(axis='y', labelsize=8, colors=TEXT)

fig.suptitle('Figure X3. Feature Importance Profile — Gradient Boosting',
             color=TEXT, fontsize=11, fontweight='bold', y=1.01)
fig.tight_layout(pad=1.5)
out3 = os.path.join(OUTDIR, 'figure_x3_feature_importance.png')
fig.savefig(out3, dpi=300, bbox_inches='tight', facecolor=BG, edgecolor='none')
plt.close(fig)
print(f"Saved: {out3}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE X4 — Predicted vs Actual (4 representative keywords, GB model)
# ══════════════════════════════════════════════════════════════════════════════
from sklearn.ensemble import GradientBoostingRegressor

kw_col_g  = [c for c in gtrends.columns if 'keyword' in c][0]
date_col  = [c for c in gtrends.columns if 'date' in c][0]
gi_col    = [c for c in gtrends.columns if 'google' in c or 'interest' in c][0]
rc_col    = [c for c in gtrends.columns if 'reddit_post_count' == c]
vc_col    = [c for c in gtrends.columns if 'view_count' == c]
cc_col    = [c for c in gtrends.columns if 'comment_count' == c]

print("\ngtrends cols:", gtrends.columns.tolist())

REPR_KWS = ['sleeping bag', 'Osprey', 'hiking backpack', 'camping tent']

fig, axes = plt.subplots(2, 2, figsize=(13, 8))
style_fig(fig)
axes_flat = axes.flatten()

def build_features(df):
    df = df.copy().sort_values(date_col).reset_index(drop=True)
    for c in ['reddit_post_count', 'view_count', 'comment_count']:
        if c not in df.columns:
            df[c] = 0
    df['reddit_post_count']  = pd.to_numeric(df['reddit_post_count'], errors='coerce').fillna(0)
    df['view_count']         = pd.to_numeric(df['view_count'], errors='coerce').fillna(0)
    df['comment_count']      = pd.to_numeric(df['comment_count'], errors='coerce').fillna(0)
    df[gi_col]               = pd.to_numeric(df[gi_col], errors='coerce')

    df['engagement_per_post']  = (df['view_count'] + df['comment_count']) / (df['reddit_post_count'] + 1)
    df['post_velocity']        = df['reddit_post_count'].diff().fillna(0)
    df['engagement_volatility']= df['engagement_per_post'].rolling(3, min_periods=1).std().fillna(0)
    df['comment_view_ratio']   = df['comment_count'] / (df['view_count'] + 1)
    df['rc_lag1']              = df['reddit_post_count'].shift(1).fillna(0)
    df['rc_lag2']              = df['reddit_post_count'].shift(2).fillna(0)
    return df.dropna(subset=[gi_col])

FEATURES = ['reddit_post_count', 'engagement_per_post', 'post_velocity',
            'engagement_volatility', 'comment_view_ratio', 'rc_lag1', 'rc_lag2']

for i, kw in enumerate(REPR_KWS):
    ax = axes_flat[i]
    style_ax(ax, title=kw.title(), xlabel='Week index', ylabel='Google Interest')

    sub = gtrends[gtrends[kw_col_g].str.lower() == kw.lower()].copy()
    if len(sub) < 8:
        ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes,
                ha='center', va='center', color=MUTED, fontsize=9)
        continue

    sub = build_features(sub)
    X = sub[FEATURES].values
    y = sub[gi_col].values
    split = int(len(sub) * 0.7)
    if split < 4 or len(sub) - split < 2:
        ax.text(0.5, 0.5, 'Split too small', transform=ax.transAxes,
                ha='center', va='center', color=MUTED, fontsize=9)
        continue

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = np.asarray(y[:split], dtype=float), np.asarray(y[split:], dtype=float)

    gb = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
    gb.fit(X_train, y_train)
    y_pred = gb.predict(X_test)

    x_all   = np.arange(len(y))
    x_test  = np.arange(split, len(y))

    ax.plot(x_all,  y,      color=SLATE, linewidth=1.5, label='Actual')
    ax.plot(x_test, y_pred, color=NAVY, linewidth=1.5, linestyle='--', label='GB Predicted')
    ax.axvline(split, color=GRAY, linestyle=':', linewidth=1)
    ax.text(split + 0.2, ax.get_ylim()[1] * 0.95, 'test→', color=GRAY, fontsize=7)
    ax.legend(facecolor='white', edgecolor=BORDER, labelcolor=TEXT, fontsize=7, framealpha=1)

fig.suptitle('Figure X4. Predicted vs Actual Google Interest — Gradient Boosting\n(dashed vertical = train/test split)',
             color=TEXT, fontsize=11, fontweight='bold', y=1.01)
fig.tight_layout(pad=1.5)
out4 = os.path.join(OUTDIR, 'figure_x4_predicted_vs_actual.png')
fig.savefig(out4, dpi=300, bbox_inches='tight', facecolor=BG, edgecolor='none')
plt.close(fig)
print(f"Saved: {out4}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — APA 7 Architecture Diagram (white background, publication quality)
# ══════════════════════════════════════════════════════════════════════════════
NAVY_APA  = '#1B3A5C'
SLATE_APA = '#4A6278'
DARK_APA  = '#111111'
FONT_APA  = 'DejaVu Sans'

BOX_W_A = 1.6
BOX_H_A = 1.1
BOX_Y_A = 1.95

stages_apa = [
    ("Data\nIngestion",        "Reddit API\nExtracts"),
    ("Data\nPreprocessing",    "Dedup &\nNormalise"),
    ("Google Trends\nExtract", "pytrends\nRetry/Backoff"),
    ("Weekly\nAlignment",      "Keyword-Week\nIndex"),
    ("TLCC\nAnalysis",         "Lag -8 to +8\nPearson r"),
    ("Feature\nEngineering",   "7 Reddit\nFeatures"),
    ("Model\nDevelopment",     "ARIMA &\nGrad. Boost"),
    ("Hypothesis\nTesting",    "RQ1-RQ4\na = 0.05"),
    ("Reporting\nOutputs",     "CSV Artifacts\n& Report"),
]

n_a   = len(stages_apa)
gap_a = (18 - n_a * BOX_W_A) / (n_a + 1)
xs_a  = [gap_a + i * (BOX_W_A + gap_a) for i in range(n_a)]

fig_a, ax_a = plt.subplots(figsize=(18, 5))
fig_a.patch.set_facecolor('#FFFFFF')
ax_a.set_facecolor('#FFFFFF')
ax_a.set_xlim(0, 18)
ax_a.set_ylim(0, 5)
ax_a.axis('off')

for i, (title, sub) in enumerate(stages_apa):
    lx = xs_a[i]
    by = BOX_Y_A - BOX_H_A / 2

    # Navy header band
    header_a = mpatches.FancyBboxPatch(
        (lx, by + BOX_H_A * 0.55), BOX_W_A, BOX_H_A * 0.45,
        boxstyle="square,pad=0", linewidth=0,
        facecolor=NAVY_APA, zorder=3)
    ax_a.add_patch(header_a)

    # White body with navy outline
    body_a = mpatches.FancyBboxPatch(
        (lx, by), BOX_W_A, BOX_H_A,
        boxstyle="square,pad=0", linewidth=1.2,
        edgecolor=NAVY_APA, facecolor='white', zorder=2)
    ax_a.add_patch(body_a)

    # Header title text (white on navy)
    ax_a.text(lx + BOX_W_A / 2, by + BOX_H_A * 0.775,
              title, ha='center', va='center',
              fontsize=7.2, fontweight='bold',
              color='white', fontfamily=FONT_APA, zorder=4, linespacing=1.3)

    # Body sub-label (dark on white)
    ax_a.text(lx + BOX_W_A / 2, by + BOX_H_A * 0.27,
              sub, ha='center', va='center',
              fontsize=6.4, color=DARK_APA,
              fontfamily=FONT_APA, zorder=4, linespacing=1.3)

    # Step number below box
    ax_a.text(lx + BOX_W_A / 2, by - 0.22,
              f'Step {i + 1}', ha='center', va='top',
              fontsize=6, color=SLATE_APA, fontfamily=FONT_APA)

    # Arrow to next box
    if i < n_a - 1:
        ax_a.annotate('',
            xy=(xs_a[i + 1], BOX_Y_A),
            xytext=(lx + BOX_W_A, BOX_Y_A),
            arrowprops=dict(arrowstyle='->', color=NAVY_APA,
                            lw=1.3, mutation_scale=12),
            zorder=5)

# APA title and note
ax_a.text(9, 4.6,
          'Figure 1. End-to-End Workflow of the Emerging Trend Detection System',
          ha='center', fontsize=9.5, fontweight='bold',
          color=DARK_APA, fontfamily=FONT_APA)
ax_a.text(9, 4.18,
          'Note. Pipeline flows left to right through nine sequential stages from raw Reddit extraction to final reporting outputs.',
          ha='center', fontsize=7.5, color=SLATE_APA,
          fontfamily=FONT_APA, style='italic')

fig_a.tight_layout(pad=0.3)
out_arch = os.path.join(OUTDIR, 'architecture_diagram_apa.png')
fig_a.savefig(out_arch, dpi=300, bbox_inches='tight',
              facecolor='white', edgecolor='none')
plt.close(fig_a)
print(f"Saved: {out_arch}")

print("\nAll figures saved to:", OUTDIR)
