# frontend/pages/4_AI_Insights.py
# ─────────────────────────────────────────────────────────────────────────────
#  ALPHA RETURN ENGINE — AI Insights Module
#  Six distinct ML lenses on the 814-fund universe:
#    1. Smart Fund Screener       — multi-factor ML scoring & ranking
#    2. Cluster Intelligence      — K-Means fund DNA mapping
#    3. Alpha Detector            — anomaly scoring via Isolation Forest
#    4. Return Forecast Engine    — gradient boosting 1yr return prediction
#    5. Efficiency Frontier       — risk-adjusted efficiency classification
#    6. Fund Manager Intelligence — manager alpha & consistency scoring
#
#  Fully offline — zero external API. All ML runs on the 814-fund database.
# ─────────────────────────────────────────────────────────────────────────────

import io
import os
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.ensemble import (
    GradientBoostingRegressor,
    IsolationForest,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# DESIGN SYSTEM
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500;700&display=swap');

:root {
  --bg-void:        #03050A;
  --bg-base:        #070B12;
  --bg-surface:     #0C1220;
  --bg-raised:      #101828;
  --bg-hover:       #141F30;
  --border-dim:     #1A2840;
  --border-mid:     #1E3050;
  --accent:         #00FFD1;
  --accent-dim:     #00FFD120;
  --accent-mid:     #00FFD160;
  --accent-glow:    0 0 20px #00FFD130, 0 0 60px #00FFD108;
  --danger:         #FF3860;
  --danger-dim:     #FF386018;
  --danger-mid:     #FF386060;
  --danger-glow:    0 0 20px #FF386030, 0 0 60px #FF386008;
  --warn:           #F5A623;
  --warn-dim:       #F5A62318;
  --gold:           #F5C842;
  --gold-dim:       #F5C84220;
  --safe:           #00C8A0;
  --safe-dim:       #00C8A018;
  --moderate:       #4895EF;
  --moderate-dim:   #4895EF18;
  --purple:         #9B72FF;
  --purple-dim:     #9B72FF18;
  --purple-glow:    0 0 20px #9B72FF30;
  --text-primary:   #E8F0FA;
  --text-secondary: #7A94B0;
  --text-dim:       #3D526A;
  --font-display:   'Bebas Neue', sans-serif;
  --font-body:      'Space Grotesk', sans-serif;
  --font-mono:      'JetBrains Mono', monospace;
  --radius-sm:      2px;
  --transition:     0.22s cubic-bezier(0.4, 0, 0.2, 1);
}

html, body, [class*="css"] {
  font-family: var(--font-body) !important;
  background-color: var(--bg-void) !important;
  color: var(--text-primary) !important;
}

body::before {
  content: '';
  position: fixed; inset: 0;
  background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.035'/%3E%3C/svg%3E");
  pointer-events: none; z-index: 0;
}

body::after {
  content: '';
  position: fixed; inset: 0;
  background-image: linear-gradient(var(--border-dim) 1px, transparent 1px),
                    linear-gradient(90deg, var(--border-dim) 1px, transparent 1px);
  background-size: 60px 60px;
  opacity: 0.25; pointer-events: none; z-index: 0;
}

::-webkit-scrollbar { width: 3px; height: 3px; }
::-webkit-scrollbar-track { background: var(--bg-void); }
::-webkit-scrollbar-thumb { background: var(--accent-mid); }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }

.block-container {
  padding: 2.5rem 3.5rem 5rem 3.5rem !important;
  max-width: 1500px !important;
  position: relative; z-index: 1;
}

#MainMenu, footer, header { visibility: hidden; }

/* ── HERO ── */
.hero-wrapper {
  position: relative; padding: 4rem 0 3rem 0;
  margin-bottom: 3rem; overflow: hidden;
}
.hero-wrapper::after {
  content: ''; position: absolute; bottom: 0; left: 0; right: 0;
  height: 1px;
  background: linear-gradient(90deg, var(--accent) 0%, var(--accent-mid) 25%, transparent 65%);
}
.hero-line {
  position: absolute; left: 0; top: 0; width: 3px; height: 100%;
  background: linear-gradient(180deg, var(--accent), transparent);
  box-shadow: var(--accent-glow);
}
@keyframes scan {
  0%   { transform: translateY(-100%); opacity: 0; }
  8%   { opacity: 1; }
  92%  { opacity: 1; }
  100% { transform: translateY(2000%); opacity: 0; }
}
.hero-scan {
  position: absolute; left: 0; right: 0; top: 0; height: 1px;
  background: linear-gradient(90deg, transparent, var(--accent-mid), transparent);
  animation: scan 7s linear infinite; pointer-events: none;
}
@keyframes pulse-dot {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.35; transform: scale(0.82); }
}
.hero-eyebrow {
  font-family: var(--font-mono); font-size: 0.62rem; font-weight: 500;
  letter-spacing: 0.32em; color: var(--accent); text-transform: uppercase;
  margin-bottom: 1rem; margin-left: 1.2rem;
  display: flex; align-items: center; gap: 0.8rem;
}
.hero-eyebrow .ai-dot {
  width: 6px; height: 6px; border-radius: 50%;
  background: var(--accent); box-shadow: var(--accent-glow);
  animation: pulse-dot 2s ease-in-out infinite; flex-shrink: 0;
}
.hero-title {
  font-family: var(--font-display); font-size: clamp(4rem, 8vw, 7.5rem);
  font-weight: 400; line-height: 0.92; letter-spacing: 0.04em;
  color: var(--text-primary); margin: 0 0 0 1rem; text-transform: uppercase;
}
.hero-title .accent-word { color: var(--accent); text-shadow: var(--accent-glow); }
.hero-subtitle {
  font-family: var(--font-mono); font-size: 0.74rem; color: var(--text-secondary);
  margin: 1.4rem 0 0 1.2rem; max-width: 540px; line-height: 1.9;
  border-left: 2px solid var(--border-mid); padding-left: 1rem;
}
.hero-bg-text {
  position: absolute; right: -1rem; top: 50%; transform: translateY(-50%);
  font-family: var(--font-display); font-size: clamp(8rem, 18vw, 16rem);
  font-weight: 400; color: transparent;
  -webkit-text-stroke: 1px var(--border-dim);
  pointer-events: none; user-select: none; letter-spacing: 0.06em; line-height: 1;
}

/* ── MODULE TABS ── */
.module-tabs {
  display: flex; gap: 0; margin-bottom: 2.5rem;
  border-bottom: 1px solid var(--border-dim); overflow-x: auto;
}
.module-tab {
  font-family: var(--font-mono); font-size: 0.6rem; letter-spacing: 0.2em;
  text-transform: uppercase; padding: 0.8rem 1.4rem;
  color: var(--text-dim); cursor: pointer; white-space: nowrap;
  border-bottom: 2px solid transparent; transition: all var(--transition);
  background: transparent; border-left: none; border-right: none; border-top: none;
}
.module-tab:hover { color: var(--text-secondary); background: var(--bg-surface); }
.module-tab.active {
  color: var(--accent); border-bottom-color: var(--accent);
  background: var(--accent-dim);
}

/* ── SECTION LABELS ── */
.section-label {
  font-family: var(--font-mono); font-size: 0.58rem; letter-spacing: 0.28em;
  color: var(--accent); text-transform: uppercase;
  margin-bottom: 1.2rem; display: flex; align-items: center; gap: 0.8rem;
}
.section-label .sl-num {
  font-size: 0.5rem; color: var(--text-dim);
  border: 1px solid var(--border-mid); padding: 0.1rem 0.35rem; letter-spacing: 0.1em;
}
.section-label::after {
  content: ''; flex: 1; height: 1px;
  background: linear-gradient(90deg, var(--border-mid), transparent); max-width: 180px;
}

/* ── METRIC GRID ── */
.metric-grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 1px; background: var(--border-dim); margin: 1.5rem 0; border: 1px solid var(--border-dim);
}
.metric-cell {
  background: var(--bg-surface); padding: 1.2rem 1.5rem;
  display: flex; flex-direction: column; gap: 0.4rem;
  transition: background var(--transition); position: relative; overflow: hidden;
}
.metric-cell::after {
  content: ''; position: absolute; bottom: 0; left: 0; height: 2px; width: 0;
  background: var(--accent); transition: width 0.4s ease;
}
.metric-cell:hover { background: var(--bg-raised); }
.metric-cell:hover::after { width: 100%; }
.metric-label { font-family: var(--font-mono); font-size: 0.56rem; letter-spacing: 0.22em; text-transform: uppercase; color: var(--text-dim); }
.metric-value { font-family: var(--font-display); font-size: 2rem; font-weight: 400; line-height: 1; color: var(--text-primary); letter-spacing: 0.02em; }
.metric-value.accent  { color: var(--accent);   text-shadow: 0 0 20px #00FFD150; }
.metric-value.warn    { color: var(--warn); }
.metric-value.danger  { color: var(--danger);  text-shadow: var(--danger-glow); }
.metric-value.safe    { color: var(--safe);    text-shadow: 0 0 20px #00C8A050; }
.metric-value.purple  { color: var(--purple);  text-shadow: var(--purple-glow); }
.metric-value.moderate{ color: var(--moderate);}

/* ── STATUS CARDS ── */
.status-card {
  border-radius: var(--radius-sm); padding: 1rem 1.3rem;
  font-family: var(--font-mono); font-size: 0.75rem;
  line-height: 1.7; margin: 1rem 0; border-left: 3px solid; position: relative; overflow: hidden;
}
.status-card::before {
  content: ''; position: absolute; inset: 0; opacity: 0.04;
  background: repeating-linear-gradient(0deg, transparent, transparent 2px, white 2px, white 3px);
  pointer-events: none;
}
.status-success { background: #051A14; border-color: var(--accent); color: var(--accent); }
.status-error   { background: #160810; border-color: var(--danger); color: #FF7090; }
.status-info    { background: #08101E; border-color: #4895EF; color: #7ABAEF; }

/* ── INSIGHT PANEL ── */
.insight-panel {
  background: var(--bg-surface); border: 1px solid var(--border-dim);
  border-left: 3px solid var(--accent); padding: 1.5rem 1.8rem;
  margin-bottom: 1rem; position: relative; overflow: hidden;
  transition: border-color var(--transition), box-shadow var(--transition);
}
.insight-panel::before {
  content: ''; position: absolute; top: 0; right: 0; width: 200px; height: 200px;
  background: radial-gradient(circle at top right, var(--accent-dim), transparent 70%);
  pointer-events: none;
}
.insight-panel:hover { box-shadow: 0 4px 40px #00000040; }
.insight-panel.purple  { border-left-color: var(--purple); }
.insight-panel.purple::before  { background: radial-gradient(circle at top right, var(--purple-dim), transparent 70%); }
.insight-panel.warn    { border-left-color: var(--warn); }
.insight-panel.safe    { border-left-color: var(--safe); }
.insight-panel.danger  { border-left-color: var(--danger); }
.insight-panel h5 {
  font-family: var(--font-body); font-size: 0.95rem; font-weight: 600;
  color: var(--text-primary); margin: 0 0 0.8rem 0;
}
.insight-panel p {
  font-family: var(--font-body) !important; font-size: 0.8rem !important;
  color: var(--text-secondary) !important; line-height: 1.85 !important; margin: 0 !important;
}

/* ── SCORE ROW ── */
.score-row { display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 0.8rem 0; }
.score-pill {
  font-family: var(--font-mono); font-size: 0.56rem; letter-spacing: 0.12em;
  text-transform: uppercase; padding: 0.2rem 0.6rem; border: 1px solid;
  background: var(--bg-raised); white-space: nowrap;
}
.sp-accent  { color: var(--accent);   border-color: var(--accent);   background: var(--accent-dim); }
.sp-purple  { color: var(--purple);   border-color: var(--purple);   background: var(--purple-dim); }
.sp-warn    { color: var(--warn);     border-color: var(--warn);     background: var(--warn-dim); }
.sp-danger  { color: var(--danger);   border-color: var(--danger);   background: var(--danger-dim); }
.sp-safe    { color: var(--safe);     border-color: var(--safe);     background: var(--safe-dim); }
.sp-neutral { color: var(--text-dim); border-color: var(--border-mid); background: transparent; }
.sp-mod     { color: var(--moderate); border-color: var(--moderate); background: var(--moderate-dim); }

/* ── RANK TABLE ── */
.rank-table-wrapper { background: var(--bg-surface); border: 1px solid var(--border-dim); overflow: hidden; }
.rank-table {
  width: 100%; border-collapse: collapse;
  font-family: var(--font-mono); font-size: 0.74rem;
}
.rank-table thead { background: var(--bg-raised); border-bottom: 1px solid var(--border-mid); }
.rank-table th {
  color: var(--text-dim); font-size: 0.56rem; letter-spacing: 0.2em;
  text-transform: uppercase; padding: 0.9rem 1.2rem;
  text-align: left; font-weight: 500; white-space: nowrap;
}
.rank-table th:first-child { color: var(--accent); }
.rank-table td {
  padding: 0.75rem 1.2rem; color: var(--text-secondary);
  border-bottom: 1px solid var(--border-dim); vertical-align: middle;
}
.rank-table tr:last-child td { border-bottom: none; }
.rank-table tbody tr { transition: background var(--transition); }
.rank-table tbody tr:hover td { background: var(--bg-hover); color: var(--text-primary); }
.rank-num { font-family: var(--font-display); font-size: 1.4rem; color: var(--text-dim); }
.rank-num.gold   { color: var(--gold);   text-shadow: 0 0 12px #F5C84260; }
.rank-num.silver { color: #9DB8CC; }
.rank-num.bronze { color: #CD7F32; }

/* ── SCORE BAR ── */
.score-bar-wrap { display: flex; align-items: center; gap: 0.7rem; }
.score-bar-bg   { flex: 1; height: 3px; background: var(--border-dim); max-width: 120px; overflow: hidden; }
.score-bar-fill { height: 3px; position: relative; border-radius: 1px; }
.score-bar-fill::after {
  content: ''; position: absolute; right: 0; top: -1px;
  width: 5px; height: 5px; border-radius: 50%; background: inherit; filter: blur(3px);
}
.score-val { font-family: var(--font-mono); font-size: 0.78rem; min-width: 40px; }

/* ── CLUSTER CARD ── */
.cluster-card {
  background: var(--bg-surface); border: 1px solid var(--border-dim);
  padding: 1.4rem 1.6rem; position: relative; overflow: hidden;
  transition: background var(--transition);
}
.cluster-card::before {
  content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
}
.cluster-card:hover { background: var(--bg-raised); }
.cluster-id {
  font-family: var(--font-display); font-size: 3.5rem; line-height: 1;
  font-weight: 400; opacity: 0.2; position: absolute; right: 1rem; top: 0.5rem;
}
.cluster-name {
  font-family: var(--font-body); font-size: 0.9rem; font-weight: 700;
  color: var(--text-primary); margin-bottom: 0.5rem;
}
.cluster-desc {
  font-family: var(--font-mono); font-size: 0.66rem;
  color: var(--text-secondary); line-height: 1.7; margin-bottom: 1rem;
}
.cluster-stats {
  display: flex; flex-wrap: wrap; gap: 1.5rem;
}
.cluster-stat { display: flex; flex-direction: column; gap: 0.15rem; }
.cluster-stat-label {
  font-family: var(--font-mono); font-size: 0.5rem;
  letter-spacing: 0.18em; text-transform: uppercase; color: var(--text-dim);
}
.cluster-stat-val {
  font-family: var(--font-mono); font-size: 0.8rem;
  color: var(--text-primary); font-weight: 500;
}

/* ── ANOMALY CARD ── */
.anomaly-high { border-left: 3px solid var(--danger) !important; }
.anomaly-med  { border-left: 3px solid var(--warn) !important; }
.anomaly-norm { border-left: 3px solid var(--safe) !important; }

/* ── DIVIDER ── */
.h-rule {
  border: none; border-top: 1px solid var(--border-dim); margin: 2.5rem 0; position: relative;
}
.h-rule::before {
  content: '◈'; position: absolute; top: -0.6rem; left: 50%;
  transform: translateX(-50%); font-size: 0.7rem; color: var(--text-dim);
  background: var(--bg-void); padding: 0 0.5rem;
}

/* ── TABLE NOTE ── */
.table-note {
  font-family: var(--font-mono); font-size: 0.58rem; color: var(--text-dim);
  margin-top: 0.75rem; line-height: 1.8; padding: 0.6rem 0;
  border-top: 1px solid var(--border-dim);
}

/* ── CHART CONTAINER ── */
.chart-container {
  background: var(--bg-surface); border: 1px solid var(--border-dim);
  padding: 1.5rem; position: relative; overflow: hidden;
}
.chart-container::before {
  content: ''; position: absolute; top: 0; left: 0; width: 30px; height: 30px;
  border-top: 2px solid var(--accent-mid); border-left: 2px solid var(--accent-mid);
}
.chart-title {
  font-family: var(--font-mono); font-size: 0.6rem; letter-spacing: 0.2em;
  text-transform: uppercase; color: var(--text-dim); margin-bottom: 1rem; padding-left: 0.3rem;
}

/* ── SELECT BOX ── */
[data-testid="stSelectbox"] > div { background: var(--bg-surface) !important; border: 1px solid var(--border-mid) !important; border-radius: var(--radius-sm) !important; }
[data-testid="stSelectbox"] label { font-family: var(--font-mono) !important; font-size: 0.64rem !important; letter-spacing: 0.1em !important; color: var(--text-secondary) !important; text-transform: uppercase !important; }

/* ── SLIDER ── */
[data-testid="stSlider"] { color: var(--accent) !important; }
[data-testid="stSlider"] label { font-family: var(--font-mono) !important; font-size: 0.64rem !important; letter-spacing: 0.1em !important; color: var(--text-secondary) !important; text-transform: uppercase !important; }

/* ── MULTISELECT ── */
[data-testid="stMultiSelect"] label { font-family: var(--font-mono) !important; font-size: 0.64rem !important; letter-spacing: 0.1em !important; color: var(--text-secondary) !important; text-transform: uppercase !important; }

/* ── BUTTON ── */
.stButton > button {
  background: transparent !important; border: 1px solid var(--accent) !important;
  color: var(--accent) !important; font-family: var(--font-mono) !important;
  font-size: 0.68rem !important; letter-spacing: 0.18em !important;
  text-transform: uppercase !important; border-radius: var(--radius-sm) !important;
  padding: 0.65rem 1.6rem !important; transition: all var(--transition) !important;
  position: relative !important; overflow: hidden !important;
}
.stButton > button::before {
  content: '' !important; position: absolute !important; inset: 0 !important;
  background: var(--accent) !important; transform: translateX(-101%) !important;
  transition: transform 0.3s ease !important; z-index: 0 !important;
}
.stButton > button:hover::before { transform: translateX(0) !important; }
.stButton > button:hover { color: var(--bg-void) !important; box-shadow: var(--accent-glow) !important; }
.stButton > button span { position: relative !important; z-index: 1 !important; }

/* ── DOWNLOAD BUTTON ── */
[data-testid="stDownloadButton"] button {
  background: var(--accent-dim) !important; border: 1px solid var(--accent-mid) !important;
  color: var(--accent) !important; font-family: var(--font-mono) !important;
  font-size: 0.67rem !important; letter-spacing: 0.15em !important;
  text-transform: uppercase !important; border-radius: var(--radius-sm) !important;
  transition: all var(--transition) !important;
}
[data-testid="stDownloadButton"] button:hover {
  background: var(--accent) !important; color: var(--bg-void) !important;
  box-shadow: var(--accent-glow) !important;
}

/* ── PROGRESS ── */
[data-testid="stProgressBar"] > div > div { background: var(--accent) !important; box-shadow: var(--accent-glow) !important; }
[data-testid="stProgressBar"] { background: var(--border-dim) !important; border-radius: 0 !important; height: 2px !important; }

/* ── TABS (Streamlit native) ── */
[data-testid="stTabs"] [role="tablist"] {
  gap: 0 !important; border-bottom: 1px solid var(--border-dim) !important; background: transparent !important;
}
[data-testid="stTabs"] button[role="tab"] {
  font-family: var(--font-mono) !important; font-size: 0.62rem !important;
  letter-spacing: 0.18em !important; text-transform: uppercase !important;
  color: var(--text-dim) !important; border-radius: 0 !important;
  padding: 0.75rem 1.4rem !important; border-bottom: 2px solid transparent !important;
  transition: all var(--transition) !important;
}
[data-testid="stTabs"] button[role="tab"]:hover { color: var(--text-secondary) !important; background: var(--bg-surface) !important; }
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
  color: var(--accent) !important; border-bottom-color: var(--accent) !important;
  background: var(--accent-dim) !important;
}
[data-testid="stTabs"] [data-testid="stVerticalBlock"] { padding-top: 1.5rem !important; }

/* ── COLUMN BORDERS ── */
[data-testid="column"]:first-child { padding-right: 1.5rem !important; border-right: 1px solid var(--border-dim); }

/* ── FOOTER ── */
.app-footer {
  display: flex; justify-content: space-between; align-items: center;
  font-family: var(--font-mono); font-size: 0.58rem; color: var(--text-dim);
  letter-spacing: 0.12em; padding: 1rem 0; border-top: 1px solid var(--border-dim); margin-top: 4rem;
}
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
.footer-dot {
  width: 4px; height: 4px; border-radius: 50%; background: var(--accent);
  display: inline-block; margin: 0 0.4rem; box-shadow: var(--accent-glow);
  animation: pulse 2.5s ease-in-out infinite;
}

p { font-family: var(--font-body) !important; font-size: 0.85rem !important; color: var(--text-secondary) !important; line-height: 1.7 !important; }
h1, h2, h3, h4, h5, h6 { font-family: var(--font-body) !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY THEME
# ─────────────────────────────────────────────────────────────────────────────
PLOT_BG  = "rgba(0,0,0,0)"
GRID_CLR = "#1A2840"
FONT_CLR = "#7A94B0"
FONT_FAM = "JetBrains Mono"
CAT_COLORS = {
    "Equity":             "#00FFD1",
    "Debt":               "#4895EF",
    "Hybrid":             "#9B72FF",
    "Other":              "#F5A623",
    "Solution Oriented":  "#FF3860",
}

def base_layout(title="", h=340):
    return dict(
        paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
        font=dict(family=FONT_FAM, color=FONT_CLR, size=11),
        title=dict(text=title, font=dict(color="#3D526A", size=10, family=FONT_FAM), x=0.01),
        xaxis=dict(gridcolor=GRID_CLR, zeroline=False, tickfont=dict(size=10), linecolor=GRID_CLR),
        yaxis=dict(gridcolor=GRID_CLR, zeroline=False, tickfont=dict(size=10), linecolor=GRID_CLR),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=GRID_CLR, borderwidth=1, font=dict(size=10)),
        margin=dict(l=10, r=10, t=36, b=10),
        height=h,
    )

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING DATA PATHS
# ─────────────────────────────────────────────────────────────────────────────
TRAINING_PATHS = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../datasets/cleaned_mutual_funds.csv"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../datasets/featured_mutual_funds.csv"),
    "cleaned_mutual_funds.csv",
    "featured_mutual_funds.csv",
]

RISK_NUM_MAP   = {1:"Low",2:"Low",3:"Moderate",4:"High",5:"High",6:"Very High"}
CLUSTER_COLORS = ["#00FFD1","#9B72FF","#F5A623","#FF3860","#4895EF","#F5C842"]

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADER
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    df = None
    for p in TRAINING_PATHS:
        try:
            df = pd.read_csv(p)
            if df is not None and len(df) > 10:
                break
        except Exception:
            continue
    if df is None:
        return None, "Could not locate training database."

    num_cols = ['expense_ratio','fund_size_cr','fund_age_yr','sortino','alpha',
                'sd','beta','sharpe','returns_1yr','returns_3yr','returns_5yr',
                'min_sip','min_lumpsum','rating']
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    df['risk_label'] = df['risk_level'].apply(
        lambda x: RISK_NUM_MAP.get(int(x) if pd.notna(x) else 3, "Moderate")
    )
    return df, None


# ─────────────────────────────────────────────────────────────────────────────
# ML BUNDLE: pre-compute everything once
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def build_ml_bundle(df_json: str):
    df = pd.read_json(io.StringIO(df_json))

    num_cols = ['expense_ratio','fund_size_cr','fund_age_yr','sortino','alpha',
                'sd','beta','sharpe','returns_1yr','returns_3yr','returns_5yr']
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # ── 1. SMART SCREENER SCORE ────────────────────────────────────────────
    # Composite score: return_momentum × risk_adj × cost_efficiency × consistency
    df_s = df.copy()

    # Fill NaN with column median for scoring only
    for c in num_cols:
        df_s[c] = df_s[c].fillna(df_s[c].median())

    scaler = MinMaxScaler()

    # Normalise each metric to [0,1]
    df_s['n_sharpe']    = scaler.fit_transform(df_s[['sharpe']].clip(lower=-2))
    df_s['n_alpha']     = scaler.fit_transform(df_s[['alpha']])
    df_s['n_sortino']   = scaler.fit_transform(df_s[['sortino']].clip(lower=-2))
    df_s['n_ret1']      = scaler.fit_transform(df_s[['returns_1yr']])
    df_s['n_ret3']      = scaler.fit_transform(df_s[['returns_3yr']])
    df_s['n_ret5']      = scaler.fit_transform(df_s[['returns_5yr']])
    df_s['n_expense_inv'] = 1 - scaler.fit_transform(df_s[['expense_ratio']])
    df_s['n_size']      = scaler.fit_transform(np.log1p(df_s[['fund_size_cr']]))
    df_s['n_age']       = scaler.fit_transform(df_s[['fund_age_yr']])
    df_s['n_rating']    = scaler.fit_transform(df_s[['rating']].fillna(0))
    # Low SD is good for debt; we don't penalise it for equity (risk neutral)
    df_s['n_sd_inv']    = 1 - scaler.fit_transform(df_s[['sd']].clip(lower=0))

    # Weighted composite
    df_s['smart_score'] = (
        df_s['n_sharpe']      * 0.20 +
        df_s['n_alpha']       * 0.15 +
        df_s['n_sortino']     * 0.15 +
        df_s['n_ret1']        * 0.10 +
        df_s['n_ret3']        * 0.12 +
        df_s['n_ret5']        * 0.08 +
        df_s['n_expense_inv'] * 0.10 +
        df_s['n_size']        * 0.04 +
        df_s['n_age']         * 0.03 +
        df_s['n_rating']      * 0.03
    ) * 100

    df['smart_score'] = df_s['smart_score'].round(1)

    # ── 2. K-MEANS CLUSTERING ──────────────────────────────────────────────
    cluster_features = ['sharpe','alpha','sd','beta','expense_ratio',
                        'returns_1yr','returns_3yr','fund_size_cr','fund_age_yr','sortino']
    cf_df = df[cluster_features].copy()
    for c in cluster_features:
        cf_df[c] = cf_df[c].fillna(cf_df[c].median())
    cf_df['fund_size_cr'] = np.log1p(cf_df['fund_size_cr'])

    std = StandardScaler()
    X_cl = std.fit_transform(cf_df)

    km = KMeans(n_clusters=6, random_state=42, n_init=20)
    df['cluster'] = km.fit_predict(X_cl)

    # Cluster profiles (means on original scale)
    cluster_profiles = df.groupby('cluster').agg(
        count=('scheme_name','count'),
        avg_sharpe=('sharpe','mean'),
        avg_alpha=('alpha','mean'),
        avg_sd=('sd','mean'),
        avg_ret1=('returns_1yr','mean'),
        avg_ret3=('returns_3yr','mean'),
        avg_expense=('expense_ratio','mean'),
        avg_size=('fund_size_cr','mean'),
        top_cat=('category', lambda x: x.value_counts().index[0]),
        top_subcat=('sub_category', lambda x: x.value_counts().index[0]),
    ).round(2)

    # Auto-name clusters based on their profile
    def name_cluster(row):
        if row['avg_sd'] < 3 and row['avg_ret3'] < 10:
            return "Capital Preservers", "Low-volatility debt & liquid funds prioritising capital safety over growth."
        elif row['avg_sd'] > 18 and row['avg_ret3'] > 25:
            return "High-Beta Growth Engines", "Aggressive equity funds — high volatility, high long-term return potential."
        elif row['avg_sharpe'] > 1.8 and row['avg_expense'] < 0.6:
            return "Efficient Alpha Generators", "Cost-lean funds delivering superior risk-adjusted returns — the sweet spot."
        elif row['avg_size'] > 8000 and row['avg_age'] if 'avg_age' in row.index else False:
            return "Large-Cap Giants", "Mega AUM flagship funds with index-like exposure."
        elif row['avg_alpha'] > 5 and row['avg_sd'] < 15:
            return "Smart Balanced Performers", "Hybrid and balanced funds generating alpha with controlled drawdown."
        elif row['avg_expense'] > 1.4:
            return "High-Cost Niche Players", "Sectoral or specialist funds with elevated cost structures."
        else:
            return "Diversified Core", "Broad-mandate funds suitable as core portfolio holdings."

    cluster_names = {}
    for cid, row in cluster_profiles.iterrows():
        name, desc = name_cluster(row)
        cluster_names[cid] = {'name': name, 'desc': desc}

    # ── 3. ISOLATION FOREST ANOMALY DETECTION ─────────────────────────────
    iso_features = ['sharpe','alpha','sd','expense_ratio','returns_1yr','beta','sortino']
    iso_df = df[iso_features].copy()
    for c in iso_features:
        iso_df[c] = iso_df[c].fillna(iso_df[c].median())

    iso = IsolationForest(contamination=0.08, random_state=42)
    df['anomaly_score'] = iso.fit_predict(iso_df)        # -1 = anomaly
    df['anomaly_raw']   = -iso.score_samples(iso_df)     # higher = more anomalous

    # Classify anomalies
    anomaly_threshold_hi = np.percentile(df['anomaly_raw'], 93)
    anomaly_threshold_md = np.percentile(df['anomaly_raw'], 80)
    df['anomaly_tier'] = df['anomaly_raw'].apply(
        lambda x: 'High' if x >= anomaly_threshold_hi
        else ('Medium' if x >= anomaly_threshold_md else 'Normal')
    )

    # ── 4. RETURN FORECAST — Gradient Boosting ────────────────────────────
    forecast_features = ['expense_ratio','fund_age_yr','sortino','alpha','sd',
                         'beta','sharpe','fund_size_cr','returns_1yr','returns_3yr']
    le_cat = LabelEncoder()
    le_sub = LabelEncoder()
    df['cat_enc'] = le_cat.fit_transform(df['category'].fillna('Unknown').astype(str))
    df['sub_enc'] = le_sub.fit_transform(df['sub_category'].fillna('Unknown').astype(str))

    all_ff = forecast_features + ['cat_enc','sub_enc']
    target_mask = df['returns_5yr'].notna()
    X_fc = df.loc[target_mask, all_ff].copy()
    y_fc = df.loc[target_mask, 'returns_5yr'].values

    imp_fc = SimpleImputer(strategy='median')
    X_fc_imp = imp_fc.fit_transform(X_fc)

    gbr = GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                     learning_rate=0.08, random_state=42)
    gbr.fit(X_fc_imp, y_fc)

    # Feature importances
    fi = dict(zip(all_ff, gbr.feature_importances_))

    # Predict for ALL funds
    X_all = df[all_ff].copy()
    X_all_imp = imp_fc.transform(X_all)
    df['predicted_5yr'] = gbr.predict(X_all_imp).round(2)

    # ── 5. EFFICIENCY TIER — Risk-Return Efficiency ────────────────────────
    # Classify funds into 4 efficiency quadrants using risk (sd) vs return (sharpe)
    sd_median     = df['sd'].median()
    sharpe_median = df['sharpe'].median()

    def eff_tier(row):
        hi_return = row['sharpe'] >= sharpe_median
        hi_risk   = row['sd']     >= sd_median
        if hi_return and not hi_risk: return "Star Performers"      # best
        if hi_return and hi_risk:     return "High-Octane"          # good but volatile
        if not hi_return and not hi_risk: return "Defensive Core"   # safe but low yield
        return "Underperformers"                                     # avoid

    df['eff_tier'] = df.apply(eff_tier, axis=1)

    # ── 6. FUND MANAGER INTELLIGENCE ──────────────────────────────────────
    mgr_stats = df.groupby('fund_manager').agg(
        fund_count=('scheme_name','count'),
        avg_alpha=('alpha','mean'),
        avg_sharpe=('sharpe','mean'),
        avg_sortino=('sortino','mean'),
        avg_ret1=('returns_1yr','mean'),
        avg_ret3=('returns_3yr','mean'),
        avg_expense=('expense_ratio','mean'),
        avg_sd=('sd','mean'),
        categories=('category', lambda x: ', '.join(sorted(x.unique()))),
        top_fund=('scheme_name','first'),
    ).round(2).reset_index()
    mgr_stats = mgr_stats[mgr_stats['fund_count'] >= 2].copy()

    # Manager score
    scaler2 = MinMaxScaler()
    mgr_stats['m_alpha']   = scaler2.fit_transform(mgr_stats[['avg_alpha']])
    mgr_stats['m_sharpe']  = scaler2.fit_transform(mgr_stats[['avg_sharpe']])
    mgr_stats['m_sortino'] = scaler2.fit_transform(mgr_stats[['avg_sortino']])
    mgr_stats['m_ret3']    = scaler2.fit_transform(mgr_stats[['avg_ret3']])
    mgr_stats['m_sd_inv']  = 1 - scaler2.fit_transform(mgr_stats[['avg_sd']].clip(lower=0))
    mgr_stats['m_exp_inv'] = 1 - scaler2.fit_transform(mgr_stats[['avg_expense']])
    mgr_stats['m_count']   = scaler2.fit_transform(mgr_stats[['fund_count']])

    mgr_stats['manager_score'] = (
        mgr_stats['m_alpha']   * 0.25 +
        mgr_stats['m_sharpe']  * 0.25 +
        mgr_stats['m_sortino'] * 0.15 +
        mgr_stats['m_ret3']    * 0.20 +
        mgr_stats['m_sd_inv']  * 0.10 +
        mgr_stats['m_exp_inv'] * 0.05
    ) * 100

    mgr_stats['manager_score'] = mgr_stats['manager_score'].round(1)
    mgr_stats = mgr_stats.sort_values('manager_score', ascending=False).reset_index(drop=True)

    return {
        'df':               df,
        'cluster_profiles': cluster_profiles,
        'cluster_names':    cluster_names,
        'feature_importance': fi,
        'mgr_stats':        mgr_stats,
        'gbr':              gbr,
        'imp_fc':           imp_fc,
        'le_cat':           le_cat,
        'le_sub':           le_sub,
        'all_ff':           all_ff,
        'sd_median':        sd_median,
        'sharpe_median':    sharpe_median,
    }


# ─────────────────────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrapper">
    <div class="hero-scan"></div>
    <div class="hero-line"></div>
    <div class="hero-eyebrow">
        <span class="ai-dot"></span>
        6 ML Engines · KMeans · IsolationForest · GradientBoosting · Screener · Efficiency · Manager Alpha
    </div>
    <h1 class="hero-title">AI<br><span class="accent-word">Insights</span><br>Engine</h1>
    <p class="hero-subtitle">
        Six independent machine-learning lenses applied simultaneously to the
        814-fund universe — delivering institutional-grade intelligence on fund DNA,
        alpha anomalies, return forecasts, risk-adjusted efficiency and fund manager
        performance. Fully offline. Zero external API.
    </p>
    <div class="hero-bg-text">AI</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA + BUILD BUNDLE
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Loading 814-fund database…"):
    df_raw, load_err = load_data()

if load_err or df_raw is None:
    st.markdown(f"""
    <div class="status-card status-error">
        ✗ Could not load training database<br>
        <span style="opacity:0.7;font-size:0.7rem;">Ensure cleaned_mutual_funds.csv is accessible at the configured path.</span>
    </div>""", unsafe_allow_html=True)
    st.stop()

with st.spinner("Running 6 ML engines on 814-fund universe…"):
    bundle = build_ml_bundle(df_raw.to_json())

df = bundle['df']

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL SUMMARY METRICS
# ─────────────────────────────────────────────────────────────────────────────
total_funds   = len(df)
total_amcs    = df['amc_name'].nunique()
total_cats    = df['category'].nunique()
avg_sharpe    = df['sharpe'].mean()
anomaly_count = (df['anomaly_tier'] != 'Normal').sum()
star_count    = (df['eff_tier'] == 'Star Performers').sum()

st.markdown(f"""
<div class="metric-grid">
    <div class="metric-cell">
        <span class="metric-label">Universe</span>
        <span class="metric-value accent">{total_funds:,}</span>
    </div>
    <div class="metric-cell">
        <span class="metric-label">Fund Houses</span>
        <span class="metric-value">{total_amcs}</span>
    </div>
    <div class="metric-cell">
        <span class="metric-label">Categories</span>
        <span class="metric-value">{total_cats}</span>
    </div>
    <div class="metric-cell">
        <span class="metric-label">Avg Sharpe</span>
        <span class="metric-value moderate">{avg_sharpe:.2f}</span>
    </div>
    <div class="metric-cell">
        <span class="metric-label">Anomalies</span>
        <span class="metric-value danger">{anomaly_count}</span>
    </div>
    <div class="metric-cell">
        <span class="metric-label">Star Funds</span>
        <span class="metric-value accent">{star_count}</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="status-card status-success">
    ✓ All 6 ML engines trained &amp; ready · KMeans · IsolationForest · GradientBoosting · SmartScreener · EfficiencyClassifier · ManagerScorer
</div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MODULE TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "01 · Smart Screener",
    "02 · Cluster DNA",
    "03 · Alpha Anomalies",
    "04 · Return Forecast",
    "05 · Efficiency Map",
    "06 · Manager Intel",
])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — SMART FUND SCREENER
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("""
    <div class="section-label"><span class="sl-num">01</span>Smart Fund Screener</div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-panel">
        <h5>Multi-Factor ML Scoring Engine</h5>
        <p>Every fund is ranked using a 10-factor composite model: Sharpe ratio (20%), alpha (15%), 
        Sortino ratio (15%), 3yr returns (12%), expense efficiency (10%), 1yr momentum (10%), 
        5yr returns (8%), AUM size (4%), fund age/maturity (3%), and star rating (3%). 
        Scores are normalised to 0–100 across the full universe before filtering.</p>
    </div>
    """, unsafe_allow_html=True)

    # Controls
    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    with col_f1:
        cat_filter = st.multiselect(
            "Filter by Category",
            options=sorted(df['category'].unique()),
            default=[]
        )
    with col_f2:
        risk_filter = st.multiselect(
            "Filter by Risk Tier",
            options=["Low","Moderate","High","Very High"],
            default=[]
        )
    with col_f3:
        min_score = st.slider("Min Smart Score", 0, 100, 40, step=5)
    with col_f4:
        top_n = st.selectbox("Show Top N", [10, 20, 30, 50, 100], index=1)

    # Apply filters
    screen_df = df.copy()
    if cat_filter:
        screen_df = screen_df[screen_df['category'].isin(cat_filter)]
    if risk_filter:
        screen_df = screen_df[screen_df['risk_label'].isin(risk_filter)]
    screen_df = screen_df[screen_df['smart_score'] >= min_score]
    screen_df = screen_df.nlargest(top_n, 'smart_score')

    n_filtered = len(screen_df)
    avg_sc     = screen_df['smart_score'].mean()
    avg_sh     = screen_df['sharpe'].mean()
    avg_al     = screen_df['alpha'].mean()

    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-cell"><span class="metric-label">Qualifying</span><span class="metric-value accent">{n_filtered}</span></div>
        <div class="metric-cell"><span class="metric-label">Avg Score</span><span class="metric-value">{avg_sc:.1f}</span></div>
        <div class="metric-cell"><span class="metric-label">Avg Sharpe</span><span class="metric-value moderate">{avg_sh:.2f}</span></div>
        <div class="metric-cell"><span class="metric-label">Avg Alpha</span><span class="metric-value accent">{avg_al:.2f}%</span></div>
    </div>
    """, unsafe_allow_html=True)

    # Table
    RISK_COLORS_CSS = {"Low":"#00C8A0","Moderate":"#4895EF","High":"#F5A623","Very High":"#FF3860"}

    rows_html = ""
    for rank, (_, row) in enumerate(screen_df.iterrows(), 1):
        rank_cls = "gold" if rank == 1 else "silver" if rank == 2 else "bronze" if rank == 3 else ""
        score_pct = min(int(row['smart_score']), 100)
        rc = RISK_COLORS_CSS.get(row['risk_label'], "#7A94B0")
        pred_5yr  = row.get('predicted_5yr', None)
        pred_str  = f"{pred_5yr:.1f}%" if pred_5yr is not None and pd.notna(pred_5yr) else "—"
        r1 = f"{row['returns_1yr']:.1f}%" if pd.notna(row['returns_1yr']) else "—"
        r3 = f"{row['returns_3yr']:.1f}%" if pd.notna(row.get('returns_3yr')) else "—"
        exp_str = f"{row['expense_ratio']:.2f}%" if pd.notna(row['expense_ratio']) else "—"
        sharpe_s = f"{row['sharpe']:.2f}" if pd.notna(row['sharpe']) else "—"
        alpha_s  = f"{row['alpha']:.2f}%" if pd.notna(row['alpha']) else "—"

        rows_html += f"""
        <tr>
            <td><span class="rank-num {rank_cls}">{rank}</span></td>
            <td style="max-width:280px;line-height:1.5;">
                <span style="color:var(--text-primary);font-weight:600;font-size:0.78rem;">{row['scheme_name']}</span><br>
                <span style="font-size:0.58rem;color:var(--text-dim);">{row['category']} · {row.get('sub_category','')}</span>
            </td>
            <td>
                <div class="score-bar-wrap">
                    <div class="score-bar-bg">
                        <div class="score-bar-fill" style="width:{score_pct}%;background:var(--accent);color:var(--accent);"></div>
                    </div>
                    <span class="score-val" style="color:var(--accent);">{row['smart_score']:.0f}</span>
                </div>
            </td>
            <td style="color:{rc};font-weight:600;font-size:0.7rem;">{row['risk_label']}</td>
            <td style="color:var(--text-secondary);">{sharpe_s}</td>
            <td style="color:var(--text-secondary);">{alpha_s}</td>
            <td style="color:var(--text-secondary);">{r1}</td>
            <td style="color:var(--text-secondary);">{r3}</td>
            <td style="color:var(--purple);">{pred_str}</td>
            <td style="color:var(--text-secondary);">{exp_str}</td>
            <td style="color:var(--text-dim);font-size:0.66rem;">{row['amc_name']}</td>
        </tr>"""

    st.markdown(f"""
    <div class="rank-table-wrapper">
    <table class="rank-table">
        <thead><tr>
            <th>#</th>
            <th>Fund Name</th>
            <th>Smart Score / 100</th>
            <th>Risk</th>
            <th>Sharpe</th>
            <th>Alpha</th>
            <th>1yr</th>
            <th>3yr</th>
            <th>ML 5yr Pred</th>
            <th>Expense</th>
            <th>AMC</th>
        </tr></thead>
        <tbody>{rows_html}</tbody>
    </table>
    </div>
    <div class="table-note">
        Smart Score = weighted composite of 10 ML-normalised factors.
        ML 5yr Pred = Gradient Boosting 5yr return forecast.
        Scores normalised across the full 814-fund universe.
    </div>
    """, unsafe_allow_html=True)

    # Score distribution chart
    st.markdown('<hr class="h-rule">', unsafe_allow_html=True)
    col_ch1, col_ch2 = st.columns(2, gap="medium")

    with col_ch1:
        st.markdown('<div class="chart-container"><div class="chart-title">Smart Score Distribution by Category</div>', unsafe_allow_html=True)
        box = px.box(
            df, x='category', y='smart_score',
            color='category', color_discrete_map=CAT_COLORS,
            labels={'smart_score':'Smart Score','category':''},
            category_orders={'category': list(CAT_COLORS.keys())},
        )
        box.update_traces(marker=dict(size=3, opacity=0.6),
                          line=dict(color="#3D526A"), fillcolor="rgba(0,255,209,0.08)")
        box.update_layout(**base_layout(), showlegend=False)
        st.plotly_chart(box, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_ch2:
        st.markdown('<div class="chart-container"><div class="chart-title">Alpha vs Sharpe — Scored Universe</div>', unsafe_allow_html=True)
        sc_df = df.dropna(subset=['alpha','sharpe'])
        scatter = px.scatter(
            sc_df, x='alpha', y='sharpe',
            color='smart_score',
            color_continuous_scale=[(0,"#1A2840"),(0.5,"#4895EF"),(1,"#00FFD1")],
            hover_data={'scheme_name':True,'category':True,'smart_score':True},
            labels={'alpha':'Alpha (%)','sharpe':'Sharpe Ratio','smart_score':'Score'},
            size_max=8,
        )
        scatter.update_traces(marker=dict(size=5, opacity=0.75, line=dict(width=0)))
        scatter.update_layout(**base_layout(), coloraxis_showscale=True, showlegend=False)
        st.plotly_chart(scatter, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Export
    st.download_button(
        "↓ Export Screener Results (.csv)",
        data=screen_df[['scheme_name','category','sub_category','smart_score','risk_label',
                         'sharpe','alpha','sortino','returns_1yr','returns_3yr','predicted_5yr',
                         'expense_ratio','fund_size_cr','amc_name']].to_csv(index=False).encode(),
        file_name="ai_screener_results.csv",
        mime="text/csv",
    )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — CLUSTER DNA
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div class="section-label"><span class="sl-num">02</span>Cluster DNA — K-Means Fund Intelligence</div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-panel purple">
        <h5>K-Means Fund DNA Mapping (k=6)</h5>
        <p>K-Means clustering on 10 normalised features (Sharpe, Alpha, SD, Beta, Expense Ratio, 
        1yr/3yr Returns, log AUM, Age, Sortino) groups all 814 funds into 6 behavioural archetypes.
        Each cluster reveals a distinct fund personality — from capital preservers to high-beta 
        growth engines — enabling instant peer-group benchmarking.</p>
    </div>
    """, unsafe_allow_html=True)

    cluster_profiles = bundle['cluster_profiles']
    cluster_names    = bundle['cluster_names']

    # Cluster cards
    cols = st.columns(3, gap="medium")
    for i, (cid, row) in enumerate(cluster_profiles.iterrows()):
        cinfo  = cluster_names.get(cid, {'name': f'Cluster {cid}', 'desc': ''})
        color  = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
        with cols[i % 3]:
            st.markdown(f"""
            <div class="cluster-card" style="border-top: none;">
                <div style="position:absolute;top:0;left:0;right:0;height:2px;background:{color};"></div>
                <div class="cluster-id" style="color:{color};">{cid}</div>
                <div class="cluster-name" style="color:{color};">{cinfo['name']}</div>
                <div class="cluster-desc">{cinfo['desc']}</div>
                <div class="cluster-stats">
                    <div class="cluster-stat">
                        <span class="cluster-stat-label">Funds</span>
                        <span class="cluster-stat-val">{int(row['count'])}</span>
                    </div>
                    <div class="cluster-stat">
                        <span class="cluster-stat-label">Sharpe</span>
                        <span class="cluster-stat-val">{row['avg_sharpe']:.2f}</span>
                    </div>
                    <div class="cluster-stat">
                        <span class="cluster-stat-label">Alpha</span>
                        <span class="cluster-stat-val">{row['avg_alpha']:.1f}%</span>
                    </div>
                    <div class="cluster-stat">
                        <span class="cluster-stat-label">Std Dev</span>
                        <span class="cluster-stat-val">{row['avg_sd']:.1f}%</span>
                    </div>
                    <div class="cluster-stat">
                        <span class="cluster-stat-label">3yr Ret</span>
                        <span class="cluster-stat-val">{row['avg_ret3']:.1f}%</span>
                    </div>
                    <div class="cluster-stat">
                        <span class="cluster-stat-label">Expense</span>
                        <span class="cluster-stat-val">{row['avg_expense']:.2f}%</span>
                    </div>
                    <div class="cluster-stat">
                        <span class="cluster-stat-label">Top Category</span>
                        <span class="cluster-stat-val">{row['top_cat']}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<hr class="h-rule">', unsafe_allow_html=True)

    # Radar chart per cluster
    col_r1, col_r2 = st.columns(2, gap="medium")

    with col_r1:
        st.markdown('<div class="chart-container"><div class="chart-title">Cluster Radar — Avg Risk-Return Metrics</div>', unsafe_allow_html=True)
        radar_metrics = ['avg_sharpe','avg_alpha','avg_ret1','avg_ret3','avg_sd','avg_expense']
        radar_labels  = ['Sharpe','Alpha','1yr Ret','3yr Ret','Std Dev','Expense']

        scaler_r = MinMaxScaler()
        radar_norm = pd.DataFrame(
            scaler_r.fit_transform(cluster_profiles[radar_metrics]),
            columns=radar_metrics, index=cluster_profiles.index
        )

        fig_radar = go.Figure()
        for cid, row in radar_norm.iterrows():
            cinfo = cluster_names.get(cid, {'name': f'C{cid}', 'desc': ''})
            vals  = row[radar_metrics].tolist()
            vals += [vals[0]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals, theta=radar_labels + [radar_labels[0]],
                name=f"C{cid}: {cinfo['name'][:18]}",
                line=dict(color=CLUSTER_COLORS[cid % len(CLUSTER_COLORS)], width=1.5),
                fill='toself', fillcolor=CLUSTER_COLORS[cid % len(CLUSTER_COLORS)],
                opacity=0.08 + 0.03 * cid,
            ))
        fig_radar.update_layout(
            **base_layout(h=380),
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, gridcolor=GRID_CLR, range=[0,1], tickfont=dict(size=8)),
                angularaxis=dict(gridcolor=GRID_CLR, tickfont=dict(size=9)),
            ),
            showlegend=True,
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_r2:
        st.markdown('<div class="chart-container"><div class="chart-title">Cluster Fund Count by Category</div>', unsafe_allow_html=True)
        heat_data = df.groupby(['cluster','category']).size().unstack(fill_value=0)
        heat_data.index = [f"C{i}: {cluster_names.get(i,{}).get('name','?')[:16]}" for i in heat_data.index]

        heatmap = go.Figure(go.Heatmap(
            z=heat_data.values,
            x=heat_data.columns.tolist(),
            y=heat_data.index.tolist(),
            colorscale=["#070B12","#1E3050","#00FFD1"], 
            hovertemplate="<b>%{y}</b><br>%{x}: %{z} funds<extra></extra>",
            text=heat_data.values,
            texttemplate="%{text}",
            textfont=dict(size=9, family=FONT_FAM, color="#E8F0FA"),
        ))
        heatmap.update_layout(**base_layout(h=380))
        st.plotly_chart(heatmap, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Scatter: cluster visualisation
    st.markdown('<hr class="h-rule">', unsafe_allow_html=True)
    st.markdown('<div class="chart-container"><div class="chart-title">K-Means Cluster Space — Sharpe vs Alpha (colour = cluster)</div>', unsafe_allow_html=True)
    sc2_df = df.dropna(subset=['alpha','sharpe']).copy()
    sc2_df['cluster_label'] = sc2_df['cluster'].apply(
        lambda x: f"C{x}: {cluster_names.get(x,{}).get('name','?')[:20]}"
    )
    cl_color_map = {
        f"C{cid}: {cluster_names.get(cid,{}).get('name','?')[:20]}": CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]
        for cid in sc2_df['cluster'].unique()
    }
    cl_fig = px.scatter(
        sc2_df, x='alpha', y='sharpe', color='cluster_label',
        color_discrete_map=cl_color_map,
        hover_data={'scheme_name':True,'category':True,'cluster_label':True},
        labels={'alpha':'Alpha (%)','sharpe':'Sharpe Ratio','cluster_label':'Cluster'},
        size_max=7,
    )
    cl_fig.update_traces(marker=dict(size=5, opacity=0.8, line=dict(width=0)))
    cl_fig.update_layout(**base_layout(h=380), showlegend=True)
    st.plotly_chart(cl_fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Fund browser per cluster
    st.markdown('<hr class="h-rule">', unsafe_allow_html=True)
    st.markdown("""<div class="section-label"><span class="sl-num">Browse</span>Funds by Cluster</div>""", unsafe_allow_html=True)
    selected_cluster = st.selectbox(
        "Select Cluster",
        options=sorted(df['cluster'].unique()),
        format_func=lambda x: f"Cluster {x} — {cluster_names.get(x,{}).get('name','?')}",
    )
    cl_funds = df[df['cluster'] == selected_cluster][
        ['scheme_name','category','sub_category','smart_score','sharpe','alpha','sd',
         'returns_1yr','returns_3yr','expense_ratio','risk_label']
    ].sort_values('smart_score', ascending=False).head(20)
    st.dataframe(cl_funds.rename(columns={
        'scheme_name':'Fund','category':'Category','sub_category':'Sub-Cat',
        'smart_score':'Score','sharpe':'Sharpe','alpha':'Alpha','sd':'SD',
        'returns_1yr':'1yr%','returns_3yr':'3yr%','expense_ratio':'Exp%',
        'risk_label':'Risk'
    }), use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — ALPHA ANOMALY DETECTOR
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("""
    <div class="section-label"><span class="sl-num">03</span>Alpha Anomaly Detector — Isolation Forest</div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-panel danger">
        <h5>Isolation Forest Anomaly Detection</h5>
        <p>An Isolation Forest (contamination=8%) identifies funds that exhibit statistically 
        unusual combinations of risk-return characteristics. High-anomaly funds may represent 
        hidden gems (unusual alpha) or landmines (high risk masking poor returns). The raw 
        anomaly score is derived from the average path length across 100 isolation trees — 
        shorter paths indicate higher isolation, i.e. more anomalous behaviour.</p>
    </div>
    """, unsafe_allow_html=True)

    hi_anom  = df[df['anomaly_tier'] == 'High'].sort_values('anomaly_raw', ascending=False)
    med_anom = df[df['anomaly_tier'] == 'Medium'].sort_values('anomaly_raw', ascending=False)
    norm_df  = df[df['anomaly_tier'] == 'Normal']

    col_a1, col_a2, col_a3 = st.columns(3)
    with col_a1:
        st.markdown(f"""
        <div class="metric-grid">
            <div class="metric-cell"><span class="metric-label">High Anomaly</span><span class="metric-value danger">{len(hi_anom)}</span></div>
        </div>""", unsafe_allow_html=True)
    with col_a2:
        st.markdown(f"""
        <div class="metric-grid">
            <div class="metric-cell"><span class="metric-label">Medium Anomaly</span><span class="metric-value warn">{len(med_anom)}</span></div>
        </div>""", unsafe_allow_html=True)
    with col_a3:
        st.markdown(f"""
        <div class="metric-grid">
            <div class="metric-cell"><span class="metric-label">Normal</span><span class="metric-value safe">{len(norm_df)}</span></div>
        </div>""", unsafe_allow_html=True)

    # Anomaly insight cards
    st.markdown("""<div class="section-label"><span class="sl-num">A</span>High-Anomaly Funds — Investigate</div>""", unsafe_allow_html=True)

    def anomaly_reason(row):
        reasons = []
        if pd.notna(row.get('sd')) and row['sd'] > 25:
            reasons.append(f"extremely high volatility (SD={row['sd']:.1f}%)")
        if pd.notna(row.get('alpha')) and abs(row['alpha']) > 10:
            reasons.append(f"extreme alpha divergence ({row['alpha']:.1f}%)")
        if pd.notna(row.get('sharpe')) and row['sharpe'] < 0.1:
            reasons.append(f"near-zero Sharpe ({row['sharpe']:.2f})")
        if pd.notna(row.get('expense_ratio')) and row['expense_ratio'] > 2.0:
            reasons.append(f"very high expense ratio ({row['expense_ratio']:.2f}%)")
        if pd.notna(row.get('returns_1yr')) and row['returns_1yr'] > 40:
            reasons.append(f"outlier 1yr return ({row['returns_1yr']:.1f}%)")
        if pd.notna(row.get('returns_1yr')) and row['returns_1yr'] < -10:
            reasons.append(f"severe 1yr drawdown ({row['returns_1yr']:.1f}%)")
        if pd.notna(row.get('beta')) and row['beta'] > 1.5:
            reasons.append(f"high market beta ({row['beta']:.2f})")
        if not reasons:
            reasons.append("unusual combination of risk-return metrics")
        return "; ".join(reasons[:3])

    for _, row in hi_anom.head(8).iterrows():
        anom_pct = min(int(row['anomaly_raw'] * 100), 100)
        reason   = anomaly_reason(row)
        r1s = f"{row['returns_1yr']:.1f}%" if pd.notna(row['returns_1yr']) else "—"
        sds = f"{row['sd']:.1f}%" if pd.notna(row['sd']) else "—"
        alps= f"{row['alpha']:.2f}%" if pd.notna(row['alpha']) else "—"
        shs = f"{row['sharpe']:.2f}" if pd.notna(row['sharpe']) else "—"

        st.markdown(f"""
        <div class="insight-panel danger">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:1rem;">
                <div>
                    <h5>{row['scheme_name']}</h5>
                    <div class="score-row">
                        <span class="score-pill sp-neutral">{row['category']}</span>
                        <span class="score-pill sp-neutral">{row.get('sub_category','')}</span>
                        <span class="score-pill {'sp-danger' if row['risk_label'] in ('High','Very High') else 'sp-warn'}">{row['risk_label']} Risk</span>
                        <span class="score-pill sp-neutral">{row['amc_name']}</span>
                    </div>
                </div>
                <div style="text-align:right;flex-shrink:0;">
                    <div style="font-family:var(--font-display);font-size:2.2rem;color:var(--danger);text-shadow:var(--danger-glow);line-height:1;">
                        {anom_pct}
                    </div>
                    <div style="font-family:var(--font-mono);font-size:0.52rem;letter-spacing:0.18em;color:var(--text-dim);text-transform:uppercase;">
                        Anomaly Score
                    </div>
                </div>
            </div>
            <div class="score-bar-wrap" style="margin:0.7rem 0;">
                <span style="font-family:var(--font-mono);font-size:0.56rem;color:var(--text-dim);letter-spacing:0.1em;text-transform:uppercase;min-width:80px;">Isolation Score</span>
                <div class="score-bar-bg" style="max-width:200px;">
                    <div class="score-bar-fill" style="width:{anom_pct}%;background:var(--danger);color:var(--danger);"></div>
                </div>
                <span style="font-family:var(--font-mono);font-size:0.72rem;color:var(--danger);">{anom_pct}/100</span>
            </div>
            <div style="display:flex;flex-wrap:wrap;gap:2rem;margin-bottom:0.8rem;padding-bottom:0.8rem;border-bottom:1px solid var(--border-dim);">
                <div><div style="font-family:var(--font-mono);font-size:0.52rem;letter-spacing:0.18em;text-transform:uppercase;color:var(--text-dim);">1yr Return</div><div style="font-family:var(--font-mono);color:var(--text-primary);font-size:0.84rem;">{r1s}</div></div>
                <div><div style="font-family:var(--font-mono);font-size:0.52rem;letter-spacing:0.18em;text-transform:uppercase;color:var(--text-dim);">Std Dev</div><div style="font-family:var(--font-mono);color:var(--text-primary);font-size:0.84rem;">{sds}</div></div>
                <div><div style="font-family:var(--font-mono);font-size:0.52rem;letter-spacing:0.18em;text-transform:uppercase;color:var(--text-dim);">Alpha</div><div style="font-family:var(--font-mono);color:var(--text-primary);font-size:0.84rem;">{alps}</div></div>
                <div><div style="font-family:var(--font-mono);font-size:0.52rem;letter-spacing:0.18em;text-transform:uppercase;color:var(--text-dim);">Sharpe</div><div style="font-family:var(--font-mono);color:var(--text-primary);font-size:0.84rem;">{shs}</div></div>
            </div>
            <p style="color:var(--text-secondary)!important;">⚠ Anomaly drivers: {reason}</p>
        </div>
        """, unsafe_allow_html=True)

    # Charts
    st.markdown('<hr class="h-rule">', unsafe_allow_html=True)
    col_ac1, col_ac2 = st.columns(2, gap="medium")

    with col_ac1:
        st.markdown('<div class="chart-container"><div class="chart-title">Anomaly Score Distribution</div>', unsafe_allow_html=True)
        anom_color_map = {"High":"#FF3860","Medium":"#F5A623","Normal":"#00C8A0"}
        hist = px.histogram(
            df, x='anomaly_raw', color='anomaly_tier',
            color_discrete_map=anom_color_map, nbins=40,
            labels={'anomaly_raw':'Anomaly Score','anomaly_tier':'Tier'},
            barmode='overlay',
        )
        hist.update_traces(opacity=0.75, marker_line_width=0)
        hist.update_layout(**base_layout(), showlegend=True)
        st.plotly_chart(hist, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_ac2:
        st.markdown('<div class="chart-container"><div class="chart-title">Anomaly Tier by Category</div>', unsafe_allow_html=True)
        anom_cat = df.groupby(['category','anomaly_tier']).size().reset_index(name='count')
        bar_anom = px.bar(
            anom_cat, x='category', y='count', color='anomaly_tier',
            color_discrete_map=anom_color_map, barmode='stack',
            labels={'count':'Funds','category':''},
        )
        bar_anom.update_layout(**base_layout(), showlegend=True)
        st.plotly_chart(bar_anom, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Anomaly scatter
    st.markdown('<div class="chart-container"><div class="chart-title">Anomaly Map — Sharpe vs 1yr Returns (size = Anomaly Score)</div>', unsafe_allow_html=True)
    anom_sc = df.dropna(subset=['sharpe','returns_1yr']).copy()
    anom_sc['size_col'] = anom_sc['anomaly_raw'] * 40 + 4
    fig_asc = px.scatter(
        anom_sc, x='returns_1yr', y='sharpe',
        color='anomaly_tier', color_discrete_map=anom_color_map,
        size='size_col', size_max=18,
        hover_data={'scheme_name':True,'category':True,'anomaly_raw':':.3f'},
        labels={'returns_1yr':'1yr Return (%)','sharpe':'Sharpe Ratio'},
    )
    fig_asc.update_layout(**base_layout(h=420), showlegend=True)
    st.plotly_chart(fig_asc, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — RETURN FORECAST ENGINE
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("""
    <div class="section-label"><span class="sl-num">04</span>Return Forecast Engine — Gradient Boosting</div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-panel">
        <h5>5-Year Return Prediction — Gradient Boosting Regressor</h5>
        <p>A GradientBoostingRegressor (200 estimators, depth=4, lr=0.08) trained on 647 funds 
        with known 5yr returns, using 12 features: expense ratio, fund age, Sortino, alpha, 
        standard deviation, beta, Sharpe, log AUM, 1yr return, 3yr return, category encoding, 
        and sub-category encoding. The model then predicts 5yr returns for all 814 funds including 
        those missing historical 5yr data — enabling forward-looking portfolio construction.</p>
    </div>
    """, unsafe_allow_html=True)

    fi = bundle['feature_importance']
    fi_df = pd.DataFrame(list(fi.items()), columns=['Feature','Importance']).sort_values('Importance', ascending=False)

    # Feature importance chart
    col_fi1, col_fi2 = st.columns([1.2, 0.8], gap="large")

    with col_fi1:
        st.markdown('<div class="chart-container"><div class="chart-title">Feature Importance — Gradient Boosting Regressor</div>', unsafe_allow_html=True)
        fi_bar = px.bar(
            fi_df, x='Importance', y='Feature', orientation='h',
            color='Importance',
            color_continuous_scale=[(0,"#1A2840"),(0.4,"#4895EF"),(1,"#00FFD1")],
            labels={'Importance':'Feature Importance','Feature':''},
        )
        fi_bar.update_traces(marker_line_width=0)
        fi_bar.update_yaxes(autorange="reversed", tickfont=dict(size=9))
        fi_bar.update_layout(**base_layout(h=360), coloraxis_showscale=False, showlegend=False)
        st.plotly_chart(fi_bar, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_fi2:
        top_features = fi_df.head(6)
        st.markdown("""<div class="section-label"><span class="sl-num">Top</span>Feature Drivers</div>""", unsafe_allow_html=True)
        for _, frow in top_features.iterrows():
            pct = int(frow['Importance'] * 100)
            st.markdown(f"""
            <div style="margin-bottom:1rem;">
                <div style="font-family:var(--font-mono);font-size:0.66rem;color:var(--text-secondary);margin-bottom:0.3rem;
                            display:flex;justify-content:space-between;">
                    <span>{frow['Feature']}</span>
                    <span style="color:var(--accent);">{frow['Importance']:.3f}</span>
                </div>
                <div class="score-bar-bg" style="max-width:100%;">
                    <div class="score-bar-fill" style="width:{min(pct*4,100)}%;background:var(--accent);color:var(--accent);height:4px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Predicted vs actual
    st.markdown('<hr class="h-rule">', unsafe_allow_html=True)
    col_fc1, col_fc2 = st.columns(2, gap="medium")

    with col_fc1:
        st.markdown('<div class="chart-container"><div class="chart-title">Predicted vs Actual 5yr Returns</div>', unsafe_allow_html=True)
        pva_df = df.dropna(subset=['returns_5yr','predicted_5yr'])
        pva = px.scatter(
            pva_df, x='returns_5yr', y='predicted_5yr',
            color='category', color_discrete_map=CAT_COLORS,
            hover_data={'scheme_name':True,'returns_5yr':':.1f','predicted_5yr':':.1f'},
            labels={'returns_5yr':'Actual 5yr Return (%)','predicted_5yr':'Predicted 5yr Return (%)'},
        )
        # Perfect prediction line
        mn = min(pva_df['returns_5yr'].min(), pva_df['predicted_5yr'].min())
        mx = max(pva_df['returns_5yr'].max(), pva_df['predicted_5yr'].max())
        pva.add_shape(type='line', x0=mn, y0=mn, x1=mx, y1=mx,
                      line=dict(color="#1E3050", dash="dot", width=1))
        pva.update_traces(marker=dict(size=5, opacity=0.8, line=dict(width=0)))
        pva.update_layout(**base_layout(), showlegend=True)
        st.plotly_chart(pva, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_fc2:
        st.markdown('<div class="chart-container"><div class="chart-title">Predicted 5yr Return Distribution by Category</div>', unsafe_allow_html=True)
        vio = px.violin(
            df, x='category', y='predicted_5yr',
            color='category', color_discrete_map=CAT_COLORS, box=True,
            labels={'predicted_5yr':'Predicted 5yr Return (%)','category':''},
        )
        vio.update_traces(opacity=0.85)
        vio.update_layout(**base_layout(), showlegend=False)
        st.plotly_chart(vio, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Top predicted outperformers (funds missing 5yr history)
    st.markdown('<hr class="h-rule">', unsafe_allow_html=True)
    st.markdown("""<div class="section-label"><span class="sl-num">★</span>ML-Predicted Outperformers — Funds Missing 5yr History</div>""", unsafe_allow_html=True)

    missing_5yr = df[df['returns_5yr'].isna()].copy()
    missing_5yr = missing_5yr.sort_values('predicted_5yr', ascending=False).head(15)

    if not missing_5yr.empty:
        rows2 = ""
        for i, (_, row) in enumerate(missing_5yr.iterrows(), 1):
            rc = {"Low":"#00C8A0","Moderate":"#4895EF","High":"#F5A623","Very High":"#FF3860"}.get(row['risk_label'],"#7A94B0")
            r1s  = f"{row['returns_1yr']:.1f}%" if pd.notna(row['returns_1yr']) else "—"
            r3s  = f"{row['returns_3yr']:.1f}%" if pd.notna(row.get('returns_3yr')) else "—"
            shs  = f"{row['sharpe']:.2f}" if pd.notna(row['sharpe']) else "—"
            alps = f"{row['alpha']:.2f}%" if pd.notna(row['alpha']) else "—"
            rows2 += f"""
            <tr>
                <td><span class="rank-num {'gold' if i==1 else 'silver' if i==2 else 'bronze' if i==3 else ''}">{i}</span></td>
                <td style="max-width:280px;line-height:1.5;">
                    <span style="color:var(--text-primary);font-weight:600;font-size:0.78rem;">{row['scheme_name']}</span><br>
                    <span style="font-size:0.58rem;color:var(--text-dim);">{row['category']} · {row.get('sub_category','')}</span>
                </td>
                <td style="color:var(--purple);font-weight:700;font-size:0.9rem;">{row['predicted_5yr']:.1f}%</td>
                <td style="color:{rc};font-weight:600;">{row['risk_label']}</td>
                <td style="color:var(--text-secondary);">{shs}</td>
                <td style="color:var(--text-secondary);">{alps}</td>
                <td style="color:var(--text-secondary);">{r1s}</td>
                <td style="color:var(--text-secondary);">{r3s}</td>
                <td style="color:var(--text-dim);font-size:0.66rem;">{row['amc_name']}</td>
            </tr>"""

        st.markdown(f"""
        <div class="rank-table-wrapper">
        <table class="rank-table">
            <thead><tr>
                <th>#</th><th>Fund</th><th>ML 5yr Forecast</th><th>Risk</th>
                <th>Sharpe</th><th>Alpha</th><th>1yr</th><th>3yr</th><th>AMC</th>
            </tr></thead>
            <tbody>{rows2}</tbody>
        </table>
        </div>
        <div class="table-note">
            These funds lack 5yr historical data. The ML model predicts their expected 5yr return 
            using fund characteristics and peer category patterns. Not investment advice.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-card status-info">All funds in this dataset have 5yr historical returns available.</div>', unsafe_allow_html=True)

    # Export
    st.download_button(
        "↓ Export Return Forecasts (.csv)",
        data=df[['scheme_name','category','sub_category','returns_1yr','returns_3yr',
                  'returns_5yr','predicted_5yr','sharpe','alpha','expense_ratio',
                  'risk_label','amc_name']].to_csv(index=False).encode(),
        file_name="ai_return_forecasts.csv",
        mime="text/csv",
    )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — EFFICIENCY MAP
# ═════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("""
    <div class="section-label"><span class="sl-num">05</span>Risk-Return Efficiency Map</div>
    """, unsafe_allow_html=True)

    EFF_COLORS = {
        "Star Performers":  "#00FFD1",
        "High-Octane":      "#9B72FF",
        "Defensive Core":   "#4895EF",
        "Underperformers":  "#FF3860",
    }

    EFF_DESCS = {
        "Star Performers":  "High Sharpe, Low Volatility — the gold standard. These funds deliver superior risk-adjusted returns with controlled drawdown. Prime allocation candidates.",
        "High-Octane":      "High Sharpe, High Volatility — strong performers but with elevated risk. Suitable for aggressive investors with long time horizons.",
        "Defensive Core":   "Low Sharpe, Low Volatility — capital preservation focus. Suitable as portfolio stabilisers, particularly for debt and liquid mandates.",
        "Underperformers":  "Low Sharpe, High Volatility — poor risk-adjusted returns with elevated risk. Review allocation and consider alternatives.",
    }

    st.markdown("""
    <div class="insight-panel">
        <h5>Efficiency Quadrant Classification</h5>
        <p>Each fund is classified into one of four quadrants by comparing its Sharpe ratio and 
        standard deviation to the universe median. This creates a 2×2 efficiency matrix — 
        Star Performers (high Sharpe / low SD) represent the optimal allocation zone, 
        while Underperformers (low Sharpe / high SD) signal review or replacement.</p>
    </div>
    """, unsafe_allow_html=True)

    eff_counts = df['eff_tier'].value_counts()

    # Quadrant cards
    ecols = st.columns(4, gap="medium")
    for i, (tier, color) in enumerate(EFF_COLORS.items()):
        cnt   = eff_counts.get(tier, 0)
        pct   = cnt / len(df) * 100
        with ecols[i]:
            st.markdown(f"""
            <div class="cluster-card">
                <div style="position:absolute;top:0;left:0;right:0;height:2px;background:{color};"></div>
                <div class="cluster-name" style="color:{color};font-size:0.82rem;">{tier}</div>
                <div style="font-family:var(--font-display);font-size:3rem;color:{color};line-height:1;
                            text-shadow:0 0 20px {color}40;margin:0.3rem 0;">{cnt}</div>
                <div style="font-family:var(--font-mono);font-size:0.58rem;color:var(--text-dim);
                            letter-spacing:0.12em;">{pct:.1f}% of universe</div>
                <div style="font-family:var(--font-body);font-size:0.7rem;color:var(--text-secondary);
                            line-height:1.6;margin-top:0.6rem;">{EFF_DESCS[tier][:80]}…</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<hr class="h-rule">', unsafe_allow_html=True)

    # Main efficiency scatter
    st.markdown('<div class="chart-container"><div class="chart-title">Efficiency Map — Std Dev (Risk) vs Sharpe (Return) — Quadrant Classification</div>', unsafe_allow_html=True)
    eff_df = df.dropna(subset=['sd','sharpe']).copy()
    sd_med = bundle['sd_median']
    sh_med = bundle['sharpe_median']

    eff_fig = px.scatter(
        eff_df, x='sd', y='sharpe',
        color='eff_tier', color_discrete_map=EFF_COLORS,
        hover_data={'scheme_name':True,'category':True,'eff_tier':True,'smart_score':':.1f'},
        labels={'sd':'Standard Deviation (Risk %)','sharpe':'Sharpe Ratio (Return)'},
        size_max=8,
        opacity=0.82,
    )
    # Quadrant lines
    eff_fig.add_vline(x=sd_med, line=dict(color="#1E3050", dash="dot", width=1))
    eff_fig.add_hline(y=sh_med, line=dict(color="#1E3050", dash="dot", width=1))
    # Quadrant labels
    xr = eff_df['sd'].quantile(0.95)
    yr = eff_df['sharpe'].quantile(0.97)
    annotations = [
        {
            "x": sd_med * 0.35,
            "y": yr * 0.9,
            "text": "STAR PERFORMERS",
            "showarrow": False,
            "font": {"color": "#00FFD1", "family": FONT_FAM, "size": 10},
        },
        {
            "x": xr * 0.85,
            "y": yr * 0.9,
            "text": "HIGH-OCTANE",
            "showarrow": False,
            "font": {"color": "#9B72FF", "family": FONT_FAM, "size": 10},
        },
        {
            "x": sd_med * 0.35,
            "y": eff_df['sharpe'].min() * 1.1,
            "text": "DEFENSIVE CORE",
            "showarrow": False,
            "font": {"color": "rgba(72, 149, 239, 0.12)", "family": FONT_FAM, "size": 10},
        },
        {
            "x": xr * 0.85,
            "y": eff_df['sharpe'].min() * 1.1,
            "text": "UNDERPERFORMERS",
            "showarrow": False,
            "font": {"color": "rgba(255, 56, 96, 0.12)", "family": FONT_FAM, "size": 10},
        },
    ]
    eff_fig.update_traces(marker=dict(size=5, line=dict(width=0)))
    eff_fig.update_layout(**base_layout(h=480), annotations=annotations, showlegend=True)
    st.plotly_chart(eff_fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Efficiency heatmap: category vs tier
    st.markdown('<hr class="h-rule">', unsafe_allow_html=True)
    col_ef1, col_ef2 = st.columns(2, gap="medium")

    with col_ef1:
        st.markdown('<div class="chart-container"><div class="chart-title">Efficiency Tier Distribution by Category</div>', unsafe_allow_html=True)
        eff_cat = df.groupby(['category','eff_tier']).size().reset_index(name='count')
        eff_bar = px.bar(
            eff_cat, x='category', y='count', color='eff_tier',
            color_discrete_map=EFF_COLORS, barmode='stack',
            labels={'count':'Funds','category':''},
        )
        eff_bar.update_layout(**base_layout(), showlegend=True)
        st.plotly_chart(eff_bar, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_ef2:
        st.markdown('<div class="chart-container"><div class="chart-title">Star Performers — Smart Score Distribution</div>', unsafe_allow_html=True)
        star_df = df[df['eff_tier']=='Star Performers']
        star_hist = px.histogram(
            star_df, x='smart_score', color='category',
            color_discrete_map=CAT_COLORS, nbins=20,
            labels={'smart_score':'Smart Score','category':'Category'},
        )
        star_hist.update_traces(opacity=0.8, marker_line_width=0)
        star_hist.update_layout(**base_layout(), showlegend=True)
        st.plotly_chart(star_hist, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Star performer table
    st.markdown('<hr class="h-rule">', unsafe_allow_html=True)
    st.markdown("""<div class="section-label"><span class="sl-num">★</span>Top Star Performers</div>""", unsafe_allow_html=True)

    star_top = df[df['eff_tier']=='Star Performers'].sort_values('smart_score', ascending=False).head(20)
    st.dataframe(star_top[['scheme_name','category','sub_category','smart_score','sharpe','alpha',
                             'sd','returns_1yr','returns_3yr','predicted_5yr','expense_ratio',
                             'risk_label','amc_name']].rename(columns={
        'scheme_name':'Fund','category':'Cat','sub_category':'Sub-Cat',
        'smart_score':'Score','sharpe':'Sharpe','alpha':'Alpha%','sd':'SD%',
        'returns_1yr':'1yr%','returns_3yr':'3yr%','predicted_5yr':'ML 5yr%',
        'expense_ratio':'Exp%','risk_label':'Risk','amc_name':'AMC',
    }), use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 6 — FUND MANAGER INTELLIGENCE
# ═════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown("""
    <div class="section-label"><span class="sl-num">06</span>Fund Manager Intelligence</div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-panel purple">
        <h5>Manager Alpha & Consistency Scoring</h5>
        <p>Each fund manager is scored across 6 dimensions — normalised to 0–100 across all managers 
        with 2+ funds: alpha generation (25%), Sharpe consistency (25%), 3yr returns (20%), 
        Sortino ratio (15%), volatility management (10%), and expense discipline (5%). 
        This creates a holistic manager score that captures both return generation and 
        risk management skill — identifying truly exceptional active managers.</p>
    </div>
    """, unsafe_allow_html=True)

    mgr_stats = bundle['mgr_stats']

    total_mgrs  = len(mgr_stats)
    elite_mgrs  = (mgr_stats['manager_score'] >= 75).sum()
    avg_mgr_sc  = mgr_stats['manager_score'].mean()
    top_mgr_sc  = mgr_stats['manager_score'].max()

    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-cell"><span class="metric-label">Managers Scored</span><span class="metric-value accent">{total_mgrs}</span></div>
        <div class="metric-cell"><span class="metric-label">Elite (Score≥75)</span><span class="metric-value purple">{elite_mgrs}</span></div>
        <div class="metric-cell"><span class="metric-label">Avg Manager Score</span><span class="metric-value">{avg_mgr_sc:.1f}</span></div>
        <div class="metric-cell"><span class="metric-label">Top Score</span><span class="metric-value accent">{top_mgr_sc:.1f}</span></div>
    </div>
    """, unsafe_allow_html=True)

    # Top manager cards
    st.markdown("""<div class="section-label"><span class="sl-num">TOP</span>Elite Fund Managers</div>""", unsafe_allow_html=True)

    for _, mgr in mgr_stats.head(8).iterrows():
        score    = mgr['manager_score']
        score_pct = int(score)
        tier_label = "Elite" if score >= 75 else "Strong" if score >= 60 else "Average"
        tier_color = "#00FFD1" if score >= 75 else "#9B72FF" if score >= 60 else "#F5A623"
        alpha_c = "#00C8A0" if mgr['avg_alpha'] > 0 else "#FF3860"

        st.markdown(f"""
        <div class="insight-panel purple">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:1.5rem;">
                <div style="flex:1;">
                    <h5 style="color:var(--text-primary);">{mgr['fund_manager']}</h5>
                    <div class="score-row">
                        <span class="score-pill sp-purple">{int(mgr['fund_count'])} Funds Managed</span>
                        <span class="score-pill sp-neutral">{mgr['categories']}</span>
                        <span class="score-pill {'sp-accent' if score>=75 else 'sp-warn'}">{tier_label}</span>
                    </div>
                    <div style="margin-top:1rem;display:flex;flex-wrap:wrap;gap:2.5rem;">
                        <div>
                            <div style="font-family:var(--font-mono);font-size:0.5rem;letter-spacing:0.18em;text-transform:uppercase;color:var(--text-dim);">Avg Alpha</div>
                            <div style="font-family:var(--font-mono);font-size:0.9rem;color:{alpha_c};font-weight:600;">{mgr['avg_alpha']:+.2f}%</div>
                        </div>
                        <div>
                            <div style="font-family:var(--font-mono);font-size:0.5rem;letter-spacing:0.18em;text-transform:uppercase;color:var(--text-dim);">Avg Sharpe</div>
                            <div style="font-family:var(--font-mono);font-size:0.9rem;color:var(--text-primary);font-weight:600;">{mgr['avg_sharpe']:.2f}</div>
                        </div>
                        <div>
                            <div style="font-family:var(--font-mono);font-size:0.5rem;letter-spacing:0.18em;text-transform:uppercase;color:var(--text-dim);">Avg 3yr Return</div>
                            <div style="font-family:var(--font-mono);font-size:0.9rem;color:var(--text-primary);font-weight:600;">{mgr['avg_ret3']:.1f}%</div>
                        </div>
                        <div>
                            <div style="font-family:var(--font-mono);font-size:0.5rem;letter-spacing:0.18em;text-transform:uppercase;color:var(--text-dim);">Avg Expense</div>
                            <div style="font-family:var(--font-mono);font-size:0.9rem;color:var(--text-secondary);">{mgr['avg_expense']:.2f}%</div>
                        </div>
                        <div>
                            <div style="font-family:var(--font-mono);font-size:0.5rem;letter-spacing:0.18em;text-transform:uppercase;color:var(--text-dim);">Avg SD</div>
                            <div style="font-family:var(--font-mono);font-size:0.9rem;color:var(--text-secondary);">{mgr['avg_sd']:.1f}%</div>
                        </div>
                    </div>
                </div>
                <div style="text-align:right;flex-shrink:0;min-width:80px;">
                    <div style="font-family:var(--font-display);font-size:2.8rem;color:{tier_color};
                                text-shadow:0 0 20px {tier_color}40;line-height:1;">{score:.0f}</div>
                    <div style="font-family:var(--font-mono);font-size:0.5rem;letter-spacing:0.18em;
                                color:var(--text-dim);text-transform:uppercase;margin-top:0.2rem;">Score / 100</div>
                    <div style="margin-top:0.5rem;">
                        <div class="score-bar-bg" style="max-width:80px;margin-left:auto;">
                            <div class="score-bar-fill" style="width:{score_pct}%;background:{tier_color};color:{tier_color};height:3px;"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Charts
    st.markdown('<hr class="h-rule">', unsafe_allow_html=True)
    col_m1, col_m2 = st.columns(2, gap="medium")

    with col_m1:
        st.markdown('<div class="chart-container"><div class="chart-title">Manager Score Distribution</div>', unsafe_allow_html=True)
        mgr_hist = px.histogram(
            mgr_stats, x='manager_score', nbins=30,
            color_discrete_sequence=["#9B72FF"],
            labels={'manager_score':'Manager Score'},
        )
        mgr_hist.add_vline(x=75, line=dict(color="#00FFD1", dash="dot", width=1))
        import numpy as np
        hist_vals, _ = np.histogram(mgr_stats['manager_score'].dropna(), bins=30)
        peak_y = int(hist_vals.max())

        mgr_hist.add_annotation(x=75, y=peak_y,
                                  text="Elite Threshold", showarrow=False,
                                  font=dict(color="#00FFD1", size=9, family=FONT_FAM), xshift=35)
        mgr_hist.update_traces(opacity=0.8, marker_line_width=0)
        mgr_hist.update_layout(**base_layout(), showlegend=False)
        st.plotly_chart(mgr_hist, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_m2:
        st.markdown('<div class="chart-container"><div class="chart-title">Top 15 Managers — Alpha vs Sharpe (size = Score)</div>', unsafe_allow_html=True)
        top15 = mgr_stats.head(15)
        mgr_sc = px.scatter(
            top15, x='avg_alpha', y='avg_sharpe',
            size='manager_score', size_max=20,
            hover_data={'fund_manager':True,'fund_count':True,'manager_score':':.1f'},
            text='fund_manager',
            labels={'avg_alpha':'Avg Alpha (%)','avg_sharpe':'Avg Sharpe'},
            color='manager_score',
            color_continuous_scale=[(0,"#4895EF"),(0.5,"#9B72FF"),(1,"#00FFD1")],
        )
        mgr_sc.update_traces(
            textposition='top center',
            textfont=dict(size=7, family=FONT_FAM, color="#5A6A80"),
            marker=dict(opacity=0.9, line=dict(width=1, color="#03050A")),
        )
        mgr_sc.update_layout(**base_layout(), coloraxis_showscale=False, showlegend=False)
        st.plotly_chart(mgr_sc, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Full manager leaderboard
    st.markdown('<hr class="h-rule">', unsafe_allow_html=True)
    st.markdown("""<div class="section-label"><span class="sl-num">All</span>Manager Leaderboard</div>""", unsafe_allow_html=True)

    rows_m = ""
    for rank, (_, mgr) in enumerate(mgr_stats.head(30).iterrows(), 1):
        sc  = mgr['manager_score']
        sc_pct = int(sc)
        tc  = "#00FFD1" if sc >= 75 else "#9B72FF" if sc >= 60 else "#F5A623" if sc >= 45 else "#FF3860"
        rk_cls = "gold" if rank==1 else "silver" if rank==2 else "bronze" if rank==3 else ""
        a_c = "#00C8A0" if mgr['avg_alpha'] > 0 else "#FF3860"
        rows_m += f"""
        <tr>
            <td><span class="rank-num {rk_cls}">{rank}</span></td>
            <td style="color:var(--text-primary);font-weight:600;font-size:0.8rem;">{mgr['fund_manager']}</td>
            <td>
                <div class="score-bar-wrap">
                    <div class="score-bar-bg">
                        <div class="score-bar-fill" style="width:{sc_pct}%;background:{tc};color:{tc};"></div>
                    </div>
                    <span style="font-family:var(--font-mono);font-size:0.76rem;color:{tc};">{sc:.0f}</span>
                </div>
            </td>
            <td style="color:var(--text-secondary);">{int(mgr['fund_count'])}</td>
            <td style="color:{a_c};font-weight:600;">{mgr['avg_alpha']:+.2f}%</td>
            <td style="color:var(--text-secondary);">{mgr['avg_sharpe']:.2f}</td>
            <td style="color:var(--text-secondary);">{mgr['avg_sortino']:.2f}</td>
            <td style="color:var(--text-secondary);">{mgr['avg_ret3']:.1f}%</td>
            <td style="color:var(--text-secondary);">{mgr['avg_sd']:.1f}%</td>
            <td style="color:var(--text-dim);font-size:0.64rem;">{mgr['avg_expense']:.2f}%</td>
            <td style="color:var(--text-dim);font-size:0.64rem;max-width:150px;">{mgr['categories']}</td>
        </tr>"""

    st.markdown(f"""
    <div class="rank-table-wrapper">
    <table class="rank-table">
        <thead><tr>
            <th>#</th><th>Fund Manager</th><th>Score / 100</th><th>Funds</th>
            <th>Avg Alpha</th><th>Sharpe</th><th>Sortino</th><th>3yr Ret</th>
            <th>Avg SD</th><th>Expense</th><th>Categories</th>
        </tr></thead>
        <tbody>{rows_m}</tbody>
    </table>
    </div>
    <div class="table-note">
        Manager Score = weighted composite: Alpha 25% · Sharpe 25% · 3yr Returns 20% · Sortino 15% · Volatility Control 10% · Cost Discipline 5%.
        Only managers with 2+ funds included. Normalised across {total_mgrs} managers.
    </div>
    """, unsafe_allow_html=True)

    # Export managers
    st.download_button(
        "↓ Export Manager Intelligence (.csv)",
        data=mgr_stats[['fund_manager','fund_count','manager_score','avg_alpha','avg_sharpe',
                         'avg_sortino','avg_ret3','avg_sd','avg_expense','categories']].to_csv(index=False).encode(),
        file_name="ai_manager_intelligence.csv",
        mime="text/csv",
    )


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    <span>◈ AI INSIGHTS ENGINE <span class="footer-dot"></span> v1.0</span>
    <span>KMEANS · ISOLATION FOREST · GRADIENT BOOSTING · SMART SCREENER · EFFICIENCY MAP · MANAGER ALPHA · 814 FUNDS · ZERO EXTERNAL API</span>
</div>
""", unsafe_allow_html=True)