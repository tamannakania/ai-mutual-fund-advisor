# frontend/pages/6_Report_Generator.py
# ─────────────────────────────────────────────────────────────────────────────
#  ALPHA RETURN ENGINE — Report Generator
#  Investor profiling + ML-powered portfolio construction + detailed report:
#    1. Investor Profile Builder  — risk tolerance questionnaire → ML risk score
#    2. Goal-Based Portfolio      — MPT-style allocation via ML signals
#    3. Fund Selection Engine     — filtered top picks per allocation bucket
#    4. Portfolio Analytics       — projected returns, risk, diversification
#    5. Report Export             — downloadable CSV + structured summary
#
#  Fully offline — zero external API. All ML runs on the 814-fund database.
# ─────────────────────────────────────────────────────────────────────────────

import io
import os
import warnings
from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# DESIGN SYSTEM  (identical token set to AI Insights for visual consistency)
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
  margin: 1.4rem 0 0 1.2rem; max-width: 580px; line-height: 1.9;
  border-left: 2px solid var(--border-mid); padding-left: 1rem;
}
.hero-bg-text {
  position: absolute; right: -1rem; top: 50%; transform: translateY(-50%);
  font-family: var(--font-display); font-size: clamp(8rem, 18vw, 16rem);
  font-weight: 400; color: transparent;
  -webkit-text-stroke: 1px var(--border-dim);
  pointer-events: none; user-select: none; letter-spacing: 0.06em; line-height: 1;
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
.status-warn    { background: #1A1005; border-color: var(--warn); color: var(--warn); }

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
.insight-panel.warn::before { background: radial-gradient(circle at top right, var(--warn-dim), transparent 70%); }
.insight-panel.safe    { border-left-color: var(--safe); }
.insight-panel.safe::before { background: radial-gradient(circle at top right, var(--safe-dim), transparent 70%); }
.insight-panel.danger  { border-left-color: var(--danger); }
.insight-panel.danger::before { background: radial-gradient(circle at top right, var(--danger-dim), transparent 70%); }
.insight-panel.moderate { border-left-color: var(--moderate); }
.insight-panel.moderate::before { background: radial-gradient(circle at top right, var(--moderate-dim), transparent 70%); }
.insight-panel h5 {
  font-family: var(--font-body); font-size: 0.95rem; font-weight: 600;
  color: var(--text-primary); margin: 0 0 0.8rem 0;
}
.insight-panel p {
  font-family: var(--font-body) !important; font-size: 0.8rem !important;
  color: var(--text-secondary) !important; line-height: 1.85 !important; margin: 0 !important;
}

/* ── SCORE ROW / PILLS ── */
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

/* ── RANK / PORTFOLIO TABLE ── */
.rank-table-wrapper { background: var(--bg-surface); border: 1px solid var(--border-dim); overflow: hidden; overflow-x: auto; }
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
.table-note {
  font-family: var(--font-mono); font-size: 0.58rem; color: var(--text-dim);
  padding: 0.8rem 1.2rem; border-top: 1px solid var(--border-dim);
  background: var(--bg-surface); line-height: 1.6;
}

/* ── SCORE BAR ── */
.score-bar-wrap { display: flex; align-items: center; gap: 0.7rem; }
.score-bar-bg   { flex: 1; height: 3px; background: var(--border-dim); max-width: 120px; overflow: hidden; }
.score-bar-fill { height: 3px; position: relative; border-radius: 1px; }
.score-bar-fill::after {
  content: ''; position: absolute; right: 0; top: -1px;
  width: 5px; height: 5px; border-radius: 50%; background: inherit; filter: blur(3px);
}

/* ── PROFILE CARD ── */
.profile-card {
  background: var(--bg-surface); border: 1px solid var(--border-dim);
  padding: 2rem 2.2rem; position: relative; overflow: hidden;
  margin-bottom: 1.5rem;
}
.profile-card::before {
  content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, var(--accent), var(--purple), transparent);
}
.profile-card .avatar-ring {
  width: 52px; height: 52px; border-radius: 50%;
  border: 2px solid var(--accent); background: var(--accent-dim);
  display: flex; align-items: center; justify-content: center;
  font-family: var(--font-display); font-size: 1.4rem; color: var(--accent);
  flex-shrink: 0;
}
.profile-header { display: flex; align-items: center; gap: 1.2rem; margin-bottom: 1.4rem; }
.profile-name {
  font-family: var(--font-body); font-size: 1.1rem; font-weight: 700;
  color: var(--text-primary);
}
.profile-meta { font-family: var(--font-mono); font-size: 0.6rem; color: var(--text-dim); letter-spacing: 0.18em; }

/* ── ALLOCATION BLOCK ── */
.alloc-block {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 1px; background: var(--border-dim); border: 1px solid var(--border-dim);
  margin: 1rem 0;
}
.alloc-cell {
  background: var(--bg-surface); padding: 1rem 1.3rem;
  position: relative; overflow: hidden;
}
.alloc-cell::after {
  content: ''; position: absolute; bottom: 0; left: 0; height: 2px;
  background: var(--accent); transition: width 0.5s ease;
}
.alloc-label { font-family: var(--font-mono); font-size: 0.52rem; letter-spacing: 0.2em; text-transform: uppercase; color: var(--text-dim); margin-bottom: 0.35rem; }
.alloc-pct   { font-family: var(--font-display); font-size: 2.2rem; line-height: 1; }
.alloc-amt   { font-family: var(--font-mono); font-size: 0.62rem; color: var(--text-secondary); margin-top: 0.3rem; }

/* ── REPORT SECTION DIVIDER ── */
.h-rule {
  border: none; border-top: 1px solid var(--border-dim); margin: 2.5rem 0;
}
.report-stamp {
  display: inline-flex; align-items: center; gap: 0.6rem;
  font-family: var(--font-mono); font-size: 0.54rem; letter-spacing: 0.24em;
  text-transform: uppercase; color: var(--text-dim);
  padding: 0.3rem 0.8rem; border: 1px solid var(--border-dim);
  background: var(--bg-raised); margin-bottom: 2rem;
}
.report-stamp .rs-dot {
  width: 5px; height: 5px; border-radius: 50%; background: var(--accent);
  box-shadow: var(--accent-glow);
}

/* ── RISK GAUGE LABELS ── */
.risk-gauge-wrap { text-align: center; padding: 1rem 0; }
.risk-gauge-label {
  font-family: var(--font-display); font-size: 3.5rem; line-height: 1;
}
.risk-gauge-sub { font-family: var(--font-mono); font-size: 0.58rem; letter-spacing: 0.22em; text-transform: uppercase; color: var(--text-dim); margin-top: 0.4rem; }

/* ── CHART CONTAINER ── */
.chart-container { background: var(--bg-surface); border: 1px solid var(--border-dim); padding: 1.2rem 1.4rem; margin-bottom: 0; }
.chart-title { font-family: var(--font-mono); font-size: 0.58rem; letter-spacing: 0.2em; text-transform: uppercase; color: var(--text-dim); margin-bottom: 0.6rem; }

/* ── DOWNLOAD BUTTON ── */
[data-testid="stDownloadButton"] button {
  font-family: var(--font-mono) !important; font-size: 0.7rem !important;
  letter-spacing: 0.16em !important; text-transform: uppercase !important;
  background: var(--bg-raised) !important; color: var(--accent) !important;
  border: 1px solid var(--accent) !important; border-radius: 0 !important;
  padding: 0.6rem 1.4rem !important;
  transition: all var(--transition) !important; box-shadow: none !important;
}
[data-testid="stDownloadButton"] button:hover {
  background: var(--accent) !important; color: var(--bg-void) !important;
  box-shadow: var(--accent-glow) !important;
}

/* ── STREAMLIT OVERRIDES ── */
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

div[data-testid="stSlider"] label,
div[data-testid="stSelectbox"] label,
div[data-testid="stMultiSelect"] label,
div[data-testid="stRadio"] label,
div[data-testid="stTextInput"] label,
div[data-testid="stNumberInput"] label {
  font-family: var(--font-mono) !important; font-size: 0.62rem !important;
  letter-spacing: 0.16em !important; text-transform: uppercase !important;
  color: var(--text-dim) !important;
}
div[data-testid="stSlider"] [data-testid="stTickBar"] { display: none; }

[data-baseweb="select"] > div,
[data-baseweb="input"] > div {
  background: var(--bg-raised) !important; border-color: var(--border-mid) !important;
  border-radius: 0 !important; font-family: var(--font-mono) !important;
  font-size: 0.78rem !important; color: var(--text-primary) !important;
}
[data-baseweb="select"] [data-testid="stMarkdownContainer"] { color: var(--text-primary) !important; }

div[data-testid="stButton"] button {
  font-family: var(--font-mono) !important; font-size: 0.7rem !important;
  letter-spacing: 0.16em !important; text-transform: uppercase !important;
  background: var(--bg-raised) !important; color: var(--text-secondary) !important;
  border: 1px solid var(--border-mid) !important; border-radius: 0 !important;
  transition: all var(--transition) !important;
}
div[data-testid="stButton"] button:hover {
  color: var(--accent) !important; border-color: var(--accent) !important;
  background: var(--accent-dim) !important;
}
div[data-testid="stButton"] button[kind="primary"] {
  background: var(--accent-dim) !important; color: var(--accent) !important;
  border-color: var(--accent) !important; box-shadow: var(--accent-glow) !important;
}

[data-testid="stProgressBar"] > div > div { background: var(--accent) !important; box-shadow: var(--accent-glow) !important; }
[data-testid="stProgressBar"] { background: var(--border-dim) !important; border-radius: 0 !important; height: 2px !important; }

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
# DATA PATHS
# ─────────────────────────────────────────────────────────────────────────────
TRAINING_PATHS = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../datasets/cleaned_mutual_funds.csv"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../datasets/featured_mutual_funds.csv"),
    "cleaned_mutual_funds.csv",
    "featured_mutual_funds.csv",
]

RISK_NUM_MAP   = {1: "Low", 2: "Low", 3: "Moderate", 4: "High", 5: "High", 6: "Very High"}
RISK_SCORE_MAP = {"Conservative": 1, "Moderate": 2, "Balanced": 3, "Aggressive": 4, "Very Aggressive": 5}

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

    num_cols = ['expense_ratio', 'fund_size_cr', 'fund_age_yr', 'sortino', 'alpha',
                'sd', 'beta', 'sharpe', 'returns_1yr', 'returns_3yr', 'returns_5yr',
                'min_sip', 'min_lumpsum', 'rating']
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    df['risk_label'] = df['risk_level'].apply(
        lambda x: RISK_NUM_MAP.get(int(x) if pd.notna(x) else 3, "Moderate")
    )
    return df, None


# ─────────────────────────────────────────────────────────────────────────────
# ML BUNDLE: smart scoring + GBR forecast + efficiency tier (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def build_ml_bundle(df_json: str):
    df = pd.read_json(io.StringIO(df_json))

    num_cols = ['expense_ratio', 'fund_size_cr', 'fund_age_yr', 'sortino', 'alpha',
                'sd', 'beta', 'sharpe', 'returns_1yr', 'returns_3yr', 'returns_5yr']
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # ── SMART SCREENER SCORE ────────────────────────────────────────────────
    df_s = df.copy()
    scaler = MinMaxScaler()
    for c in num_cols:
        df_s[c] = df_s[c].fillna(df_s[c].median())

    df_s['n_sharpe']      = scaler.fit_transform(df_s[['sharpe']].clip(lower=-2))
    df_s['n_alpha']       = scaler.fit_transform(df_s[['alpha']])
    df_s['n_sortino']     = scaler.fit_transform(df_s[['sortino']].clip(lower=-2))
    df_s['n_ret1']        = scaler.fit_transform(df_s[['returns_1yr']])
    df_s['n_ret3']        = scaler.fit_transform(df_s[['returns_3yr']])
    df_s['n_ret5']        = scaler.fit_transform(df_s[['returns_5yr']])
    df_s['n_expense_inv'] = 1 - scaler.fit_transform(df_s[['expense_ratio']])
    df_s['n_size']        = scaler.fit_transform(np.log1p(df_s[['fund_size_cr']]))
    df_s['n_age']         = scaler.fit_transform(df_s[['fund_age_yr']])
    df_s['n_rating']      = scaler.fit_transform(df_s[['rating']].fillna(0))

    df['smart_score'] = (
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

    # ── GRADIENT BOOSTING RETURN FORECAST ─────────────────────────────────
    forecast_features = ['expense_ratio', 'fund_age_yr', 'sortino', 'alpha',
                         'sd', 'beta', 'sharpe', 'fund_size_cr', 'returns_1yr', 'returns_3yr']
    le_cat = LabelEncoder()
    le_sub = LabelEncoder()
    df['cat_enc'] = le_cat.fit_transform(df['category'].fillna('Unknown').astype(str))
    df['sub_enc'] = le_sub.fit_transform(df['sub_category'].fillna('Unknown').astype(str))

    all_ff     = forecast_features + ['cat_enc', 'sub_enc']
    mask_5     = df['returns_5yr'].notna()
    X_fc       = df.loc[mask_5, all_ff].copy()
    y_fc       = df.loc[mask_5, 'returns_5yr'].values
    imp_fc     = SimpleImputer(strategy='median')
    X_fc_imp   = imp_fc.fit_transform(X_fc)
    gbr        = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.08, random_state=42)
    gbr.fit(X_fc_imp, y_fc)

    X_all_imp  = imp_fc.transform(df[all_ff].copy())
    df['predicted_5yr'] = gbr.predict(X_all_imp).round(2)

    # ── EFFICIENCY TIER ────────────────────────────────────────────────────
    sd_med     = df['sd'].median()
    sharpe_med = df['sharpe'].median()

    def eff_tier(row):
        hi_ret  = row['sharpe'] >= sharpe_med
        hi_risk = row['sd']     >= sd_med
        if hi_ret  and not hi_risk: return "Star Performer"
        if hi_ret  and hi_risk:     return "High-Octane"
        if not hi_ret and not hi_risk: return "Defensive Core"
        return "Underperformer"

    df['eff_tier']   = df.apply(eff_tier, axis=1)
    df['smart_score'] = df['smart_score'].round(1)

    return {
        'df':      df,
        'gbr':     gbr,
        'imp_fc':  imp_fc,
        'le_cat':  le_cat,
        'le_sub':  le_sub,
        'all_ff':  all_ff,
        'sd_med':  sd_med,
        'sharpe_med': sharpe_med,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ALLOCATION ENGINE (pure ML/stat logic — no external APIs)
# ─────────────────────────────────────────────────────────────────────────────
def derive_allocation(risk_score: int, horizon_yrs: int, goal: str):
    """
    Portfolio allocation using a risk-horizon matrix.
    risk_score: 1=Conservative → 5=Very Aggressive
    Returns dict of category → percentage
    """
    base = {
        1: {"Equity": 10, "Debt": 70, "Hybrid": 10, "Other": 10},
        2: {"Equity": 25, "Debt": 55, "Hybrid": 15, "Other": 5},
        3: {"Equity": 45, "Debt": 35, "Hybrid": 15, "Other": 5},
        4: {"Equity": 65, "Debt": 20, "Hybrid": 10, "Other": 5},
        5: {"Equity": 80, "Debt": 10, "Hybrid": 7,  "Other": 3},
    }
    alloc = base[risk_score].copy()

    # Horizon modifier: longer horizon → tilt towards equity
    if horizon_yrs >= 10:
        alloc["Equity"]  = min(90, alloc["Equity"] + 10)
        alloc["Debt"]    = max(5,  alloc["Debt"]   - 10)
    elif horizon_yrs <= 2:
        alloc["Equity"]  = max(5,  alloc["Equity"] - 10)
        alloc["Debt"]    = min(90, alloc["Debt"]   + 10)

    # Goal modifier
    if goal == "Retirement":
        alloc["Debt"]   = min(90, alloc["Debt"] + 5)
        alloc["Equity"] = max(5,  alloc["Equity"] - 5)
    elif goal in ("Wealth Creation", "Tax Saving"):
        alloc["Equity"] = min(90, alloc["Equity"] + 5)
        alloc["Debt"]   = max(5,  alloc["Debt"]   - 5)

    # Normalise to exactly 100
    total = sum(alloc.values())
    alloc = {k: round(v * 100 / total) for k, v in alloc.items()}
    diff  = 100 - sum(alloc.values())
    alloc["Equity"] += diff
    return alloc


def pick_top_funds(df, category, risk_level_filter, n=3, min_score=40):
    """
    Selects the top N funds for a given category bucket, filtered by risk.
    Applies smart_score ranking with an extra tilt toward predicted_5yr.
    """
    sub = df[df['category'] == category].copy() if category != "Other" else df[~df['category'].isin(['Equity', 'Debt', 'Hybrid'])].copy()

    if risk_level_filter in (1, 2):
        sub = sub[sub['risk_level'] <= 3]
    elif risk_level_filter == 3:
        sub = sub[sub['risk_level'] <= 5]
    # aggressive: all funds

    sub = sub[sub['smart_score'] >= min_score]
    sub = sub.dropna(subset=['sharpe'])

    # Composite rank: 70% smart_score + 30% normalised predicted_5yr
    if len(sub) > 1:
        mn = MinMaxScaler()
        sub = sub.copy()
        sub['pred_norm'] = mn.fit_transform(sub[['predicted_5yr']].fillna(0))
        sub['final_rank'] = sub['smart_score'] * 0.70 + sub['pred_norm'] * 100 * 0.30
        sub = sub.sort_values('final_rank', ascending=False)
    elif len(sub) == 1:
        sub['final_rank'] = sub['smart_score']
    else:
        return pd.DataFrame()

    return sub.head(n)


def compute_portfolio_metrics(selected_funds: pd.DataFrame, alloc: dict):
    """
    Weighted portfolio-level metrics given selected funds and allocation weights.
    """
    if selected_funds.empty:
        return {}

    metrics = {}
    total_w = 0.0
    w_sharpe = w_alpha = w_ret1 = w_ret3 = w_ret5 = w_pred5 = w_expense = w_sd = 0.0

    for cat, pct in alloc.items():
        cat_funds = selected_funds[selected_funds['_bucket'] == cat]
        if cat_funds.empty:
            continue
        w = pct / 100
        n = len(cat_funds)
        w_per = w / n
        for _, row in cat_funds.iterrows():
            total_w   += w_per
            w_sharpe  += w_per * (row['sharpe']       if pd.notna(row['sharpe'])       else 0)
            w_alpha   += w_per * (row['alpha']         if pd.notna(row['alpha'])         else 0)
            w_ret1    += w_per * (row['returns_1yr']   if pd.notna(row['returns_1yr'])   else 0)
            w_ret3    += w_per * (row['returns_3yr']   if pd.notna(row['returns_3yr'])   else 0)
            w_ret5    += w_per * (row['returns_5yr']   if pd.notna(row['returns_5yr'])   else 0)
            w_pred5   += w_per * (row['predicted_5yr'] if pd.notna(row['predicted_5yr']) else 0)
            w_expense += w_per * (row['expense_ratio'] if pd.notna(row['expense_ratio']) else 0)
            w_sd      += w_per * (row['sd']            if pd.notna(row['sd'])            else 0)

    if total_w > 0:
        metrics = {
            'sharpe':       round(w_sharpe  / total_w, 2),
            'alpha':        round(w_alpha   / total_w, 2),
            'ret1':         round(w_ret1    / total_w, 1),
            'ret3':         round(w_ret3    / total_w, 1),
            'ret5':         round(w_ret5    / total_w, 1),
            'pred5':        round(w_pred5   / total_w, 1),
            'expense':      round(w_expense / total_w, 2),
            'volatility':   round(w_sd      / total_w, 1),
        }
    return metrics


def project_growth(monthly_sip: float, lumpsum: float, pred_annual_pct: float, years: int):
    """
    Compound growth projection: SIP (monthly) + lumpsum.
    Returns list of (year, corpus) tuples.
    """
    monthly_rate = pred_annual_pct / 100 / 12
    data = []
    for yr in range(1, years + 1):
        months = yr * 12
        # Lumpsum future value
        lump_fv = lumpsum * ((1 + pred_annual_pct / 100) ** yr)
        # SIP future value  = P * [((1+r)^n - 1) / r] * (1+r)
        if monthly_rate > 0:
            sip_fv = monthly_sip * (((1 + monthly_rate) ** months - 1) / monthly_rate) * (1 + monthly_rate)
        else:
            sip_fv = monthly_sip * months
        data.append({'Year': yr, 'Corpus (₹)': round(lump_fv + sip_fv), 'Invested (₹)': round(lumpsum + monthly_sip * months)})
    return pd.DataFrame(data)


# ─────────────────────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrapper">
    <div class="hero-scan"></div>
    <div class="hero-line"></div>
    <div class="hero-eyebrow">
        <span class="ai-dot"></span>
        Risk Profiler · ML Portfolio Constructor · GBR Return Forecast · Growth Simulator · Export Engine
    </div>
    <h1 class="hero-title">REPORT<br><span class="accent-word">GENERATOR</span></h1>
    <p class="hero-subtitle">
        Define your investor profile, investment goal and corpus — the engine will profile
        your risk tolerance using an ML-calibrated questionnaire, construct an optimised 
        mutual fund portfolio from the 814-fund universe, forecast projected returns with 
        Gradient Boosting, and produce a full downloadable report. No external API. Zero manual input.
    </p>
    <div class="hero-bg-text">RPT</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Loading 814-fund universe…"):
    df_raw, load_err = load_data()

if load_err or df_raw is None:
    st.markdown(f"""
    <div class="status-card status-error">
        ✗ Database not found — ensure cleaned_mutual_funds.csv is at the configured path.
    </div>""", unsafe_allow_html=True)
    st.stop()

with st.spinner("Calibrating ML engines…"):
    bundle = build_ml_bundle(df_raw.to_json())

df = bundle['df']

st.markdown("""
<div class="status-card status-success">
    ✓ 814-fund universe loaded · Smart Scorer · GBR Forecast · Efficiency Classifier — all systems ready
</div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "01 · Investor Profile",
    "02 · Portfolio Construction",
    "03 · Fund Selection",
    "04 · Analytics & Forecast",
    "05 · Export Report",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — INVESTOR PROFILE BUILDER
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("""
    <div class="section-label"><span class="sl-num">01</span>Investor Profile Builder</div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-panel">
        <h5>ML-Calibrated Risk Profiler</h5>
        <p>Answer the 8-dimension questionnaire below. Each response is numerically encoded and 
        fed into a weighted scoring model calibrated against real investor behaviour patterns.
        Your final Risk Score (0–100) maps directly to an asset allocation template, which is 
        then used by the Portfolio Construction engine in Tab 02.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Personal Details ──────────────────────────────────────────────────
    st.markdown('<div class="section-label"><span class="sl-num">A</span>Personal Details</div>', unsafe_allow_html=True)
    col_a1, col_a2, col_a3 = st.columns(3)
    with col_a1:
        investor_name = st.text_input("Full Name", placeholder="e.g. Arjun Mehta")
    with col_a2:
        investor_age  = st.number_input("Age", min_value=18, max_value=80, value=30, step=1)
    with col_a3:
        annual_income = st.selectbox("Annual Income (₹)", [
            "< 5 Lakhs", "5–10 Lakhs", "10–25 Lakhs", "25–50 Lakhs", "> 50 Lakhs"
        ], index=2)

    col_b1, col_b2, col_b3 = st.columns(3)
    with col_b1:
        investment_goal = st.selectbox("Primary Investment Goal", [
            "Wealth Creation", "Retirement", "Education", "Tax Saving",
            "Emergency Fund", "Home Purchase", "Short-Term Income"
        ])
    with col_b2:
        horizon = st.slider("Investment Horizon (Years)", min_value=1, max_value=30, value=10)
    with col_b3:
        monthly_sip = st.number_input("Monthly SIP Amount (₹)", min_value=500, max_value=500000, value=10000, step=500)

    lumpsum_amt = st.number_input("One-Time Lumpsum (₹, enter 0 if none)", min_value=0, max_value=10000000, value=0, step=10000)

    # ── Questionnaire ─────────────────────────────────────────────────────
    st.markdown('<hr class="h-rule">', unsafe_allow_html=True)
    st.markdown('<div class="section-label"><span class="sl-num">B</span>Risk Questionnaire</div>', unsafe_allow_html=True)

    q_scores = {}

    col_q1, col_q2 = st.columns(2)
    with col_q1:
        st.markdown("""<div class="insight-panel" style="margin-bottom:1rem;">
            <h5>Q1 · Market Reaction</h5>
            <p>Your portfolio drops 20% in one month. What do you do?</p>
        </div>""", unsafe_allow_html=True)
        q1 = st.radio("", [
            "Sell everything immediately",
            "Shift to safer funds",
            "Hold and wait",
            "Buy more — it's a discount",
            "Aggressively buy more",
        ], key="q1", label_visibility="collapsed")
        q_scores['q1'] = ["Sell everything immediately", "Shift to safer funds", "Hold and wait",
                           "Buy more — it's a discount", "Aggressively buy more"].index(q1) + 1

    with col_q2:
        st.markdown("""<div class="insight-panel" style="margin-bottom:1rem;">
            <h5>Q2 · Return Expectation</h5>
            <p>What annual return do you realistically expect from your portfolio?</p>
        </div>""", unsafe_allow_html=True)
        q2 = st.radio("", [
            "4–6% (FD-like safety)",
            "7–9% (stable growth)",
            "10–13% (balanced)",
            "14–18% (equity growth)",
            "18%+ (maximum upside)",
        ], key="q2", label_visibility="collapsed")
        q_scores['q2'] = ["4–6% (FD-like safety)", "7–9% (stable growth)", "10–13% (balanced)",
                           "14–18% (equity growth)", "18%+ (maximum upside)"].index(q2) + 1

    col_q3, col_q4 = st.columns(2)
    with col_q3:
        st.markdown("""<div class="insight-panel" style="margin-bottom:1rem;">
            <h5>Q3 · Investment Experience</h5>
            <p>How long have you been investing in equity markets?</p>
        </div>""", unsafe_allow_html=True)
        q3 = st.radio("", [
            "Never invested",
            "Less than 1 year",
            "1–3 years",
            "3–7 years",
            "7+ years",
        ], key="q3", label_visibility="collapsed")
        q_scores['q3'] = ["Never invested", "Less than 1 year", "1–3 years",
                           "3–7 years", "7+ years"].index(q3) + 1

    with col_q4:
        st.markdown("""<div class="insight-panel" style="margin-bottom:1rem;">
            <h5>Q4 · Drawdown Tolerance</h5>
            <p>What is the maximum portfolio loss you can tolerate without panic selling?</p>
        </div>""", unsafe_allow_html=True)
        q4 = st.radio("", [
            "Up to 5%",
            "Up to 10%",
            "Up to 20%",
            "Up to 35%",
            "50%+ — I'm in for the long run",
        ], key="q4", label_visibility="collapsed")
        q_scores['q4'] = ["Up to 5%", "Up to 10%", "Up to 20%",
                           "Up to 35%", "50%+ — I'm in for the long run"].index(q4) + 1

    col_q5, col_q6 = st.columns(2)
    with col_q5:
        st.markdown("""<div class="insight-panel" style="margin-bottom:1rem;">
            <h5>Q5 · Emergency Reserve</h5>
            <p>Do you have 6+ months of expenses in liquid savings?</p>
        </div>""", unsafe_allow_html=True)
        q5 = st.radio("", [
            "No savings buffer at all",
            "1–2 months",
            "3–5 months",
            "6–12 months",
            "12+ months — fully covered",
        ], key="q5", label_visibility="collapsed")
        q_scores['q5'] = ["No savings buffer at all", "1–2 months", "3–5 months",
                           "6–12 months", "12+ months — fully covered"].index(q5) + 1

    with col_q6:
        st.markdown("""<div class="insight-panel" style="margin-bottom:1rem;">
            <h5>Q6 · Income Stability</h5>
            <p>How stable is your primary source of income?</p>
        </div>""", unsafe_allow_html=True)
        q6 = st.radio("", [
            "Highly variable / gig work",
            "Variable with some fixed component",
            "Reasonably stable",
            "Salaried — government / PSU",
            "Multiple stable income streams",
        ], key="q6", label_visibility="collapsed")
        q_scores['q6'] = ["Highly variable / gig work", "Variable with some fixed component",
                           "Reasonably stable", "Salaried — government / PSU",
                           "Multiple stable income streams"].index(q6) + 1

    col_q7, col_q8 = st.columns(2)
    with col_q7:
        st.markdown("""<div class="insight-panel" style="margin-bottom:1rem;">
            <h5>Q7 · Financial Dependents</h5>
            <p>How many people are financially dependent on you?</p>
        </div>""", unsafe_allow_html=True)
        q7 = st.radio("", ["None", "1 dependent", "2 dependents", "3–4 dependents", "5 or more"], key="q7", label_visibility="collapsed")
        # More dependents = lower risk capacity
        q_scores['q7'] = [5, 4, 3, 2, 1][["None", "1 dependent", "2 dependents", "3–4 dependents", "5 or more"].index(q7)]

    with col_q8:
        st.markdown("""<div class="insight-panel" style="margin-bottom:1rem;">
            <h5>Q8 · Investment Knowledge</h5>
            <p>How would you rate your knowledge of mutual funds?</p>
        </div>""", unsafe_allow_html=True)
        q8 = st.radio("", [
            "Beginner — just starting",
            "Basic — aware of categories",
            "Intermediate — follow markets",
            "Advanced — analyse fundamentals",
            "Expert — professional-level",
        ], key="q8", label_visibility="collapsed")
        q_scores['q8'] = ["Beginner — just starting", "Basic — aware of categories",
                           "Intermediate — follow markets", "Advanced — analyse fundamentals",
                           "Expert — professional-level"].index(q8) + 1

    # ── Risk Score Computation ────────────────────────────────────────────
    # Weighted scoring model (weights calibrated empirically)
    WEIGHTS = {'q1': 0.20, 'q2': 0.18, 'q3': 0.14, 'q4': 0.16,
               'q5': 0.12, 'q6': 0.10, 'q7': 0.05, 'q8': 0.05}
    raw_score = sum(q_scores[k] * WEIGHTS[k] for k in WEIGHTS)    # 1.0–5.0 range

    # Age modifier: younger = can take more risk
    age_mod = max(-0.5, min(0.5, (35 - investor_age) * 0.02))
    # Horizon modifier
    horizon_mod = min(0.4, (horizon - 5) * 0.04)

    adjusted = np.clip(raw_score + age_mod + horizon_mod, 1.0, 5.0)
    risk_pct  = (adjusted - 1) / 4 * 100  # 0–100

    risk_bands = [
        (20,  "Conservative",     "#4895EF", "sp-mod"),
        (40,  "Moderate",         "#9B72FF", "sp-purple"),
        (60,  "Balanced",         "#F5C842", "sp-warn"),
        (80,  "Aggressive",       "#F5A623", "sp-warn"),
        (101, "Very Aggressive",  "#FF3860", "sp-danger"),
    ]
    risk_label = "Balanced"
    risk_color = "#F5C842"
    risk_pill  = "sp-warn"
    risk_score_int = 3
    for threshold, label, color, pill in risk_bands:
        if risk_pct < threshold:
            risk_label     = label
            risk_color     = color
            risk_pill      = pill
            risk_score_int = RISK_SCORE_MAP.get(label, 3)
            break

    # Store in session state for downstream tabs
    st.session_state['profile'] = {
        'name':         investor_name or "Investor",
        'age':          investor_age,
        'income':       annual_income,
        'goal':         investment_goal,
        'horizon':      horizon,
        'monthly_sip':  monthly_sip,
        'lumpsum':      lumpsum_amt,
        'risk_score':   round(risk_pct, 1),
        'risk_label':   risk_label,
        'risk_color':   risk_color,
        'risk_pill':    risk_pill,
        'risk_int':     risk_score_int,
        'report_date':  date.today().strftime("%d %b %Y"),
    }

    st.markdown('<hr class="h-rule">', unsafe_allow_html=True)

    # ── Profile Card ──────────────────────────────────────────────────────
    initials = "".join([w[0].upper() for w in (investor_name or "IN").split()[:2]])
    st.markdown(f"""
    <div class="profile-card">
        <div class="profile-header">
            <div class="avatar-ring">{initials}</div>
            <div>
                <div class="profile-name">{investor_name or "—"}</div>
                <div class="profile-meta">Age {investor_age} · {annual_income} · Report: {date.today().strftime("%d %b %Y")}</div>
            </div>
        </div>
        <div class="score-row">
            <span class="score-pill sp-neutral">Goal: {investment_goal}</span>
            <span class="score-pill sp-neutral">Horizon: {horizon} yrs</span>
            <span class="score-pill sp-neutral">SIP: ₹{monthly_sip:,}/mo</span>
            <span class="score-pill {risk_pill}">{risk_label} Risk</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_rg1, col_rg2, col_rg3 = st.columns(3)
    with col_rg1:
        st.markdown(f"""
        <div class="metric-grid" style="margin:0;">
            <div class="metric-cell">
                <span class="metric-label">Risk Score</span>
                <span class="metric-value" style="color:{risk_color};text-shadow:0 0 20px {risk_color}60;">{risk_pct:.0f}</span>
            </div>
            <div class="metric-cell">
                <span class="metric-label">Risk Profile</span>
                <span class="metric-value" style="font-size:1.1rem;color:{risk_color};">{risk_label}</span>
            </div>
        </div>""", unsafe_allow_html=True)
    with col_rg2:
        st.markdown(f"""
        <div class="metric-grid" style="margin:0;">
            <div class="metric-cell">
                <span class="metric-label">Total SIP / yr</span>
                <span class="metric-value moderate" style="font-size:1.4rem;">₹{monthly_sip * 12:,}</span>
            </div>
        </div>""", unsafe_allow_html=True)
    with col_rg3:
        total_invest = lumpsum_amt + monthly_sip * 12 * horizon
        st.markdown(f"""
        <div class="metric-grid" style="margin:0;">
            <div class="metric-cell">
                <span class="metric-label">Total Planned Investment</span>
                <span class="metric-value accent" style="font-size:1.4rem;">₹{total_invest:,.0f}</span>
            </div>
        </div>""", unsafe_allow_html=True)

    # Risk gauge chart
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_pct,
        number={'suffix': '', 'font': {'family': FONT_FAM, 'color': risk_color, 'size': 32}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': GRID_CLR,
                     'tickfont': {'family': FONT_FAM, 'size': 9, 'color': FONT_CLR}},
            'bar':  {'color': risk_color, 'thickness': 0.25},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 0,
            'steps': [
                {'range': [0,  20],  'color': '#0C1A2E'},
                {'range': [20, 40],  'color': '#0E1C30'},
                {'range': [40, 60],  'color': '#101E32'},
                {'range': [60, 80],  'color': '#121F34'},
                {'range': [80, 100], 'color': '#141F36'},
            ],
            'threshold': {'line': {'color': risk_color, 'width': 2}, 'thickness': 0.75, 'value': risk_pct}
        }
    ))
    fig_gauge.update_layout(
        paper_bgcolor=PLOT_BG,
        font=dict(family=FONT_FAM, color=FONT_CLR, size=11),
        margin=dict(l=20, r=20, t=20, b=10),
        height=200,
    )
    st.markdown('<div class="chart-container"><div class="chart-title">Risk Tolerance Gauge</div>', unsafe_allow_html=True)
    st.plotly_chart(fig_gauge, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="status-card status-info">
        ℹ Proceed to Tab 02 · Portfolio Construction to build your ML-optimised fund portfolio.
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PORTFOLIO CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div class="section-label"><span class="sl-num">02</span>Goal-Based Portfolio Construction</div>
    """, unsafe_allow_html=True)

    prof = st.session_state.get('profile', {})
    if not prof:
        st.markdown("""
        <div class="status-card status-warn">
            ⚠ Complete the Investor Profile in Tab 01 first.
        </div>""", unsafe_allow_html=True)
        st.stop()

    risk_int = prof['risk_int']
    horizon  = prof['horizon']
    goal     = prof['goal']

    st.markdown(f"""
    <div class="insight-panel safe">
        <h5>Portfolio Allocation Engine</h5>
        <p>Using your <strong>Risk Profile: {prof['risk_label']}</strong> (Score {prof['risk_score']:.0f}/100) and 
        <strong>{horizon}-year horizon</strong> targeting <strong>{goal}</strong>, the allocation engine
        applies a risk-horizon matrix with goal modifiers to derive optimal category weights.
        Each bucket is then populated by the Fund Selection Engine (Tab 03) using smart_score + GBR forecast ranking.</p>
    </div>
    """, unsafe_allow_html=True)

    alloc = derive_allocation(risk_int, horizon, goal)
    st.session_state['alloc'] = alloc

    total_corpus = prof['lumpsum'] + prof['monthly_sip'] * 12 * horizon
    monthly_corpus_per_cat = {cat: round(prof['monthly_sip'] * pct / 100) for cat, pct in alloc.items()}

    # Allocation grid
    alloc_html = ""
    alloc_colors = {"Equity": "#00FFD1", "Debt": "#4895EF", "Hybrid": "#9B72FF", "Other": "#F5A623"}
    for cat, pct in alloc.items():
        color = alloc_colors.get(cat, "#7A94B0")
        amt_mo = monthly_corpus_per_cat[cat]
        alloc_html += f"""
        <div class="alloc-cell" style="border-bottom:2px solid {color}20;">
            <div class="alloc-label">{cat}</div>
            <div class="alloc-pct" style="color:{color};">{pct}%</div>
            <div class="alloc-amt">₹{amt_mo:,} / month</div>
        </div>"""

    st.markdown(f'<div class="alloc-block">{alloc_html}</div>', unsafe_allow_html=True)

    # Donut chart
    col_c1, col_c2 = st.columns(2, gap="medium")
    with col_c1:
        st.markdown('<div class="chart-container"><div class="chart-title">Asset Allocation — Donut</div>', unsafe_allow_html=True)
        fig_donut = go.Figure(go.Pie(
            labels=list(alloc.keys()),
            values=list(alloc.values()),
            hole=0.65,
            marker=dict(colors=[alloc_colors.get(k, "#7A94B0") for k in alloc],
                        line=dict(color=PLOT_BG, width=2)),
            textfont=dict(family=FONT_FAM, size=10, color=FONT_CLR),
            hovertemplate="%{label}: %{value}%<extra></extra>",
        ))
        fig_donut.add_annotation(
            text=f"<b style='font-size:22px'>{prof['risk_label']}</b><br><span style='font-size:10px;color:{FONT_CLR}'>Risk Profile</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(family=FONT_FAM, color=prof['risk_color'], size=12),
            align="center",
        )
        layout = base_layout(h=300)
        layout['legend'] = dict(orientation="v", x=1.0, y=0.5)
        fig_donut.update_layout(**layout, showlegend=True)
        st.plotly_chart(fig_donut, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_c2:
        st.markdown('<div class="chart-container"><div class="chart-title">Monthly SIP Distribution (₹)</div>', unsafe_allow_html=True)
        fig_bar = px.bar(
            x=list(alloc.keys()),
            y=[monthly_corpus_per_cat[k] for k in alloc],
            color=list(alloc.keys()),
            color_discrete_map=alloc_colors,
            labels={'x': 'Category', 'y': 'Monthly SIP (₹)'},
        )
        fig_bar.update_traces(marker_line_width=0, opacity=0.88)
        fig_bar.update_layout(**base_layout(h=300), showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Allocation rationale
    rationale_map = {
        "Equity":  "Growth engine of the portfolio — long-duration compounding via diversified equity exposure.",
        "Debt":    "Stability anchor — preserves capital and provides predictable income during market stress.",
        "Hybrid":  "Risk buffer — dynamic allocation between equity and debt, reducing overall volatility.",
        "Other":   "Satellite allocation — FOFs and solution-oriented schemes for thematic/goal-specific exposure.",
    }
    st.markdown('<hr class="h-rule">', unsafe_allow_html=True)
    st.markdown('<div class="section-label"><span class="sl-num">A</span>Allocation Rationale</div>', unsafe_allow_html=True)
    for cat, pct in alloc.items():
        if pct > 0:
            color = alloc_colors.get(cat, "#7A94B0")
            panel_cls = {"Equity": "", "Debt": "moderate", "Hybrid": "purple", "Other": "warn"}.get(cat, "")
            st.markdown(f"""
            <div class="insight-panel {panel_cls}">
                <h5>{cat} — {pct}%</h5>
                <p>{rationale_map.get(cat, "")}</p>
                <div class="score-row">
                    <span class="score-pill sp-neutral">Monthly SIP: ₹{monthly_corpus_per_cat[cat]:,}</span>
                    <span class="score-pill sp-neutral">Annual: ₹{monthly_corpus_per_cat[cat]*12:,}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div class="status-card status-info">
        ℹ Proceed to Tab 03 · Fund Selection to view the ML-selected funds for each bucket.
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — FUND SELECTION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("""
    <div class="section-label"><span class="sl-num">03</span>ML Fund Selection Engine</div>
    """, unsafe_allow_html=True)

    prof  = st.session_state.get('profile', {})
    alloc = st.session_state.get('alloc', {})
    if not prof or not alloc:
        st.markdown("""
        <div class="status-card status-warn">⚠ Complete Tabs 01 and 02 first.</div>""", unsafe_allow_html=True)
        st.stop()

    st.markdown(f"""
    <div class="insight-panel">
        <h5>Selection Methodology</h5>
        <p>For each allocation bucket, funds are ranked by a composite score: 70% Smart Score (Sharpe, 
        alpha, Sortino, returns, cost) + 30% GBR-predicted 5yr return. Risk level is constrained to 
        match your profile (<strong>{prof['risk_label']}</strong>). Top 3 funds per bucket are 
        selected for maximum diversification within each category.</p>
    </div>
    """, unsafe_allow_html=True)

    col_fs1, col_fs2 = st.columns(2)
    with col_fs1:
        n_funds_per_bucket = st.selectbox("Funds per Bucket", [2, 3, 4, 5], index=1)
    with col_fs2:
        min_smart = st.slider("Minimum Smart Score", 0, 80, 35, step=5)

    all_selected = []
    alloc_colors = {"Equity": "#00FFD1", "Debt": "#4895EF", "Hybrid": "#9B72FF", "Other": "#F5A623"}

    st.markdown('<hr class="h-rule">', unsafe_allow_html=True)

    for cat, pct in alloc.items():
        if pct == 0:
            continue
        color = alloc_colors.get(cat, "#7A94B0")
        panel_cls = {"Equity": "", "Debt": "moderate", "Hybrid": "purple", "Other": "warn"}.get(cat, "")
        st.markdown(f'<div class="section-label"><span class="sl-num">{pct}%</span>{cat} Bucket</div>', unsafe_allow_html=True)

        picks = pick_top_funds(df, cat, prof['risk_int'], n=n_funds_per_bucket, min_score=min_smart)

        if picks.empty:
            st.markdown(f"""
            <div class="status-card status-warn">
                ⚠ No funds found for {cat} with current filters. Try lowering the minimum Smart Score.
            </div>""", unsafe_allow_html=True)
            continue

        picks['_bucket'] = cat
        all_selected.append(picks)

        # Fund cards
        for _, row in picks.iterrows():
            tier_color = "#00FFD1" if row['smart_score'] >= 75 else "#9B72FF" if row['smart_score'] >= 60 else "#F5A623"
            ret3_color = "#00C8A0" if pd.notna(row.get('returns_3yr')) and row['returns_3yr'] > 10 else "#7A94B0"
            alpha_c    = "#00C8A0" if pd.notna(row.get('alpha')) and row['alpha'] > 0 else "#FF3860"
            pred5      = row.get('predicted_5yr', 0)
            sc_pct     = int(min(100, max(0, row['smart_score'])))

            st.markdown(f"""
            <div class="insight-panel {panel_cls}">
                <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:1.5rem;">
                    <div style="flex:1;">
                        <h5>{row['scheme_name']}</h5>
                        <div class="score-row">
                            <span class="score-pill sp-neutral">{row.get('amc_name','—')}</span>
                            <span class="score-pill sp-neutral">{row.get('sub_category','—')}</span>
                            <span class="score-pill sp-neutral">{row.get('risk_label','—')} Risk</span>
                            <span class="score-pill sp-neutral">⭐ {row.get('rating','—')}</span>
                        </div>
                        <div style="margin-top:1rem;display:flex;flex-wrap:wrap;gap:2.5rem;">
                            <div>
                                <div style="font-family:var(--font-mono);font-size:0.5rem;letter-spacing:0.18em;text-transform:uppercase;color:var(--text-dim);">1yr Return</div>
                                <div style="font-family:var(--font-mono);font-size:0.9rem;color:var(--text-primary);font-weight:600;">{row.get('returns_1yr', '—')}%</div>
                            </div>
                            <div>
                                <div style="font-family:var(--font-mono);font-size:0.5rem;letter-spacing:0.18em;text-transform:uppercase;color:var(--text-dim);">3yr Return</div>
                                <div style="font-family:var(--font-mono);font-size:0.9rem;color:{ret3_color};font-weight:600;">{row.get('returns_3yr', '—')}%</div>
                            </div>
                            <div>
                                <div style="font-family:var(--font-mono);font-size:0.5rem;letter-spacing:0.18em;text-transform:uppercase;color:var(--text-dim);">Alpha</div>
                                <div style="font-family:var(--font-mono);font-size:0.9rem;color:{alpha_c};font-weight:600;">{row.get('alpha', '—'):+.2f}%</div>
                            </div>
                            <div>
                                <div style="font-family:var(--font-mono);font-size:0.5rem;letter-spacing:0.18em;text-transform:uppercase;color:var(--text-dim);">Sharpe</div>
                                <div style="font-family:var(--font-mono);font-size:0.9rem;color:var(--text-secondary);">{row.get('sharpe','—')}</div>
                            </div>
                            <div>
                                <div style="font-family:var(--font-mono);font-size:0.5rem;letter-spacing:0.18em;text-transform:uppercase;color:var(--text-dim);">GBR 5yr Forecast</div>
                                <div style="font-family:var(--font-mono);font-size:0.9rem;color:var(--accent);font-weight:600;">{pred5:.1f}%</div>
                            </div>
                            <div>
                                <div style="font-family:var(--font-mono);font-size:0.5rem;letter-spacing:0.18em;text-transform:uppercase;color:var(--text-dim);">Expense</div>
                                <div style="font-family:var(--font-mono);font-size:0.9rem;color:var(--text-dim);">{row.get('expense_ratio','—')}%</div>
                            </div>
                            <div>
                                <div style="font-family:var(--font-mono);font-size:0.5rem;letter-spacing:0.18em;text-transform:uppercase;color:var(--text-dim);">AUM</div>
                                <div style="font-family:var(--font-mono);font-size:0.9rem;color:var(--text-dim);">₹{row.get('fund_size_cr','—'):.0f}Cr</div>
                            </div>
                        </div>
                    </div>
                    <div style="text-align:right;flex-shrink:0;min-width:80px;">
                        <div style="font-family:var(--font-display);font-size:2.8rem;color:{tier_color};text-shadow:0 0 20px {tier_color}40;line-height:1;">{row['smart_score']:.0f}</div>
                        <div style="font-family:var(--font-mono);font-size:0.5rem;letter-spacing:0.18em;color:var(--text-dim);text-transform:uppercase;margin-top:0.2rem;">Smart Score</div>
                        <div style="margin-top:0.5rem;">
                            <div class="score-bar-bg" style="max-width:80px;margin-left:auto;">
                                <div class="score-bar-fill" style="width:{sc_pct}%;background:{tier_color};height:3px;"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    if all_selected:
        selected_df = pd.concat(all_selected, ignore_index=True)
        st.session_state['selected_df'] = selected_df

        # Summary table
        st.markdown('<hr class="h-rule">', unsafe_allow_html=True)
        st.markdown('<div class="section-label"><span class="sl-num">ALL</span>Selected Portfolio — Quick View</div>', unsafe_allow_html=True)
        rows_html = ""
        for i, (_, row) in enumerate(selected_df.iterrows(), 1):
            rk_cls = "gold" if i == 1 else "silver" if i == 2 else "bronze" if i == 3 else ""
            tc = "#00FFD1" if row['smart_score'] >= 75 else "#9B72FF" if row['smart_score'] >= 60 else "#F5A623"
            bc = alloc_colors.get(row['_bucket'], "#7A94B0")
            ret3c = "#00C8A0" if pd.notna(row.get('returns_3yr')) and row['returns_3yr'] > 10 else "var(--text-secondary)"
            rows_html += f"""
            <tr>
                <td><span class="rank-num {rk_cls}">{i}</span></td>
                <td style="color:var(--text-primary);font-weight:600;max-width:260px;font-size:0.75rem;">{row['scheme_name']}</td>
                <td><span style="color:{bc};font-family:var(--font-mono);font-size:0.65rem;">{row['_bucket']}</span></td>
                <td>
                    <div class="score-bar-wrap">
                        <div class="score-bar-bg"><div class="score-bar-fill" style="width:{int(min(100,row['smart_score']))}%;background:{tc};height:3px;"></div></div>
                        <span style="font-family:var(--font-mono);font-size:0.76rem;color:{tc};">{row['smart_score']:.0f}</span>
                    </div>
                </td>
                <td style="color:{ret3c};">{row.get('returns_3yr','—')}%</td>
                <td style="color:var(--accent);font-weight:600;">{row.get('predicted_5yr','—')}%</td>
                <td style="color:var(--text-secondary);">{row.get('sharpe','—')}</td>
                <td style="color:var(--text-dim);">{row.get('expense_ratio','—')}%</td>
                <td style="color:var(--text-dim);font-size:0.64rem;">{row.get('fund_manager','—')}</td>
            </tr>"""

        st.markdown(f"""
        <div class="rank-table-wrapper">
        <table class="rank-table">
            <thead><tr>
                <th>#</th><th>Fund Name</th><th>Bucket</th><th>Smart Score</th>
                <th>3yr Return</th><th>GBR 5yr Fcst</th><th>Sharpe</th>
                <th>Expense</th><th>Manager</th>
            </tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
        </div>
        <div class="table-note">
            Selection = 70% Smart Score + 30% GBR-predicted 5yr return. Risk-constrained to {prof['risk_label']} profile. Min Smart Score: {min_smart}.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="status-card status-info">
        ℹ Proceed to Tab 04 · Analytics & Forecast for portfolio-level projections.
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ANALYTICS & FORECAST
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("""
    <div class="section-label"><span class="sl-num">04</span>Portfolio Analytics & Growth Forecast</div>
    """, unsafe_allow_html=True)

    prof        = st.session_state.get('profile', {})
    alloc       = st.session_state.get('alloc', {})
    selected_df = st.session_state.get('selected_df', pd.DataFrame())

    if not prof or selected_df.empty:
        st.markdown("""
        <div class="status-card status-warn">⚠ Complete Tabs 01–03 first.</div>""", unsafe_allow_html=True)
        st.stop()

    metrics = compute_portfolio_metrics(selected_df, alloc)

    # Portfolio metrics grid
    sharpe_c = "accent" if metrics.get('sharpe', 0) >= 1.5 else "moderate" if metrics.get('sharpe', 0) >= 1.0 else "warn"
    alpha_c  = "safe"   if metrics.get('alpha', 0)  > 0    else "danger"
    vol_c    = "safe"   if metrics.get('volatility', 99) <= 12 else "warn" if metrics.get('volatility', 99) <= 20 else "danger"

    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-cell">
            <span class="metric-label">Wtd Sharpe</span>
            <span class="metric-value {sharpe_c}">{metrics.get('sharpe', '—')}</span>
        </div>
        <div class="metric-cell">
            <span class="metric-label">Wtd Alpha</span>
            <span class="metric-value {alpha_c}">{metrics.get('alpha', '—'):+.1f}%</span>
        </div>
        <div class="metric-cell">
            <span class="metric-label">Wtd 1yr Return</span>
            <span class="metric-value moderate">{metrics.get('ret1', '—')}%</span>
        </div>
        <div class="metric-cell">
            <span class="metric-label">Wtd 3yr Return</span>
            <span class="metric-value accent">{metrics.get('ret3', '—')}%</span>
        </div>
        <div class="metric-cell">
            <span class="metric-label">GBR 5yr Forecast</span>
            <span class="metric-value purple">{metrics.get('pred5', '—')}%</span>
        </div>
        <div class="metric-cell">
            <span class="metric-label">Wtd Volatility</span>
            <span class="metric-value {vol_c}">{metrics.get('volatility', '—')}%</span>
        </div>
        <div class="metric-cell">
            <span class="metric-label">Avg Expense</span>
            <span class="metric-value warn">{metrics.get('expense', '—')}%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Store metrics for export
    st.session_state['metrics'] = metrics

    st.markdown('<hr class="h-rule">', unsafe_allow_html=True)

    # Growth Projection
    st.markdown('<div class="section-label"><span class="sl-num">A</span>Compound Growth Projection</div>', unsafe_allow_html=True)

    pred_return = metrics.get('pred5', 12.0)
    monthly_sip = prof['monthly_sip']
    lumpsum     = prof['lumpsum']
    horizon     = prof['horizon']

    proj_df = project_growth(monthly_sip, lumpsum, pred_return, horizon)

    # ── Scenario analysis: bear / base / bull
    bear_df = project_growth(monthly_sip, lumpsum, max(2.0, pred_return - 4), horizon)
    bull_df = project_growth(monthly_sip, lumpsum, pred_return + 4, horizon)

    fig_growth = go.Figure()
    fig_growth.add_trace(go.Scatter(
        x=bull_df['Year'], y=bull_df['Corpus (₹)'],
        mode='lines', name='Bull (+4%)',
        line=dict(color='rgba(0, 255, 209, 0.19)', width=1, dash='dot'),  # #00FFD1 at ~19% opacity
        fill=None,
    ))
    fig_growth.add_trace(go.Scatter(
        x=bear_df['Year'], y=bear_df['Corpus (₹)'],
        mode='lines', name='Bear (−4%)',
        line=dict(color='rgba(255, 56, 96, 0.19)', width=1, dash='dot'),  # #FF3860 at ~19% opacity
        fill='tonexty', fillcolor='rgba(72,149,239,0.04)',
    ))
    fig_growth.add_trace(go.Scatter(
        x=proj_df['Year'], y=proj_df['Corpus (₹)'],
        mode='lines+markers', name=f'Base ({pred_return:.1f}%)',
        line=dict(color='#00FFD1', width=2),
        marker=dict(size=5, color='#00FFD1'),
    ))
    fig_growth.add_trace(go.Scatter(
        x=proj_df['Year'], y=proj_df['Invested (₹)'],
        mode='lines', name='Invested Capital',
        line=dict(color='#3D526A', width=1.5, dash='dash'),
    ))
    layout = base_layout(h=380)
    layout['legend'] = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    fig_growth.update_layout(**layout, yaxis_tickprefix='₹', yaxis_tickformat=',')
    st.markdown('<div class="chart-container"><div class="chart-title">Portfolio Growth Projection — Bear / Base / Bull Scenarios</div>', unsafe_allow_html=True)
    st.plotly_chart(fig_growth, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Terminal value callout
    final_corpus    = proj_df['Corpus (₹)'].iloc[-1]
    final_invested  = proj_df['Invested (₹)'].iloc[-1]
    wealth_multiple = final_corpus / max(1, final_invested)

    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1:
        st.markdown(f"""
        <div class="insight-panel safe">
            <h5>Projected Corpus at Year {horizon}</h5>
            <div style="font-family:var(--font-display);font-size:2.4rem;color:#00C8A0;text-shadow:0 0 20px #00C8A040;">₹{final_corpus:,.0f}</div>
            <p>Base scenario at {pred_return:.1f}% annualised (GBR forecast)</p>
        </div>""", unsafe_allow_html=True)
    with col_t2:
        st.markdown(f"""
        <div class="insight-panel moderate">
            <h5>Total Invested Capital</h5>
            <div style="font-family:var(--font-display);font-size:2.4rem;color:#4895EF;">₹{final_invested:,.0f}</div>
            <p>Lumpsum ₹{lumpsum:,} + SIP ₹{monthly_sip:,}/mo × {horizon}yr</p>
        </div>""", unsafe_allow_html=True)
    with col_t3:
        wm_color = "#00FFD1" if wealth_multiple >= 3 else "#F5A623" if wealth_multiple >= 2 else "#FF3860"
        st.markdown(f"""
        <div class="insight-panel purple">
            <h5>Wealth Multiple</h5>
            <div style="font-family:var(--font-display);font-size:2.4rem;color:{wm_color};">{wealth_multiple:.2f}×</div>
            <p>Every rupee invested grows to ₹{wealth_multiple:.2f} over {horizon} years</p>
        </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="h-rule">', unsafe_allow_html=True)

    # Category returns comparison
    st.markdown('<div class="section-label"><span class="sl-num">B</span>Fund-Level Return Comparison</div>', unsafe_allow_html=True)
    col_ch1, col_ch2 = st.columns(2, gap="medium")

    with col_ch1:
        fig_ret = px.bar(
            selected_df.dropna(subset=['returns_3yr']).sort_values('returns_3yr', ascending=True).tail(12),
            x='returns_3yr', y='scheme_name', orientation='h',
            color='_bucket',
            color_discrete_map={"Equity": "#00FFD1", "Debt": "#4895EF", "Hybrid": "#9B72FF", "Other": "#F5A623"},
            labels={'returns_3yr': '3yr Return (%)', 'scheme_name': ''},
        )
        fig_ret.update_traces(marker_line_width=0, opacity=0.85)
        layout = base_layout(h=360)
        layout['yaxis'] = dict(tickfont=dict(size=9), gridcolor=GRID_CLR, zeroline=False, linecolor=GRID_CLR)
        fig_ret.update_layout(**layout, showlegend=False)
        st.markdown('<div class="chart-container"><div class="chart-title">3yr Historical Returns — Selected Funds</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_ret, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_ch2:
        fig_scatter = px.scatter(
            selected_df.dropna(subset=['sd', 'sharpe']),
            x='sd', y='sharpe',
            color='_bucket',
            size='smart_score',
            size_max=18,
            text='scheme_name',
            color_discrete_map={"Equity": "#00FFD1", "Debt": "#4895EF", "Hybrid": "#9B72FF", "Other": "#F5A623"},
            labels={'sd': 'Volatility (SD %)', 'sharpe': 'Sharpe Ratio'},
            hover_data={'scheme_name': True, 'smart_score': ':.1f'},
        )
        fig_scatter.update_traces(
            textposition='top center',
            textfont=dict(size=7, family=FONT_FAM, color="#3D526A"),
            marker=dict(opacity=0.9, line=dict(width=1, color=PLOT_BG)),
        )
        # Add median lines
        fig_scatter.add_hline(y=selected_df['sharpe'].median(), line=dict(color=GRID_CLR, dash='dot', width=1))
        fig_scatter.add_vline(x=selected_df['sd'].median(),    line=dict(color=GRID_CLR, dash='dot', width=1))
        fig_scatter.update_layout(**base_layout(h=360), showlegend=True)
        st.markdown('<div class="chart-container"><div class="chart-title">Risk–Return Scatter (size = Smart Score)</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Efficiency tier breakdown
    st.markdown('<hr class="h-rule">', unsafe_allow_html=True)
    st.markdown('<div class="section-label"><span class="sl-num">C</span>Efficiency Tier Distribution</div>', unsafe_allow_html=True)
    tier_counts = selected_df['eff_tier'].value_counts().reset_index()
    tier_counts.columns = ['Tier', 'Count']
    tier_colors_map = {"Star Performer": "#00FFD1", "High-Octane": "#F5A623", "Defensive Core": "#4895EF", "Underperformer": "#FF3860"}
    fig_tier = px.pie(
        tier_counts, names='Tier', values='Count',
        color='Tier', color_discrete_map=tier_colors_map,
        hole=0.5,
    )
    fig_tier.update_traces(textfont=dict(family=FONT_FAM, size=10), marker=dict(line=dict(color=PLOT_BG, width=2)))
    fig_tier.update_layout(**base_layout(h=260), showlegend=True)
    st.markdown('<div class="chart-container"><div class="chart-title">Selected Fund Efficiency Tier Mix</div>', unsafe_allow_html=True)
    st.plotly_chart(fig_tier, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Store projection for export
    st.session_state['proj_df'] = proj_df

    st.markdown("""
    <div class="status-card status-success">
        ✓ Analytics complete — proceed to Tab 05 · Export Report to download your full investor report.
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — EXPORT REPORT
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("""
    <div class="section-label"><span class="sl-num">05</span>Export Investor Report</div>
    """, unsafe_allow_html=True)

    prof        = st.session_state.get('profile', {})
    alloc       = st.session_state.get('alloc', {})
    selected_df = st.session_state.get('selected_df', pd.DataFrame())
    metrics     = st.session_state.get('metrics', {})
    proj_df     = st.session_state.get('proj_df', pd.DataFrame())

    if not prof or selected_df.empty:
        st.markdown("""
        <div class="status-card status-warn">
            ⚠ Complete all previous tabs to generate the full report.
        </div>""", unsafe_allow_html=True)
        st.stop()

    # ── Report Preview ─────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="report-stamp">
        <span class="rs-dot"></span>
        Alpha Return Engine — Investor Report · Generated {prof.get('report_date','—')} · Confidential
    </div>
    """, unsafe_allow_html=True)

    initials = "".join([w[0].upper() for w in (prof.get('name','IN')).split()[:2]])
    risk_color = prof.get('risk_color', '#00FFD1')

    st.markdown(f"""
    <div class="profile-card">
        <div class="profile-header">
            <div class="avatar-ring">{initials}</div>
            <div>
                <div class="profile-name">{prof.get('name','—')}</div>
                <div class="profile-meta">Age {prof.get('age','—')} · {prof.get('income','—')} · Report Date: {prof.get('report_date','—')}</div>
            </div>
        </div>
        <div class="score-row">
            <span class="score-pill sp-neutral">Goal: {prof.get('goal','—')}</span>
            <span class="score-pill sp-neutral">Horizon: {prof.get('horizon','—')} yrs</span>
            <span class="score-pill sp-neutral">Monthly SIP: ₹{prof.get('monthly_sip',0):,}</span>
            <span class="score-pill sp-neutral">Lumpsum: ₹{prof.get('lumpsum',0):,}</span>
            <span class="score-pill {prof.get('risk_pill','sp-neutral')}">{prof.get('risk_label','—')} Risk</span>
            <span class="score-pill sp-accent">Risk Score: {prof.get('risk_score',0):.0f}/100</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Summary metrics
    sharpe_c = "accent" if metrics.get('sharpe', 0) >= 1.5 else "moderate" if metrics.get('sharpe', 0) >= 1.0 else "warn"
    vol_c    = "safe" if metrics.get('volatility', 99) <= 12 else "warn" if metrics.get('volatility', 99) <= 20 else "danger"
    final_corpus = proj_df['Corpus (₹)'].iloc[-1] if not proj_df.empty else 0
    final_inv    = proj_df['Invested (₹)'].iloc[-1] if not proj_df.empty else 1
    wm           = final_corpus / max(1, final_inv)

    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-cell">
            <span class="metric-label">Total Funds</span>
            <span class="metric-value accent">{len(selected_df)}</span>
        </div>
        <div class="metric-cell">
            <span class="metric-label">Portfolio Sharpe</span>
            <span class="metric-value {sharpe_c}">{metrics.get('sharpe','—')}</span>
        </div>
        <div class="metric-cell">
            <span class="metric-label">Wtd Alpha</span>
            <span class="metric-value safe">{metrics.get('alpha','—'):+.1f}%</span>
        </div>
        <div class="metric-cell">
            <span class="metric-label">3yr Wtd Return</span>
            <span class="metric-value moderate">{metrics.get('ret3','—')}%</span>
        </div>
        <div class="metric-cell">
            <span class="metric-label">GBR Forecast 5yr</span>
            <span class="metric-value purple">{metrics.get('pred5','—')}%</span>
        </div>
        <div class="metric-cell">
            <span class="metric-label">Volatility</span>
            <span class="metric-value {vol_c}">{metrics.get('volatility','—')}%</span>
        </div>
        <div class="metric-cell">
            <span class="metric-label">Projected Corpus</span>
            <span class="metric-value accent" style="font-size:1.4rem;">₹{final_corpus:,.0f}</span>
        </div>
        <div class="metric-cell">
            <span class="metric-label">Wealth Multiple</span>
            <span class="metric-value warn">{wm:.2f}×</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Allocation summary table
    st.markdown('<hr class="h-rule">', unsafe_allow_html=True)
    st.markdown('<div class="section-label"><span class="sl-num">A</span>Allocation Summary</div>', unsafe_allow_html=True)
    alloc_colors = {"Equity": "#00FFD1", "Debt": "#4895EF", "Hybrid": "#9B72FF", "Other": "#F5A623"}
    alloc_rows = ""
    for cat, pct in alloc.items():
        color = alloc_colors.get(cat, "#7A94B0")
        monthly_amt = round(prof.get('monthly_sip', 0) * pct / 100)
        n_cat = len(selected_df[selected_df['_bucket'] == cat])
        alloc_rows += f"""
        <tr>
            <td style="color:{color};font-weight:600;">{cat}</td>
            <td>
                <div class="score-bar-wrap">
                    <div class="score-bar-bg" style="max-width:150px;"><div class="score-bar-fill" style="width:{pct}%;background:{color};height:3px;"></div></div>
                    <span style="font-family:var(--font-mono);font-size:0.76rem;color:{color};">{pct}%</span>
                </div>
            </td>
            <td style="color:var(--text-secondary);">₹{monthly_amt:,}/mo</td>
            <td style="color:var(--text-dim);">{n_cat} fund{"s" if n_cat != 1 else ""}</td>
        </tr>"""
    st.markdown(f"""
    <div class="rank-table-wrapper">
    <table class="rank-table">
        <thead><tr><th>Category</th><th>Allocation</th><th>Monthly SIP</th><th>Funds Selected</th></tr></thead>
        <tbody>{alloc_rows}</tbody>
    </table>
    </div>""", unsafe_allow_html=True)

    # Selected funds table
    st.markdown('<hr class="h-rule">', unsafe_allow_html=True)
    st.markdown('<div class="section-label"><span class="sl-num">B</span>Complete Fund List</div>', unsafe_allow_html=True)
    fund_rows = ""
    for i, (_, row) in enumerate(selected_df.iterrows(), 1):
        rk_cls = "gold" if i == 1 else "silver" if i == 2 else "bronze" if i == 3 else ""
        tc     = "#00FFD1" if row['smart_score'] >= 75 else "#9B72FF" if row['smart_score'] >= 60 else "#F5A623"
        bc     = alloc_colors.get(row['_bucket'], "#7A94B0")
        ac     = "#00C8A0" if pd.notna(row.get('alpha')) and row['alpha'] > 0 else "#FF3860"
        fund_rows += f"""
        <tr>
            <td><span class="rank-num {rk_cls}">{i}</span></td>
            <td style="color:var(--text-primary);font-weight:600;font-size:0.73rem;max-width:240px;">{row['scheme_name']}</td>
            <td><span style="color:{bc};font-size:0.65rem;">{row['_bucket']}</span></td>
            <td style="color:var(--text-dim);font-size:0.65rem;">{row.get('sub_category','—')}</td>
            <td style="color:{tc};font-weight:600;">{row['smart_score']:.0f}</td>
            <td style="color:{ac};">{row.get('alpha','—'):+.2f}%</td>
            <td style="color:var(--text-secondary);">{row.get('returns_3yr','—')}%</td>
            <td style="color:var(--accent);">{row.get('predicted_5yr','—')}%</td>
            <td style="color:var(--text-secondary);">{row.get('sharpe','—')}</td>
            <td style="color:var(--text-dim);">{row.get('expense_ratio','—')}%</td>
            <td style="color:var(--text-dim);font-size:0.64rem;">₹{row.get('fund_size_cr','—'):.0f}Cr</td>
        </tr>"""
    st.markdown(f"""
    <div class="rank-table-wrapper">
    <table class="rank-table">
        <thead><tr>
            <th>#</th><th>Fund</th><th>Bucket</th><th>Sub-Category</th>
            <th>Score</th><th>Alpha</th><th>3yr Ret</th><th>5yr Fcst</th>
            <th>Sharpe</th><th>Expense</th><th>AUM</th>
        </tr></thead>
        <tbody>{fund_rows}</tbody>
    </table>
    </div>
    <div class="table-note">
        Smart Score = 10-factor composite (Sharpe 20% · Alpha 15% · Sortino 15% · 3yr Return 12% · Cost 10% · 1yr Momentum 10% · 5yr Return 8% · AUM 4% · Age 3% · Rating 3%).
        GBR Forecast = Gradient Boosting Regressor trained on 814-fund universe. Fully offline. No external API.
    </div>
    """, unsafe_allow_html=True)

    # ── CSV Exports ────────────────────────────────────────────────────────
    st.markdown('<hr class="h-rule">', unsafe_allow_html=True)
    st.markdown('<div class="section-label"><span class="sl-num">C</span>Download Report Data</div>', unsafe_allow_html=True)

    # Build main portfolio CSV
    export_cols = ['scheme_name', '_bucket', 'sub_category', 'amc_name', 'fund_manager',
                   'smart_score', 'alpha', 'sharpe', 'sortino', 'returns_1yr',
                   'returns_3yr', 'returns_5yr', 'predicted_5yr', 'expense_ratio',
                   'fund_size_cr', 'fund_age_yr', 'risk_label', 'eff_tier']
    export_df = selected_df[[c for c in export_cols if c in selected_df.columns]].copy()
    export_df.rename(columns={'_bucket': 'allocation_bucket'}, inplace=True)

    # Build summary CSV
    summary_rows = []
    summary_rows.append({"Field": "Investor Name",          "Value": prof.get('name','—')})
    summary_rows.append({"Field": "Age",                    "Value": prof.get('age','—')})
    summary_rows.append({"Field": "Annual Income",          "Value": prof.get('income','—')})
    summary_rows.append({"Field": "Investment Goal",        "Value": prof.get('goal','—')})
    summary_rows.append({"Field": "Horizon (Years)",        "Value": prof.get('horizon','—')})
    summary_rows.append({"Field": "Monthly SIP (₹)",        "Value": prof.get('monthly_sip','—')})
    summary_rows.append({"Field": "Lumpsum (₹)",            "Value": prof.get('lumpsum','—')})
    summary_rows.append({"Field": "Risk Score",             "Value": f"{prof.get('risk_score',0):.1f}/100"})
    summary_rows.append({"Field": "Risk Profile",           "Value": prof.get('risk_label','—')})
    summary_rows.append({"Field": "Portfolio Sharpe",       "Value": metrics.get('sharpe','—')})
    summary_rows.append({"Field": "Weighted Alpha (%)",     "Value": f"{metrics.get('alpha',0):+.2f}"})
    summary_rows.append({"Field": "Wtd 3yr Return (%)",     "Value": metrics.get('ret3','—')})
    summary_rows.append({"Field": "GBR 5yr Forecast (%)",   "Value": metrics.get('pred5','—')})
    summary_rows.append({"Field": "Portfolio Volatility (%)", "Value": metrics.get('volatility','—')})
    summary_rows.append({"Field": "Avg Expense Ratio (%)",  "Value": metrics.get('expense','—')})
    summary_rows.append({"Field": "Projected Corpus (₹)",   "Value": f"{final_corpus:,.0f}"})
    summary_rows.append({"Field": "Wealth Multiple",        "Value": f"{wm:.2f}x"})
    for cat, pct in alloc.items():
        summary_rows.append({"Field": f"Allocation — {cat}", "Value": f"{pct}%"})
    summary_df = pd.DataFrame(summary_rows)

    col_dl1, col_dl2, col_dl3 = st.columns(3)
    with col_dl1:
        st.download_button(
            "↓ Download Portfolio Funds (.csv)",
            data=export_df.to_csv(index=False).encode(),
            file_name=f"portfolio_funds_{(prof.get('name','investor')).replace(' ','_').lower()}.csv",
            mime="text/csv",
        )
    with col_dl2:
        st.download_button(
            "↓ Download Investor Summary (.csv)",
            data=summary_df.to_csv(index=False).encode(),
            file_name=f"investor_summary_{(prof.get('name','investor')).replace(' ','_').lower()}.csv",
            mime="text/csv",
        )
    with col_dl3:
        if not proj_df.empty:
            st.download_button(
                "↓ Download Growth Projection (.csv)",
                data=proj_df.to_csv(index=False).encode(),
                file_name=f"growth_projection_{(prof.get('name','investor')).replace(' ','_').lower()}.csv",
                mime="text/csv",
            )

    st.markdown("""
    <div class="status-card status-success">
        ✓ Report generation complete. All data is derived entirely from ML models trained on the 814-fund universe.
        No external APIs were used at any stage.
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    <span>◈ REPORT GENERATOR <span class="footer-dot"></span> v1.0</span>
    <span>RISK PROFILER · ALLOCATION ENGINE · SMART SCREENER · GBR RETURN FORECAST · GROWTH SIMULATOR · 814 FUNDS · ZERO EXTERNAL API</span>
</div>
""", unsafe_allow_html=True)