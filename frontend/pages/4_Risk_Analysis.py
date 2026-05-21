# frontend/pages/3_Risk_Analysis.py
# ─────────────────────────────────────────────────────────────────────────────
#  ALPHA RETURN ENGINE — Risk Analysis Module
#  ML-powered risk profiling trained on 814-fund database.
#  Works identically to the Return Prediction page: upload any CSV/XLSX
#  portfolio, the engine fuzzy-matches funds against training data, fills
#  missing features from category statistics, and runs a trained
#  Random-Forest classifier to predict composite risk scores — then
#  delivers institutional-grade risk intelligence dashboards.
# ─────────────────────────────────────────────────────────────────────────────

import io
import os
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

# NOTE: Do not call st.set_page_config() in page files when using frontend/app.py.
# Sidebar visibility/state should be controlled centrally.

# ─────────────────────────────────────────────────────────────────────────────
# DESIGN SYSTEM — full token set matching Alpha Return Engine
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
  --warn-mid:       #F5A62360;
  --gold:           #F5C842;
  --gold-dim:       #F5C84220;
  --safe:           #00C8A0;
  --safe-dim:       #00C8A018;
  --moderate:       #4895EF;
  --moderate-dim:   #4895EF18;
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
::-webkit-scrollbar-thumb { background: var(--danger-mid); }
::-webkit-scrollbar-thumb:hover { background: var(--danger); }

.block-container {
  padding: 2.5rem 3.5rem 5rem 3.5rem !important;
  max-width: 1500px !important;
  position: relative; z-index: 1;
}

#MainMenu, footer, header { visibility: hidden; }

/* ── HERO ── */
.hero-wrapper {
  position: relative;
  padding: 4rem 0 3rem 0;
  margin-bottom: 3rem;
  overflow: hidden;
}

.hero-wrapper::after {
  content: '';
  position: absolute;
  bottom: 0; left: 0; right: 0;
  height: 1px;
  background: linear-gradient(90deg, var(--danger) 0%, var(--danger-mid) 25%, transparent 65%);
}

.hero-line {
  position: absolute; left: 0; top: 0;
  width: 3px; height: 100%;
  background: linear-gradient(180deg, var(--danger), transparent);
  box-shadow: var(--danger-glow);
}

@keyframes scan {
  0%   { transform: translateY(-100%); opacity: 0; }
  8%   { opacity: 1; }
  92%  { opacity: 1; }
  100% { transform: translateY(2000%); opacity: 0; }
}

.hero-scan {
  position: absolute; left: 0; right: 0; top: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--danger-mid), transparent);
  animation: scan 7s linear infinite;
  pointer-events: none;
}

@keyframes threat-pulse {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.35; transform: scale(0.82); }
}

.hero-eyebrow {
  font-family: var(--font-mono);
  font-size: 0.62rem;
  font-weight: 500;
  letter-spacing: 0.32em;
  color: var(--danger);
  text-transform: uppercase;
  margin-bottom: 1rem;
  margin-left: 1.2rem;
  display: flex; align-items: center; gap: 0.8rem;
}

.hero-eyebrow .threat-dot {
  width: 6px; height: 6px; border-radius: 50%;
  background: var(--danger); box-shadow: var(--danger-glow);
  animation: threat-pulse 2s ease-in-out infinite; flex-shrink: 0;
}

.hero-title {
  font-family: var(--font-display);
  font-size: clamp(4rem, 8vw, 7.5rem);
  font-weight: 400;
  line-height: 0.92;
  letter-spacing: 0.04em;
  color: var(--text-primary);
  margin: 0 0 0 1rem;
  text-transform: uppercase;
}

.hero-title .danger-word {
  color: var(--danger); text-shadow: var(--danger-glow);
}

.hero-subtitle {
  font-family: var(--font-mono);
  font-size: 0.74rem; color: var(--text-secondary);
  margin: 1.4rem 0 0 1.2rem;
  max-width: 520px; line-height: 1.9;
  border-left: 2px solid var(--border-mid);
  padding-left: 1rem;
}

.hero-bg-text {
  position: absolute; right: -1rem; top: 50%;
  transform: translateY(-50%);
  font-family: var(--font-display);
  font-size: clamp(8rem, 18vw, 16rem);
  font-weight: 400; color: transparent;
  -webkit-text-stroke: 1px var(--border-dim);
  pointer-events: none; user-select: none;
  letter-spacing: 0.06em; line-height: 1;
}

/* ── SECTION LABELS ── */
.section-label {
  font-family: var(--font-mono);
  font-size: 0.58rem; letter-spacing: 0.28em;
  color: var(--danger); text-transform: uppercase;
  margin-bottom: 1.2rem;
  display: flex; align-items: center; gap: 0.8rem;
}

.section-label .sl-num {
  font-size: 0.5rem; color: var(--text-dim);
  border: 1px solid var(--border-mid);
  padding: 0.1rem 0.35rem; letter-spacing: 0.1em;
}

.section-label::after {
  content: ''; flex: 1; height: 1px;
  background: linear-gradient(90deg, var(--border-mid), transparent);
  max-width: 180px;
}

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] {
  background: var(--bg-surface) !important;
  border: 1px dashed var(--border-mid) !important;
  border-radius: var(--radius-sm) !important;
  padding: 1.2rem !important;
  transition: border-color var(--transition), background var(--transition) !important;
}
[data-testid="stFileUploader"]:hover {
  border-color: var(--danger-mid) !important;
  background: var(--bg-raised) !important;
}
[data-testid="stFileUploaderFileName"] {
  font-family: var(--font-mono) !important;
  font-size: 0.72rem !important; color: var(--danger) !important;
}

/* ── TEXT INPUTS ── */
[data-testid="stTextInput"] input {
  background: var(--bg-surface) !important;
  border: 1px solid var(--border-mid) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text-primary) !important;
  font-family: var(--font-mono) !important;
  font-size: 0.74rem !important; padding: 0.6rem 0.9rem !important;
  transition: border-color var(--transition) !important;
}
[data-testid="stTextInput"] input:focus {
  border-color: var(--danger) !important;
  box-shadow: 0 0 0 3px var(--danger-dim) !important;
  outline: none !important;
}
[data-testid="stTextInput"] label {
  font-family: var(--font-mono) !important;
  font-size: 0.64rem !important; letter-spacing: 0.1em !important;
  color: var(--text-secondary) !important; text-transform: uppercase !important;
}

/* ── METRIC GRID ── */
.metric-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 1px; background: var(--border-dim);
  margin: 1.5rem 0; border: 1px solid var(--border-dim);
}
.metric-cell {
  background: var(--bg-surface); padding: 1.2rem 1.5rem;
  display: flex; flex-direction: column; gap: 0.4rem;
  transition: background var(--transition);
  position: relative; overflow: hidden;
}
.metric-cell::after {
  content: ''; position: absolute;
  bottom: 0; left: 0; height: 2px; width: 0;
  background: var(--danger); transition: width 0.4s ease;
}
.metric-cell:hover { background: var(--bg-raised); }
.metric-cell:hover::after { width: 100%; }
.metric-label {
  font-family: var(--font-mono); font-size: 0.56rem;
  letter-spacing: 0.22em; text-transform: uppercase; color: var(--text-dim);
}
.metric-value {
  font-family: var(--font-display); font-size: 2rem;
  font-weight: 400; line-height: 1;
  color: var(--text-primary); letter-spacing: 0.02em;
}
.metric-value.accent { color: var(--accent); text-shadow: 0 0 20px #00FFD150; }
.metric-value.warn   { color: var(--warn); }
.metric-value.danger { color: var(--danger); text-shadow: var(--danger-glow); }
.metric-value.safe   { color: var(--safe); text-shadow: 0 0 20px #00C8A050; }
.metric-value.moderate { color: var(--moderate); }

/* ── STATUS CARDS ── */
.status-card {
  border-radius: var(--radius-sm); padding: 1rem 1.3rem;
  font-family: var(--font-mono); font-size: 0.75rem;
  line-height: 1.7; margin: 1rem 0; border-left: 3px solid;
  position: relative; overflow: hidden;
}
.status-card::before {
  content: ''; position: absolute; inset: 0; opacity: 0.04;
  background: repeating-linear-gradient(0deg, transparent, transparent 2px, white 2px, white 3px);
  pointer-events: none;
}
.status-success { background: #051A14; border-color: var(--accent); color: var(--accent); }
.status-error   { background: #160810; border-color: var(--danger); color: #FF7090; }
.status-warning { background: #141006; border-color: var(--gold); color: var(--gold); }
.status-info    { background: #08101E; border-color: #4895EF; color: #7ABAEF; }

/* ── RISK BANNER ── */
.risk-banner {
  padding: 1.4rem 2rem; border: 1px solid;
  display: flex; align-items: center; gap: 2rem;
  margin: 1.5rem 0; position: relative; overflow: hidden;
}
.risk-banner::before {
  content: ''; position: absolute; inset: 0;
  background: linear-gradient(90deg, currentColor, transparent);
  opacity: 0.04; pointer-events: none;
}
.risk-banner-level {
  font-family: var(--font-display); font-size: 2.8rem;
  letter-spacing: 0.06em; line-height: 1; white-space: nowrap;
}
.risk-banner-score { display: flex; flex-direction: column; gap: 0.3rem; }
.risk-banner-score-label {
  font-family: var(--font-mono); font-size: 0.54rem;
  letter-spacing: 0.2em; text-transform: uppercase; color: var(--text-dim);
}
.risk-banner-score-bar { width: 200px; height: 4px; background: var(--border-dim); }
.risk-banner-score-fill {
  height: 4px; position: relative;
}
.risk-banner-score-fill::after {
  content: ''; position: absolute; right: 0; top: -2px;
  width: 8px; height: 8px; border-radius: 50%;
  background: currentColor; box-shadow: 0 0 8px currentColor;
}
.risk-banner-meta {
  font-family: var(--font-mono); font-size: 0.72rem;
  color: var(--text-secondary); line-height: 1.8;
}

/* ── KPI GRID ── */
.kpi-grid {
  display: grid; grid-template-columns: repeat(5, 1fr);
  gap: 1px; background: var(--border-dim);
  border: 1px solid var(--border-dim); margin: 1.5rem 0;
}
.kpi-cell {
  background: var(--bg-surface); padding: 1.3rem 1.5rem;
  display: flex; flex-direction: column; gap: 0.4rem;
  position: relative; overflow: hidden;
  transition: background var(--transition);
}
.kpi-cell::before {
  content: ''; position: absolute; top: 0; right: 0;
  width: 20px; height: 20px;
  border-top: 1px solid var(--border-mid); border-right: 1px solid var(--border-mid);
  opacity: 0.6;
}
.kpi-cell::after {
  content: ''; position: absolute; bottom: 0; left: 0;
  height: 2px; width: 0; transition: width 0.4s ease;
}
.kpi-cell:hover { background: var(--bg-raised); }
.kpi-cell:hover::after { width: 100%; }
.kpi-cell.safe::after    { background: var(--safe); }
.kpi-cell.moderate::after{ background: var(--moderate); }
.kpi-cell.high::after    { background: var(--warn); }
.kpi-cell.very-high::after { background: var(--danger); }
.kpi-cell.overall::after { background: var(--accent); }
.kpi-label {
  font-family: var(--font-mono); font-size: 0.54rem;
  letter-spacing: 0.22em; text-transform: uppercase; color: var(--text-dim);
}
.kpi-value {
  font-family: var(--font-display); font-size: 2.2rem;
  font-weight: 400; line-height: 1; letter-spacing: 0.02em;
}
.kpi-value.c-safe     { color: var(--safe);     text-shadow: 0 0 20px #00C8A050; }
.kpi-value.c-moderate { color: var(--moderate); text-shadow: 0 0 20px #4895EF50; }
.kpi-value.c-high     { color: var(--warn);     text-shadow: 0 0 20px #F5A62350; }
.kpi-value.c-very-high{ color: var(--danger);   text-shadow: var(--danger-glow); }
.kpi-value.c-overall  { color: var(--text-primary); }
.kpi-badge {
  font-family: var(--font-mono); font-size: 0.54rem;
  letter-spacing: 0.12em; text-transform: uppercase;
  padding: 0.2rem 0.5rem; border: 1px solid; display: inline-block; width: fit-content;
}
.kpi-badge.safe     { color: var(--safe);     border-color: var(--safe);     background: var(--safe-dim); }
.kpi-badge.moderate { color: var(--moderate); border-color: var(--moderate); background: var(--moderate-dim); }
.kpi-badge.high     { color: var(--warn);     border-color: var(--warn);     background: var(--warn-dim); }
.kpi-badge.very-high{ color: var(--danger);   border-color: var(--danger);   background: var(--danger-dim); }
.kpi-badge.neutral  { color: var(--text-dim); border-color: var(--border-mid); background: transparent; }

/* ── PREDICTION TABLE ── */
.pred-table-wrapper {
  background: var(--bg-surface); border: 1px solid var(--border-dim); overflow: hidden;
}
.pred-table {
  width: 100%; border-collapse: collapse;
  font-family: var(--font-mono); font-size: 0.74rem;
}
.pred-table thead { background: var(--bg-raised); border-bottom: 1px solid var(--border-mid); }
.pred-table th {
  color: var(--text-dim); font-size: 0.56rem; letter-spacing: 0.2em;
  text-transform: uppercase; padding: 0.9rem 1.2rem;
  text-align: left; font-weight: 500; white-space: nowrap;
}
.pred-table th:first-child { color: var(--danger); }
.pred-table td {
  padding: 0.75rem 1.2rem; color: var(--text-secondary);
  border-bottom: 1px solid var(--border-dim); vertical-align: middle;
}
.pred-table tr:last-child td { border-bottom: none; }
.pred-table tbody tr { transition: background var(--transition); }
.pred-table tbody tr:hover td { background: var(--bg-hover); color: var(--text-primary); }

.badge-pos  { color: var(--safe);     font-weight: 600; }
.badge-neg  { color: var(--danger);   font-weight: 600; }
.badge-neut { color: var(--warn);     font-weight: 600; }
.badge-mod  { color: var(--moderate); font-weight: 600; }

.match-badge {
  display: inline-flex; align-items: center;
  font-family: var(--font-mono); font-size: 0.5rem;
  letter-spacing: 0.14em; padding: 0.15rem 0.5rem;
  border: 1px solid; text-transform: uppercase;
  margin-left: 0.5rem; vertical-align: middle; border-radius: 1px;
}
.match-exact   { border-color: var(--accent);  color: var(--accent);  background: var(--accent-dim); }
.match-partial { border-color: var(--gold);    color: var(--gold);    background: var(--gold-dim); }
.match-cat     { border-color: #4895EF;        color: #7ABAEF;        background: #4895EF15; }

/* ── INSIGHT CARDS ── */
.insight-card {
  background: var(--bg-surface); border: 1px solid var(--border-dim);
  border-left: 3px solid var(--danger);
  padding: 1.5rem 1.8rem; margin-bottom: 1rem;
  position: relative; overflow: hidden;
  transition: border-color var(--transition), box-shadow var(--transition);
}
.insight-card::before {
  content: ''; position: absolute; top: 0; right: 0;
  width: 200px; height: 200px;
  background: radial-gradient(circle at top right, var(--danger-dim), transparent 70%);
  pointer-events: none;
}
.insight-card:hover { box-shadow: 0 4px 40px #00000030; }
.insight-card-header {
  display: flex; align-items: flex-start;
  justify-content: space-between; gap: 1rem; margin-bottom: 1.2rem;
}
.insight-card h5 {
  font-family: var(--font-body); font-size: 0.92rem; font-weight: 600;
  color: var(--text-primary); margin: 0; letter-spacing: 0.01em; line-height: 1.3;
}
.insight-rec-badge {
  font-family: var(--font-display); font-size: 1.1rem;
  letter-spacing: 0.12em; padding: 0.2rem 0.8rem;
  border: 1px solid currentColor; white-space: nowrap; flex-shrink: 0;
}
.insight-row {
  display: flex; flex-wrap: wrap; gap: 2rem;
  margin-bottom: 1.2rem; padding-bottom: 1.2rem;
  border-bottom: 1px solid var(--border-dim);
}
.insight-kv { display: flex; flex-direction: column; gap: 0.25rem; }
.insight-kv-label {
  font-family: var(--font-mono); font-size: 0.54rem;
  letter-spacing: 0.2em; text-transform: uppercase; color: var(--text-dim);
}
.insight-kv-value {
  font-family: var(--font-mono); font-size: 0.84rem;
  color: var(--text-primary); font-weight: 500;
}
.insight-text {
  font-family: var(--font-body); font-size: 0.8rem;
  color: var(--text-secondary); line-height: 1.85; margin-bottom: 0.6rem;
}
.insight-reason {
  font-family: var(--font-mono); font-size: 0.66rem;
  color: var(--text-dim); line-height: 1.6; padding-top: 0.6rem;
}
.insight-footer {
  font-family: var(--font-mono); font-size: 0.56rem;
  color: var(--text-dim); margin-top: 0.8rem; letter-spacing: 0.08em;
}

/* ── RISK SCORE GAUGE BAR ── */
.conf-bar-wrap { display: flex; align-items: center; gap: 0.6rem; }
.conf-bar-bg   { flex: 1; height: 2px; background: var(--border-dim); max-width: 100px; overflow: hidden; }
.conf-bar-fill { height: 2px; position: relative; }
.conf-bar-fill::after {
  content: ''; position: absolute; right: 0; top: 0;
  width: 4px; height: 2px; background: inherit; filter: blur(4px);
}

/* ── REC CARDS ── */
.rec-card {
  background: var(--bg-surface); border: 1px solid var(--border-dim);
  border-left: 3px solid var(--danger);
  padding: 1.2rem 1.5rem; margin-bottom: 0.8rem;
  position: relative; overflow: hidden;
  transition: background var(--transition);
}
.rec-card::before {
  content: '→'; position: absolute; right: 1.5rem; top: 50%;
  transform: translateY(-50%);
  font-family: var(--font-mono); font-size: 0.8rem; color: var(--border-mid);
  transition: color var(--transition), right var(--transition);
}
.rec-card:hover { background: var(--bg-raised); }
.rec-card:hover::before { color: var(--danger); right: 1.2rem; }
.rec-card-icon {
  font-family: var(--font-mono); font-size: 0.58rem;
  letter-spacing: 0.15em; text-transform: uppercase;
  color: var(--danger); margin-bottom: 0.4rem;
}
.rec-card-text {
  font-family: var(--font-body); font-size: 0.82rem;
  color: var(--text-primary); line-height: 1.6;
}
.rec-card.safe { border-left-color: var(--safe); }
.rec-card.safe .rec-card-icon { color: var(--safe); }
.rec-card.safe:hover::before { color: var(--safe); }
.rec-card.warn { border-left-color: var(--warn); }
.rec-card.warn .rec-card-icon { color: var(--warn); }
.rec-card.warn:hover::before { color: var(--warn); }

/* ── CHART CONTAINER ── */
.chart-container {
  background: var(--bg-surface); border: 1px solid var(--border-dim);
  padding: 1.5rem; position: relative; overflow: hidden;
}
.chart-container::before {
  content: ''; position: absolute; top: 0; left: 0;
  width: 30px; height: 30px;
  border-top: 2px solid var(--danger-mid); border-left: 2px solid var(--danger-mid);
}
.chart-title {
  font-family: var(--font-mono); font-size: 0.6rem;
  letter-spacing: 0.2em; text-transform: uppercase;
  color: var(--text-dim); margin-bottom: 1rem; padding-left: 0.3rem;
}

/* ── UPLOAD HINT ── */
.upload-hint {
  font-family: var(--font-mono); font-size: 0.62rem;
  color: var(--text-dim); margin-top: 0.8rem;
  line-height: 2; padding: 0.7rem 0.9rem;
  background: var(--bg-surface); border-left: 2px solid var(--border-mid);
}
.upload-hint span.hl     { color: var(--text-secondary); }
.upload-hint span.accent { color: var(--danger); }

/* ── DIVIDERS ── */
.h-rule {
  border: none; border-top: 1px solid var(--border-dim); margin: 2.5rem 0; position: relative;
}
.h-rule::before {
  content: '◈'; position: absolute; top: -0.6rem; left: 50%;
  transform: translateX(-50%); font-size: 0.7rem; color: var(--text-dim);
  background: var(--bg-void); padding: 0 0.5rem;
}

/* ── EMPTY STATE ── */
.empty-state {
  height: 420px; display: flex; flex-direction: column;
  align-items: center; justify-content: center;
  border: 1px solid var(--border-dim); color: var(--text-dim);
  font-family: var(--font-mono); font-size: 0.68rem;
  letter-spacing: 0.15em; text-align: center; gap: 0.6rem;
  background: var(--bg-surface); position: relative; overflow: hidden;
}
.empty-state::before {
  content: ''; position: absolute; inset: 0;
  background: repeating-linear-gradient(45deg, transparent, transparent 20px,
    var(--border-dim) 20px, var(--border-dim) 21px);
  opacity: 0.15;
}
.empty-state-icon { font-size: 2.8rem; opacity: 0.15; line-height: 1; position: relative; }
.empty-state-label { position: relative; font-size: 0.68rem; }
.empty-state-hint  { position: relative; font-size: 0.58rem; opacity: 0.5; margin-top: 0.2rem; }

/* ── BUTTON ── */
.stButton > button {
  background: transparent !important; border: 1px solid var(--danger) !important;
  color: var(--danger) !important; font-family: var(--font-mono) !important;
  font-size: 0.68rem !important; letter-spacing: 0.18em !important;
  text-transform: uppercase !important; border-radius: var(--radius-sm) !important;
  padding: 0.65rem 1.6rem !important; transition: all var(--transition) !important;
  position: relative !important; overflow: hidden !important;
}
.stButton > button::before {
  content: '' !important; position: absolute !important; inset: 0 !important;
  background: var(--danger) !important; transform: translateX(-101%) !important;
  transition: transform 0.3s ease !important; z-index: 0 !important;
}
.stButton > button:hover::before { transform: translateX(0) !important; }
.stButton > button:hover {
  color: var(--bg-void) !important; box-shadow: var(--danger-glow) !important;
}
.stButton > button span { position: relative !important; z-index: 1 !important; }

/* ── DOWNLOAD BUTTON ── */
[data-testid="stDownloadButton"] button {
  background: var(--danger-dim) !important; border: 1px solid var(--danger-mid) !important;
  color: var(--danger) !important; font-family: var(--font-mono) !important;
  font-size: 0.67rem !important; letter-spacing: 0.15em !important;
  text-transform: uppercase !important; border-radius: var(--radius-sm) !important;
  transition: all var(--transition) !important;
}
[data-testid="stDownloadButton"] button:hover {
  background: var(--danger) !important; color: var(--bg-void) !important;
  box-shadow: var(--danger-glow) !important;
}

/* ── PROGRESS BAR ── */
[data-testid="stProgressBar"] > div > div {
  background: var(--danger) !important; box-shadow: var(--danger-glow) !important;
}
[data-testid="stProgressBar"] {
  background: var(--border-dim) !important; border-radius: 0 !important; height: 2px !important;
}

/* ── DATA FRAME ── */
[data-testid="stDataFrame"] {
  border: 1px solid var(--border-dim) !important; border-radius: 0 !important;
}
[data-testid="stDataFrame"] table {
  font-family: var(--font-mono) !important; font-size: 0.72rem !important;
}

/* ── TABLE NOTE ── */
.table-note {
  font-family: var(--font-mono); font-size: 0.58rem; color: var(--text-dim);
  margin-top: 0.75rem; line-height: 1.8; padding: 0.6rem 0;
  border-top: 1px solid var(--border-dim);
  display: flex; flex-wrap: wrap; gap: 0.8rem; align-items: center;
}

/* ── FOOTER ── */
.app-footer {
  display: flex; justify-content: space-between; align-items: center;
  font-family: var(--font-mono); font-size: 0.58rem;
  color: var(--text-dim); letter-spacing: 0.12em;
  padding: 1rem 0; border-top: 1px solid var(--border-dim); margin-top: 4rem;
}
@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.3; }
}
.footer-dot {
  width: 4px; height: 4px; border-radius: 50%;
  background: var(--danger); display: inline-block; margin: 0 0.4rem;
  box-shadow: var(--danger-glow); animation: pulse 2.5s ease-in-out infinite;
}

/* ── COLUMN LAYOUT ── */
[data-testid="column"]:first-child {
  padding-right: 1.5rem !important; border-right: 1px solid var(--border-dim);
}

p {
  font-family: var(--font-body) !important;
  font-size: 0.85rem !important; color: var(--text-secondary) !important;
  line-height: 1.7 !important;
}
h1, h2, h3, h4, h5, h6 { font-family: var(--font-body) !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY THEME
# ─────────────────────────────────────────────────────────────────────────────
PLOT_BG    = "rgba(0,0,0,0)"
GRID_CLR   = "#1A2840"
FONT_CLR   = "#7A94B0"
FONT_FAM   = "JetBrains Mono"
RISK_ORDER = ["Low", "Moderate", "High", "Very High"]
RISK_COLORS = {
    "Low":       "#00C8A0",
    "Moderate":  "#4895EF",
    "High":      "#F5A623",
    "Very High": "#FF3860",
}

def base_layout(title="", h=320):
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
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
RISK_NUM_MAP = {1: "Low", 2: "Low", 3: "Moderate", 4: "High", 5: "High", 6: "Very High"}
RISK_LABEL_MAP = {"Low": 1, "Moderate": 3, "High": 5, "Very High": 6,
                  "Very Low": 1, "Low to Moderate": 2, "Moderate to High": 4, "Moderately High": 4}
RISK_SCORES = {"Low": 1, "Moderate": 2, "High": 3, "Very High": 4}

TRAINING_PATHS = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/cleaned_mutual_funds.csv"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/featured_mutual_funds.csv"),
    "cleaned_mutual_funds.csv",
    "featured_mutual_funds.csv",
]

# ─────────────────────────────────────────────────────────────────────────────
# ML TRAINING — Risk Classifier
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def train_risk_model(training_df_json: str):
    """
    Train a GradientBoosting risk classifier on the 814-fund training database.
    Returns a bundle dict identical in spirit to the return-prediction bundle.
    """
    df = pd.read_json(io.StringIO(training_df_json))

    num_cols = ['expense_ratio', 'fund_size_cr', 'fund_age_yr',
                'sortino', 'alpha', 'sd', 'beta', 'sharpe',
                'returns_1yr', 'returns_3yr', 'returns_5yr']
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Map numeric risk_level → 4-class label
    df['risk_label'] = df['risk_level'].apply(lambda x: RISK_NUM_MAP.get(int(x) if pd.notna(x) else 3, "Moderate"))

    le_cat = LabelEncoder()
    le_sub = LabelEncoder()
    df['cat_enc'] = le_cat.fit_transform(df['category'].fillna('Unknown').astype(str))
    df['sub_enc'] = le_sub.fit_transform(df['sub_category'].fillna('Unknown').astype(str))

    feature_cols = [c for c in num_cols if c in df.columns]
    all_feature_cols = feature_cols + ['cat_enc', 'sub_enc']

    X = df[all_feature_cols].copy()
    y = df['risk_label'].values

    imp = SimpleImputer(strategy='median')
    X_imp = imp.fit_transform(X)

    model = GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                        learning_rate=0.08, random_state=42)
    model.fit(X_imp, y)

    # Category stats for fallback imputation
    cat_medians = df.groupby('category')[feature_cols].median()
    sub_medians  = df.groupby('sub_category')[feature_cols].median() if 'sub_category' in df.columns else pd.DataFrame()
    cat_risk_dist = df.groupby('category')['risk_label'].value_counts(normalize=True).unstack(fill_value=0)
    cat_avgs = df.groupby('category')[num_cols].mean()

    return {
        'model':          model,
        'imputer':        imp,
        'training_df':    df,
        'feature_cols':   feature_cols,
        'all_feature_cols': all_feature_cols,
        'le_cat':         le_cat,
        'le_sub':         le_sub,
        'cat_medians':    cat_medians,
        'sub_medians':    sub_medians,
        'cat_risk_dist':  cat_risk_dist,
        'cat_avgs':       cat_avgs,
    }


@st.cache_data(show_spinner=False)
def load_training_data(path: str):
    try:
        df = pd.read_csv(path)
        return df, None
    except Exception as e:
        return None, str(e)


# ─────────────────────────────────────────────────────────────────────────────
# FUND LOOKUP (identical to return-prediction page)
# ─────────────────────────────────────────────────────────────────────────────
def lookup_fund(name: str, training_df: pd.DataFrame):
    if training_df is None or training_df.empty or not name:
        return None, None
    name_l = str(name).lower().strip()
    mask = training_df['scheme_name'].str.lower().str.strip() == name_l
    if mask.any():
        return training_df[mask].iloc[0].to_dict(), 'exact'
    words = [w for w in name_l.split() if len(w) >= 4]
    best, best_score = None, 0
    for _, row in training_df.iterrows():
        sn = str(row['scheme_name']).lower()
        s  = sum(1 for w in words if w in sn)
        if s > best_score:
            best_score, best = s, row.to_dict()
    if best_score >= 1:
        return best, 'partial'
    return None, None


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE BUILDER FOR RISK MODEL
# ─────────────────────────────────────────────────────────────────────────────
def build_risk_features(user_row: dict, matched: dict | None, bundle: dict):
    feature_cols     = bundle['feature_cols']
    all_feature_cols = bundle['all_feature_cols']
    cat_medians      = bundle['cat_medians']
    le_cat           = bundle['le_cat']
    le_sub           = bundle['le_sub']

    cat = None
    sub = None
    for src in [user_row, matched or {}]:
        if not cat and pd.notna(src.get('category', None)):
            cat = str(src['category'])
        if not sub and pd.notna(src.get('sub_category', None)):
            sub = str(src.get('sub_category', ''))

    row = {}

    # Seed with global medians as baseline (prevents KeyError on sparse user data)
    global_medians = bundle['training_df'][feature_cols].median()
    for f in feature_cols:
        row[f] = global_medians[f]

    # Override with category medians
    if cat and cat in cat_medians.index:
        for f in feature_cols:
            row[f] = cat_medians.loc[cat, f]

    # Override with matched training row values
    if matched:
        for f in feature_cols:
            v = matched.get(f)
            if v is not None and pd.notna(v):
                try:
                    row[f] = float(v)
                except (ValueError, TypeError):
                    pass

    # Override with user-provided values
    for f in feature_cols:
        v = user_row.get(f)
        if v is not None:
            try:
                fv = float(v)
                if pd.notna(fv):
                    row[f] = fv
            except (ValueError, TypeError):
                pass

    # Map text risk_level if present
    rl_raw = user_row.get('risk_level', '')
    if isinstance(rl_raw, str):
        mapped = RISK_LABEL_MAP.get(rl_raw.strip(), None)
        if mapped:
            row['risk_level_numeric'] = mapped

    try:
        row['cat_enc'] = int(le_cat.transform([cat])[0]) if cat and cat in le_cat.classes_ else 0
    except Exception:
        row['cat_enc'] = 0
    try:
        row['sub_enc'] = int(le_sub.transform([sub])[0]) if sub and sub in le_sub.classes_ else 0
    except Exception:
        row['sub_enc'] = 0

    X_df = pd.DataFrame([row])[all_feature_cols]
    return X_df, cat, sub


# ─────────────────────────────────────────────────────────────────────────────
# RISK INSIGHT GENERATOR
# ─────────────────────────────────────────────────────────────────────────────
def generate_risk_insight(fund_name: str, pred_risk: str, pred_proba: dict,
                           matched: dict | None, user_row: dict, cat: str | None,
                           bundle: dict) -> dict:
    cat_avgs     = bundle.get('cat_avgs')
    cat_risk_dist = bundle.get('cat_risk_dist')

    risk_score   = RISK_SCORES.get(pred_risk, 2)
    risk_pct     = int((risk_score - 1) / 3 * 100)
    risk_color   = {"Low": "#00C8A0", "Moderate": "#4895EF", "High": "#F5A623", "Very High": "#FF3860"}.get(pred_risk, "#7A94B0")

    # Category-level expected risk
    cat_exp_risk = "Moderate"
    if cat and cat_risk_dist is not None and cat in cat_risk_dist.index:
        cat_exp_risk = cat_risk_dist.loc[cat].idxmax()

    # Confidence from match quality + proba spread
    top_prob   = max(pred_proba.values())
    match_type = 'exact' if matched and str(matched.get('scheme_name', '')).lower() == fund_name.lower() \
                 else ('partial' if matched else 'category')

    conf_score = 0
    if match_type == 'exact':   conf_score += 3
    elif match_type == 'partial': conf_score += 2
    if top_prob > 0.7:  conf_score += 1
    conf_label  = ['Low', 'Low', 'Medium', 'Medium', 'High'][min(conf_score, 4)]
    conf_pct    = {0: 20, 1: 35, 2: 55, 3: 72, 4: 90}[min(conf_score, 4)]

    # Key metrics for insight narrative
    expense_ratio = user_row.get('expense_ratio') or (matched or {}).get('expense_ratio')
    sharpe        = (matched or {}).get('sharpe')
    sd            = (matched or {}).get('sd')
    beta          = (matched or {}).get('beta')
    ret_5yr       = user_row.get('returns_5yr') or (matched or {}).get('returns_5yr')

    try: expense_ratio = float(expense_ratio) if expense_ratio else None
    except: expense_ratio = None
    try: sharpe = float(sharpe) if sharpe else None
    except: sharpe = None
    try: sd = float(sd) if sd else None
    except: sd = None
    try: beta = float(beta) if beta else None
    except: beta = None
    try: ret_5yr = float(ret_5yr) if ret_5yr else None
    except: ret_5yr = None

    cat_str = cat or "this category"
    parts   = []

    if match_type in ('exact', 'partial') and sd is not None:
        parts.append(
            f"Standard deviation of {sd:.1f}% indicates "
            f"{'elevated' if sd > 15 else 'moderate' if sd > 5 else 'low'} price volatility."
        )

    if beta is not None:
        if beta > 1.1:
            parts.append(f"Beta of {beta:.2f} signals amplified market sensitivity — this fund moves more than the benchmark.")
        elif beta < 0.7:
            parts.append(f"Beta of {beta:.2f} indicates defensive positioning relative to the market.")
        else:
            parts.append(f"Beta of {beta:.2f} tracks broadly in line with market movements.")

    if sharpe is not None:
        if sharpe > 1.5:
            parts.append(f"Sharpe ratio of {sharpe:.2f} reflects strong risk-adjusted returns for the risk taken.")
        elif sharpe < 0.5:
            parts.append(f"Sharpe ratio of {sharpe:.2f} is below average — returns may not justify the volatility.")

    if expense_ratio is not None:
        if expense_ratio > 1.5:
            parts.append(f"Expense ratio of {expense_ratio:.2f}% is above average and will compound negatively over time.")
        elif expense_ratio < 0.5:
            parts.append(f"Lean expense ratio of {expense_ratio:.2f}% is a structural cost advantage.")

    if ret_5yr is not None:
        parts.append(f"5-year historical return of {ret_5yr:.1f}%.")

    if cat_exp_risk != pred_risk:
        parts.append(
            f"Predicted risk level ({pred_risk}) diverges from typical {cat_str} funds ({cat_exp_risk}) — "
            "review fund-specific characteristics before allocation."
        )

    if not parts:
        parts.append(f"Risk profile assessed as {pred_risk} based on category statistics for {cat_str} funds.")

    insight_text = " ".join(parts)

    # Recommendation
    if pred_risk == "Very High":
        action = "Reduce"
        reason = f"Very-high risk exposure in {cat_str} — suitable only for aggressive investors with long horizons."
    elif pred_risk == "High":
        action = "Monitor"
        reason = f"High risk requires active monitoring; ensure this aligns with your overall portfolio risk budget."
    elif pred_risk == "Moderate":
        action = "Hold"
        reason = f"Moderate risk profile is appropriate for balanced portfolios targeting medium-term growth."
    else:
        action = "Safe"
        reason = f"Low-risk fund suitable for capital preservation and short-to-medium term goals."

    return {
        'risk_level':        pred_risk,
        'risk_color':        risk_color,
        'risk_score':        risk_score,
        'risk_pct':          risk_pct,
        'confidence':        conf_label,
        'conf_pct':          conf_pct,
        'action':            action,
        'insight_text':      insight_text,
        'reason':            reason,
        'matched_scheme':    (matched or {}).get('scheme_name', '—'),
        'match_type':        match_type,
        'cat':               cat or '—',
        'proba':             pred_proba,
        'sharpe':            sharpe,
        'sd':                sd,
        'beta':              beta,
        'expense_ratio':     expense_ratio,
        'ret_5yr':           ret_5yr,
        'top_prob':          top_prob,
    }


# ─────────────────────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrapper">
    <div class="hero-scan"></div>
    <div class="hero-line"></div>
    <div class="hero-eyebrow">
        <span class="threat-dot"></span>
        ML Risk Intelligence · Gradient Boosting Classifier · 814-Fund Database
    </div>
    <h1 class="hero-title">Portfolio<br><span class="danger-word">Risk</span><br>Engine</h1>
    <p class="hero-subtitle">
        Upload any portfolio file. The engine fuzzy-matches each fund
        against the 814-fund training database, fills missing features
        from category statistics, and runs a trained classifier to
        predict risk tier — then delivers institutional-grade
        risk intelligence across every holding.
    </p>
    <div class="hero-bg-text">⚠</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1.05, 1.95], gap="large")

# ─────────────────────────────────────────────────────────────────────────────
# LEFT PANEL
# ─────────────────────────────────────────────────────────────────────────────
with left_col:
    st.markdown("""
    <div class="section-label"><span class="sl-num">01</span>Portfolio File</div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drop portfolio file",
        type=["csv", "xlsx"],
        label_visibility="collapsed",
    )

    st.markdown("""
    <div class="upload-hint">
        Accepted: <span class="hl">.csv</span> &nbsp;·&nbsp; <span class="hl">.xlsx</span><br>
        Needs a fund name column — everything else is optional.<br>
        Risk is predicted from category statistics + any provided metrics.<br>
        <span class="accent">↑</span> No external API — fully offline ML classifier.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="h-rule">', unsafe_allow_html=True)

    st.markdown("""
    <div class="section-label"><span class="sl-num">02</span>Training Database</div>
    """, unsafe_allow_html=True)

    default_train = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../datasets/cleaned_mutual_funds.csv")
    )
    train_path = st.text_input(
        "Training data (.csv)",
        value=default_train,
        help="Path to cleaned_mutual_funds.csv"
    )

    st.markdown('<hr class="h-rule">', unsafe_allow_html=True)

    st.markdown("""
    <div class="section-label"><span class="sl-num">03</span>Column Mapping</div>
    """, unsafe_allow_html=True)

    fund_col_hint = st.text_input(
        "Fund / share name column",
        value="fund_name",
        help="Column in the uploaded file containing fund names for DB lookup.",
    )

    run_btn = st.button("▶  Run Risk Analysis", use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# RIGHT PANEL
# ─────────────────────────────────────────────────────────────────────────────
with right_col:
    if not uploaded_file:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon">⚠</div>
            <div class="empty-state-label">No Portfolio Loaded</div>
            <div class="empty-state-hint">Upload a portfolio file to begin risk analysis</div>
        </div>
        """, unsafe_allow_html=True)

    else:
        # ── Parse uploaded file ──
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            n_rows, n_cols = df.shape
            n_nulls        = int(df.isnull().sum().sum())
            numeric_cols   = df.select_dtypes(include="number").columns.tolist()

            st.markdown(f"""
            <div class="metric-grid">
                <div class="metric-cell">
                    <span class="metric-label">Rows</span>
                    <span class="metric-value accent">{n_rows:,}</span>
                </div>
                <div class="metric-cell">
                    <span class="metric-label">Columns</span>
                    <span class="metric-value">{n_cols}</span>
                </div>
                <div class="metric-cell">
                    <span class="metric-label">Numeric</span>
                    <span class="metric-value">{len(numeric_cols)}</span>
                </div>
                <div class="metric-cell">
                    <span class="metric-label">Nulls</span>
                    <span class="metric-value {'accent' if n_nulls == 0 else 'warn'}">{n_nulls}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="section-label"><span class="sl-num">Preview</span>Dataset</div>
            """, unsafe_allow_html=True)
            st.dataframe(df.head(8), use_container_width=True, hide_index=False)

            # ── Run analysis ──
            if run_btn:

                # Load training data
                train_df, train_err = load_training_data(train_path)
                if train_err or train_df is None:
                    # Try fallback paths
                    for fp in TRAINING_PATHS:
                        train_df, train_err = load_training_data(fp)
                        if train_df is not None:
                            break

                if train_df is None:
                    st.markdown(f"""
                    <div class="status-card status-error">
                        ✗ Could not load training database<br>
                        <span style="opacity:0.7;font-size:0.7rem;">Provide path to cleaned_mutual_funds.csv</span>
                    </div>""", unsafe_allow_html=True)
                    st.stop()

                # Train model (cached)
                with st.spinner("Training risk classifier on 814-fund database…"):
                    bundle = train_risk_model(train_df.to_json())

                model     = bundle['model']
                imp       = bundle['imputer']
                train_df_ = bundle['training_df']
                n_train   = len(train_df_)

                st.markdown(
                    f'<div class="status-card status-success">✓ Gradient Boosting classifier trained on {n_train} funds · '
                    f'Using <strong>{fund_col_hint}</strong> as fund identifier</div>',
                    unsafe_allow_html=True,
                )

                # Detect fund column
                fund_col = None
                for c in [fund_col_hint, "fund_name", "scheme_name", "name",
                          "fund", "share_name", "stock_name", "instrument"]:
                    if c in df.columns:
                        fund_col = c
                        break
                if not fund_col:
                    str_cols = df.select_dtypes(include="object").columns.tolist()
                    if str_cols:
                        fund_col = str_cols[0]

                if not fund_col:
                    st.markdown("""
                    <div class="status-card status-warning">
                        ⚠ Cannot find fund name column. Enter column name in "Fund / share name column".
                    </div>""", unsafe_allow_html=True)
                    st.stop()

                # ── Predict per fund ──
                results  = []
                progress = st.progress(0, text="Initialising risk pipeline…")

                for i, (_, row) in enumerate(df.iterrows()):
                    fund_name = str(row.get(fund_col, f"Fund #{i+1}"))
                    user_row  = row.to_dict()

                    matched, mtype = lookup_fund(fund_name, train_df_)
                    X_df, cat, sub = build_risk_features(user_row, matched, bundle)

                    try:
                        X_imp   = imp.transform(X_df)
                        pred_lbl = model.predict(X_imp)[0]
                        probas   = dict(zip(model.classes_, model.predict_proba(X_imp)[0].tolist()))
                    except Exception:
                        # Fallback: use category or user-provided risk level
                        cat_str  = cat or ""
                        rl_raw   = user_row.get('risk_level', '')
                        if isinstance(rl_raw, str) and rl_raw:
                            num = RISK_LABEL_MAP.get(rl_raw.strip(), 2)
                            pred_lbl = RISK_NUM_MAP.get(num, "Moderate")
                        else:
                            pred_lbl = {"Equity": "High", "Debt": "Low",
                                        "Hybrid": "Moderate", "Other": "Moderate"}.get(cat_str, "Moderate")
                        probas = {pred_lbl: 0.5}

                    insight = generate_risk_insight(fund_name, pred_lbl, probas,
                                                    matched, user_row, cat, bundle)

                    results.append({
                        'fund_name':    fund_name,
                        'match_type':   mtype or 'category',
                        **insight,
                    })

                    progress.progress((i + 1) / n_rows, text=f"Assessing risk · {fund_name[:50]}…")

                progress.empty()

                # ─────────────────────────────────────────────────────────────
                # SUMMARY METRICS
                # ─────────────────────────────────────────────────────────────
                counts = {r: sum(1 for x in results if x['risk_level'] == r) for r in RISK_ORDER}
                avg_score = np.mean([RISK_SCORES.get(x['risk_level'], 2) for x in results])
                matches   = sum(1 for x in results if x['match_type'] in ('exact', 'partial'))

                if avg_score <= 1.5:   portfolio_risk, pr_color = "LOW RISK",       "#00C8A0"
                elif avg_score <= 2.5: portfolio_risk, pr_color = "MODERATE RISK",  "#4895EF"
                elif avg_score <= 3.2: portfolio_risk, pr_color = "HIGH RISK",      "#F5A623"
                else:                  portfolio_risk, pr_color = "VERY HIGH RISK", "#FF3860"

                score_pct = int(((avg_score - 1) / 3) * 100)

                # ── Risk Banner ──
                st.markdown(f"""
                <div class="risk-banner" style="border-color:{pr_color}40;color:{pr_color};">
                    <div class="risk-banner-level">{portfolio_risk}</div>
                    <div class="risk-banner-score">
                        <span class="risk-banner-score-label">Composite Risk Score</span>
                        <div class="risk-banner-score-bar">
                            <div class="risk-banner-score-fill"
                                 style="width:{score_pct}%;background:{pr_color};color:{pr_color};"></div>
                        </div>
                        <span style="font-family:var(--font-mono);font-size:0.62rem;color:{pr_color};margin-top:0.2rem;">
                            {avg_score:.2f} / 4.00
                        </span>
                    </div>
                    <div class="risk-banner-meta">
                        {n_rows} funds analysed &nbsp;·&nbsp;
                        {counts['High'] + counts['Very High']} high-exposure positions &nbsp;·&nbsp;
                        {counts['Low']} defensive positions &nbsp;·&nbsp;
                        {matches} DB matches
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # ── KPI Grid ──
                st.markdown("""
                <div class="section-label"><span class="sl-num">01</span>Risk Distribution</div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class="kpi-grid">
                    <div class="kpi-cell safe">
                        <span class="kpi-label">Low Risk</span>
                        <span class="kpi-value c-safe">{counts['Low']}</span>
                        <span class="kpi-badge safe">Defensive</span>
                    </div>
                    <div class="kpi-cell moderate">
                        <span class="kpi-label">Moderate</span>
                        <span class="kpi-value c-moderate">{counts['Moderate']}</span>
                        <span class="kpi-badge moderate">Balanced</span>
                    </div>
                    <div class="kpi-cell high">
                        <span class="kpi-label">High Risk</span>
                        <span class="kpi-value c-high">{counts['High']}</span>
                        <span class="kpi-badge high">Aggressive</span>
                    </div>
                    <div class="kpi-cell very-high">
                        <span class="kpi-label">Very High</span>
                        <span class="kpi-value c-very-high">{counts['Very High']}</span>
                        <span class="kpi-badge very-high">Volatile</span>
                    </div>
                    <div class="kpi-cell overall">
                        <span class="kpi-label">Portfolio Overall</span>
                        <span class="kpi-value c-overall" style="font-size:1.2rem;line-height:1.2;margin-top:0.2rem;">
                            {portfolio_risk}
                        </span>
                        <span class="kpi-badge neutral">{avg_score:.2f} / 4.0</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # ─────────────────────────────────────────────────────────────
                # PREDICTIONS TABLE
                # ─────────────────────────────────────────────────────────────
                st.markdown('<hr class="h-rule">', unsafe_allow_html=True)
                st.markdown("""
                <div class="section-label"><span class="sl-num">02</span>Risk Predictions</div>
                """, unsafe_allow_html=True)

                RISK_ROW_CLASSES = {
                    "Low":       "badge-pos",
                    "Moderate":  "badge-mod",
                    "High":      "badge-neut",
                    "Very High": "badge-neg",
                }

                rows_html = ""
                for r in results:
                    rc = RISK_ROW_CLASSES.get(r['risk_level'], "")
                    match_badge = {
                        'exact':    '<span class="match-badge match-exact">Exact</span>',
                        'partial':  '<span class="match-badge match-partial">Partial</span>',
                        'category': '<span class="match-badge match-cat">Category</span>',
                    }.get(r['match_type'], '')

                    action_cls = {"Reduce": "badge-neg", "Monitor": "badge-neut",
                                  "Hold": "badge-mod", "Safe": "badge-pos"}.get(r['action'], "")

                    # Top 2 probabilities
                    sorted_proba = sorted(r['proba'].items(), key=lambda x: -x[1])[:2]
                    proba_str = " · ".join(f"{k} {v*100:.0f}%" for k, v in sorted_proba)

                    sd_str  = f"{r['sd']:.1f}%" if r['sd'] is not None else "—"
                    beta_str = f"{r['beta']:.2f}" if r['beta'] is not None else "—"
                    sharpe_str = f"{r['sharpe']:.2f}" if r['sharpe'] is not None else "—"

                    rows_html += f"""
                    <tr>
                        <td style="max-width:260px;line-height:1.5;">
                            <span style="color:var(--text-primary);font-weight:500;">{r['fund_name']}</span>
                            {match_badge}<br>
                            <span style="font-size:0.58rem;color:var(--text-dim);">{r['cat']}</span>
                        </td>
                        <td class="{rc}" style="font-weight:700;letter-spacing:0.04em;">{r['risk_level']}</td>
                        <td style="font-size:0.66rem;color:var(--text-secondary);">{sd_str}</td>
                        <td style="font-size:0.66rem;color:var(--text-secondary);">{beta_str}</td>
                        <td style="font-size:0.66rem;color:var(--text-secondary);">{sharpe_str}</td>
                        <td style="font-size:0.56rem;color:var(--text-dim);max-width:150px;">{proba_str}</td>
                        <td style="font-size:0.64rem;color:var(--text-dim);">{r['confidence']}</td>
                        <td class="{action_cls}" style="font-weight:600;letter-spacing:0.05em;">{r['action']}</td>
                    </tr>"""

                st.markdown(f"""
                <div class="pred-table-wrapper">
                <table class="pred-table">
                    <thead><tr>
                        <th>Fund / Instrument</th>
                        <th>Risk Tier</th>
                        <th>Std Dev</th>
                        <th>Beta</th>
                        <th>Sharpe</th>
                        <th>Prob Distribution</th>
                        <th>Confidence</th>
                        <th>Action</th>
                    </tr></thead>
                    <tbody>{rows_html}</tbody>
                </table>
                </div>
                <div class="table-note">
                    Risk predicted by Gradient Boosting classifier trained on 814-fund database.
                    &nbsp;
                    <span class="match-badge match-exact">Exact</span>&nbsp; direct DB match
                    &nbsp;
                    <span class="match-badge match-partial">Partial</span>&nbsp; fuzzy name match
                    &nbsp;
                    <span class="match-badge match-cat">Category</span>&nbsp; category-level inference
                </div>
                """, unsafe_allow_html=True)

                # ─────────────────────────────────────────────────────────────
                # CHARTS
                # ─────────────────────────────────────────────────────────────
                st.markdown('<hr class="h-rule">', unsafe_allow_html=True)
                st.markdown("""
                <div class="section-label"><span class="sl-num">03</span>Risk Visualisations</div>
                """, unsafe_allow_html=True)

                chart_df = pd.DataFrame([{
                    'fund_name':     r['fund_name'],
                    'risk_level':    r['risk_level'],
                    'confidence':    r['conf_pct'],
                    'risk_score':    r['risk_score'],
                    'expense_ratio': r['expense_ratio'],
                    'sharpe':        r['sharpe'],
                    'sd':            r['sd'],
                    'beta':          r['beta'],
                    'ret_5yr':       r['ret_5yr'],
                } for r in results])

                col_l, col_r = st.columns(2, gap="medium")

                # ── Donut ──
                with col_l:
                    st.markdown('<div class="chart-container"><div class="chart-title">Risk Tier Distribution</div>', unsafe_allow_html=True)
                    rc_df = chart_df['risk_level'].value_counts().reset_index()
                    rc_df.columns = ['risk_level', 'count']
                    pie = go.Figure(go.Pie(
                        labels=rc_df['risk_level'], values=rc_df['count'],
                        hole=0.62,
                        marker=dict(
                            colors=[RISK_COLORS.get(r, "#7A94B0") for r in rc_df['risk_level']],
                            line=dict(color="#03050A", width=3),
                        ),
                        textfont=dict(family=FONT_FAM, size=10, color="#7A94B0"),
                        hovertemplate="<b>%{label}</b><br>%{value} funds (%{percent})<extra></extra>",
                    ))
                    pie.add_annotation(
                        text=f"<b>{n_rows}</b><br><span style='font-size:9px'>FUNDS</span>",
                        x=0.5, y=0.5,
                        font=dict(family=FONT_FAM, color="#E8F0FA", size=16),
                        showarrow=False,
                    )
# base_layout() already sets a legend dict; only pass showlegend here to avoid
# "multiple values for keyword argument 'legend'".
                    pie.update_layout(**base_layout(), showlegend=True)
                    st.plotly_chart(pie, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                # ── Risk Score Bar ──
                with col_r:
                    st.markdown('<div class="chart-container"><div class="chart-title">Risk Score per Fund</div>', unsafe_allow_html=True)
                    bar_df = chart_df.sort_values('risk_score', ascending=False).head(20)
                    bar = px.bar(
                        bar_df, x='fund_name', y='risk_score',
                        color='risk_level', color_discrete_map=RISK_COLORS,
                        labels={'risk_score': 'Risk Score (1–4)', 'fund_name': ''},
                        text='risk_level',
                        category_orders={'risk_level': RISK_ORDER},
                    )
                    bar.update_traces(
                        marker_line_color="#03050A", marker_line_width=1,
                        textfont=dict(family=FONT_FAM, size=9, color="#E8F0FA"),
                        textposition="outside",
                    )
                    bar.update_xaxes(tickangle=-35, tickfont=dict(size=8))
                    bar.update_layout(**base_layout(), showlegend=False)
                    st.plotly_chart(bar, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                # ── Scatter: Expense vs Risk ──
                col_bl, col_br = st.columns(2, gap="medium")

                with col_bl:
                    scatter_df = chart_df.dropna(subset=['expense_ratio', 'ret_5yr'])
                    if not scatter_df.empty:
                        st.markdown('<div class="chart-container"><div class="chart-title">Expense Ratio vs 5yr Return by Risk Tier</div>', unsafe_allow_html=True)
                        sc = px.scatter(
                            scatter_df, x='expense_ratio', y='ret_5yr',
                            color='risk_level', color_discrete_map=RISK_COLORS,
                            hover_data={'fund_name': True, 'risk_level': True,
                                        'expense_ratio': ':.2f', 'ret_5yr': ':.1f'},
                            labels={'expense_ratio': 'Expense Ratio (%)', 'ret_5yr': '5yr Return (%)'},
                            category_orders={'risk_level': RISK_ORDER},
                        )
                        sc.update_traces(marker=dict(size=10, opacity=0.85, line=dict(width=1, color="#03050A")))
                        sc.update_layout(**base_layout(), showlegend=True)
                        st.plotly_chart(sc, use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)

                with col_br:
                    # ── Probability Confidence Stacked Bar ──
                    prob_rows = []
                    for r in results:
                        for tier, p in r['proba'].items():
                            prob_rows.append({'fund': r['fund_name'][:22], 'tier': tier, 'prob': round(p*100, 1)})
                    prob_df = pd.DataFrame(prob_rows)
                    if not prob_df.empty:
                        st.markdown('<div class="chart-container"><div class="chart-title">Classifier Confidence Distribution</div>', unsafe_allow_html=True)
                        stacked = px.bar(
                            prob_df, x='fund', y='prob', color='tier',
                            color_discrete_map=RISK_COLORS,
                            labels={'prob': 'Probability (%)', 'fund': ''},
                            barmode='stack',
                            category_orders={'tier': RISK_ORDER},
                        )
                        stacked.update_xaxes(tickangle=-35, tickfont=dict(size=8))
                        stacked.update_layout(**base_layout(), showlegend=True)
                        st.plotly_chart(stacked, use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)

                # ─────────────────────────────────────────────────────────────
                # HIGH-RISK HOLDINGS TABLE
                # ─────────────────────────────────────────────────────────────
                st.markdown('<hr class="h-rule">', unsafe_allow_html=True)
                st.markdown("""
                <div class="section-label"><span class="sl-num">04</span>High-Exposure Holdings</div>
                """, unsafe_allow_html=True)

                high_risk_results = [r for r in results if r['risk_level'] in ('High', 'Very High')]
                if not high_risk_results:
                    st.markdown("""
                    <div class="rec-card safe">
                        <div class="rec-card-icon">✓ All Clear</div>
                        <div class="rec-card-text">
                            No high or very-high risk holdings detected in this portfolio.
                        </div>
                    </div>""", unsafe_allow_html=True)
                else:
                    hr_df = pd.DataFrame([{
                        'Fund':          r['fund_name'],
                        'Category':      r['cat'],
                        'Risk Tier':     r['risk_level'],
                        'Std Dev':       f"{r['sd']:.1f}%" if r['sd'] else "—",
                        'Beta':          f"{r['beta']:.2f}" if r['beta'] else "—",
                        'Sharpe':        f"{r['sharpe']:.2f}" if r['sharpe'] else "—",
                        'Expense Ratio': f"{r['expense_ratio']:.2f}%" if r['expense_ratio'] else "—",
                        '5yr Return':    f"{r['ret_5yr']:.1f}%" if r['ret_5yr'] else "—",
                        'Confidence':    r['confidence'],
                        'Action':        r['action'],
                    } for r in high_risk_results])
                    st.dataframe(hr_df, use_container_width=True, hide_index=True)

                # ─────────────────────────────────────────────────────────────
                # AI RECOMMENDATIONS
                # ─────────────────────────────────────────────────────────────
                st.markdown('<hr class="h-rule">', unsafe_allow_html=True)
                st.markdown("""
                <div class="section-label"><span class="sl-num">05</span>AI Risk Recommendations</div>
                """, unsafe_allow_html=True)

                recs = []
                if counts['Very High'] >= 2:
                    recs.append((
                        f"{counts['Very High']} very-high-risk positions detected. "
                        "Consider trimming small-cap / sectoral exposure to reduce tail-risk concentration.",
                        "danger"
                    ))
                if counts['High'] >= 3:
                    recs.append((
                        f"{counts['High']} high-risk funds create elevated drawdown risk. "
                        "Rebalancing toward large-cap or hybrid funds would improve risk-adjusted returns.",
                        "danger"
                    ))
                if counts['Low'] == 0:
                    recs.append((
                        "No defensive positions found. Adding 10–20% debt/liquid allocation "
                        "can buffer volatility during equity market corrections.",
                        "warn"
                    ))

                high_expense = [r for r in results if r['expense_ratio'] and r['expense_ratio'] > 1.5]
                if len(high_expense) >= 2:
                    recs.append((
                        f"{len(high_expense)} funds have expense ratios above 1.5%. "
                        "High costs compound negatively over a 5-year horizon — review direct-plan alternatives.",
                        "warn"
                    ))

                low_sharpe = [r for r in results if r['sharpe'] and r['sharpe'] < 0.5]
                if len(low_sharpe) >= 2:
                    recs.append((
                        f"{len(low_sharpe)} funds show below-average Sharpe ratios (<0.5). "
                        "These funds may not be adequately compensating investors for the risk taken.",
                        "warn"
                    ))

                high_beta = [r for r in results if r['beta'] and r['beta'] > 1.2]
                if len(high_beta) >= 2:
                    recs.append((
                        f"{len(high_beta)} funds have beta above 1.2 — highly market-sensitive positions "
                        "that amplify both gains and drawdowns in volatile markets.",
                        "warn"
                    ))

                if not recs:
                    recs.append((
                        "Portfolio risk structure appears well-balanced. "
                        "Continue monitoring for category drift and rebalance if allocations shift beyond target thresholds.",
                        "safe"
                    ))

                sev_map = {
                    "danger": ("⚠ Alert",    "var(--danger)"),
                    "warn":   ("◈ Advisory", "var(--warn)"),
                    "safe":   ("✓ Status",   "var(--safe)"),
                }
                for text, sev in recs:
                    icon, color = sev_map.get(sev, ("◈", "var(--text-dim)"))
                    cls = {"danger": "", "warn": "warn", "safe": "safe"}.get(sev, "")
                    st.markdown(f"""
                    <div class="rec-card {cls}" style="border-left-color:{color};">
                        <div class="rec-card-icon" style="color:{color};">{icon}</div>
                        <div class="rec-card-text">{text}</div>
                    </div>""", unsafe_allow_html=True)

                # ─────────────────────────────────────────────────────────────
                # FUND-LEVEL INSIGHT CARDS
                # ─────────────────────────────────────────────────────────────
                st.markdown('<hr class="h-rule">', unsafe_allow_html=True)
                st.markdown("""
                <div class="section-label"><span class="sl-num">06</span>Fund-Level Risk Insights</div>
                """, unsafe_allow_html=True)

                for r in results:
                    color = r['risk_color']
                    conf_color = {'High': '#00FFD1', 'Medium': '#F5A623', 'Low': '#FF3860'}.get(r['confidence'], '#3D526A')
                    conf_w     = r['conf_pct']

                    sd_str  = f"{r['sd']:.1f}%" if r['sd'] is not None else "—"
                    beta_str = f"{r['beta']:.2f}" if r['beta'] is not None else "—"
                    sharpe_str = f"{r['sharpe']:.2f}" if r['sharpe'] is not None else "—"
                    er_str  = f"{r['expense_ratio']:.2f}%" if r['expense_ratio'] is not None else "—"
                    ret_str = f"{r['ret_5yr']:.1f}%" if r['ret_5yr'] is not None else "—"

                    top2 = sorted(r['proba'].items(), key=lambda x: -x[1])[:2]
                    proba_str = " · ".join(f"{k}: {v*100:.0f}%" for k, v in top2)

                    st.markdown(f"""
                    <div class="insight-card" style="border-left-color:{color};">
                        <div class="insight-card-header">
                            <h5>{r['fund_name']}</h5>
                            <span class="insight-rec-badge" style="color:{color};border-color:{color}40;background:{color}10;">
                                {r['action']}
                            </span>
                        </div>
                        <div class="insight-row">
                            <div class="insight-kv">
                                <span class="insight-kv-label">Category</span>
                                <span class="insight-kv-value">{r['cat']}</span>
                            </div>
                            <div class="insight-kv">
                                <span class="insight-kv-label">Risk Tier</span>
                                <span class="insight-kv-value" style="color:{color};">{r['risk_level']}</span>
                            </div>
                            <div class="insight-kv">
                                <span class="insight-kv-label">Std Dev</span>
                                <span class="insight-kv-value">{sd_str}</span>
                            </div>
                            <div class="insight-kv">
                                <span class="insight-kv-label">Beta</span>
                                <span class="insight-kv-value">{beta_str}</span>
                            </div>
                            <div class="insight-kv">
                                <span class="insight-kv-label">Sharpe</span>
                                <span class="insight-kv-value">{sharpe_str}</span>
                            </div>
                            <div class="insight-kv">
                                <span class="insight-kv-label">Expense</span>
                                <span class="insight-kv-value">{er_str}</span>
                            </div>
                            <div class="insight-kv">
                                <span class="insight-kv-label">5yr Return</span>
                                <span class="insight-kv-value">{ret_str}</span>
                            </div>
                            <div class="insight-kv">
                                <span class="insight-kv-label">Confidence</span>
                                <div class="conf-bar-wrap">
                                    <span style="color:{conf_color};font-family:var(--font-mono);font-size:0.72rem;">{r['confidence']}</span>
                                    <div class="conf-bar-bg">
                                        <div class="conf-bar-fill" style="width:{conf_w}%;background:{conf_color};"></div>
                                    </div>
                                    <span style="font-family:var(--font-mono);font-size:0.56rem;color:var(--text-dim);">{conf_w}%</span>
                                </div>
                            </div>
                        </div>
                        <div class="insight-text">{r['insight_text']}</div>
                        <div class="insight-reason">→ {r['reason']}</div>
                        <div class="insight-footer">
                            Classifier probabilities · {proba_str}
                            &nbsp;&nbsp;|&nbsp;&nbsp;
                            DB match · {r['matched_scheme']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # ─────────────────────────────────────────────────────────────
                # EXPORT
                # ─────────────────────────────────────────────────────────────
                export_rows = [{
                    'Fund':               r['fund_name'],
                    'Category':           r['cat'],
                    'Match Type':         r['match_type'],
                    'Matched DB Scheme':  r['matched_scheme'],
                    'Predicted Risk Tier':r['risk_level'],
                    'Risk Score (1-4)':   r['risk_score'],
                    'Classifier Confidence': r['confidence'],
                    'Confidence %':       r['conf_pct'],
                    'Std Dev':            r['sd'],
                    'Beta':               r['beta'],
                    'Sharpe':             r['sharpe'],
                    'Expense Ratio':      r['expense_ratio'],
                    '5yr Return':         r['ret_5yr'],
                    'Action':             r['action'],
                    'Reason':             r['reason'],
                    'Insight':            r['insight_text'],
                    'Top Probabilities':  " | ".join(f"{k}:{v*100:.0f}%" for k, v in sorted(r['proba'].items(), key=lambda x: -x[1])[:3]),
                } for r in results]

                csv_data = pd.DataFrame(export_rows).to_csv(index=False).encode()
                st.download_button(
                    label="↓  Export Risk Analysis (.csv)",
                    data=csv_data,
                    file_name="alpha_risk_analysis.csv",
                    mime="text/csv",
                )

        except Exception as e:
            st.markdown(f"""
            <div class="status-card status-error">
                ✗ Failed to process file<br>
                <span style="opacity:0.7;font-size:0.7rem;">{e}</span>
            </div>""", unsafe_allow_html=True)
            import traceback
            st.code(traceback.format_exc(), language="text")

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    <span>◈ RISK ANALYSIS ENGINE <span class="footer-dot"></span> v5.0</span>
    <span>GRADIENT BOOSTING CLASSIFIER · 814 FUNDS · ZERO EXTERNAL API</span>
</div>
""", unsafe_allow_html=True)