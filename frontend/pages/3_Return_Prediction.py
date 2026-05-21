import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Alpha Return Engine",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500;700&family=Bebas+Neue&display=swap');

/* ═══════════════════ RESET & TOKENS ═══════════════════ */
:root {
  --bg-void:      #03050A;
  --bg-base:      #070B12;
  --bg-surface:   #0C1220;
  --bg-raised:    #101828;
  --bg-hover:     #141F30;

  --border-dim:   #1A2840;
  --border-mid:   #1E3050;
  --border-glow:  #00FFD140;

  --accent:       #00FFD1;
  --accent-dim:   #00FFD120;
  --accent-mid:   #00FFD160;
  --accent-glow:  0 0 20px #00FFD130, 0 0 60px #00FFD108;

  --gold:         #F5C842;
  --gold-dim:     #F5C84220;
  --danger:       #FF3860;
  --danger-dim:   #FF386020;

  --text-primary:   #E8F0FA;
  --text-secondary: #7A94B0;
  --text-dim:       #3D526A;
  --text-accent:    #00FFD1;

  --font-display: 'Bebas Neue', sans-serif;
  --font-body:    'Space Grotesk', sans-serif;
  --font-mono:    'JetBrains Mono', monospace;

  --radius-sm: 2px;
  --radius-md: 4px;
  --transition: 0.22s cubic-bezier(0.4, 0, 0.2, 1);
}

/* ═══════════════════ BASE ═══════════════════ */
html, body, [class*="css"] {
  font-family: var(--font-body) !important;
  background-color: var(--bg-void) !important;
  color: var(--text-primary) !important;
}

/* Subtle noise texture on body */
body::before {
  content: '';
  position: fixed;
  inset: 0;
  background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.035'/%3E%3C/svg%3E");
  pointer-events: none;
  z-index: 0;
}

/* Grid lines background */
body::after {
  content: '';
  position: fixed;
  inset: 0;
  background-image:
    linear-gradient(var(--border-dim) 1px, transparent 1px),
    linear-gradient(90deg, var(--border-dim) 1px, transparent 1px);
  background-size: 60px 60px;
  opacity: 0.3;
  pointer-events: none;
  z-index: 0;
}

::-webkit-scrollbar { width: 3px; height: 3px; }
::-webkit-scrollbar-track { background: var(--bg-void); }
::-webkit-scrollbar-thumb { background: var(--accent-mid); border-radius: 0; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }

.block-container {
  padding: 2.5rem 3.5rem 5rem 3.5rem !important;
  max-width: 1500px !important;
  position: relative;
  z-index: 1;
}

#MainMenu, footer, header { visibility: hidden; }

/* ═══════════════════ HERO ═══════════════════ */
.hero-wrapper {
  position: relative;
  padding: 4rem 0 3rem 0;
  margin-bottom: 3rem;
  overflow: hidden;
}

/* Glowing horizontal rule */
.hero-wrapper::after {
  content: '';
  position: absolute;
  bottom: 0; left: 0; right: 0;
  height: 1px;
  background: linear-gradient(90deg,
    var(--accent) 0%,
    var(--accent-mid) 30%,
    transparent 70%
  );
}

/* Vertical accent line */
.hero-line {
  position: absolute;
  left: 0; top: 0;
  width: 3px; height: 100%;
  background: linear-gradient(180deg, var(--accent), transparent);
  box-shadow: var(--accent-glow);
}

.hero-eyebrow {
  font-family: var(--font-mono);
  font-size: 0.62rem;
  font-weight: 500;
  letter-spacing: 0.32em;
  color: var(--accent);
  text-transform: uppercase;
  margin-bottom: 1rem;
  margin-left: 1.2rem;
  display: flex;
  align-items: center;
  gap: 0.8rem;
}

.hero-eyebrow::before {
  content: '';
  width: 20px;
  height: 1px;
  background: var(--accent);
  display: inline-block;
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

.hero-title .accent-word {
  color: var(--accent);
  text-shadow: var(--accent-glow);
}

.hero-subtitle {
  font-family: var(--font-mono);
  font-size: 0.74rem;
  color: var(--text-secondary);
  margin: 1.4rem 0 0 1.2rem;
  max-width: 520px;
  line-height: 1.9;
  border-left: 2px solid var(--border-mid);
  padding-left: 1rem;
}

/* Large background counter text */
.hero-bg-text {
  position: absolute;
  right: -2rem; top: 50%;
  transform: translateY(-50%);
  font-family: var(--font-display);
  font-size: clamp(8rem, 18vw, 16rem);
  font-weight: 400;
  color: transparent;
  -webkit-text-stroke: 1px var(--border-dim);
  pointer-events: none;
  user-select: none;
  letter-spacing: 0.06em;
  line-height: 1;
}

/* Scanning line animation on hero */
@keyframes scan {
  0%   { transform: translateY(-100%); opacity: 0; }
  10%  { opacity: 1; }
  90%  { opacity: 1; }
  100% { transform: translateY(2000%); opacity: 0; }
}

.hero-scan {
  position: absolute;
  left: 0; right: 0; top: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--accent-mid), transparent);
  animation: scan 6s linear infinite;
  pointer-events: none;
}

/* ═══════════════════ SECTION LABELS ═══════════════════ */
.section-label {
  font-family: var(--font-mono);
  font-size: 0.58rem;
  letter-spacing: 0.28em;
  color: var(--accent);
  text-transform: uppercase;
  margin-bottom: 1.2rem;
  display: flex;
  align-items: center;
  gap: 0.8rem;
}

.section-label .sl-num {
  font-size: 0.5rem;
  color: var(--text-dim);
  border: 1px solid var(--border-mid);
  padding: 0.1rem 0.35rem;
  letter-spacing: 0.1em;
}

.section-label::after {
  content: '';
  flex: 1;
  height: 1px;
  background: linear-gradient(90deg, var(--border-mid), transparent);
  max-width: 180px;
}

/* ═══════════════════ FILE UPLOADER ═══════════════════ */
[data-testid="stFileUploader"] {
  background: var(--bg-surface) !important;
  border: 1px dashed var(--border-mid) !important;
  border-radius: var(--radius-sm) !important;
  padding: 1.2rem !important;
  transition: border-color var(--transition), background var(--transition) !important;
  position: relative;
}

[data-testid="stFileUploader"]:hover {
  border-color: var(--accent-mid) !important;
  background: var(--bg-raised) !important;
}

[data-testid="stFileUploadDropzone"] {
  background: transparent !important;
}

[data-testid="stFileUploaderFileName"] {
  font-family: var(--font-mono) !important;
  font-size: 0.72rem !important;
  color: var(--accent) !important;
}

/* ═══════════════════ TEXT INPUTS ═══════════════════ */
[data-testid="stTextInput"] input {
  background: var(--bg-surface) !important;
  border: 1px solid var(--border-mid) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text-primary) !important;
  font-family: var(--font-mono) !important;
  font-size: 0.74rem !important;
  padding: 0.6rem 0.9rem !important;
  transition: border-color var(--transition) !important;
}

[data-testid="stTextInput"] input:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px var(--accent-dim) !important;
  outline: none !important;
}

[data-testid="stTextInput"] label {
  font-family: var(--font-mono) !important;
  font-size: 0.64rem !important;
  letter-spacing: 0.1em !important;
  color: var(--text-secondary) !important;
  text-transform: uppercase !important;
}

/* ═══════════════════ STATUS CARDS ═══════════════════ */
.status-card {
  border-radius: var(--radius-sm);
  padding: 1rem 1.3rem;
  font-family: var(--font-mono);
  font-size: 0.75rem;
  line-height: 1.7;
  margin: 1rem 0;
  border-left: 3px solid;
  position: relative;
  overflow: hidden;
}

.status-card::before {
  content: '';
  position: absolute;
  inset: 0;
  opacity: 0.04;
  background: repeating-linear-gradient(
    0deg,
    transparent,
    transparent 2px,
    white 2px,
    white 3px
  );
  pointer-events: none;
}

.status-success {
  background: #051A14;
  border-color: var(--accent);
  color: var(--accent);
  box-shadow: inset 4px 0 12px #00FFD108;
}

.status-error {
  background: #160810;
  border-color: var(--danger);
  color: #FF7090;
  box-shadow: inset 4px 0 12px #FF386008;
}

.status-warning {
  background: #141006;
  border-color: var(--gold);
  color: var(--gold);
  box-shadow: inset 4px 0 12px #F5C84208;
}

.status-info {
  background: #08101E;
  border-color: #4895EF;
  color: #7ABAEF;
  box-shadow: inset 4px 0 12px #4895EF08;
}

/* ═══════════════════ METRIC GRID ═══════════════════ */
.metric-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 1px;
  background: var(--border-dim);
  margin: 1.5rem 0;
  border: 1px solid var(--border-dim);
}

.metric-cell {
  background: var(--bg-surface);
  padding: 1.2rem 1.5rem;
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
  transition: background var(--transition);
  position: relative;
  overflow: hidden;
}

.metric-cell::after {
  content: '';
  position: absolute;
  bottom: 0; left: 0;
  height: 2px; width: 0;
  background: var(--accent);
  transition: width 0.4s ease;
}

.metric-cell:hover {
  background: var(--bg-raised);
}

.metric-cell:hover::after {
  width: 100%;
}

.metric-label {
  font-family: var(--font-mono);
  font-size: 0.56rem;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  color: var(--text-dim);
}

.metric-value {
  font-family: var(--font-display);
  font-size: 2rem;
  font-weight: 400;
  line-height: 1;
  color: var(--text-primary);
  letter-spacing: 0.02em;
}

.metric-value.accent {
  color: var(--accent);
  text-shadow: 0 0 20px #00FFD150;
}

.metric-value.warn   { color: var(--gold); }
.metric-value.danger { color: var(--danger); }

/* ═══════════════════ DATA TABLE (streamlit native) ═══════════════════ */
[data-testid="stDataFrame"] {
  border: 1px solid var(--border-dim) !important;
  border-radius: var(--radius-sm) !important;
}

[data-testid="stDataFrame"] table {
  font-family: var(--font-mono) !important;
  font-size: 0.72rem !important;
}

/* ═══════════════════ PREDICTION TABLE ═══════════════════ */
.pred-table-wrapper {
  background: var(--bg-surface);
  border: 1px solid var(--border-dim);
  overflow: hidden;
}

.pred-table {
  width: 100%;
  border-collapse: collapse;
  font-family: var(--font-mono);
  font-size: 0.74rem;
}

.pred-table thead {
  background: var(--bg-raised);
  border-bottom: 1px solid var(--border-mid);
}

.pred-table th {
  color: var(--text-dim);
  font-size: 0.56rem;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  padding: 0.9rem 1.2rem;
  text-align: left;
  font-weight: 500;
  white-space: nowrap;
}

.pred-table th:first-child {
  color: var(--accent);
}

.pred-table td {
  padding: 0.75rem 1.2rem;
  color: var(--text-secondary);
  border-bottom: 1px solid var(--border-dim);
  vertical-align: middle;
}

.pred-table tr:last-child td { border-bottom: none; }

.pred-table tbody tr {
  transition: background var(--transition);
}

.pred-table tbody tr:hover td {
  background: var(--bg-hover);
  color: var(--text-primary);
}

/* Return value badges */
.badge-pos  { color: var(--accent);  font-weight: 600; }
.badge-neg  { color: var(--danger);  font-weight: 600; }
.badge-neut { color: var(--gold);    font-weight: 600; }

/* Match quality badges */
.match-badge {
  display: inline-flex;
  align-items: center;
  font-family: var(--font-mono);
  font-size: 0.5rem;
  letter-spacing: 0.14em;
  padding: 0.15rem 0.5rem;
  border: 1px solid;
  text-transform: uppercase;
  margin-left: 0.5rem;
  vertical-align: middle;
  border-radius: 1px;
}

.match-exact   { border-color: var(--accent);  color: var(--accent);  background: var(--accent-dim); }
.match-partial { border-color: var(--gold);    color: var(--gold);    background: var(--gold-dim); }
.match-cat     { border-color: #4895EF;        color: #7ABAEF;        background: #4895EF15; }

/* ═══════════════════ INSIGHT CARDS ═══════════════════ */
.insight-card {
  background: var(--bg-surface);
  border: 1px solid var(--border-dim);
  border-left: 3px solid var(--accent);
  padding: 1.5rem 1.8rem;
  margin-bottom: 1rem;
  position: relative;
  overflow: hidden;
  transition: border-color var(--transition), box-shadow var(--transition);
}

.insight-card::before {
  content: '';
  position: absolute;
  top: 0; right: 0;
  width: 200px; height: 200px;
  background: radial-gradient(circle at top right, var(--accent-dim), transparent 70%);
  pointer-events: none;
}

.insight-card:hover {
  border-color: var(--border-mid);
  box-shadow: 0 4px 40px #00000030;
}

.insight-card-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 1rem;
  margin-bottom: 1.2rem;
}

.insight-card h5 {
  font-family: var(--font-body);
  font-size: 0.92rem;
  font-weight: 600;
  color: var(--text-primary);
  margin: 0;
  letter-spacing: 0.01em;
  line-height: 1.3;
}

.insight-rec-badge {
  font-family: var(--font-display);
  font-size: 1.1rem;
  letter-spacing: 0.12em;
  padding: 0.2rem 0.8rem;
  border: 1px solid currentColor;
  white-space: nowrap;
  flex-shrink: 0;
}

.insight-row {
  display: flex;
  flex-wrap: wrap;
  gap: 2rem;
  margin-bottom: 1.2rem;
  padding-bottom: 1.2rem;
  border-bottom: 1px solid var(--border-dim);
}

.insight-kv {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.insight-kv-label {
  font-family: var(--font-mono);
  font-size: 0.54rem;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: var(--text-dim);
}

.insight-kv-value {
  font-family: var(--font-mono);
  font-size: 0.84rem;
  color: var(--text-primary);
  font-weight: 500;
}

.insight-text {
  font-family: var(--font-body);
  font-size: 0.8rem;
  color: var(--text-secondary);
  line-height: 1.85;
  margin-bottom: 0.6rem;
}

.insight-reason {
  font-family: var(--font-mono);
  font-size: 0.66rem;
  color: var(--text-dim);
  line-height: 1.6;
  padding-top: 0.6rem;
}

.insight-footer {
  font-family: var(--font-mono);
  font-size: 0.56rem;
  color: var(--text-dim);
  margin-top: 0.8rem;
  letter-spacing: 0.08em;
}

/* Confidence bar */
.conf-bar-wrap {
  display: flex;
  align-items: center;
  gap: 0.6rem;
}

.conf-bar-bg {
  flex: 1;
  height: 2px;
  background: var(--border-dim);
  max-width: 100px;
  position: relative;
  overflow: hidden;
}

.conf-bar-fill {
  height: 2px;
  position: relative;
}

.conf-bar-fill::after {
  content: '';
  position: absolute;
  right: 0; top: 0;
  width: 4px; height: 2px;
  background: inherit;
  filter: blur(4px);
}

/* ═══════════════════ DIVIDERS ═══════════════════ */
.h-rule {
  border: none;
  border-top: 1px solid var(--border-dim);
  margin: 2.5rem 0;
  position: relative;
}

.h-rule::before {
  content: '◈';
  position: absolute;
  top: -0.6rem;
  left: 50%;
  transform: translateX(-50%);
  font-size: 0.7rem;
  color: var(--text-dim);
  background: var(--bg-void);
  padding: 0 0.5rem;
}

/* ═══════════════════ BUTTON ═══════════════════ */
.stButton > button {
  background: transparent !important;
  border: 1px solid var(--accent) !important;
  color: var(--accent) !important;
  font-family: var(--font-mono) !important;
  font-size: 0.68rem !important;
  letter-spacing: 0.18em !important;
  text-transform: uppercase !important;
  border-radius: var(--radius-sm) !important;
  padding: 0.65rem 1.6rem !important;
  transition: all var(--transition) !important;
  position: relative !important;
  overflow: hidden !important;
}

.stButton > button::before {
  content: '' !important;
  position: absolute !important;
  inset: 0 !important;
  background: var(--accent) !important;
  transform: translateX(-101%) !important;
  transition: transform 0.3s ease !important;
  z-index: 0 !important;
}

.stButton > button:hover::before {
  transform: translateX(0) !important;
}

.stButton > button:hover {
  color: var(--bg-void) !important;
  box-shadow: var(--accent-glow) !important;
}

.stButton > button span {
  position: relative !important;
  z-index: 1 !important;
}

/* ═══════════════════ DOWNLOAD BUTTON ═══════════════════ */
[data-testid="stDownloadButton"] button {
  background: var(--accent-dim) !important;
  border: 1px solid var(--accent-mid) !important;
  color: var(--accent) !important;
  font-family: var(--font-mono) !important;
  font-size: 0.67rem !important;
  letter-spacing: 0.15em !important;
  text-transform: uppercase !important;
  border-radius: var(--radius-sm) !important;
  transition: all var(--transition) !important;
}

[data-testid="stDownloadButton"] button:hover {
  background: var(--accent) !important;
  color: var(--bg-void) !important;
  box-shadow: var(--accent-glow) !important;
}

/* ═══════════════════ PROGRESS BAR ═══════════════════ */
[data-testid="stProgressBar"] > div > div {
  background: var(--accent) !important;
  box-shadow: var(--accent-glow) !important;
}

[data-testid="stProgressBar"] {
  background: var(--border-dim) !important;
  border-radius: 0 !important;
  height: 2px !important;
}

/* ═══════════════════ EMPTY STATE ═══════════════════ */
.empty-state {
  height: 420px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  border: 1px solid var(--border-dim);
  color: var(--text-dim);
  font-family: var(--font-mono);
  font-size: 0.68rem;
  letter-spacing: 0.15em;
  text-align: center;
  gap: 0.6rem;
  background: var(--bg-surface);
  position: relative;
  overflow: hidden;
}

.empty-state::before {
  content: '';
  position: absolute;
  inset: 0;
  background:
    repeating-linear-gradient(
      45deg,
      transparent,
      transparent 20px,
      var(--border-dim) 20px,
      var(--border-dim) 21px
    );
  opacity: 0.15;
}

.empty-state-icon {
  font-size: 2.8rem;
  opacity: 0.15;
  line-height: 1;
  position: relative;
}

.empty-state-label {
  position: relative;
  font-size: 0.68rem;
}

.empty-state-hint {
  position: relative;
  font-size: 0.58rem;
  opacity: 0.5;
  margin-top: 0.2rem;
}

/* ═══════════════════ TABLE FOOTER NOTE ═══════════════════ */
.table-note {
  font-family: var(--font-mono);
  font-size: 0.58rem;
  color: var(--text-dim);
  margin-top: 0.75rem;
  line-height: 1.8;
  padding: 0.6rem 0;
  border-top: 1px solid var(--border-dim);
  display: flex;
  flex-wrap: wrap;
  gap: 0.8rem;
  align-items: center;
}

/* ═══════════════════ UPLOAD HELPER TEXT ═══════════════════ */
.upload-hint {
  font-family: var(--font-mono);
  font-size: 0.62rem;
  color: var(--text-dim);
  margin-top: 0.8rem;
  line-height: 2;
  padding: 0.7rem 0.9rem;
  background: var(--bg-surface);
  border-left: 2px solid var(--border-mid);
}

.upload-hint span.hl { color: var(--text-secondary); }
.upload-hint span.accent { color: var(--accent); }

/* ═══════════════════ LEFT PANEL CARD ═══════════════════ */
.panel-card {
  background: var(--bg-surface);
  border: 1px solid var(--border-dim);
  padding: 1.3rem 1.4rem;
  margin-bottom: 0;
}

/* ═══════════════════ FOOTER ═══════════════════ */
.app-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-family: var(--font-mono);
  font-size: 0.58rem;
  color: var(--text-dim);
  letter-spacing: 0.12em;
  padding: 1rem 0;
  border-top: 1px solid var(--border-dim);
  margin-top: 4rem;
}

.footer-dot {
  width: 4px; height: 4px;
  border-radius: 50%;
  background: var(--accent);
  display: inline-block;
  margin: 0 0.4rem;
  box-shadow: var(--accent-glow);
  animation: pulse 2.5s ease-in-out infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.3; }
}

/* ═══════════════════ COLUMN LAYOUT POLISH ═══════════════════ */
[data-testid="column"]:first-child {
  padding-right: 1.5rem !important;
  border-right: 1px solid var(--border-dim);
}

/* ═══════════════════ STALE STREAMLIT ELEMENT OVERRIDES ═══════════════════ */
.stAlert { border-radius: var(--radius-sm) !important; }

p {
  font-family: var(--font-body) !important;
  font-size: 0.85rem !important;
  color: var(--text-secondary) !important;
  line-height: 1.7 !important;
}

h1, h2, h3, h4, h5, h6 {
  font-family: var(--font-body) !important;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL BUNDLE — load once, cache forever
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_model_bundle(path: str):
    try:
        return joblib.load(path), None
    except FileNotFoundError:
        return None, f"Model file not found: {path}"
    except Exception as e:
        return None, str(e)


# ══════════════════════════════════════════════════════════════════════════════
# LOOKUP — fuzzy match fund name against training DB
# ══════════════════════════════════════════════════════════════════════════════
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


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE BUILDER
# ══════════════════════════════════════════════════════════════════════════════
RISK_MAP = {
    'low': 1, 'very low': 1,
    'low to moderate': 2, 'low-moderate': 2,
    'moderate': 3, 'medium': 3,
    'moderately high': 4, 'moderate to high': 4,
    'high': 5,
    'very high': 6, 'very-high': 6,
}

def build_features(user_row: dict, matched: dict | None, bundle: dict) -> tuple:
    feature_cols = bundle['feature_cols']
    cat_medians  = bundle['cat_medians']
    sub_medians  = bundle['sub_medians']
    le_cat       = bundle['le_cat']
    le_sub       = bundle['le_sub']

    row = {}
    cat = None
    sub = None
    for src in [user_row, matched or {}]:
        if not cat and pd.notna(src.get('category')): cat = str(src['category'])
        if not sub and pd.notna(src.get('sub_category')): sub = str(src.get('sub_category', ''))

    if cat and cat in cat_medians.index:
        for f in feature_cols:
            row[f] = cat_medians.loc[cat, f]

    used_from_match = []
    if matched:
        for f in feature_cols:
            v = matched.get(f)
            if v is not None and pd.notna(v):
                row[f] = float(v)
                used_from_match.append(f)

    for f in feature_cols:
        v = user_row.get(f)
        if v is not None and pd.notna(v):
            try:
                row[f] = float(v)
            except (ValueError, TypeError):
                pass

    rl_raw = user_row.get('risk_level', '')
    if isinstance(rl_raw, str):
        mapped = RISK_MAP.get(rl_raw.lower().strip())
        if mapped:
            row['risk_level'] = mapped

    try:
        row['cat_enc'] = int(le_cat.transform([cat])[0]) if cat and cat in le_cat.classes_ else 0
    except Exception:
        row['cat_enc'] = 0
    try:
        row['sub_enc'] = int(le_sub.transform([sub])[0]) if sub and sub in le_sub.classes_ else 0
    except Exception:
        row['sub_enc'] = 0

    X_df = pd.DataFrame([row])[feature_cols + ['cat_enc', 'sub_enc']]
    return X_df, cat, sub, used_from_match


# ══════════════════════════════════════════════════════════════════════════════
# INSIGHT GENERATOR
# ══════════════════════════════════════════════════════════════════════════════
def generate_insight(fund_name: str, preds: dict, matched: dict | None,
                     user_row: dict, cat: str | None, bundle: dict) -> dict:
    cat_avgs = bundle.get('cat_avgs')

    p1, p3, p5 = preds['returns_1yr'], preds['returns_3yr'], preds['returns_5yr']

    if cat and cat_avgs is not None and cat in cat_avgs.index:
        b1 = cat_avgs.loc[cat, 'returns_1yr']
        b3 = cat_avgs.loc[cat, 'returns_3yr']
        b5 = cat_avgs.loc[cat, 'returns_5yr']
    else:
        b1, b3, b5 = 3.9, 18.5, 9.5

    score = 0
    score += 1 if p1 > b1 else (-1 if p1 < b1 * 0.7 else 0)
    score += 2 if p3 > b3 else (-2 if p3 < b3 * 0.7 else 0)
    score += 2 if p5 > b5 else (-2 if p5 < b5 * 0.7 else 0)

    if score >= 3:
        rec, rec_color = 'Buy',   '#00FFD1'
    elif score >= 0:
        rec, rec_color = 'Hold',  '#F5C842'
    else:
        rec, rec_color = 'Avoid', '#FF3860'

    expense_ratio = user_row.get('expense_ratio') or (matched or {}).get('expense_ratio') or 1.0
    sharpe_val    = (matched or {}).get('sharpe')
    try:
        sharpe_val = float(sharpe_val) if sharpe_val is not None else None
    except Exception:
        sharpe_val = None

    conf_score = 0
    if matched:                       conf_score += 2
    if p3 > 0 and p5 > 0:            conf_score += 1
    if sharpe_val and sharpe_val > 1: conf_score += 1

    confidence = ['Low', 'Low', 'Medium', 'Medium', 'High'][min(conf_score, 4)]
    conf_pct   = {0: 20, 1: 35, 2: 55, 3: 72, 4: 90}[min(conf_score, 4)]

    rl_raw = user_row.get('risk_level') or (matched or {}).get('risk_level')
    risk_int = None
    if isinstance(rl_raw, str):
        risk_int = RISK_MAP.get(rl_raw.lower().strip())
    elif rl_raw is not None:
        try: risk_int = int(rl_raw)
        except Exception: pass

    risk_labels = {1: 'Very Low', 2: 'Low-Moderate', 3: 'Moderate',
                   4: 'Moderately High', 5: 'High', 6: 'Very High'}
    risk_label = risk_labels.get(risk_int, 'Moderate')

    vs_bench_1 = "above" if p1 > b1 else "below"
    vs_bench_3 = "above" if p3 > b3 else "below"
    outperform = abs(p3 - b3)

    cat_str = cat or "this category"
    insight_parts = [
        f"Predicted 3-year return of {p3:.1f}% is {vs_bench_3} the {cat_str} "
        f"category average of {b3:.1f}% (Δ {outperform:.1f}pp).",
        f"Short-term 1-year outlook of {p1:.1f}% is {vs_bench_1} the category benchmark of {b1:.1f}%.",
    ]

    try:
        er = float(expense_ratio)
        if er > 1.5:
            insight_parts.append(f"Expense ratio of {er:.2f}% is elevated — factor this into net returns.")
        elif er < 0.5:
            insight_parts.append(f"Low expense ratio of {er:.2f}% is a cost efficiency advantage.")
    except Exception:
        pass

    if matched:
        hist_5yr = matched.get('returns_5yr')
        if hist_5yr is not None and pd.notna(hist_5yr):
            try:
                insight_parts.append(
                    f"Historical 5-year return in training data: {float(hist_5yr):.1f}% "
                    f"(model predicts {p5:.1f}%)."
                )
            except Exception:
                pass

    insight_text = " ".join(insight_parts)

    if rec == 'Buy':
        reason = f"Predicted returns outperform {cat_str} benchmarks across multiple horizons."
    elif rec == 'Hold':
        reason = f"Predicted returns are broadly in line with {cat_str} category averages."
    else:
        reason = f"Predicted returns lag {cat_str} category averages — consider alternatives."

    matched_name = (matched or {}).get('scheme_name', '—')
    match_type   = 'exact' if matched and matched.get('scheme_name','').lower() == fund_name.lower() \
                   else ('partial' if matched else 'category')

    return {
        'recommendation': rec,
        'rec_color': rec_color,
        'confidence': confidence,
        'conf_pct': conf_pct,
        'risk_label': risk_label,
        'insight_text': insight_text,
        'recommendation_reason': reason,
        'matched_scheme': matched_name,
        'match_type': match_type,
        'benchmark_1yr': b1,
        'benchmark_3yr': b3,
        'benchmark_5yr': b5,
    }


# ══════════════════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-wrapper">
    <div class="hero-scan"></div>
    <div class="hero-line"></div>
    <div class="hero-eyebrow">Quantitative Analytics · ML-Powered · Fully Offline</div>
    <h1 class="hero-title">Alpha<br><span class="accent-word">Return</span><br>Engine</h1>
    <p class="hero-subtitle">
        Upload any portfolio file. The engine matches each fund against an
        814-fund training database, fills missing features from category
        statistics, and runs a Gradient Boosting model to predict
        1yr · 3yr · 5yr returns with benchmarked insights.
    </p>
    <div class="hero-bg-text">α</div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT
# ══════════════════════════════════════════════════════════════════════════════
left_col, right_col = st.columns([1.05, 1.95], gap="large")

with left_col:
    st.markdown("""
    <div class="section-label">
        <span class="sl-num">01</span>Portfolio File
    </div>""", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drop portfolio file",
        type=["csv", "xlsx"],
        label_visibility="collapsed",
    )

    st.markdown("""
    <div class="upload-hint">
        Accepted: <span class="hl">.csv</span> &nbsp;·&nbsp; <span class="hl">.xlsx</span><br>
        Needs a fund / share <em>name</em> column — everything else is optional.<br>
        <span class="accent">↑</span> No external API — fully offline ML.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="h-rule">', unsafe_allow_html=True)

    st.markdown("""
    <div class="section-label">
        <span class="sl-num">02</span>Model & Training Data
    </div>""", unsafe_allow_html=True)

    default_model = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../ml_models/best_return_model.pkl")
    )
    model_path = st.text_input("Model file (.pkl)", value=default_model)

    st.markdown('<hr class="h-rule">', unsafe_allow_html=True)

    st.markdown("""
    <div class="section-label">
        <span class="sl-num">03</span>Column Mapping
    </div>""", unsafe_allow_html=True)

    fund_col_hint = st.text_input(
        "Fund / share name column",
        value="fund_name",
        help="Column in the uploaded file that contains the fund or share name.",
    )

    run_btn = st.button("▶  Run ML Analysis", use_container_width=True)


with right_col:
    if not uploaded_file:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon">◈</div>
            <div class="empty-state-label">No Data Loaded</div>
            <div class="empty-state-hint">Upload a portfolio file to begin analysis</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            n_rows, n_cols = df.shape
            n_nulls = int(df.isnull().sum().sum())
            numeric_cols = df.select_dtypes(include="number").columns.tolist()

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
            <div class="section-label">
                <span class="sl-num">Preview</span>Dataset
            </div>""", unsafe_allow_html=True)
            st.dataframe(df.head(8), use_container_width=True, hide_index=False)

            if run_btn:
                bundle, model_err = load_model_bundle(model_path)
                if model_err:
                    st.markdown(f"""
                    <div class="status-card status-error">
                        ✗ Could not load model<br>
                        <span style="opacity:0.7;font-size:0.7rem;">{model_err}</span><br>
                        <span style="opacity:0.8;font-size:0.68rem;">This usually happens due to model pickle incompatibility across pandas versions.</span>
                    </div>""", unsafe_allow_html=True)
                    st.stop()

                if isinstance(bundle, dict):
                    models      = bundle['models']
                    imp         = bundle['imputer']
                    training_df = bundle['training_df']
                else:
                    models = bundle
                    imp = None
                    training_df = None

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
                        ⚠ Cannot find a fund name column. Enter the column name in "Fund / share name column".
                    </div>""", unsafe_allow_html=True)
                    st.stop()

                st.markdown(
                    f'<div class="status-card status-success">✓ Model loaded &nbsp;·&nbsp; Using column <strong>{fund_col}</strong> as fund identifier</div>',
                    unsafe_allow_html=True,
                )

                st.markdown("""
                <div class="section-label" style="margin-top:1.5rem;">
                    <span class="sl-num">04</span>Predictions &amp; Insights
                </div>""", unsafe_allow_html=True)

                results = []
                progress = st.progress(0, text="Initialising ML pipeline…")

                for i, (_, row) in enumerate(df.iterrows()):
                    fund_name = str(row.get(fund_col, f"Fund #{i+1}"))
                    user_row  = row.to_dict()

                    if training_df is not None and imp is not None and isinstance(bundle, dict) and isinstance(models, dict):
                        matched, mtype = lookup_fund(fund_name, training_df)

                        X_df, cat, sub, used_fields = build_features(user_row, matched, bundle)

                        X_imp = imp.transform(X_df)
                        preds = {t: float(models[t].predict(X_imp)[0])
                                 for t in ['returns_1yr', 'returns_3yr', 'returns_5yr']}

                        insight = generate_insight(fund_name, preds, matched, user_row, cat, bundle)

                        results.append({
                            'fund_name':  fund_name,
                            'category':   cat or '—',
                            'pred_1yr':   preds['returns_1yr'],
                            'pred_3yr':   preds['returns_3yr'],
                            'pred_5yr':   preds['returns_5yr'],
                            **insight,
                            'used_fields': used_fields,
                        })
                    else:
                        p1 = float(user_row.get('returns_1yr')) if 'returns_1yr' in user_row and pd.notna(user_row.get('returns_1yr')) else 0.0
                        p3 = float(user_row.get('returns_3yr')) if 'returns_3yr' in user_row and pd.notna(user_row.get('returns_3yr')) else 0.0
                        if 'returns_5yr' in user_row and pd.notna(user_row.get('returns_5yr')):
                            p5 = float(user_row.get('returns_5yr'))
                        elif p3 != 0:
                            p5 = p3
                        else:
                            p5 = 9.5

                        cat = user_row.get('category') if 'category' in user_row else None
                        preds = {'returns_1yr': p1, 'returns_3yr': p3, 'returns_5yr': p5}
                        rec = 'Hold'
                        if p5 >= 12.2:
                            rec = 'Buy'
                        elif p5 <= 7.0:
                            rec = 'Avoid'

                        results.append({
                            'fund_name': fund_name,
                            'category': cat or '—',
                            'pred_1yr': p1,
                            'pred_3yr': p3,
                            'pred_5yr': p5,
                            'recommendation': rec,
                            'rec_color': '#F5C842' if rec=='Hold' else ('#00FFD1' if rec=='Buy' else '#FF3860'),
                            'confidence': 'Low',
                            'conf_pct': 20,
                            'risk_label': 'Moderate',
                            'insight_text': 'Model bundle unavailable — using fallback baseline from available data columns.',
                            'recommendation_reason': 'Low-confidence baseline estimate (no compatible ML feature bundle loaded).',
                            'matched_scheme': '—',
                            'match_type': 'error',
                            'benchmark_1yr': 3.9,
                            'benchmark_3yr': 18.5,
                            'benchmark_5yr': 9.5,
                            'used_fields': [],
                        })

                    progress.progress((i + 1) / n_rows,
                                      text=f"Predicting · {fund_name[:50]}…")

                progress.empty()

                # ── Summary metrics ──
                buys   = sum(1 for r in results if r['recommendation'] == 'Buy')
                holds  = sum(1 for r in results if r['recommendation'] == 'Hold')
                avoids = sum(1 for r in results if r['recommendation'] == 'Avoid')
                avg_3yr = np.mean([r['pred_3yr'] for r in results])
                matches = sum(1 for r in results if r['match_type'] in ('exact', 'partial'))

                st.markdown(f"""
                <div class="metric-grid">
                    <div class="metric-cell">
                        <span class="metric-label">Avg 3yr Pred</span>
                        <span class="metric-value accent">{avg_3yr:.1f}%</span>
                    </div>
                    <div class="metric-cell">
                        <span class="metric-label">Buy Signals</span>
                        <span class="metric-value accent">{buys}</span>
                    </div>
                    <div class="metric-cell">
                        <span class="metric-label">Hold Signals</span>
                        <span class="metric-value warn">{holds}</span>
                    </div>
                    <div class="metric-cell">
                        <span class="metric-label">Avoid Signals</span>
                        <span class="metric-value danger">{avoids}</span>
                    </div>
                    <div class="metric-cell">
                        <span class="metric-label">DB Matches</span>
                        <span class="metric-value">{matches}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # ── Prediction table ──
                def fmt_ret(v, b=None):
                    cls  = "badge-pos" if v >= 0 else "badge-neg"
                    sign = "▲" if v >= 0 else "▼"
                    bench = (f"<br><span style='font-size:0.54rem;color:#3D526A;'>vs {b:.1f}%</span>"
                             if b is not None else "")
                    return f"<td class='{cls}'>{sign} {v:.1f}%{bench}</td>"

                rows_html = ""
                for r in results:
                    rec = r['recommendation']
                    rec_cls = {"Buy": "badge-pos", "Hold": "badge-neut", "Avoid": "badge-neg"}.get(rec, "")
                    match_badge = {
                        'exact':   '<span class="match-badge match-exact">Exact</span>',
                        'partial': '<span class="match-badge match-partial">Partial</span>',
                        'category':'<span class="match-badge match-cat">Category</span>',
                    }.get(r['match_type'], '')

                    rows_html += f"""
                    <tr>
                        <td style="max-width:260px;line-height:1.5;">
                            <span style="color:var(--text-primary);font-weight:500;">{r['fund_name']}</span>
                            {match_badge}<br>
                            <span style="font-size:0.58rem;color:var(--text-dim);">{r['category']}</span>
                        </td>
                        {fmt_ret(r['pred_1yr'], r['benchmark_1yr'])}
                        {fmt_ret(r['pred_3yr'], r['benchmark_3yr'])}
                        {fmt_ret(r['pred_5yr'], r['benchmark_5yr'])}
                        <td style="font-size:0.64rem;color:var(--text-dim);">{r['confidence']}</td>
                        <td class="{rec_cls}" style="font-weight:600;letter-spacing:0.05em;">{rec}</td>
                    </tr>"""

                st.markdown(f"""
                <div class="pred-table-wrapper">
                <table class="pred-table">
                    <thead><tr>
                        <th>Fund / Instrument</th>
                        <th>1 Yr</th>
                        <th>3 Yr</th>
                        <th>5 Yr</th>
                        <th>Confidence</th>
                        <th>Signal</th>
                    </tr></thead>
                    <tbody>{rows_html}</tbody>
                </table>
                </div>
                <div class="table-note">
                    Benchmarks are category averages from the 814-fund training dataset.
                    &nbsp;
                    <span class="match-badge match-exact">Exact</span>&nbsp; direct DB match
                    &nbsp;
                    <span class="match-badge match-partial">Partial</span>&nbsp; fuzzy name match
                    &nbsp;
                    <span class="match-badge match-cat">Category</span>&nbsp; category-level inference
                </div>
                """, unsafe_allow_html=True)

                # ── Per-fund insight cards ──
                st.markdown('<hr class="h-rule">', unsafe_allow_html=True)
                st.markdown("""
                <div class="section-label">
                    <span class="sl-num">05</span>Fund-Level Insights
                </div>""", unsafe_allow_html=True)

                for r in results:
                    color = r['rec_color']
                    conf_color = {'High': '#00FFD1', 'Medium': '#F5C842', 'Low': '#FF3860'}.get(r['confidence'], '#3D526A')
                    conf_w = r['conf_pct']

                    st.markdown(f"""
                    <div class="insight-card" style="border-left-color:{color};">
                        <div class="insight-card-header">
                            <h5>{r['fund_name']}</h5>
                            <span class="insight-rec-badge" style="color:{color};border-color:{color}40;background:{color}10;">
                                {r['recommendation']}
                            </span>
                        </div>
                        <div class="insight-row">
                            <div class="insight-kv">
                                <span class="insight-kv-label">Category</span>
                                <span class="insight-kv-value">{r['category']}</span>
                            </div>
                            <div class="insight-kv">
                                <span class="insight-kv-label">1yr</span>
                                <span class="insight-kv-value" style="color:{color};">{r['pred_1yr']:.1f}%</span>
                            </div>
                            <div class="insight-kv">
                                <span class="insight-kv-label">3yr</span>
                                <span class="insight-kv-value" style="color:{color};">{r['pred_3yr']:.1f}%</span>
                            </div>
                            <div class="insight-kv">
                                <span class="insight-kv-label">5yr</span>
                                <span class="insight-kv-value" style="color:{color};">{r['pred_5yr']:.1f}%</span>
                            </div>
                            <div class="insight-kv">
                                <span class="insight-kv-label">Risk</span>
                                <span class="insight-kv-value">{r['risk_label']}</span>
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
                        <div class="insight-reason">→ {r['recommendation_reason']}</div>
                        <div class="insight-footer">DB match · {r['matched_scheme']}</div>
                    </div>
                    """, unsafe_allow_html=True)

                # ── Export ──
                export_rows = [{
                    'Fund':                     r['fund_name'],
                    'Category':                 r['category'],
                    'Match Type':               r['match_type'],
                    'Matched DB Scheme':        r['matched_scheme'],
                    'Predicted Return 1yr (%)': round(r['pred_1yr'], 2),
                    'Predicted Return 3yr (%)': round(r['pred_3yr'], 2),
                    'Predicted Return 5yr (%)': round(r['pred_5yr'], 2),
                    'Confidence':               r['confidence'],
                    'Risk Label':               r['risk_label'],
                    'Recommendation':           r['recommendation'],
                    'Reason':                   r['recommendation_reason'],
                    'Insight':                  r['insight_text'],
                } for r in results]

                csv_data = pd.DataFrame(export_rows).to_csv(index=False).encode()
                st.download_button(
                    label="↓  Export Predictions (.csv)",
                    data=csv_data,
                    file_name="alpha_ml_predictions.csv",
                    mime="text/csv",
                )

        except Exception as e:
            st.markdown(f"""
            <div class="status-card status-error">
                ✗ Failed to process file<br>
                <span style="opacity:0.7;font-size:0.7rem;">{e}</span>
            </div>""", unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    <span>◈ ALPHA RETURN ENGINE <span class="footer-dot"></span> v5.0</span>
    <span>GRADIENT BOOSTING · 814 FUNDS · ZERO EXTERNAL API</span>
</div>
""", unsafe_allow_html=True)