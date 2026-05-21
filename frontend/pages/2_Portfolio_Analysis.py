# frontend/pages/2_Portfolio_Analysis.py

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Mutual Fund Portfolio Analyzer",
    page_icon="📊",
    layout="wide"
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
    background-color: #080C14;
    color: #D4DCE8;
}

.stApp {
    background: #080C14;
}

#MainMenu, footer, header {
    visibility: hidden;
}

.block-container {
    padding-top: 2rem;
    max-width: 1400px;
}

/* TITLE */

.main-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.5rem;
    font-weight: 800;
    color: #EAEFF6;
    margin-bottom: 0.3rem;
}

.sub-title {
    color: #5A6A80;
    font-size: 0.8rem;
    margin-bottom: 2rem;
}

/* SECTION */

.section-label {
    display: inline-block;
    padding: 5px 12px;
    border-radius: 5px;
    background: rgba(0,200,160,0.08);
    color: #00C8A0;
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 12px;
}

/* KPI CARD */

.kpi-card {
    background: linear-gradient(135deg, #0E1620 0%, #0A1018 100%);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 1.3rem;
    height: 100%;
}

.kpi-label {
    color: #5A6A80;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 10px;
}

.kpi-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: #EAEFF6;
}

.kpi-badge {
    margin-top: 10px;
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.6rem;
    background: rgba(0,200,160,0.08);
    color: #00C8A0;
}

/* CHART CARD */

.chart-card {
    background: #0E1620;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 1rem;
    margin-bottom: 1rem;
}

/* RECOMMENDATION */

.rec-card {
    background: #0E1620;
    border-left: 4px solid #00C8A0;
    padding: 1rem;
    border-radius: 10px;
    margin-bottom: 10px;
    color: #D4DCE8;
}

/* HR */

.hr-line {
    border-top: 1px solid rgba(255,255,255,0.06);
    margin: 2rem 0;
}

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# FUNCTIONS
# ─────────────────────────────────────────────────────────────

def kpi(col, label, value, badge):

    col.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-badge">{badge}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

PLOT_BG = "rgba(0,0,0,0)"
GRID_COLOR = "rgba(255,255,255,0.05)"
FONT_COLOR = "#8A9BB0"

PALETTE = [
    "#00C8A0",
    "#0078FF",
    "#F5A623",
    "#FF4E6A",
    "#A78BFA"
]

def base_layout():

    return dict(
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(
            family="DM Mono",
            color=FONT_COLOR
        ),
        xaxis=dict(gridcolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR),
        margin=dict(l=10, r=10, t=20, b=10)
    )

# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────

st.markdown(
    '<div class="section-label">AI Portfolio Intelligence</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-title">AI Mutual Fund Portfolio Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Upload your portfolio and get instant AI-powered insights</div>',
    unsafe_allow_html=True
)

st.markdown('<div class="hr-line"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# UPLOAD SECTION
# ─────────────────────────────────────────────────────────────

left, right = st.columns([1, 1.2])

with left:

    st.markdown(
        '<div class="section-label">Upload Portfolio</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload CSV or Excel File",
        type=["csv", "xlsx"]
    )

    st.info("Accepted formats: CSV and Excel")

with right:

    st.markdown(
        '<div class="section-label">Smart Detection Engine</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    ### AI Engine Automatically Detects:

    - Different column names
    - Returns/CAGR columns
    - Expense ratios
    - Investment amounts
    - Risk levels
    - Portfolio categories
    - Unknown schemas
    """)

st.markdown('<div class="hr-line"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# MAIN ANALYSIS
# ─────────────────────────────────────────────────────────────

if uploaded_file is None:

    st.info(
        "Upload a CSV or Excel portfolio file to begin AI-powered analysis."
    )
    st.stop()

try:

    if uploaded_file.name.endswith(".csv"):

        df = pd.read_csv(uploaded_file)

    else:

        df = pd.read_excel(uploaded_file)

    st.session_state["portfolio_df"] = df

    st.success("Portfolio uploaded successfully")

except Exception as e:

    st.error(f"Error reading file: {e}")
    st.stop()

# ─────────────────────────────────────────────────────────
# INDUSTRY LEVEL PORTFOLIO ENGINE
# ─────────────────────────────────────────────────────────

# CLEAN COLUMN NAMES
df.columns = [
    str(col).strip().lower()
    for col in df.columns
]

# REMOVE EMPTY ROWS
df = df.dropna(how="all")

# SMART COLUMN IDENTIFICATION

def find_column(possible_keywords):

    for col in df.columns:

        col_clean = str(col).lower()

        for keyword in possible_keywords:

            if keyword in col_clean:

                return col

    return None

# DETECT REAL-WORLD COLUMNS

fund_col = find_column([
    "scheme",
    "fund",
    "name"
])

returns_col = find_column([
    "xirr",
    "return",
    "cagr",
    "performance"
])

investment_col = find_column([
    "aum",
    "current value",
    "current_value",
    "investment",
    "amount",
    "portfolio",
    "value"
])

category_col = find_column([
    "category",
    "type"
])

expense_col = find_column([
    "expense",
    "ter",
    "ratio"
])

# VALIDATION

if fund_col is None:

    st.error("Unable to detect fund names.")
    st.stop()

if returns_col is None:

    st.error("Unable to detect return/XIRR column.")
    st.stop()

if investment_col is None:

    st.error("Unable to detect investment/AUM column.")
    st.stop()

# CREATE FALLBACKS

if category_col is None:

    df["generated_category"] = "Mutual Fund"

    category_col = "generated_category"

if expense_col is None:

    df["generated_expense"] = 1.0

    expense_col = "generated_expense"

# ─────────────────────────────────────────────────────────
# CLEAN NUMERIC VALUES
# ─────────────────────────────────────────────────────────

def clean_numeric(series):

    return pd.to_numeric(

        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.extract(r'(-?\d+\.?\d*)')[0],

        errors="coerce"
    )

df[returns_col] = clean_numeric(
    df[returns_col]
)

df[investment_col] = clean_numeric(
    df[investment_col]
)

df[expense_col] = clean_numeric(
    df[expense_col]
)

# REMOVE INVALID ROWS

df = df.dropna(
    subset=[
        returns_col,
        investment_col
    ]
)

# REMOVE TOTAL ROWS

df = df[
    ~df[fund_col]
    .astype(str)
    .str.contains(
        "total",
        case=False,
        na=False
    )
]

# REMOVE OUTLIERS

df = df[
    (df[returns_col] > -100) &
    (df[returns_col] < 300)
]

if df.empty:

    st.error("No valid portfolio records found.")
    st.stop()

# ─────────────────────────────────────────────────────────
# REAL-TIME WEIGHTED ANALYSIS
# ─────────────────────────────────────────────────────────

# PORTFOLIO WEIGHTS

df["weight"] = (
    df[investment_col]
    / df[investment_col].sum()
)

# WEIGHTED RETURN

avg_return = np.average(
    df[returns_col],
    weights=df["weight"]
)

# WEIGHTED EXPENSE

avg_expense = np.average(
    df[expense_col],
    weights=df["weight"]
)

# VOLATILITY

volatility = np.sqrt(
    np.average(
        (
            df[returns_col]
            - avg_return
        ) ** 2,
        weights=df["weight"]
    )
)

# TOTAL INVESTMENT

total_investment = (
    df[investment_col].sum()
)

# TOTAL FUNDS

total_funds = len(df)

# CATEGORY COUNT

categories = (
    df[category_col]
    .nunique()
)

# SHARPE-LIKE SCORE

risk_free_rate = 6

sharpe_ratio = (
    avg_return - risk_free_rate
) / max(volatility, 1)

# DIVERSIFICATION

largest_category_weight = (
    df.groupby(category_col)["weight"]
    .sum()
    .max()
)

diversification_score = (
    100 - largest_category_weight * 100
)

# ─────────────────────────────────────────────────────────
# AI HEALTH SCORE
# ─────────────────────────────────────────────────────────

score = 0

# RETURNS

if avg_return >= 18:
    score += 40

elif avg_return >= 12:
    score += 30

elif avg_return >= 8:
    score += 20

else:
    score += 10

# EXPENSE

if avg_expense <= 0.7:
    score += 30

elif avg_expense <= 1.5:
    score += 20

else:
    score += 10

# RISK

if volatility <= 8:
    score += 30

elif volatility <= 15:
    score += 20

else:
    score += 10

# DIVERSIFICATION BONUS

score += diversification_score * 0.2

score = int(
    min(score, 100)
)

# ─────────────────────────────────────────────────────────
# PORTFOLIO RISK LABEL
# ─────────────────────────────────────────────────────────

if volatility >= 18:

    portfolio_risk = "High Risk"

elif volatility >= 10:

    portfolio_risk = "Moderate Risk"

else:

    portfolio_risk = "Low Risk"

# ─────────────────────────────────────────────────────────
# DATA PREVIEW
# ─────────────────────────────────────────────────────────

st.markdown(
    '<div class="section-label">Portfolio Data</div>',
    unsafe_allow_html=True
)

st.dataframe(
    df,
    use_container_width=True
)

csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇ Download Cleaned Portfolio",
    data=csv,
    file_name="cleaned_portfolio.csv",
    mime="text/csv"
)

st.markdown('<div class="hr-line"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# KPI SECTION
# ─────────────────────────────────────────────────────────

st.markdown(
    '<div class="section-label">Portfolio Summary</div>',
    unsafe_allow_html=True
)

c1, c2, c3, c4, c5 = st.columns(5)

kpi(
    c1,
    "Investment",
    f"₹{total_investment:,.0f}",
    "Portfolio Value"
)

kpi(
    c2,
    "Funds",
    total_funds,
    "Holdings"
)

kpi(
    c3,
    "Weighted Return",
    f"{avg_return:.2f}%",
    "AI Weighted CAGR"
)

kpi(
    c4,
    "Expense Ratio",
    f"{avg_expense:.2f}%",
    "Weighted Cost"
)

kpi(
    c5,
    "Health Score",
    f"{score}/100",
    portfolio_risk
)

st.markdown('<div class="hr-line"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────

st.markdown(
    '<div class="section-label">Portfolio Visualizations</div>',
    unsafe_allow_html=True
)

left_chart, right_chart = st.columns(2)

with left_chart:

    pie = px.pie(
        df,
        names=category_col,
        values=investment_col,
        hole=0.55,
        color_discrete_sequence=PALETTE
    )

    pie.update_layout(**base_layout())

    st.plotly_chart(
        pie,
        use_container_width=True
    )

with right_chart:

    hist = px.histogram(
        df,
        x=returns_col,
        nbins=20,
        color_discrete_sequence=["#0078FF"]
    )

    hist.update_layout(**base_layout())

    st.plotly_chart(
        hist,
        use_container_width=True
    )

# ─────────────────────────────────────────────────────────
# SCATTER CHART
# ─────────────────────────────────────────────────────────

scatter = px.scatter(
    df,
    x=expense_col,
    y=returns_col,
    color=category_col,
    size=investment_col,
    hover_data=[fund_col],
    color_discrete_sequence=PALETTE
)

scatter.update_layout(**base_layout())

st.plotly_chart(
    scatter,
    use_container_width=True
)

st.markdown('<div class="hr-line"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# AI INSIGHTS
# ─────────────────────────────────────────────────────────

st.markdown(
    '<div class="section-label">AI Insights</div>',
    unsafe_allow_html=True
)

if avg_return >= 18:

    st.success(
        f"Excellent weighted portfolio return detected ({avg_return:.2f}%)"
    )

elif avg_return >= 12:

    st.info(
        f"Stable portfolio return detected ({avg_return:.2f}%)"
    )

else:

    st.warning(
        f"Portfolio return is below benchmark ({avg_return:.2f}%)"
    )

if volatility >= 18:

    st.error(
        f"High volatility detected ({volatility:.2f})"
    )

elif volatility >= 12:

    st.warning(
        f"Moderate volatility detected ({volatility:.2f})"
    )

else:

    st.success(
        f"Portfolio volatility is under control ({volatility:.2f})"
    )

if avg_expense >= 1.8:

    st.error(
        f"Expense ratio is too high ({avg_expense:.2f}%)"
    )

elif avg_expense >= 1.2:

    st.warning(
        f"Moderate expense ratio ({avg_expense:.2f}%)"
    )

else:

    st.success(
        f"Expense ratio is efficient ({avg_expense:.2f}%)"
    )

if sharpe_ratio >= 1:

    st.success(
        f"Excellent risk-adjusted return profile ({sharpe_ratio:.2f})"
    )

else:

    st.warning(
        f"Risk-adjusted performance can be improved ({sharpe_ratio:.2f})"
    )

st.markdown('<div class="hr-line"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# AI RECOMMENDATIONS
# ─────────────────────────────────────────────────────────

st.markdown(
    '<div class="section-label">AI Recommendations</div>',
    unsafe_allow_html=True
)

recommendations = []

if avg_return < 12:

    recommendations.append(
        "Consider replacing underperforming funds with better CAGR performers."
    )

if avg_expense > 1.5:

    recommendations.append(
        "Switch to lower expense ratio direct plans."
    )

if volatility > 15:

    recommendations.append(
        "Reduce equity concentration and add debt/hybrid exposure."
    )

if categories < 3:

    recommendations.append(
        "Increase category diversification."
    )

if largest_category_weight > 0.50:

    recommendations.append(
        "Portfolio is heavily concentrated in one category."
    )

if sharpe_ratio < 1:

    recommendations.append(
        "Improve risk-adjusted returns by balancing asset allocation."
    )

if not recommendations:

    recommendations.append(
        "Portfolio looks healthy. Continue long-term disciplined investing."
    )

for rec in recommendations:

    st.markdown(
        f"""
        <div class="rec-card">
        {rec}
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown('<div class="hr-line"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────

st.markdown("""
<div style="
    text-align:center;
    color:#5A6A80;
    font-size:0.7rem;
    padding:1rem;
">
    AI Mutual Fund Portfolio Analyzer • Real-Time Portfolio Intelligence Engine
</div>
""", unsafe_allow_html=True)
