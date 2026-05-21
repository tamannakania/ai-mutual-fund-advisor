import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="AI Mutual Fund Intelligence Platform",
    page_icon="📈",
    layout="wide"
)

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

import os

# Resolve path relative to this file, so Streamlit works regardless of CWD
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_PATH = os.path.normpath(os.path.join(_BASE_DIR, "../../datasets/cleaned_mutual_funds.csv"))

df = pd.read_csv(_DATA_PATH)

df.columns = df.columns.str.lower()

# ---------------------------------------------------
# CALCULATIONS
# ---------------------------------------------------

avg_return = df["returns_5yr"].mean()
avg_expense = df["expense_ratio"].mean()
best_return = df["returns_5yr"].max()
categories = df["category"].nunique()

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------

st.markdown("""
<style>

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

.stApp {
    background: linear-gradient(to right, #020617, #0f172a);
    color: white;
}

/* TITLE */

.main-title {
    font-size: 52px;
    font-weight: 800;
    color: white;
    margin-bottom: 0;
}

.sub-title {
    color: #94A3B8;
    font-size: 20px;
    margin-top: 0;
}

/* KPI CARDS */

.kpi-card {
    background: linear-gradient(
        135deg,
        rgba(59,130,246,0.25),
        rgba(147,51,234,0.25)
    );

    padding: 25px;
    border-radius: 25px;

    border: 1px solid rgba(255,255,255,0.08);

    box-shadow: 0 8px 32px rgba(0,0,0,0.35);

    transition: 0.3s;
}

.kpi-card:hover {
    transform: translateY(-6px);
}

.kpi-title {
    color: #CBD5E1;
    font-size: 16px;
    margin-bottom: 10px;
}

.kpi-value {
    font-size: 38px;
    font-weight: bold;
    color: white;
}

/* SECTION TITLE */

.section-title {
    font-size: 30px;
    font-weight: 700;
    color: white;
    margin-bottom: 20px;
}

/* GLASS CARD */

.glass {
    background: rgba(255,255,255,0.04);

    border-radius: 24px;

    padding: 25px;

    border: 1px solid rgba(255,255,255,0.05);
}

/* DATAFRAME */

[data-testid="stDataFrame"] {
    border-radius: 18px;
    overflow: hidden;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.title("📊 AI MF Platform")

st.sidebar.markdown("---")

st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Portfolio Analytics",
        "Risk Intelligence",
        "Return Prediction",
        "AI Recommendations",
        "Reports"
    ]
)

st.sidebar.markdown("---")

st.sidebar.success("AI Engine Running")

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------

st.markdown("""
<p class='main-title'>
AI Mutual Fund Intelligence Platform
</p>

<p class='sub-title'>
Advanced Financial Analytics • AI Insights • Smart Portfolio Intelligence
</p>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------
# HERO SECTION
# ---------------------------------------------------

hero1, hero2 = st.columns([2,1])

with hero1:

    st.markdown("""
    <div class='glass'>

    <h2 style='color:white;'>
    Smart Wealth Intelligence
    </h2>

    <p style='color:#CBD5E1; font-size:18px;'>

    Analyze mutual fund performance, identify investment risks,
    optimize diversification, and unlock AI-driven financial insights.

    </p>

    </div>
    """, unsafe_allow_html=True)

with hero2:

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=avg_return,

        title={'text': "Portfolio Health"},

        gauge={
            'axis': {'range': [0, 25]},
            'bar': {'color': "#3B82F6"},

            'steps': [
                {'range': [0, 8], 'color': "#1E293B"},
                {'range': [8, 15], 'color': "#334155"},
                {'range': [15, 25], 'color': "#475569"}
            ]
        }
    ))

    gauge.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        height=250
    )

    st.plotly_chart(
        gauge,
        use_container_width=True
    )

# ---------------------------------------------------
# KPI SECTION
# ---------------------------------------------------

st.markdown("<br>", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-title'>Average Return</div>
        <div class='kpi-value'>{avg_return:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-title'>Expense Ratio</div>
        <div class='kpi-value'>{avg_expense:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-title'>Best Fund Return</div>
        <div class='kpi-value'>{best_return:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-title'>Fund Categories</div>
        <div class='kpi-value'>{categories}</div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------
# CHARTS
# ---------------------------------------------------

st.markdown("<br><br>", unsafe_allow_html=True)

left, right = st.columns(2)

# ---------------------------------------------------
# CATEGORY CHART
# ---------------------------------------------------

with left:

    st.markdown(
        "<p class='section-title'>Fund Categories</p>",
        unsafe_allow_html=True
    )

    category_data = (
        df["category"]
        .value_counts()
        .reset_index()
    )

    category_data.columns = [
        "Category",
        "Count"
    ]

    bar_chart = px.bar(
        category_data,
        x="Category",
        y="Count",
        color="Count",
        text_auto=True
    )

    bar_chart.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        height=450
    )

    st.plotly_chart(
        bar_chart,
        use_container_width=True
    )

# ---------------------------------------------------
# RISK PIE CHART
# ---------------------------------------------------

with right:

    st.markdown(
        "<p class='section-title'>Risk Distribution</p>",
        unsafe_allow_html=True
    )

    pie_chart = px.pie(
        df,
        names="risk_level",
        hole=0.7
    )

    pie_chart.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        height=450
    )

    st.plotly_chart(
        pie_chart,
        use_container_width=True
    )

# ---------------------------------------------------
# ADVANCED RETURNS ANALYSIS
# ---------------------------------------------------

st.markdown("<br>", unsafe_allow_html=True)

st.markdown(
    "<p class='section-title'>Advanced Returns Analytics</p>",
    unsafe_allow_html=True
)

analysis_df = df.sort_values(
    by="returns_5yr"
).reset_index(drop=True)

fig = go.Figure()

# RETURNS LINE

fig.add_trace(
    go.Scatter(
        x=analysis_df.index,
        y=analysis_df["returns_5yr"],
        mode='lines',
        fill='tozeroy',
        name='Returns',

        line=dict(
            color='#3B82F6',
            width=4
        )
    )
)

# AVERAGE RETURN LINE

fig.add_hline(
    y=avg_return,

    line_dash="dash",

    line_color="#F59E0B",

    annotation_text=f"Average Return ({avg_return:.2f}%)",

    annotation_position="top right"
)

# LAYOUT

fig.update_layout(

    paper_bgcolor='rgba(0,0,0,0)',

    plot_bgcolor='rgba(0,0,0,0)',

    font_color='white',

    xaxis_title="Mutual Funds",

    yaxis_title="5-Year Returns (%)",

    height=550,

    hovermode="x unified",

    margin=dict(
        l=20,
        r=20,
        t=40,
        b=20
    )
)

# GRID STYLING

fig.update_xaxes(
    showgrid=False
)

fig.update_yaxes(
    gridcolor='rgba(255,255,255,0.08)'
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ---------------------------------------------------
# ANALYTICS INSIGHTS
# ---------------------------------------------------

best_fund = df.loc[
    df["returns_5yr"].idxmax()
]

worst_fund = df.loc[
    df["returns_5yr"].idxmin()
]

col1, col2 = st.columns(2)

with col1:

    st.success(
        f"""
        Best Performing Fund:
        {best_fund['scheme_name']}
        
        Return: {best_fund['returns_5yr']:.2f}%
        """
    )

with col2:

    st.warning(
        f"""
        Lowest Performing Fund:
        {worst_fund['scheme_name']}
        
        Return: {worst_fund['returns_5yr']:.2f}%
        """
    )
# ---------------------------------------------------
# TOP FUNDS TABLE
# ---------------------------------------------------

st.markdown("<br>", unsafe_allow_html=True)

st.markdown(
    "<p class='section-title'>Top Performing Funds</p>",
    unsafe_allow_html=True
)

top_funds = df.sort_values(
    by="returns_5yr",
    ascending=False
).head(10)

columns_to_show = [
    col for col in [
        "scheme_name",
        "category",
        "returns_5yr",
        "expense_ratio",
        "risk_level"
    ]
    if col in top_funds.columns
]

st.dataframe(
    top_funds[columns_to_show],
    use_container_width=True,
    height=380
)

# ---------------------------------------------------
# AI INSIGHTS
# ---------------------------------------------------

st.markdown("<br>", unsafe_allow_html=True)

st.markdown(
    "<p class='section-title'>AI Investment Insights</p>",
    unsafe_allow_html=True
)

best_category = (
    df.groupby("category")["returns_5yr"]
    .mean()
    .sort_values(ascending=False)
    .index[0]
)

st.success(
    f"AI detected that '{best_category}' funds are generating the strongest long-term performance."
)

if avg_expense > 1.5:

    st.warning(
        "High expense ratios detected. Lower-cost funds may improve long-term profitability."
    )

else:

    st.info(
        "Expense ratios appear optimized for sustainable wealth generation."
    )
