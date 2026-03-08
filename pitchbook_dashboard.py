"""
pitchbook_dashboard.py
======================
Interactive Sell-Side PitchBook Dashboard for LSCG – TechVista Solutions Pvt. Ltd.

Run with:
    streamlit run pitchbook_dashboard.py

Sections (sidebar navigation):
  1. Executive Summary
  2. Market & Industry Analysis
  3. Company Overview (Target Analysis)
  4. Financial Statements
  5. Valuation Analysis
  6. Deal Structure & Recommendation
  7. Potential Buyers
  8. Risk Analysis & Mitigation
  9. Transaction Timeline
 10. Appendix / Data Download
"""

import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ────────────────────────────────────────────────────────────────
# Page configuration
# ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LSCG PitchBook – TechVista Solutions (Sell-Side)",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "LSCG_Financial_Data.xlsx")

# ────────────────────────────────────────────────────────────────
# Data loading (cached)
# ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(path: str) -> dict[str, pd.DataFrame]:
    """Load all sheets from the financial data workbook."""
    sheets = {}
    xls = pd.ExcelFile(path, engine="openpyxl")
    for name in xls.sheet_names:
        sheets[name] = pd.read_excel(xls, sheet_name=name)
    return sheets


def _try_load():
    if not os.path.exists(DATA_PATH):
        st.error(
            f"Data file not found at `{DATA_PATH}`. "
            "Run `python data/generate_financial_data.py` first."
        )
        st.stop()
    return load_data(DATA_PATH)


DATA = _try_load()
inc = DATA["Income Statement"]
bs = DATA["Balance Sheet"]
cf = DATA["Cash Flow"]
ratios = DATA["Key Ratios"]
comps = DATA["Public Comps"]
precedents = DATA["Precedent Transactions"]
dcf_assumptions = DATA["DCF Assumptions"]
seg_rev = DATA["Segment Revenue"]
geo_rev = DATA["Geographic Revenue"]

HIST_YEARS = [2021, 2022, 2023, 2024, 2025]
PROJ_YEARS = [2026, 2027, 2028]
ALL_YEARS = HIST_YEARS + PROJ_YEARS

# ────────────────────────────────────────────────────────────────
# Sidebar navigation
# ────────────────────────────────────────────────────────────────
st.sidebar.image(
    "https://img.icons8.com/fluency/96/combo-chart.png", width=64
)
st.sidebar.title("📘 LSCG PitchBook")
st.sidebar.markdown("**TechVista Solutions Pvt. Ltd.**")
st.sidebar.markdown("*Sell-Side Advisory*")
st.sidebar.markdown("---")

SECTIONS = [
    "1. Executive Summary",
    "2. Market & Industry Analysis",
    "3. Company Overview",
    "4. Financial Statements",
    "5. Valuation Analysis",
    "6. Deal Structure & Recommendation",
    "7. Potential Buyers",
    "8. Risk Analysis & Mitigation",
    "9. Transaction Timeline",
    "10. Appendix & Downloads",
]

section = st.sidebar.radio("Navigate to Section", SECTIONS)

st.sidebar.markdown("---")
st.sidebar.caption("Prepared by LSCG Advisory Group")
st.sidebar.caption("Confidential – For Discussion Purposes Only")


# ═══════════════════════════════════════════════════════════════════
# HELPER UTILITIES
# ═══════════════════════════════════════════════════════════════════
def _fmt_cr(val):
    """Format number as ₹ Cr string."""
    return f"₹ {val:,.0f} Cr"


def _kpi_row(metrics: list[tuple[str, str]]):
    """Render a row of KPI cards."""
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics):
        col.metric(label, value)


# ═══════════════════════════════════════════════════════════════════
# SECTION 1 – EXECUTIVE SUMMARY
# ═══════════════════════════════════════════════════════════════════
if section == SECTIONS[0]:
    st.title("📋 Executive Summary")
    st.markdown("### Situation Overview")
    st.markdown(
        """
        **TechVista Solutions Pvt. Ltd.** is a high-growth, mid-market IT services and
        digital-transformation company headquartered in Pune, India. The company has
        demonstrated consistent top-line growth (~20 % CAGR over FY21-FY25) with expanding
        EBITDA margins, a diversified client base across India, North America, and Europe,
        and a strong order book.

        **LSCG Advisory Group** has been engaged to evaluate and execute a **sell-side
        mandate** to maximise shareholder value through a structured transaction process.
        """
    )

    st.markdown("### Key Investment Highlights")
    highlights = [
        "🚀 **Revenue CAGR ~20 %** (FY21-FY25) with projected continuation through FY28",
        "📈 **EBITDA margin expansion** from 36 % (FY21) to 44 % (FY25), demonstrating operating leverage",
        "🌍 **Diversified geographic presence** – India (40 %), North America (30 %), Europe (20 %), RoW (10 %)",
        "💡 **Strong positioning** in Cloud/SaaS and Digital Transformation – fastest-growing segments",
        "🏦 **Low leverage** – Net Debt/EBITDA < 0.1x, providing flexibility for acquirers",
        "👥 **Experienced management team** with deep domain expertise",
    ]
    for h in highlights:
        st.markdown(f"- {h}")

    st.markdown("---")
    st.markdown("### Snapshot – FY25")

    latest_idx = inc[inc["Year"] == 2025].index[0]
    _kpi_row([
        ("Revenue", _fmt_cr(inc.loc[latest_idx, "Revenue"])),
        ("EBITDA", _fmt_cr(inc.loc[latest_idx, "EBITDA"])),
        ("EBITDA Margin", f"{ratios.loc[latest_idx, 'EBITDA Margin (%)']:.1f} %"),
        ("Net Income", _fmt_cr(inc.loc[latest_idx, "PAT (Net Income)"])),
        ("Total Assets", _fmt_cr(bs.loc[latest_idx, "Total Assets"])),
        ("Debt/Equity", f"{ratios.loc[latest_idx, 'Debt-to-Equity']:.2f}x"),
    ])

    # Revenue trend mini-chart
    fig = px.bar(
        inc, x="Year", y="Revenue",
        color=inc["Year"].apply(lambda y: "Projected" if y >= 2026 else "Historical"),
        color_discrete_map={"Historical": "#1f77b4", "Projected": "#aec7e8"},
        title="Revenue Trend (₹ Cr)",
        text="Revenue",
    )
    fig.update_layout(showlegend=True, legend_title_text="", xaxis_title="", yaxis_title="₹ Cr")
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "**Recommendation:** LSCG recommends a structured broad-auction sell-side process "
        "targeting strategic acquirers and select private-equity sponsors to maximise value."
    )


# ═══════════════════════════════════════════════════════════════════
# SECTION 2 – MARKET & INDUSTRY ANALYSIS
# ═══════════════════════════════════════════════════════════════════
elif section == SECTIONS[1]:
    st.title("🌐 Market & Industry Analysis")

    st.markdown("### Indian IT Services Industry Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            | Parameter | Value |
            |-----------|-------|
            | Market Size (FY25) | **~$254 Bn** |
            | Projected Size (FY28) | **~$350 Bn** |
            | CAGR (FY25-28) | **~11 %** |
            | Employment | **5.4 M+** |
            | IT Exports / GDP | **~7.5 %** |
            """
        )
    with col2:
        # Market size bar chart
        market_data = pd.DataFrame({
            "Year": ["FY23", "FY24", "FY25E", "FY26E", "FY27E", "FY28E"],
            "Market Size ($Bn)": [210, 230, 254, 283, 315, 350],
        })
        fig = px.bar(
            market_data, x="Year", y="Market Size ($Bn)",
            title="Indian IT Market Size",
            color_discrete_sequence=["#2ca02c"],
            text="Market Size ($Bn)",
        )
        fig.update_layout(yaxis_title="$ Billion")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Key Growth Drivers")
    drivers = pd.DataFrame({
        "Driver": [
            "Cloud Migration & SaaS Adoption",
            "Digital Transformation",
            "AI / ML Integration",
            "Cybersecurity Demand",
            "Global Outsourcing Trend",
        ],
        "Impact": ["High", "High", "Medium-High", "Medium", "Medium"],
        "Growth Rate": ["25-30 %", "20-25 %", "30-35 %", "18-22 %", "10-15 %"],
    })
    st.table(drivers)

    st.markdown("### Competitive Landscape")
    fig = px.scatter(
        comps, x="Revenue (₹ Cr)", y="EBITDA Margin (%)",
        size="EV/EBITDA (x)", color="Company",
        title="Peer Comparison – Revenue vs EBITDA Margin (bubble = EV/EBITDA)",
        hover_data=["P/E (x)", "Net Debt/EBITDA (x)"],
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Regulatory & Macro Environment")
    st.markdown(
        """
        - **Government Incentives:** PLI scheme for IT hardware; Software Technology Parks of India (STPI) benefits.
        - **SEZ Policy:** Continued tax benefits for exports from SEZs until FY25, gradual phase-out.
        - **Data Localisation:** Evolving regulatory framework around data privacy (Digital Personal Data Protection Act, 2023).
        - **Macro Tailwinds:** Stable INR, rising domestic digital spend, G20 presidency boosting India's tech profile.
        """
    )


# ═══════════════════════════════════════════════════════════════════
# SECTION 3 – COMPANY OVERVIEW (TARGET ANALYSIS)
# ═══════════════════════════════════════════════════════════════════
elif section == SECTIONS[2]:
    st.title("🏢 Company Overview – TechVista Solutions")

    st.markdown("### Business Description")
    st.markdown(
        """
        **TechVista Solutions Pvt. Ltd.** is a full-spectrum IT services company established
        in 2012, offering IT Consulting, Managed Services, Cloud & SaaS, and Digital
        Transformation solutions to enterprise clients across BFSI, Healthcare, Manufacturing,
        and Retail verticals.

        | Attribute | Details |
        |-----------|---------|
        | **Founded** | 2012 |
        | **Headquarters** | Pune, Maharashtra, India |
        | **Employees** | ~4,500 |
        | **Key Clients** | 120+ enterprise clients across 18 countries |
        | **Delivery Centres** | Pune, Hyderabad, Bengaluru, Dallas (US), London (UK) |
        | **Certifications** | ISO 27001, CMMI Level 5, SOC 2 Type II |
        """
    )

    st.markdown("### Revenue by Segment")
    year_filter = st.select_slider("Select Year", options=ALL_YEARS, value=2025)
    seg_yr = seg_rev[seg_rev["Year"] == year_filter]
    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(
            seg_yr, values="Revenue", names="Segment",
            title=f"Revenue by Segment – FY{year_filter}",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.pie(
            geo_rev[geo_rev["Year"] == year_filter],
            values="Revenue", names="Geography",
            title=f"Revenue by Geography – FY{year_filter}",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Segment Revenue Trend")
    fig = px.area(
        seg_rev, x="Year", y="Revenue", color="Segment",
        title="Segment Revenue Over Time (₹ Cr)",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(yaxis_title="₹ Cr")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### SWOT Analysis")
    sw_col1, sw_col2 = st.columns(2)
    with sw_col1:
        st.success("**Strengths**")
        st.markdown(
            """
            - Strong revenue growth (~20 % CAGR)
            - Diversified client base & geography
            - High EBITDA margins with expansion trend
            - CMMI Level 5 & ISO certified
            - Low attrition relative to peers
            """
        )
        st.error("**Weaknesses**")
        st.markdown(
            """
            - Relatively smaller scale vs. tier-1 IT players
            - Limited proprietary IP / products
            - Concentration in India for delivery
            - Brand recognition outside India still developing
            """
        )
    with sw_col2:
        st.info("**Opportunities**")
        st.markdown(
            """
            - Cloud & AI / ML services adoption accelerating
            - Cross-selling into existing client base
            - Expansion into APAC and Middle East
            - Strategic acquisitions in niche areas
            """
        )
        st.warning("**Threats**")
        st.markdown(
            """
            - Intense competition from tier-1 and global players
            - Currency fluctuation risk (INR/USD)
            - Regulatory changes in data privacy
            - Talent acquisition & wage inflation pressures
            """
        )


# ═══════════════════════════════════════════════════════════════════
# SECTION 4 – FINANCIAL STATEMENTS
# ═══════════════════════════════════════════════════════════════════
elif section == SECTIONS[3]:
    st.title("📊 Financial Statements – Interactive Dashboard")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Income Statement", "Balance Sheet", "Cash Flow", "Key Ratios",
    ])

    # ── Income Statement ────────────────────────────────────────
    with tab1:
        st.subheader("Income Statement (₹ Cr)")
        st.dataframe(inc.style.format({
            c: "{:,.1f}" for c in inc.columns if c != "Year"
        }), use_container_width=True)

        metric_options = [c for c in inc.columns if c != "Year"]
        selected_metrics = st.multiselect(
            "Select metrics to chart",
            metric_options,
            default=["Revenue", "EBITDA", "PAT (Net Income)"],
        )
        if selected_metrics:
            fig = go.Figure()
            for m in selected_metrics:
                fig.add_trace(go.Bar(x=inc["Year"].astype(str), y=inc[m], name=m))
            fig.update_layout(
                title="Income Statement Metrics", barmode="group",
                xaxis_title="Year", yaxis_title="₹ Cr",
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Balance Sheet ────────────────────────────────────────────
    with tab2:
        st.subheader("Balance Sheet (₹ Cr)")
        st.dataframe(bs.style.format({
            c: "{:,.1f}" for c in bs.columns if c != "Year"
        }), use_container_width=True)

        # Stacked bar – Assets composition
        asset_cols = ["Cash & Equivalents", "Accounts Receivable", "Inventory",
                      "PP&E (Net)", "Intangible Assets", "Other Assets"]
        fig = go.Figure()
        for ac in asset_cols:
            fig.add_trace(go.Bar(x=bs["Year"].astype(str), y=bs[ac], name=ac))
        fig.update_layout(
            title="Asset Composition", barmode="stack",
            xaxis_title="Year", yaxis_title="₹ Cr",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Liabilities vs Equity
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=bs["Year"].astype(str), y=bs["Total Liabilities"], name="Total Liabilities"))
        fig2.add_trace(go.Bar(x=bs["Year"].astype(str), y=bs["Shareholders' Equity"], name="Equity"))
        fig2.update_layout(
            title="Liabilities vs Equity", barmode="group",
            xaxis_title="Year", yaxis_title="₹ Cr",
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Cash Flow ────────────────────────────────────────────────
    with tab3:
        st.subheader("Cash Flow Statement (₹ Cr)")
        st.dataframe(cf.style.format({
            c: "{:,.1f}" for c in cf.columns if c != "Year"
        }), use_container_width=True)

        fig = go.Figure()
        fig.add_trace(go.Bar(x=cf["Year"].astype(str), y=cf["Cash from Operations (CFO)"], name="CFO", marker_color="#2ca02c"))
        fig.add_trace(go.Bar(x=cf["Year"].astype(str), y=cf["Cash from Investing (CFI)"], name="CFI", marker_color="#d62728"))
        fig.add_trace(go.Bar(x=cf["Year"].astype(str), y=cf["Cash from Financing (CFF)"], name="CFF", marker_color="#ff7f0e"))
        fig.add_trace(go.Scatter(x=cf["Year"].astype(str), y=cf["Net Change in Cash"], name="Net Change", mode="lines+markers"))
        fig.update_layout(
            title="Cash Flow Breakdown", barmode="group",
            xaxis_title="Year", yaxis_title="₹ Cr",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Key Ratios ───────────────────────────────────────────────
    with tab4:
        st.subheader("Key Financial Ratios")
        st.dataframe(ratios.style.format({
            c: "{:.1f}" for c in ratios.columns if c not in ("Year", "Revenue Growth (%)")
        }), use_container_width=True)

        ratio_choice = st.selectbox(
            "Select ratio to visualise",
            [c for c in ratios.columns if c != "Year"],
        )
        fig = px.line(
            ratios, x="Year", y=ratio_choice,
            markers=True, title=ratio_choice,
        )
        fig.update_layout(yaxis_title=ratio_choice)
        st.plotly_chart(fig, use_container_width=True)

        # Margin comparison
        fig2 = go.Figure()
        for m in ["Gross Margin (%)", "EBITDA Margin (%)", "Net Margin (%)"]:
            fig2.add_trace(go.Scatter(
                x=ratios["Year"].astype(str), y=ratios[m],
                mode="lines+markers", name=m,
            ))
        fig2.update_layout(title="Margin Progression", yaxis_title="%", xaxis_title="Year")
        st.plotly_chart(fig2, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# SECTION 5 – VALUATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════
elif section == SECTIONS[4]:
    st.title("💰 Valuation Analysis")

    val_tab1, val_tab2, val_tab3, val_tab4 = st.tabs([
        "Public Comps", "Precedent Transactions", "DCF Analysis", "Valuation Summary",
    ])

    # ── Public Comps ─────────────────────────────────────────────
    with val_tab1:
        st.subheader("Public Company Comparables")
        st.dataframe(comps.style.format({
            "Revenue (₹ Cr)": "{:,.0f}",
            "EBITDA Margin (%)": "{:.1f}",
            "Gross Margin (%)": "{:.1f}",
            "Net Margin (%)": "{:.1f}",
            "EV/EBITDA (x)": "{:.1f}",
            "EV/Revenue (x)": "{:.1f}",
            "Net Debt/EBITDA (x)": "{:.2f}",
            "P/E (x)": "{:.1f}",
        }), use_container_width=True)

        st.markdown("#### Comparable Multiples Summary")
        numeric_cols = ["EV/EBITDA (x)", "EV/Revenue (x)", "P/E (x)"]
        summary_data = []
        for c in numeric_cols:
            summary_data.append({
                "Multiple": c,
                "Mean": f"{comps[c].mean():.1f}x",
                "Median": f"{comps[c].median():.1f}x",
                "Min": f"{comps[c].min():.1f}x",
                "Max": f"{comps[c].max():.1f}x",
            })
        st.table(pd.DataFrame(summary_data))

        fig = px.bar(
            comps, x="Company", y="EV/EBITDA (x)",
            color="Company", title="EV/EBITDA by Peer Company",
        )
        fig.add_hline(y=comps["EV/EBITDA (x)"].median(), line_dash="dash",
                       annotation_text=f"Median: {comps['EV/EBITDA (x)'].median():.1f}x")
        st.plotly_chart(fig, use_container_width=True)

    # ── Precedent Transactions ───────────────────────────────────
    with val_tab2:
        st.subheader("Precedent M&A Transactions")
        st.dataframe(precedents.style.format({
            "Enterprise Value (₹ Cr)": "{:,.0f}",
            "EV/EBITDA (x)": "{:.1f}",
            "EV/Revenue (x)": "{:.1f}",
            "EV/EBIT (x)": "{:.1f}",
        }), use_container_width=True)

        fig = px.bar(
            precedents, x="Target / Acquirer", y="EV/EBITDA (x)",
            color="Target / Acquirer",
            title="Precedent Transaction EV/EBITDA Multiples",
        )
        fig.add_hline(y=precedents["EV/EBITDA (x)"].median(), line_dash="dash",
                       annotation_text=f"Median: {precedents['EV/EBITDA (x)'].median():.1f}x")
        st.plotly_chart(fig, use_container_width=True)

    # ── DCF Analysis ─────────────────────────────────────────────
    with val_tab3:
        st.subheader("DCF Analysis – Key Assumptions")
        st.table(dcf_assumptions)

        st.markdown("#### Adjust Key Parameters")
        col1, col2, col3 = st.columns(3)
        with col1:
            wacc_input = st.slider("WACC (%)", 8.0, 18.0, 12.36, 0.25)
        with col2:
            tgr_input = st.slider("Terminal Growth Rate (%)", 1.0, 6.0, 4.0, 0.25)
        with col3:
            base_ebitda = inc[inc["Year"] == 2025]["EBITDA"].values[0]
            ebitda_input = st.number_input("Base EBITDA (₹ Cr)", value=float(base_ebitda), step=10.0)

        # Simplified DCF valuation
        proj_ebitda = []
        g_rates = [0.18, 0.16, 0.14, 0.12, 0.10]  # declining growth
        e = ebitda_input
        for g in g_rates:
            e = e * (1 + g)
            proj_ebitda.append(e)

        pv_sum = 0
        for i, pe in enumerate(proj_ebitda):
            pv_sum += pe / ((1 + wacc_input / 100) ** (i + 1))

        terminal_value = proj_ebitda[-1] * (1 + tgr_input / 100) / (wacc_input / 100 - tgr_input / 100)
        pv_terminal = terminal_value / ((1 + wacc_input / 100) ** len(proj_ebitda))

        enterprise_value = pv_sum + pv_terminal
        net_debt_val = bs[bs["Year"] == 2025]["Total Debt"].values[0] - bs[bs["Year"] == 2025]["Cash & Equivalents"].values[0]
        equity_value = enterprise_value - net_debt_val

        st.markdown("#### DCF Output")
        _kpi_row([
            ("PV of Projected EBITDA", _fmt_cr(round(pv_sum))),
            ("Terminal Value", _fmt_cr(round(terminal_value))),
            ("PV of Terminal Value", _fmt_cr(round(pv_terminal))),
            ("Enterprise Value", _fmt_cr(round(enterprise_value))),
            ("Net Debt", _fmt_cr(round(net_debt_val))),
            ("Equity Value", _fmt_cr(round(equity_value))),
        ])

        # Sensitivity table
        st.markdown("#### Sensitivity Analysis – Equity Value (₹ Cr)")
        wacc_range = [wacc_input - 1.5, wacc_input - 0.75, wacc_input, wacc_input + 0.75, wacc_input + 1.5]
        tgr_range = [tgr_input - 1.0, tgr_input - 0.5, tgr_input, tgr_input + 0.5, tgr_input + 1.0]

        sens = {}
        for tg in tgr_range:
            row_vals = []
            for w in wacc_range:
                if w / 100 <= tg / 100:
                    row_vals.append("N/A")
                    continue
                pv_s = sum(pe / ((1 + w / 100) ** (j + 1)) for j, pe in enumerate(proj_ebitda))
                tv = proj_ebitda[-1] * (1 + tg / 100) / (w / 100 - tg / 100)
                pv_t = tv / ((1 + w / 100) ** len(proj_ebitda))
                ev = pv_s + pv_t - net_debt_val
                row_vals.append(f"{ev:,.0f}")
            sens[f"TGR {tg:.1f}%"] = row_vals

        sens_df = pd.DataFrame(sens, index=[f"WACC {w:.2f}%" for w in wacc_range]).T
        st.table(sens_df)

    # ── Valuation Summary ────────────────────────────────────────
    with val_tab4:
        st.subheader("Valuation Summary – Football Field")

        median_ev_ebitda = comps["EV/EBITDA (x)"].median()
        mean_ev_ebitda = comps["EV/EBITDA (x)"].mean()
        prec_median = precedents["EV/EBITDA (x)"].median()

        fy25_ebitda = inc[inc["Year"] == 2025]["EBITDA"].values[0]

        methods = ["Public Comps\n(Median)", "Public Comps\n(Mean)", "Precedent Txns\n(Median)", "DCF"]
        low = [
            fy25_ebitda * (median_ev_ebitda - 2),
            fy25_ebitda * (mean_ev_ebitda - 2),
            fy25_ebitda * (prec_median - 2),
            equity_value * 0.85,
        ]
        high = [
            fy25_ebitda * (median_ev_ebitda + 2),
            fy25_ebitda * (mean_ev_ebitda + 2),
            fy25_ebitda * (prec_median + 2),
            equity_value * 1.15,
        ]
        mid = [(l + h) / 2 for l, h in zip(low, high)]

        fig = go.Figure()
        for i, method in enumerate(methods):
            fig.add_trace(go.Bar(
                x=[high[i] - low[i]], y=[method], base=[low[i]],
                orientation="h", name=method,
                text=f"₹{mid[i]:,.0f} Cr", textposition="inside",
                marker_color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"][i],
            ))
        fig.update_layout(
            title="Valuation Range – Football Field Chart (₹ Cr)",
            xaxis_title="Enterprise / Equity Value (₹ Cr)",
            showlegend=False, height=400,
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            f"""
            **Indicative Valuation Range:** ₹ {min(low):,.0f} Cr – ₹ {max(high):,.0f} Cr

            Based on the convergence of all three methodologies, LSCG recommends an
            **indicative enterprise-value range of ₹ {fy25_ebitda * (median_ev_ebitda - 1):,.0f} Cr
            to ₹ {fy25_ebitda * (median_ev_ebitda + 1):,.0f} Cr** (implied EV/EBITDA of
            {median_ev_ebitda - 1:.1f}x – {median_ev_ebitda + 1:.1f}x on FY25 EBITDA).
            """
        )


# ═══════════════════════════════════════════════════════════════════
# SECTION 6 – DEAL STRUCTURE & RECOMMENDATION
# ═══════════════════════════════════════════════════════════════════
elif section == SECTIONS[5]:
    st.title("🤝 Deal Structure & Recommendation")

    st.markdown("### Recommended Transaction Structure")
    st.success(
        "**Primary Recommendation:** Broad-auction **sell-side** process targeting "
        "strategic acquirers in the IT-services sector, supplemented by outreach to "
        "select PE sponsors with sector focus."
    )

    st.markdown(
        """
        | Component | Recommendation |
        |-----------|---------------|
        | **Transaction Type** | 100 % equity sale (outright acquisition) |
        | **Process** | Two-stage auction: indicative bids → final binding bids |
        | **Target Buyer Profile** | Tier-1/2 IT companies (strategic) + PE sponsors |
        | **Valuation Basis** | EV/EBITDA-based, cross-referenced with DCF |
        | **Consideration** | Primarily cash; partial stock considered for strategic buyers |
        | **Earn-Out** | Optional 12-month earn-out tied to revenue milestones |
        | **Management Retention** | Lock-in for key management (24 months post-close) |
        | **Exclusivity** | Post final-bid, 45-day exclusivity for due diligence & documentation |
        """
    )

    st.markdown("### Mode of Financing (Buyer Perspective)")
    financing = pd.DataFrame({
        "Source": ["Cash on Balance Sheet", "Senior Debt", "Mezzanine / Sub-Debt", "Equity (Sponsor)"],
        "% of EV": [20, 40, 15, 25],
        "Notes": [
            "Acquirer's available cash reserves",
            "Term loan / bond issuance; 4-5x Debt/EBITDA capacity",
            "Higher-yield subordinated debt",
            "PE equity contribution or strategic buyer stock",
        ],
    })
    st.table(financing)

    fig = px.pie(
        financing, values="% of EV", names="Source",
        title="Illustrative Financing Mix",
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# SECTION 7 – POTENTIAL BUYERS
# ═══════════════════════════════════════════════════════════════════
elif section == SECTIONS[6]:
    st.title("🎯 Potential Buyers / Investors")

    st.markdown("### Strategic Buyers")
    strategic = pd.DataFrame({
        "Buyer": ["Infosys Ltd.", "Wipro Ltd.", "HCL Technologies", "Tech Mahindra",
                   "L&T Technology Services", "Mphasis Ltd."],
        "Rationale": [
            "Expand mid-market client base; fill cloud-consulting gap",
            "Tuck-in acquisition for digital services; geographic synergies in Europe",
            "Strengthen SaaS capabilities and delivery presence in Pune",
            "Complementary capabilities in manufacturing vertical",
            "Expand into BFSI and healthcare verticals",
            "Cross-sell managed-services offerings to TechVista's client base",
        ],
        "Strategic Fit": ["★★★★★", "★★★★☆", "★★★★☆", "★★★☆☆", "★★★★☆", "★★★☆☆"],
        "Estimated Capacity": ["High", "High", "High", "Medium", "Medium", "Medium"],
    })
    st.table(strategic)

    st.markdown("### Financial Sponsors (PE)")
    financial = pd.DataFrame({
        "Sponsor": ["Blackstone", "Carlyle Group", "Baring Private Equity", "Advent International",
                     "Warburg Pincus", "KKR"],
        "Sector Focus": [
            "Technology & IT Services", "Technology & Business Services",
            "Technology", "Technology & Healthcare",
            "Technology & Financial Services", "Technology",
        ],
        "Recent Deals in Sector": [
            "Mphasis acquisition", "Hexaware acquisition",
            "NIIT Tech acquisition", "Cyient DLM",
            "Lenskart, Tradesmart", "Vini Cosmetics, Max Healthcare",
        ],
        "Fit": ["★★★★★", "★★★★★", "★★★★☆", "★★★★☆", "★★★☆☆", "★★★☆☆"],
    })
    st.table(financial)


# ═══════════════════════════════════════════════════════════════════
# SECTION 8 – RISK ANALYSIS & MITIGATION
# ═══════════════════════════════════════════════════════════════════
elif section == SECTIONS[7]:
    st.title("⚠️ Risk Analysis & Mitigation")

    risks = pd.DataFrame({
        "Risk Category": [
            "Execution Risk",
            "Regulatory Risk",
            "Integration Risk",
            "Financing Risk",
            "Market Timing",
            "Key-Person Risk",
            "Client Concentration",
            "Litigation / IP Risk",
        ],
        "Severity": [
            "Medium", "Medium", "High", "Low", "Medium",
            "Medium-High", "Low", "Low",
        ],
        "Mitigation Strategy": [
            "Structured two-stage auction; experienced LSCG deal team; tight timeline discipline",
            "Early engagement with legal counsel; proactive regulatory filings",
            "Detailed integration playbook prepared pre-close; management retention agreements",
            "Pre-qualify buyer financing capacity; staple financing package from LSCG",
            "Flexible timing; ability to pause/accelerate based on market conditions",
            "Management lock-in clauses; incentive alignment via earn-out",
            "Demonstrate diversified client base (top-10 < 35 % of revenue)",
            "Comprehensive IP audit in data room; reps & warranties insurance",
        ],
    })

    severity_colors = {
        "Low": "🟢", "Medium": "🟡", "Medium-High": "🟠", "High": "🔴",
    }
    risks["Indicator"] = risks["Severity"].map(severity_colors)

    st.dataframe(
        risks[["Indicator", "Risk Category", "Severity", "Mitigation Strategy"]],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Risk Heat Map")
    risk_heat = pd.DataFrame({
        "Risk": risks["Risk Category"],
        "Likelihood": [3, 2, 3, 1, 3, 3, 1, 1],
        "Impact": [3, 3, 4, 2, 3, 4, 2, 2],
    })
    fig = px.scatter(
        risk_heat, x="Likelihood", y="Impact", text="Risk",
        size=[40] * len(risk_heat),
        color="Impact",
        color_continuous_scale="RdYlGn_r",
        title="Risk Heat Map (Likelihood vs Impact)",
    )
    fig.update_traces(textposition="top center")
    fig.update_layout(
        xaxis=dict(range=[0, 5], title="Likelihood (1=Low, 5=High)"),
        yaxis=dict(range=[0, 5], title="Impact (1=Low, 5=High)"),
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# SECTION 9 – TRANSACTION TIMELINE
# ═══════════════════════════════════════════════════════════════════
elif section == SECTIONS[8]:
    st.title("📅 Transaction Timeline")

    timeline = pd.DataFrame({
        "Phase": [
            "Phase 1: Preparation",
            "Phase 1: Preparation",
            "Phase 1: Preparation",
            "Phase 2: Marketing",
            "Phase 2: Marketing",
            "Phase 2: Marketing",
            "Phase 3: Due Diligence",
            "Phase 3: Due Diligence",
            "Phase 4: Negotiation & Close",
            "Phase 4: Negotiation & Close",
            "Phase 4: Negotiation & Close",
        ],
        "Activity": [
            "Engagement & data collection",
            "Prepare Confidential Information Memorandum (CIM)",
            "Set up Virtual Data Room (VDR)",
            "Distribute teaser to potential buyers",
            "Management presentations",
            "Receive indicative (non-binding) bids",
            "Grant data-room access to shortlisted buyers",
            "Receive final binding bids",
            "Select preferred bidder; exclusivity period",
            "Negotiate SPA & ancillary documents",
            "Signing & Closing",
        ],
        "Start (Week)": [1, 2, 3, 5, 7, 9, 10, 14, 15, 16, 19],
        "End (Week)": [2, 4, 5, 7, 9, 10, 14, 15, 16, 19, 20],
        "Duration (Weeks)": [1, 2, 2, 2, 2, 1, 4, 1, 1, 3, 1],
    })

    st.dataframe(timeline, use_container_width=True, hide_index=True)

    # Gantt-style chart
    fig = px.timeline(
        timeline.assign(
            Start=timeline["Start (Week)"].apply(lambda w: pd.Timestamp("2026-01-06") + pd.Timedelta(weeks=w - 1)),
            Finish=timeline["End (Week)"].apply(lambda w: pd.Timestamp("2026-01-06") + pd.Timedelta(weeks=w)),
        ),
        x_start="Start", x_end="Finish", y="Activity",
        color="Phase",
        title="Transaction Process – Gantt Chart (~20 Weeks)",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=550)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Key Milestones & Decision Points")
    milestones = [
        ("Week 4", "CIM & Data Room ready"),
        ("Week 7", "Teaser distributed; initial interest gauged"),
        ("Week 10", "Indicative bids received – shortlist 3-5 parties"),
        ("Week 15", "Final binding bids received"),
        ("Week 16", "Preferred bidder selected – begin exclusivity"),
        ("Week 20", "Target signing & closing"),
    ]
    for wk, desc in milestones:
        st.markdown(f"- **{wk}:** {desc}")


# ═══════════════════════════════════════════════════════════════════
# SECTION 10 – APPENDIX & DOWNLOADS
# ═══════════════════════════════════════════════════════════════════
elif section == SECTIONS[9]:
    st.title("📎 Appendix & Data Downloads")

    st.markdown("### Download Financial Data")
    with open(DATA_PATH, "rb") as f:
        st.download_button(
            "📥 Download LSCG Financial Data (Excel)",
            data=f.read(),
            file_name="LSCG_Financial_Data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml",
        )

    st.markdown("### All Data Sheets")
    sheet_choice = st.selectbox("Select sheet to view", list(DATA.keys()))
    st.dataframe(DATA[sheet_choice], use_container_width=True)

    st.markdown("### Bank Credentials – LSCG Advisory Group")
    st.markdown(
        """
        | Credential | Details |
        |-----------|---------|
        | **Founded** | 2008 |
        | **Offices** | Mumbai, Delhi, Singapore, London |
        | **Sector Focus** | Technology, Healthcare, Consumer, Industrials |
        | **Deals Closed (Last 3 Years)** | 45+ |
        | **Total Deal Value** | $12 Bn+ |
        | **Key Team** | 8 MDs, 25 VPs, 60+ analysts |
        | **League Table Rank** | Top-5 in mid-market Indian M&A (FY23-FY25) |
        """
    )

    st.markdown("### Recent Relevant Transactions by LSCG")
    recent_deals = pd.DataFrame({
        "Deal": [
            "CloudServe India – Sell-side to PE sponsor",
            "DataBridge Analytics – Buy-side for strategic acquirer",
            "NetPulse Technologies – IPO advisory",
            "DigiCore Services – Sell-side (cross-border)",
            "SmartOps Pvt. Ltd. – Restructuring advisory",
        ],
        "Year": [2024, 2024, 2023, 2023, 2022],
        "Deal Size (₹ Cr)": [3500, 2200, 5000, 1800, 900],
        "Sector": ["IT Services", "Data Analytics", "IT Services", "IT Services", "IT Services"],
    })
    st.table(recent_deals)

    st.markdown("---")
    st.caption("**Disclaimer:** This presentation is confidential and intended solely for "
               "discussion purposes. It does not constitute an offer or solicitation.")


# ────────────────────────────────────────────────────────────────
# Footer
# ────────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("📘 *LSCG PitchBook v1.0*")
st.sidebar.markdown("🗓 Prepared: March 2026")
