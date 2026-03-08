"""
generate_financial_data.py
--------------------------
Generates a sample Excel workbook with multiple sheets of financial data
for the LSCG PitchBook sell-side analysis of "TechVista Solutions Pvt. Ltd."
Run once to produce data/LSCG_Financial_Data.xlsx
"""

import pandas as pd
import numpy as np
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__))
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "LSCG_Financial_Data.xlsx")

# ── Historical years + projections ──────────────────────────────────────────
HIST_YEARS = [2021, 2022, 2023, 2024, 2025]
PROJ_YEARS = [2026, 2027, 2028]
ALL_YEARS = HIST_YEARS + PROJ_YEARS


def _income_statement() -> pd.DataFrame:
    """Revenue, COGS, EBITDA, EBIT, PBT, PAT."""
    revenue = [1200, 1450, 1780, 2100, 2520, 3024, 3568, 4140]
    cogs_pct = [0.42, 0.41, 0.40, 0.39, 0.38, 0.37, 0.36, 0.35]
    sga_pct = [0.22, 0.21, 0.20, 0.19, 0.18, 0.17, 0.17, 0.16]
    da_pct = [0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04]
    interest = [45, 42, 38, 35, 30, 28, 25, 22]
    tax_rate = 0.25

    rows = []
    for i, yr in enumerate(ALL_YEARS):
        rev = revenue[i]
        cogs = round(rev * cogs_pct[i], 1)
        gross_profit = round(rev - cogs, 1)
        sga = round(rev * sga_pct[i], 1)
        ebitda = round(gross_profit - sga, 1)
        da = round(rev * da_pct[i], 1)
        ebit = round(ebitda - da, 1)
        pbt = round(ebit - interest[i], 1)
        tax = round(pbt * tax_rate, 1)
        pat = round(pbt - tax, 1)
        rows.append({
            "Year": yr,
            "Revenue": rev,
            "COGS": cogs,
            "Gross Profit": gross_profit,
            "SG&A Expenses": sga,
            "EBITDA": ebitda,
            "Depreciation & Amortisation": da,
            "EBIT": ebit,
            "Interest Expense": interest[i],
            "PBT": pbt,
            "Tax": tax,
            "PAT (Net Income)": pat,
        })
    return pd.DataFrame(rows)


def _balance_sheet() -> pd.DataFrame:
    """Simplified balance sheet."""
    total_assets = [3200, 3600, 4100, 4700, 5400, 6200, 7100, 8100]
    cash = [320, 400, 500, 620, 780, 950, 1150, 1400]
    receivables = [200, 240, 290, 340, 400, 470, 550, 640]
    inventory = [150, 170, 200, 230, 260, 300, 340, 380]
    ppe = [1800, 2000, 2200, 2500, 2800, 3100, 3500, 3900]
    intangibles = [400, 430, 470, 510, 560, 620, 680, 750]
    other_assets_calc = lambda i: total_assets[i] - cash[i] - receivables[i] - inventory[i] - ppe[i] - intangibles[i]

    total_liab = [1600, 1700, 1850, 2000, 2150, 2350, 2550, 2750]
    debt = [800, 780, 750, 700, 650, 600, 540, 480]
    payables = [180, 210, 250, 290, 340, 400, 460, 530]
    other_liab_calc = lambda i: total_liab[i] - debt[i] - payables[i]
    equity_calc = lambda i: total_assets[i] - total_liab[i]

    rows = []
    for i, yr in enumerate(ALL_YEARS):
        rows.append({
            "Year": yr,
            "Cash & Equivalents": cash[i],
            "Accounts Receivable": receivables[i],
            "Inventory": inventory[i],
            "PP&E (Net)": ppe[i],
            "Intangible Assets": intangibles[i],
            "Other Assets": other_assets_calc(i),
            "Total Assets": total_assets[i],
            "Accounts Payable": payables[i],
            "Total Debt": debt[i],
            "Other Liabilities": other_liab_calc(i),
            "Total Liabilities": total_liab[i],
            "Shareholders' Equity": equity_calc(i),
            "Total Liabilities & Equity": total_assets[i],
        })
    return pd.DataFrame(rows)


def _cash_flow() -> pd.DataFrame:
    """Simplified cash-flow statement."""
    pat = _income_statement()["PAT (Net Income)"].tolist()
    da = _income_statement()["Depreciation & Amortisation"].tolist()
    capex = [180, 200, 220, 260, 300, 340, 380, 420]
    wc_change = [-20, -25, -30, -35, -40, -45, -50, -55]
    debt_repay = [0, -20, -30, -50, -50, -50, -60, -60]
    dividends = [0, 0, -20, -30, -40, -50, -60, -70]

    rows = []
    for i, yr in enumerate(ALL_YEARS):
        cfo = round(pat[i] + da[i] + wc_change[i], 1)
        cfi = round(-capex[i], 1)
        cff = round(debt_repay[i] + dividends[i], 1)
        net = round(cfo + cfi + cff, 1)
        rows.append({
            "Year": yr,
            "Net Income": pat[i],
            "Depreciation & Amortisation": da[i],
            "Changes in Working Capital": wc_change[i],
            "Cash from Operations (CFO)": cfo,
            "Capital Expenditure": -capex[i],
            "Cash from Investing (CFI)": cfi,
            "Debt Repayment": debt_repay[i],
            "Dividends Paid": dividends[i],
            "Cash from Financing (CFF)": cff,
            "Net Change in Cash": net,
        })
    return pd.DataFrame(rows)


def _key_ratios() -> pd.DataFrame:
    inc = _income_statement()
    bs = _balance_sheet()
    rows = []
    for i, yr in enumerate(ALL_YEARS):
        rev = inc.loc[i, "Revenue"]
        ebitda = inc.loc[i, "EBITDA"]
        pat = inc.loc[i, "PAT (Net Income)"]
        ta = bs.loc[i, "Total Assets"]
        eq = bs.loc[i, "Shareholders' Equity"]
        debt = bs.loc[i, "Total Debt"]
        rows.append({
            "Year": yr,
            "Revenue Growth (%)": round((rev / inc.loc[i - 1, "Revenue"] - 1) * 100, 1) if i > 0 else None,
            "Gross Margin (%)": round(inc.loc[i, "Gross Profit"] / rev * 100, 1),
            "EBITDA Margin (%)": round(ebitda / rev * 100, 1),
            "Net Margin (%)": round(pat / rev * 100, 1),
            "ROE (%)": round(pat / eq * 100, 1),
            "ROA (%)": round(pat / ta * 100, 1),
            "Debt-to-Equity": round(debt / eq, 2),
            "Current Ratio": round(
                (bs.loc[i, "Cash & Equivalents"] + bs.loc[i, "Accounts Receivable"] + bs.loc[i, "Inventory"])
                / (bs.loc[i, "Accounts Payable"] + bs.loc[i, "Other Liabilities"]),
                2,
            ),
            "EV/EBITDA (implied)": round((eq + debt - bs.loc[i, "Cash & Equivalents"]) / ebitda, 1),
        })
    return pd.DataFrame(rows)


def _public_comps() -> pd.DataFrame:
    """Comparable public companies."""
    data = [
        ("Infosys Ltd.", 95000, 22.5, 28.1, 18.5, 21.3, 12.4, 0.05, 28.5),
        ("Wipro Ltd.", 65000, 18.2, 23.4, 14.8, 17.0, 10.2, 0.22, 24.1),
        ("TCS Ltd.", 145000, 25.8, 31.2, 20.3, 24.8, 15.1, 0.02, 32.6),
        ("HCL Technologies", 52000, 20.1, 26.0, 16.2, 19.5, 11.8, 0.12, 26.3),
        ("Mphasis Ltd.", 14000, 19.5, 24.8, 15.1, 18.2, 10.8, 0.18, 22.5),
        ("Persistent Systems", 8500, 21.3, 27.5, 17.0, 20.1, 12.0, 0.08, 30.2),
        ("Coforge Ltd.", 9200, 17.8, 22.1, 13.5, 16.0, 9.5, 0.30, 20.8),
        ("L&T Technology", 11500, 20.8, 25.5, 16.5, 19.0, 11.2, 0.15, 27.1),
    ]
    cols = [
        "Company", "Revenue (₹ Cr)", "EBITDA Margin (%)",
        "Gross Margin (%)", "Net Margin (%)", "EV/EBITDA (x)",
        "EV/Revenue (x)", "Net Debt/EBITDA (x)", "P/E (x)",
    ]
    return pd.DataFrame(data, columns=cols)


def _precedent_transactions() -> pd.DataFrame:
    """Precedent M&A transactions in the IT-services space."""
    data = [
        ("Mphasis / Blackstone", "Apr 2021", 22500, 18.5, 13.2, 2.8),
        ("Mindtree / L&T Infotech", "Nov 2022", 17800, 20.2, 15.0, 3.2),
        ("NIIT Tech / Baring PE", "Mar 2020", 4900, 16.0, 11.5, 2.5),
        ("Hexaware / Carlyle", "Sep 2020", 7600, 19.0, 14.8, 3.0),
        ("Majesco / Thoma Bravo", "Jul 2020", 3200, 15.5, 10.2, 2.2),
        ("Rackspace / Apollo", "Dec 2021", 10500, 17.8, 12.5, 2.6),
        ("Zensar / RPG Group", "Jan 2023", 5800, 18.0, 13.0, 2.4),
        ("Cyient DLM / Advent", "Jun 2023", 4200, 16.5, 11.0, 2.1),
    ]
    cols = [
        "Target / Acquirer", "Date", "Enterprise Value (₹ Cr)",
        "EV/EBITDA (x)", "EV/Revenue (x)", "EV/EBIT (x)",
    ]
    return pd.DataFrame(data, columns=cols)


def _dcf_assumptions() -> pd.DataFrame:
    """DCF model key assumptions."""
    data = {
        "Parameter": [
            "Risk-Free Rate (%)", "Equity Risk Premium (%)", "Beta (levered)",
            "Cost of Equity (%)", "Cost of Debt (pre-tax) (%)", "Tax Rate (%)",
            "Debt Weight (%)", "Equity Weight (%)", "WACC (%)",
            "Terminal Growth Rate (%)", "Projection Period (years)",
        ],
        "Value": [7.0, 6.5, 1.10, 14.15, 9.0, 25.0, 25.0, 75.0, 12.36, 4.0, 5],
    }
    return pd.DataFrame(data)


def _segment_revenue() -> pd.DataFrame:
    """Revenue by business segment."""
    rows = []
    segments = {
        "IT Consulting": [420, 508, 623, 735, 882, 1059, 1249, 1449],
        "Managed Services": [360, 435, 534, 630, 756, 907, 1070, 1242],
        "Cloud & SaaS": [240, 290, 356, 420, 504, 605, 714, 828],
        "Digital Transformation": [180, 217, 267, 315, 378, 453, 535, 621],
    }
    for seg, vals in segments.items():
        for i, yr in enumerate(ALL_YEARS):
            rows.append({"Year": yr, "Segment": seg, "Revenue": vals[i]})
    return pd.DataFrame(rows)


def _geographic_revenue() -> pd.DataFrame:
    """Revenue by geography."""
    rows = []
    geos = {
        "India": [480, 580, 712, 840, 1008, 1210, 1427, 1656],
        "North America": [360, 435, 534, 630, 756, 907, 1070, 1242],
        "Europe": [240, 290, 356, 420, 504, 605, 714, 828],
        "Rest of World": [120, 145, 178, 210, 252, 302, 357, 414],
    }
    for geo, vals in geos.items():
        for i, yr in enumerate(ALL_YEARS):
            rows.append({"Year": yr, "Geography": geo, "Revenue": vals[i]})
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        _income_statement().to_excel(writer, sheet_name="Income Statement", index=False)
        _balance_sheet().to_excel(writer, sheet_name="Balance Sheet", index=False)
        _cash_flow().to_excel(writer, sheet_name="Cash Flow", index=False)
        _key_ratios().to_excel(writer, sheet_name="Key Ratios", index=False)
        _public_comps().to_excel(writer, sheet_name="Public Comps", index=False)
        _precedent_transactions().to_excel(writer, sheet_name="Precedent Transactions", index=False)
        _dcf_assumptions().to_excel(writer, sheet_name="DCF Assumptions", index=False)
        _segment_revenue().to_excel(writer, sheet_name="Segment Revenue", index=False)
        _geographic_revenue().to_excel(writer, sheet_name="Geographic Revenue", index=False)
    print(f"✅ Financial data written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
