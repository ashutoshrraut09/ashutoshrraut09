## LSCG PitchBook – Interactive Sell-Side Advisory Dashboard

### TechVista Solutions Pvt. Ltd. – Sell-Side Mandate

An interactive Streamlit-based **PitchBook dashboard** built for LSCG Advisory Group's
sell-side engagement with TechVista Solutions Pvt. Ltd., a mid-market Indian IT services company.

---

### Project Structure

```
├── app.py                              # Original PDF keyword extractor app
├── pitchbook_dashboard.py              # Main PitchBook interactive dashboard
├── data/
│   ├── generate_financial_data.py      # Script to generate sample financial data
│   └── LSCG_Financial_Data.xlsx        # Financial data workbook (9 sheets)
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```

### Dashboard Sections

Navigate via the sidebar to explore each section:

| # | Section | Description |
|---|---------|-------------|
| 1 | **Executive Summary** | Situation overview, investment highlights, FY25 KPIs |
| 2 | **Market & Industry Analysis** | IT market size, growth drivers, competitive landscape |
| 3 | **Company Overview** | Business description, segment/geography revenue, SWOT |
| 4 | **Financial Statements** | Interactive Income Statement, Balance Sheet, Cash Flow, Ratios |
| 5 | **Valuation Analysis** | Public comps, precedent transactions, DCF with sensitivity |
| 6 | **Deal Structure** | Transaction recommendation, financing mix |
| 7 | **Potential Buyers** | Strategic acquirers and PE sponsors |
| 8 | **Risk Analysis** | Risk matrix with severity, mitigation strategies, heat map |
| 9 | **Transaction Timeline** | Gantt chart with 20-week process plan and milestones |
| 10 | **Appendix & Downloads** | Data downloads, bank credentials, recent deals |

### Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate financial data (if not already present)
python data/generate_financial_data.py

# 3. Launch the PitchBook dashboard
streamlit run pitchbook_dashboard.py
```

### Financial Data Sheets

The Excel workbook (`data/LSCG_Financial_Data.xlsx`) contains:

- **Income Statement** – Revenue through PAT (FY21-FY28)
- **Balance Sheet** – Assets, liabilities, equity (FY21-FY28)
- **Cash Flow** – CFO, CFI, CFF breakdown (FY21-FY28)
- **Key Ratios** – Margins, ROE, ROA, leverage, multiples
- **Public Comps** – 8 comparable public companies
- **Precedent Transactions** – 8 recent M&A deals
- **DCF Assumptions** – WACC, beta, growth rates
- **Segment Revenue** – Revenue by business segment
- **Geographic Revenue** – Revenue by geography

### Key Features

- **Interactive charts** built with Plotly for drill-down exploration
- **Dynamic sliders** to adjust DCF parameters (WACC, terminal growth)
- **Sensitivity analysis** table for valuation stress-testing
- **Football-field valuation** chart comparing multiple methodologies
- **Gantt chart** for transaction timeline visualisation
- **Risk heat map** for likelihood vs. impact assessment
- **One-click Excel download** of all financial data

---

*Confidential – For Discussion Purposes Only | LSCG Advisory Group*
