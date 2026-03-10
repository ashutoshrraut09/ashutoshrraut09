"""
Geopolitical Risk Analysis Report Generator

Generates an Excel workbook and a Word document analyzing geopolitical risks
for critical mineral mining operations in:
  - Mali (Gold)
  - DRC (Cobalt)
  - Zimbabwe (Platinum)

Parameters assessed:
  - Geography, State and Non-State Actors, Likelihood of Occurrence (%),
    Velocity, Impact, Supply Chain

Scenarios analyzed:
  1. Government controls all mines — no private building
  2. Constant interest rate and legislative changes
  3. International companies must establish infrastructure to continue mining
"""

import os
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.chart import BarChart, Reference, BarChart3D
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

# ---------------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------------

COUNTRIES = [
    {
        "country": "Mali",
        "mineral": "Gold",
        "geography": (
            "Landlocked West African nation; mining concentrated in Kayes, "
            "Sikasso and Koulikoro regions. Sahel climate with limited "
            "transport infrastructure linking mines to Bamako and export ports "
            "in Senegal/Côte d'Ivoire."
        ),
        "state_actors": (
            "Military transitional government (CNSP); Ministry of Mines; "
            "SOMILO (state mining entity). Non-state: JNIM & ISGS armed "
            "groups; artisanal mining cooperatives; Wagner Group/Africa Corps."
        ),
        "likelihood": 72,
        "velocity": "High — political instability can escalate within weeks",
        "impact": "Severe — Mali is Africa's 3rd-largest gold producer",
        "supply_chain": (
            "Barrick Gold, B2Gold, Hummingbird Resources operate major mines. "
            "Gold exported via Dakar and Abidjan ports. Disruption affects "
            "~70 tonnes/year of global supply."
        ),
        "scenarios": {
            "Government controls all mines": 65,
            "Interest rate & legislative changes": 55,
            "Infrastructure requirement for intl companies": 70,
        },
    },
    {
        "country": "DRC",
        "mineral": "Cobalt",
        "geography": (
            "Central African nation; cobalt mining concentrated in the "
            "Copperbelt (Katanga/Haut-Katanga & Lualaba provinces). Tropical "
            "climate with poor road networks; exports via Dar es Salaam and "
            "Durban corridors."
        ),
        "state_actors": (
            "Presidency & National Assembly; Gécamines (state mining company); "
            "Entreprise Générale du Cobalt (EGC). Non-state: M23, ADF and "
            "other armed groups in eastern provinces; Chinese-backed private "
            "mining firms; artisanal miners (ASM) — estimated 200k workers."
        ),
        "likelihood": 78,
        "velocity": "Very High — conflict zones can shut operations overnight",
        "impact": (
            "Critical — DRC produces ~73% of global cobalt, essential for "
            "lithium-ion batteries"
        ),
        "supply_chain": (
            "CMOC, Glencore (Mutanda & Kamoto), Chemaf supply global market. "
            "China refines >80% of DRC cobalt. EV and electronics industries "
            "directly exposed."
        ),
        "scenarios": {
            "Government controls all mines": 75,
            "Interest rate & legislative changes": 60,
            "Infrastructure requirement for intl companies": 80,
        },
    },
    {
        "country": "Zimbabwe",
        "mineral": "Platinum",
        "geography": (
            "Southern African nation; platinum group metals (PGMs) mined on "
            "the Great Dyke geological formation stretching ~550 km. Semi-arid "
            "climate; road and rail links to Beira (Mozambique) and South "
            "African ports."
        ),
        "state_actors": (
            "ZANU-PF government; Ministry of Mines & Mining Development; "
            "ZMDC (Zimbabwe Mining Development Corp). Non-state: opposition "
            "parties; war veterans associations; South African labour unions "
            "influencing cross-border workers."
        ),
        "likelihood": 58,
        "velocity": "Moderate — policy shifts typically signalled in advance",
        "impact": (
            "High — Zimbabwe holds the world's 2nd-largest platinum reserves "
            "after South Africa"
        ),
        "supply_chain": (
            "Zimplats (Impala Platinum), Unki (Anglo American Platinum), "
            "Mimosa (Sibanye-Stillwater). PGMs critical for catalytic "
            "converters, hydrogen fuel cells and electronics."
        ),
        "scenarios": {
            "Government controls all mines": 50,
            "Interest rate & legislative changes": 65,
            "Infrastructure requirement for intl companies": 55,
        },
    },
]

SCENARIO_LABELS = [
    "Government controls all mines",
    "Interest rate & legislative changes",
    "Infrastructure requirement for intl companies",
]

# ---------------------------------------------------------------------------
# EXCEL GENERATION
# ---------------------------------------------------------------------------

HEADER_FONT = Font(bold=True, color="FFFFFF", size=11)
HEADER_FILL = PatternFill(start_color="2F5496", end_color="2F5496", fill_type="solid")
SUBHEADER_FILL = PatternFill(start_color="D6E4F0", end_color="D6E4F0", fill_type="solid")
THIN_BORDER = Border(
    left=Side(style="thin"),
    right=Side(style="thin"),
    top=Side(style="thin"),
    bottom=Side(style="thin"),
)


def _style_header_row(ws, row, max_col):
    for col in range(1, max_col + 1):
        cell = ws.cell(row=row, column=col)
        cell.font = HEADER_FONT
        cell.fill = HEADER_FILL
        cell.alignment = Alignment(horizontal="center", wrap_text=True)
        cell.border = THIN_BORDER


def _style_data_cell(ws, row, col):
    cell = ws.cell(row=row, column=col)
    cell.border = THIN_BORDER
    cell.alignment = Alignment(wrap_text=True, vertical="top")


def _auto_col_width(ws, min_width=12, max_width=45):
    for col_cells in ws.columns:
        col_letter = get_column_letter(col_cells[0].column)
        best = min_width
        for cell in col_cells:
            if cell.value:
                best = max(best, min(len(str(cell.value)), max_width))
        ws.column_dimensions[col_letter].width = best + 2


def generate_excel(output_path: str):
    wb = Workbook()

    # ---- Sheet 1: Risk Overview ----
    ws1 = wb.active
    ws1.title = "Risk Overview"

    headers = [
        "Country", "Mineral", "Geography", "State & Non-State Actors",
        "Likelihood (%)", "Velocity", "Impact", "Supply Chain",
    ]
    ws1.append(headers)
    _style_header_row(ws1, 1, len(headers))

    for c in COUNTRIES:
        row = [
            c["country"], c["mineral"], c["geography"], c["state_actors"],
            c["likelihood"], c["velocity"], c["impact"], c["supply_chain"],
        ]
        ws1.append(row)

    for r in range(2, len(COUNTRIES) + 2):
        for col in range(1, len(headers) + 1):
            _style_data_cell(ws1, r, col)

    _auto_col_width(ws1)

    # ---- Sheet 2: Scenario Analysis ----
    ws2 = wb.create_sheet("Scenario Analysis")

    scenario_headers = ["Country", "Mineral"] + SCENARIO_LABELS
    ws2.append(scenario_headers)
    _style_header_row(ws2, 1, len(scenario_headers))

    for c in COUNTRIES:
        row = [c["country"], c["mineral"]]
        for s in SCENARIO_LABELS:
            row.append(c["scenarios"][s])
        ws2.append(row)

    for r in range(2, len(COUNTRIES) + 2):
        for col in range(1, len(scenario_headers) + 1):
            _style_data_cell(ws2, r, col)

    _auto_col_width(ws2)

    # Add a clustered bar chart for scenarios
    chart = BarChart()
    chart.type = "col"
    chart.grouping = "clustered"
    chart.title = "Scenario Likelihood by Country (%)"
    chart.y_axis.title = "Likelihood (%)"
    chart.x_axis.title = "Country"
    chart.y_axis.scaling.min = 0
    chart.y_axis.scaling.max = 100

    cats = Reference(ws2, min_col=1, min_row=2, max_row=len(COUNTRIES) + 1)
    for i, label in enumerate(SCENARIO_LABELS, start=3):
        vals = Reference(ws2, min_col=i, min_row=1, max_row=len(COUNTRIES) + 1)
        chart.add_data(vals, titles_from_data=True)

    chart.set_categories(cats)
    chart.shape = 4
    chart.width = 22
    chart.height = 14
    ws2.add_chart(chart, "A7")

    # ---- Sheet 3: Likelihood Comparison ----
    ws3 = wb.create_sheet("Likelihood Comparison")
    ws3.append(["Country", "Mineral", "Overall Likelihood (%)"])
    _style_header_row(ws3, 1, 3)
    for c in COUNTRIES:
        ws3.append([c["country"], c["mineral"], c["likelihood"]])
    for r in range(2, len(COUNTRIES) + 2):
        for col in range(1, 4):
            _style_data_cell(ws3, r, col)
    _auto_col_width(ws3)

    bar = BarChart()
    bar.type = "col"
    bar.title = "Overall Geopolitical Risk Likelihood (%)"
    bar.y_axis.title = "Likelihood (%)"
    bar.y_axis.scaling.min = 0
    bar.y_axis.scaling.max = 100
    cats = Reference(ws3, min_col=1, min_row=2, max_row=4)
    vals = Reference(ws3, min_col=3, min_row=1, max_row=4)
    bar.add_data(vals, titles_from_data=True)
    bar.set_categories(cats)
    bar.width = 18
    bar.height = 12
    ws3.add_chart(bar, "A7")

    wb.save(output_path)
    print(f"Excel report saved to: {output_path}")


# ---------------------------------------------------------------------------
# MATPLOTLIB CHART IMAGES (for Word document)
# ---------------------------------------------------------------------------

def _make_scenario_chart(save_path: str):
    """Bar chart comparing scenarios across countries."""
    x = np.arange(len(COUNTRIES))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["#2F5496", "#C55A11", "#548235"]
    for i, scenario in enumerate(SCENARIO_LABELS):
        vals = [c["scenarios"][scenario] for c in COUNTRIES]
        bars = ax.bar(x + i * width, vals, width, label=scenario, color=colors[i])
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{bar.get_height()}%", ha="center", va="bottom", fontsize=9,
            )

    ax.set_xlabel("Country — Mineral")
    ax.set_ylabel("Likelihood of Occurrence (%)")
    ax.set_title("Scenario Analysis — Likelihood by Country")
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"{c['country']}\n({c['mineral']})" for c in COUNTRIES])
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _make_likelihood_chart(save_path: str):
    """Bar chart of overall likelihood."""
    labels = [f"{c['country']}\n({c['mineral']})" for c in COUNTRIES]
    values = [c["likelihood"] for c in COUNTRIES]
    colors = ["#FFD700", "#4682B4", "#C0C0C0"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.5)
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"{v}%", ha="center", va="bottom", fontweight="bold",
        )
    ax.set_ylabel("Likelihood of Occurrence (%)")
    ax.set_title("Overall Geopolitical Risk Likelihood")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _make_radar_chart(save_path: str):
    """Radar chart comparing countries on key parameters."""
    categories = ["Likelihood", "Velocity", "Impact", "Supply Chain\nExposure", "Political\nInstability"]
    # Normalized scores (0-100) for radar comparison
    data = {
        "Mali (Gold)": [72, 80, 75, 65, 85],
        "DRC (Cobalt)": [78, 90, 95, 90, 88],
        "Zimbabwe (Platinum)": [58, 55, 70, 60, 50],
    }

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = ["#FFD700", "#4682B4", "#C0C0C0"]

    for (label, values), color in zip(data.items(), colors):
        vals = values + values[:1]
        ax.plot(angles, vals, "o-", linewidth=2, label=label, color=color)
        ax.fill(angles, vals, alpha=0.15, color=color)

    ax.set_thetagrids(np.degrees(angles[:-1]), categories)
    ax.set_ylim(0, 100)
    ax.set_title("Multi-Parameter Geopolitical Risk Comparison", pad=20, fontsize=13)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# WORD DOCUMENT GENERATION
# ---------------------------------------------------------------------------

def _add_heading(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.color.rgb = RGBColor(0x2F, 0x54, 0x96)
    return h


def _add_country_table(doc, country_data):
    """Add a formatted table for one country."""
    table = doc.add_table(rows=7, cols=2, style="Light Grid Accent 1")
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    rows_data = [
        ("Country — Mineral", f"{country_data['country']} — {country_data['mineral']}"),
        ("Geography", country_data["geography"]),
        ("State & Non-State Actors", country_data["state_actors"]),
        ("Likelihood of Occurrence", f"{country_data['likelihood']}%"),
        ("Velocity", country_data["velocity"]),
        ("Impact", country_data["impact"]),
        ("Supply Chain", country_data["supply_chain"]),
    ]

    for i, (label, value) in enumerate(rows_data):
        table.rows[i].cells[0].text = label
        table.rows[i].cells[1].text = str(value)
        # Bold the label column
        for paragraph in table.rows[i].cells[0].paragraphs:
            for run in paragraph.runs:
                run.bold = True


def _add_scenario_table(doc):
    """Add the scenario comparison table."""
    table = doc.add_table(
        rows=len(COUNTRIES) + 1,
        cols=len(SCENARIO_LABELS) + 2,
        style="Light Grid Accent 1",
    )
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    # Header row
    header_labels = ["Country", "Mineral"] + SCENARIO_LABELS
    for j, label in enumerate(header_labels):
        cell = table.rows[0].cells[j]
        cell.text = label
        for paragraph in cell.paragraphs:
            for run in paragraph.runs:
                run.bold = True

    # Data rows
    for i, c in enumerate(COUNTRIES, start=1):
        table.rows[i].cells[0].text = c["country"]
        table.rows[i].cells[1].text = c["mineral"]
        for j, s in enumerate(SCENARIO_LABELS, start=2):
            table.rows[i].cells[j].text = f"{c['scenarios'][s]}%"


def generate_word(output_path: str, chart_dir: str):
    doc = Document()

    # Title
    title = doc.add_heading("Geopolitical Risk Analysis — Critical Minerals in Africa", level=0)
    for run in title.runs:
        run.font.color.rgb = RGBColor(0x2F, 0x54, 0x96)

    doc.add_paragraph(
        "This report presents a geopolitical risk assessment for three critical "
        "mineral supply chains: Gold (Mali), Cobalt (DRC), and Platinum (Zimbabwe). "
        "It evaluates geography, state and non-state actors, likelihood of disruption, "
        "velocity of change, impact severity, and supply chain exposure across three "
        "forward-looking scenarios."
    )

    # -------------------------------------------------------------------
    # SECTION 1: Country-level risk profiles
    # -------------------------------------------------------------------
    _add_heading(doc, "1. Country Risk Profiles", level=1)

    for c in COUNTRIES:
        _add_heading(doc, f"{c['country']} — {c['mineral']}", level=2)
        _add_country_table(doc, c)
        doc.add_paragraph("")  # spacer

    # -------------------------------------------------------------------
    # SECTION 2: Overall Likelihood Comparison
    # -------------------------------------------------------------------
    _add_heading(doc, "2. Overall Likelihood Comparison", level=1)
    doc.add_paragraph(
        "The chart below compares the overall likelihood of geopolitical disruption "
        "for each country-mineral pair. DRC (Cobalt) carries the highest risk at 78%, "
        "driven by armed conflict and extreme dependence of the global cobalt supply "
        "chain on a single country. Mali (Gold) follows at 72% due to military "
        "governance and Sahel instability. Zimbabwe (Platinum) is assessed at 58%, "
        "moderated by relatively more stable institutional frameworks."
    )

    likelihood_path = os.path.join(chart_dir, "likelihood_chart.png")
    _make_likelihood_chart(likelihood_path)
    doc.add_picture(likelihood_path, width=Inches(5.5))
    last_paragraph = doc.paragraphs[-1]
    last_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # -------------------------------------------------------------------
    # SECTION 3: Multi-parameter radar comparison
    # -------------------------------------------------------------------
    _add_heading(doc, "3. Multi-Parameter Risk Comparison", level=1)
    doc.add_paragraph(
        "The radar chart compares each country across five normalized risk dimensions: "
        "likelihood, velocity of change, impact severity, supply chain exposure, and "
        "political instability. DRC dominates in nearly every dimension, underscoring "
        "its status as the highest-risk critical mineral jurisdiction."
    )

    radar_path = os.path.join(chart_dir, "radar_chart.png")
    _make_radar_chart(radar_path)
    doc.add_picture(radar_path, width=Inches(5.0))
    last_paragraph = doc.paragraphs[-1]
    last_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # -------------------------------------------------------------------
    # SECTION 4: Scenario Analysis
    # -------------------------------------------------------------------
    _add_heading(doc, "4. Scenario Analysis", level=1)

    doc.add_paragraph(
        "Three forward-looking scenarios are evaluated for each country. "
        "Likelihood percentages reflect the assessed probability of each "
        "scenario materialising within a 3–5 year horizon."
    )

    # Scenario 1
    _add_heading(doc, "Scenario 1: Government Controls All Mines — No Private Building", level=2)
    doc.add_paragraph(
        "Under this scenario, host governments nationalise or assume direct control "
        "of all mining operations, barring new private-sector investment. "
        "Mali (65%): The military junta has already revised mining codes to increase "
        "state ownership stakes; full nationalisation is plausible. "
        "DRC (75%): Gécamines and EGC have been empowered to monopolise artisanal "
        "cobalt trade; broader nationalisation aligns with political rhetoric. "
        "Zimbabwe (50%): Indigenisation laws exist but enforcement is inconsistent; "
        "full government takeover faces pushback from international investors."
    )

    # Scenario 2
    _add_heading(doc, "Scenario 2: Constant Interest Rate & Legislative Changes", level=2)
    doc.add_paragraph(
        "Frequent shifts in fiscal policy, royalty rates, and mining legislation "
        "create an unstable operating environment. "
        "Mali (55%): Post-coup legislative uncertainty, though mining code reform "
        "is ongoing. "
        "DRC (60%): The 2018 Mining Code revision raised royalties significantly; "
        "further changes are anticipated as cobalt strategic importance grows. "
        "Zimbabwe (65%): Hyperinflation history, multi-currency regime, and "
        "frequent statutory instrument changes make this the highest-scoring "
        "country for this scenario."
    )

    # Scenario 3
    _add_heading(doc, "Scenario 3: International Companies Must Establish Infrastructure", level=2)
    doc.add_paragraph(
        "International mining companies are required to build transport, energy, "
        "and social infrastructure as a condition of continued operations. "
        "Mali (70%): Limited existing infrastructure in the Sahel raises costs; "
        "government leverage is increasing. "
        "DRC (80%): Massive infrastructure deficit in Katanga; the Lobito Corridor "
        "project illustrates the scale of investment required. "
        "Zimbabwe (55%): Great Dyke is relatively well-served; South Africa's "
        "neighbouring infrastructure provides an alternative."
    )

    # Scenario comparison table
    _add_heading(doc, "Scenario Comparison Table", level=2)
    _add_scenario_table(doc)
    doc.add_paragraph("")

    # Scenario chart
    scenario_path = os.path.join(chart_dir, "scenario_chart.png")
    _make_scenario_chart(scenario_path)
    doc.add_picture(scenario_path, width=Inches(5.5))
    last_paragraph = doc.paragraphs[-1]
    last_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # -------------------------------------------------------------------
    # SECTION 5: Key Takeaways
    # -------------------------------------------------------------------
    _add_heading(doc, "5. Key Takeaways & Recommendations", level=1)

    takeaways = [
        (
            "DRC (Cobalt) is the highest-risk jurisdiction across all parameters. "
            "Diversification of cobalt sourcing (e.g., Indonesia, Australia) and "
            "investment in recycling are strategic imperatives."
        ),
        (
            "Mali (Gold) faces acute near-term risk from military governance and "
            "Sahel security deterioration. Companies should stress-test supply "
            "chains against a full nationalisation scenario."
        ),
        (
            "Zimbabwe (Platinum) presents moderate risk; however, legislative "
            "volatility warrants close monitoring, especially as hydrogen economy "
            "demand for PGMs grows."
        ),
        (
            "All three countries show elevated risk under the infrastructure "
            "mandate scenario. International operators should engage in early-stage "
            "public-private partnership negotiations."
        ),
    ]
    for t in takeaways:
        doc.add_paragraph(t, style="List Bullet")

    doc.save(output_path)
    print(f"Word report saved to: {output_path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    output_dir = os.path.dirname(os.path.abspath(__file__))
    chart_dir = os.path.join(output_dir, "charts")
    os.makedirs(chart_dir, exist_ok=True)

    excel_path = os.path.join(output_dir, "Geopolitical_Risk_Analysis.xlsx")
    word_path = os.path.join(output_dir, "Geopolitical_Risk_Analysis.docx")

    generate_excel(excel_path)
    generate_word(word_path, chart_dir)

    print("\nDone. Generated files:")
    print(f"  • {excel_path}")
    print(f"  • {word_path}")


if __name__ == "__main__":
    main()
