"""
Convert docs/overview.md to docs/overview.pdf using fpdf2.
Run: python scripts/md_to_pdf.py
Requires: pip install fpdf2
"""
import re
from pathlib import Path

try:
    from fpdf import FPDF
except ImportError:
    print("Install fpdf2: pip install fpdf2")
    raise

# Paths
ROOT = Path(__file__).resolve().parent.parent
MD_PATH = ROOT / "docs" / "overview.md"
PDF_PATH = ROOT / "docs" / "overview.pdf"


class PDF(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        self.set_font("Helvetica", "", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 6, "Master Thesis Overview", align="C")
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def section_title(self, text: str):
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 8, text)
        self.ln(2)

    def body(self, text: str):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(0, 0, 0)
        # Replace markdown bold with clean text for PDF
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
        self.multi_cell(0, 6, text)
        self.ln(2)

    def bullet(self, text: str, indent: int = 5):
        self.set_font("Helvetica", "", 10)
        self.set_x(10 + indent)
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
        self.multi_cell(0, 6, "  - " + text)
        self.ln(1)

    def h2(self, text: str):
        self.ln(4)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 7, text)
        self.ln(2)

    def table_row(self, cells: list, header: bool = False):
        col_w = 190 / len(cells)
        self.set_font("Helvetica", "B" if header else "", 9)
        for c in cells:
            c = re.sub(r"\*\*(.+?)\*\*", r"\1", str(c))
            self.cell(col_w, 7, c[:45] + ("..." if len(c) > 45 else ""), border=1)
        self.ln()

    def horizontal_rule(self):
        self.ln(3)


def main():
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, "Master Thesis Overview")
    pdf.ln(6)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 6, "Topic: Decision Logic for ODD Exit - A Reinforcement Learning Approach for MRM Selection")
    pdf.ln(4)
    pdf.body(
        "Short, simple summary so anyone (including you) can understand what you will do, "
        "how hard it is, and what technical issues you might face."
    )
    pdf.horizontal_rule()

    # What You Will Do
    pdf.h2("What You Will Do (Simple Terms)")
    pdf.body("In one sentence:")
    pdf.body(
        "When a self-driving car can no longer drive safely (e.g. roadworks, bad weather), it must exit normal operation "
        "and choose a safe way to stop (e.g. brake in lane, pull over, change lane then brake). Your thesis is: build and "
        "train a small \"brain\" (RL) that chooses which of these stopping strategies to use, using risk and traffic info, "
        "and compare it to simple rules. You do this in a simulator (esmini), not a real car."
    )
    pdf.body("In steps:")
    steps = [
        "Phase 0 - Lock scope with your supervisor: which 3 stopping types (MRMs), what \"safely stopped\" (MRC) means, how you trigger \"exit,\" what data the sim gives you.",
        "Phase 1 - Get the sim talking to Python over UDP: read car state, send \"brake\" (or similar), trigger \"exit\" at a chosen time, log everything. No learning yet.",
        "Phase 2 - Add: risk (DRF + DARA), simple rule-based choices (e.g. \"if very risky -> brake\"), a safety shield that can block bad choices, run several scenarios, log success/collision/time-to-stop.",
        "Phase 3 - Turn that into an RL problem (Gym env), train an agent (e.g. PPO) to pick the MRM, use risk in the observation and reward, compare baseline vs RL and run ablations (with/without risk, with/without shield).",
    ]
    for s in steps:
        pdf.bullet(s)
    pdf.body(
        "So in one sentence: you will design and implement a risk-aware decision logic for \"how to stop safely when leaving the driving domain,\" "
        "first as rules + shield, then as an RL agent, and evaluate both in esmini."
    )
    pdf.horizontal_rule()

    # How Hard
    pdf.h2("How Hard Is It?")
    pdf.body("Medium overall. You're not inventing new RL theory; you're applying known ideas (PPO, Gym env, reward design) to a concrete ODD-exit problem.")
    pdf.body("Hard parts:")
    for s in [
        "Getting esmini + UDP + your code to run reliably (Phase 1).",
        "Reward design and training stability (Phase 3).",
        "Making sure risk, baseline, and shield are consistent and interpretable (Phase 2).",
    ]:
        pdf.bullet(s)
    pdf.body("Easier parts:")
    for s in [
        "Risk formulas (DRF/DARA) and rule-based baseline are well specified in your docs.",
        "RL part is standard (discrete actions, PPO, existing papers to follow).",
    ]:
        pdf.bullet(s)
    pdf.body("Difficulty is mainly integration and debugging, not deep math.")
    pdf.horizontal_rule()

    # Dummy data
    pdf.h2("Can You Use Dummy / Fake Data If Something Doesn't Work?")
    pdf.body("Short answer: yes, but only in a controlled way, and you must say so clearly.")
    pdf.body("Simulator / UDP / esmini: If the real esmini UDP feed is broken or delayed, you can use recorded or scripted state sequences to develop and debug your Python pipeline. In the thesis, describe this as a \"simplified/offline test environment\" and report final results with the real simulator (or clearly state the limitation).")
    pdf.body("Risk (DRF/DARA): Implement the real formulas. You can temporarily fix or clamp values (e.g. cap risk to [0,1]) so training doesn't explode, but don't \"fake\" risk with random numbers for main results.")
    pdf.body("RL training / evaluation: You must not fake success rates or rewards. You can use fewer scenarios or shorter episodes for quick experiments, or dummy scenarios to check the env, then move to real scenarios for thesis results.")
    pdf.body("Rule of thumb:")
    pdf.bullet("Fake/dummy OK: to get the pipeline running or to unit-test parts of the code.")
    pdf.bullet("Not OK: to invent or hide data in the final results.")
    pdf.bullet("Always: document what is real vs synthetic/dummy and what the limitations are.")
    pdf.horizontal_rule()

    # Technical difficulties table
    pdf.h2("Technical Difficulties You Might Face")
    rows = [
        ("Area", "Difficulty", "What can go wrong"),
        ("esmini + UDP", "High", "Wrong port/format/timing; codec and timeouts critical."),
        ("Observation / state", "Medium", "Sim may give different fields; need robust parser."),
        ("Risk (DRF/DARA)", "Low-Medium", "Formulas clear; validate on hand-made cases."),
        ("Shield / MRM rules", "Medium", "Edge cases; downgrade-only logic must match thesis."),
        ("RL env", "Medium", "reset/step must be clean; reward and done flags correct."),
        ("Training", "Medium", "PPO can be unstable; reward scale matters."),
        ("Scenarios", "Medium", "OpenSCENARIO fiddly; need scenarios that trigger ODD exit."),
    ]
    for i, row in enumerate(rows):
        pdf.table_row(list(row), header=(i == 0))
    pdf.ln(3)
    pdf.body(
        "Biggest single risk: esmini + UDP not behaving as assumed. That's why Phase 1 is so important and why your plan says "
        "\"only codec should change\" if the format differs - so you can swap in a mock UDP feed (dummy data) until the real one works."
    )
    pdf.horizontal_rule()

    # One-sentence summary
    pdf.h2("One-Sentence Summary")
    pdf.body(
        "You will build a small risk-aware \"brain\" that chooses how the car should safely stop when it leaves its normal driving domain, "
        "implement it first as rules + safety shield, then as an RL agent, run both in the esmini simulator, and compare them; "
        "you may use dummy or simplified data to get the pipeline working, but final results and conclusions must be based on what you actually measured and clearly documented."
    )

    PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(PDF_PATH))
    print(f"Wrote {PDF_PATH}")


if __name__ == "__main__":
    main()
