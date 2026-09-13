"""
sample_queries.py -- 10 representative stakeholder queries for app.py (RetroFit IRL v4)

HOW TO USE
----------
Option A -- "Ask" tab, free-text box:
    Copy any query string from SAMPLE_QUERIES into the text box and click "Run analysis".

Option B -- preset selectbox (already wired into app.py):
    Choosing a preset fills the query box; the Interface Agent then extracts the constraints.

These are the ten practitioner-phrased queries of the paper (Appendix, Table "Practitioner-
phrased queries"; wlca_rag/evaluation.py: PRACTITIONER_QUERIES). They use the eight-band
EPC / BER scale (A0, A, B, C, D, E, F, G) and the three retrofit scenarios:
    S1  NZEB fabric + heat pump + 4 kWp PV
    S2  NZEB fabric + heat pump
    S3  NZEB fabric + condensing gas boiler

QUERY COVERAGE
--------------
Query  1 -- Semi-detached  D  1967-1982   Heat pump           Target A
Query  2 -- Semi-detached  -  boiler now  Heat pump           Whole-life carbon
Query  3 -- Bungalow       -              HP vs HP + PV       Embodied trade-off
Query  4 -- Terraced       -  1983-1993   Boiler vs HP        Cost per tonne saved
Query  5 -- (any)          G              (any)               Target B
Query  6 -- Semi-detached  -              Boiler upgrade      Target B, payback
Query  7 -- Detached       C              Gas vs all-electric Whole-life comparison
Query  8 -- Terraced       -  has PV      HP with / without PV Rating reached
Query  9 -- (any)          -              HP vs HP + PV       Least benefit from PV
Query 10 -- Bungalow       E              (any)               Quickest carbon payback

PERIOD GROUPS ACCEPTED BY THE INTERFACE AGENT
---------------------------------------------
Before 1967 | 1967-1982 | 1983-1993 | 1994-2004 | 2005-2009 | 2010-2013 | 2014-Onwards
"""

from __future__ import annotations
from typing import Dict, List

# ---------------------------------------------------------------------------
# QUERY CATALOGUE
# ---------------------------------------------------------------------------

SAMPLE_QUERIES: List[Dict] = [

    # Query 1 ----------------------------------------------------------------
    {
        "id": 1,
        "label": "1 - Semi-detached D (1970s), heat pump, can it reach A",
        "category": "Rating -- current band, period and target band",
        "pathway": "S2",
        "archetype": "semi",
        "current_epc": "D",
        "target_epc": "A",
        "period": "1967-1982",
        "query": (
            "My BER cert says D for a 1970s semi in Dublin. "
            "Would an air-to-water heat pump get me into the A band?"
        ),
        "notes": (
            "Interface Agent: semi, period 1967-1982, EPC pre D, post >= A, pathway S2. "
            "Rating agent reports the scenarios that reach A or better."
        ),
    },

    # Query 2 ----------------------------------------------------------------
    {
        "id": 2,
        "label": "2 - Semi-detached, gas boiler now, deep retrofit with heat pump",
        "category": "Whole-life -- existing system named, pathway S2",
        "pathway": "S2",
        "archetype": "semi",
        "current_epc": None,
        "target_epc": None,
        "period": None,
        "query": (
            "Three-bed semi, gas boiler at the moment, thinking of the SEAI one-stop-shop "
            "deep retrofit with a heat pump - what's the carbon over the life of the house?"
        ),
        "notes": "Routes to the whole-life domain; existing heating boiler is a hard constraint.",
    },

    # Query 3 ----------------------------------------------------------------
    {
        "id": 3,
        "label": "3 - Bungalow, heat pump with or without solar panels",
        "category": "Embodied trade-off -- S1 versus S2",
        "pathway": ["S1", "S2"],
        "archetype": "bungalow",
        "current_epc": None,
        "target_epc": None,
        "period": None,
        "query": (
            "Detached bungalow, oil heating, poorly insulated - is it worth putting solar "
            "panels on as well as a heat pump or does the embodied carbon cancel it out?"
        ),
        "notes": "Comparison query; the bungalow is the archetype where PV pays back most.",
    },

    # Query 4 ----------------------------------------------------------------
    {
        "id": 4,
        "label": "4 - Mid-terrace 1985, boiler upgrade or heat pump",
        "category": "Scenario comparison -- S2 versus S3",
        "pathway": ["S2", "S3"],
        "archetype": "terraced",
        "current_epc": None,
        "target_epc": None,
        "period": "1983-1993",
        "query": (
            "For a mid-terrace built in 1985, which option is cheapest per tonne of carbon "
            "saved: boiler upgrade or ASHP?"
        ),
        "notes": "Year 1985 resolves to the 1983-1993 period group.",
    },

    # Query 5 ----------------------------------------------------------------
    {
        "id": 5,
        "label": "5 - G-rated cottage to at least a B",
        "category": "Rating -- no archetype, band anchor and target",
        "pathway": None,
        "archetype": None,
        "current_epc": "G",
        "target_epc": "B",
        "period": None,
        "query": "What grant-eligible measures bring a G-rated cottage to at least a B?",
        "notes": "No archetype constraint: every archetype admitted; S3 reaches B for all four.",
    },

    # Query 6 ----------------------------------------------------------------
    {
        "id": 6,
        "label": "6 - Semi-detached, condensing boiler, is B achievable and payback",
        "category": "Rating + payback -- pathway S3",
        "pathway": "S3",
        "archetype": "semi",
        "current_epc": None,
        "target_epc": "B",
        "period": None,
        "query": (
            "Semi-detached, want a B or better with a new condensing boiler and insulation - "
            "is that achievable and what is the payback?"
        ),
        "notes": "S3 reaches B for the semi-detached; payback from the whole-life agent.",
    },

    # Query 7 ----------------------------------------------------------------
    {
        "id": 7,
        "label": "7 - Detached C, keep gas versus all-electric with PV",
        "category": "Whole-life comparison -- S1 versus S3",
        "pathway": ["S1", "S3"],
        "archetype": "detached",
        "current_epc": "C",
        "target_epc": None,
        "period": None,
        "query": (
            "Large detached house, C at present. Compare keeping gas versus going "
            "all-electric with PV on whole-life emissions."
        ),
        "notes": "Largest S3 penalty of the four archetypes (114 versus 30 tCO2e).",
    },

    # Query 8 ----------------------------------------------------------------
    {
        "id": 8,
        "label": "8 - Terraced with existing solar, does a heat pump still make sense",
        "category": "Existing PV baseline -- S1 versus S2, rating reached",
        "pathway": ["S1", "S2"],
        "archetype": "terraced",
        "current_epc": None,
        "target_epc": None,
        "period": None,
        "query": (
            "Terraced house, already has 2 kW of solar. Does a heat pump still make sense "
            "and what rating would it reach?"
        ),
        "notes": "Existing-PV dwellings only; S1 reaches A0 for the terraced archetype.",
    },

    # Query 9 ----------------------------------------------------------------
    {
        "id": 9,
        "label": "9 - Which house type benefits least from adding PV",
        "category": "Cross-archetype comparison -- S1 versus S2",
        "pathway": ["S1", "S2"],
        "archetype": None,
        "current_epc": None,
        "target_epc": None,
        "period": None,
        "query": "Which house type benefits least from adding PV to a heat pump retrofit?",
        "notes": "Semi-detached: S1 exceeds S2 by 1.1 tCO2e (PV credit below its embodied burden).",
    },

    # Query 10 ---------------------------------------------------------------
    {
        "id": 10,
        "label": "10 - Bungalow E, quickest carbon payback",
        "category": "Payback -- band anchor, all scenarios",
        "pathway": None,
        "archetype": "bungalow",
        "current_epc": "E",
        "target_epc": None,
        "period": None,
        "query": "Bungalow, E rating, elderly owner - quickest carbon payback option?",
        "notes": "Objective payback; all three scenarios admitted and ranked.",
    },
]


# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------

def get_query_by_id(query_id: int) -> Dict | None:
    """Return the query dict for a given 1-based integer ID, or None."""
    return next((q for q in SAMPLE_QUERIES if q["id"] == query_id), None)


def get_query_strings() -> List[str]:
    """Return only the natural-language query strings, in order."""
    return [q["query"] for q in SAMPLE_QUERIES]


def get_labels() -> List[str]:
    """Return only the short labels, suitable for a selectbox."""
    return [q["label"] for q in SAMPLE_QUERIES]


def query_summary_table() -> str:
    """Plain-text summary table of all 10 queries for terminal output."""
    lines = [
        f"{'ID':<4} {'Pathway':<12} {'EPC / period':<18} {'Archetype':<14}",
        "-" * 52,
    ]
    for q in SAMPLE_QUERIES:
        pw = q["pathway"] if isinstance(q["pathway"], str) else ("+".join(q["pathway"]) if q["pathway"] else "any")
        anchor = q.get("current_epc") or (f"built {q['period']}" if q.get("period") else "-")
        if q.get("target_epc"):
            anchor += f" -> {q['target_epc']}"
        lines.append(f"{q['id']:<4} {pw:<12} {anchor:<18} {q['archetype'] or 'any':<14}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(query_summary_table())
