"""
paper_test_queries.py -- Free-text stakeholder queries from the paper's evaluation set

Ten of the 34 free-text queries in the 114-query headline set (Supplementary Table S14,
notebooks4/eval_freeform.py). They were written blind, in the language of SEAI advisers and
homeowners, and avoid the controlled vocabulary of the Interface Agent ("air-to-water unit"
instead of heat pump, "panels" instead of PV, "cert" instead of rating). They are the queries
on which the LLM-RAG configuration separates from the deterministic configuration.

DO NOT use these as dashboard presets -- sample_queries.py serves that purpose.
They are exposed in the "Audit & methodology" tab so a stakeholder can try them.

QUERY COVERAGE (two per category)
---------------------------------
F01  Operational   Bungalow      S1          electricity per m2 after the full job
F04  Operational   Semi          any         what a D becomes after a deep job
F08  Embodied      Semi          S2          materials carbon of insulation + electric unit
F11  Embodied      Semi          S1          manufacturing carbon of panels
F16  Whole-life    Terraced      S2          sixty-year carbon, no panels
F19  Whole-life    Detached      S1          nineties house, full package
F22  Rating        Semi          S1          top zero-emission grade or at least A
F26  Rating        Bungalow      any         highest grade from an F
F31  Scenario      Semi          any         fastest carbon payback at F
F34  Scenario      Detached      any         already electric, lowest lifetime carbon
"""

from __future__ import annotations
from typing import Dict, List

PAPER_TEST_QUERIES: List[Dict] = [
    {"id": "F01", "category": "Operational", "archetype": "bungalow", "pathway": "S1",
     "query": "How much electricity per square metre will a one-storey house use after the full job with panels on the roof"},
    {"id": "F04", "category": "Operational", "archetype": "semi", "pathway": None,
     "query": "What does a D on the cert turn into after a deep job on a semi"},
    {"id": "F08", "category": "Embodied", "archetype": "semi", "pathway": "S2",
     "query": "Materials carbon of insulating a semi and swapping the boiler for an electric unit"},
    {"id": "F11", "category": "Embodied", "archetype": "semi", "pathway": "S1",
     "query": "Does adding panels add a lot of manufacturing carbon on a semi"},
    {"id": "F16", "category": "Whole-life", "archetype": "terraced", "pathway": "S2",
     "query": "Sixty-year carbon for a terraced house with an air-to-water unit and no panels"},
    {"id": "F19", "category": "Whole-life", "archetype": "detached", "pathway": "S1",
     "query": "Whole-of-life figure for a nineties detached house after the full package with panels"},
    {"id": "F22", "category": "Rating", "archetype": "semi", "pathway": "S1",
     "query": "Can a semi get to the top zero-emission grade, or at least an A, with an electric unit and panels"},
    {"id": "F26", "category": "Rating", "archetype": "bungalow", "pathway": None,
     "query": "Highest grade reachable for a bungalow from an F"},
    {"id": "F31", "category": "Scenario", "archetype": "semi", "pathway": None,
     "query": "Which option pays back its carbon fastest on a semi at F"},
    {"id": "F34", "category": "Scenario", "archetype": "detached", "pathway": None,
     "query": "Lowest lifetime carbon for a detached house that already has an electric heating unit"},
]


def get_query_strings() -> List[str]:
    return [q["query"] for q in PAPER_TEST_QUERIES]


def get_labels() -> List[str]:
    return [f"{q['id']} ({q['category']}) - {q['query'][:60]}..." for q in PAPER_TEST_QUERIES]
