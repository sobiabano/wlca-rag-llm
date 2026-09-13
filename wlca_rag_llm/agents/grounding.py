"""Grounding and Verification (Section 3.6.2): completeness, consistency and provenance checks
applied to every ranked candidate before a carbon agent reads it. Rule based; shared by both
configurations."""
from __future__ import annotations
import pandas as pd

from ..retrieval.interface_agent import RetrievalConstraints, record_satisfies

_REQUIRED_FIELDS = ["case_id", "archetype_id", "scenario_id", "building_type", "pathway",
                    "BER_rating", "BER_rating_after", "building_area_m2", "delivered_kwh_yr",
                    "operational_carbon_60y_tCO2e", "total_embodied_60y_tCO2e", "life_cycle_60y_tCO2e",
                    "no_retrofit_operational_carbon_60y_tCO2e", "dataset_version", "simulation_id",
                    "inventory_source", "emission_factor_source"]


def ground_and_verify(cand: pd.DataFrame, c: RetrievalConstraints, kbs: dict) -> tuple[pd.DataFrame, list[dict]]:
    log = []
    keep = []
    arch_ids = set(kbs["archetype"]["archetype_id"])
    scen_ids = set(kbs["embodied"]["scenario_id"])
    for _, r in cand.iterrows():
        complete = all(pd.notna(r.get(f)) for f in _REQUIRED_FIELDS)
        pre_ok, post_ok = record_satisfies(r, c)
        cat_ok = all(r[k] == v for k, v in c.categorical.items())
        pw = c.retrofit.get("pathway")
        pw_ok = True if (pw is None or "pathway" in c.relaxations) else (r["pathway"] in (pw if isinstance(pw, list) else [pw]))
        consistent = pre_ok and post_ok and cat_ok and pw_ok and (r["archetype_id"] in arch_ids) and (r["scenario_id"] in scen_ids)
        provenance = bool(r["dataset_version"]) and bool(r["simulation_id"]) and bool(r["inventory_source"])
        ok = complete and consistent and provenance
        log.append(dict(case_id=r["case_id"], completeness=complete, consistency=consistent, provenance=provenance, promoted=ok))
        if ok:
            keep.append(r.name)
    return cand.loc[keep], log
