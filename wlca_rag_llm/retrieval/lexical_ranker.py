"""
Lexical ranker (Section 3.6.1): Jaccard vocabulary overlap + keyword bonus + percentile-rank
weighting (Eq. jaccard, Eq. total_score). Deterministic configuration -- no embeddings, no API key.

Also holds the shared per-domain text representation used by both rankers (record_text,
DOMAIN_FIELDS): the lexical ranker tokenises it, the semantic ranker embeds it.
"""
from __future__ import annotations
import re

import numpy as np
import pandas as pd

from .. import registry as R

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-]+")
_STOP = {"the", "a", "an", "for", "with", "and", "of", "to", "in", "is", "what", "show", "which",
         "house", "home", "all", "cases", "case", "me", "that", "after", "retrofit", "retrofitted",
         "give", "does", "do", "how", "much", "on", "by", "at", "from", "are", "it", "this", "its"}

# Controlled vocabulary kappa(q): terms that must appear in a record's text to earn the bonus
_KEY_TERMS = {"heatpump", "solarpv", "boiler", "gas", "semi-detached", "detached", "bungalow",
              "terraced", "embodied", "operational", "whole-life", "payback", "a0", "a", "b", "c",
              "d", "e", "f", "g", "a2", "a3", "b2"}

# Domain text representations (one per carbon domain)
DOMAIN_FIELDS = {
    "archetype":   ["building_type_label", "period", "BER_rating", "BER_rating_after", "pathway",
                    "pathway_label", "heating_system", "pre_heating", "pv_option", "pre_retrofit_baseline", "pathway_applicable"],
    "operational": ["building_type_label", "pathway_label", "BER_rating", "BER_rating_after",
                    "delivered_EUI_kWh_m2", "primary_EUI_kWh_m2", "operational_carbon_60y_tCO2e",
                    "no_retrofit_operational_carbon_60y_tCO2e", "operational_carbon_60y_tCO2e_per_m2"],
    "embodied":    ["building_type_label", "pathway_label", "wall_embodied_tCO2e", "roof_embodied_tCO2e",
                    "glazing_floor_embodied_tCO2e", "fabric_embodied_60y_tCO2e", "heatpump_embodied_60y_tCO2e",
                    "pv_embodied_60y_tCO2e", "boiler_embodied_60y_tCO2e", "system_embodied_60y_tCO2e",
                    "total_embodied_60y_tCO2e", "total_embodied_60y_tCO2e_per_m2"],
    "wlca":        ["building_type_label", "period", "pathway_label", "BER_rating", "BER_rating_after",
                    "life_cycle_60y_tCO2e", "life_cycle_60y_tCO2e_per_m2", "operational_carbon_60y_tCO2e",
                    "total_embodied_60y_tCO2e", "carbon_saving_60y_tCO2e", "net_wlca_vs_bau_tCO2e",
                    "carbon_payback_years"],
}
_FIELD_WORDS = {
    "building_type_label": "archetype", "period": "built", "pre_heating": "existing heating", "pathway_applicable": "pathway applicable", "BER_rating": "pre-retrofit EPC rating",
    "BER_rating_after": "post-retrofit EPC rating achieved", "pathway": "pathway", "pathway_label": "retrofit pathway",
    "heating_system": "heating", "pv_option": "solar PV", "pre_retrofit_baseline": "existing PV baseline",
    "delivered_EUI_kWh_m2": "delivered energy use intensity kWh/m2/yr",
    "primary_EUI_kWh_m2": "primary energy use intensity kWh/m2/yr",
    "operational_carbon_60y_tCO2e": "operational carbon 60-year tCO2e",
    "no_retrofit_operational_carbon_60y_tCO2e": "business-as-usual no-retrofit carbon tCO2e",
    "operational_carbon_60y_tCO2e_per_m2": "operational carbon per m2",
    "wall_embodied_tCO2e": "wall insulation embodied carbon tCO2e", "roof_embodied_tCO2e": "roof insulation embodied carbon tCO2e",
    "glazing_floor_embodied_tCO2e": "glazing and floor embodied carbon tCO2e",
    "fabric_embodied_60y_tCO2e": "fabric embodied carbon tCO2e", "heatpump_embodied_60y_tCO2e": "heat pump embodied carbon tCO2e",
    "pv_embodied_60y_tCO2e": "solar PV panel embodied carbon tCO2e", "boiler_embodied_60y_tCO2e": "boiler embodied carbon tCO2e",
    "system_embodied_60y_tCO2e": "system embodied carbon tCO2e", "total_embodied_60y_tCO2e": "total embodied carbon 60-year tCO2e",
    "total_embodied_60y_tCO2e_per_m2": "embodied carbon per m2",
    "life_cycle_60y_tCO2e": "whole-life carbon WLCA 60-year tCO2e", "life_cycle_60y_tCO2e_per_m2": "whole-life carbon per m2",
    "carbon_saving_60y_tCO2e": "carbon saving versus baseline tCO2e", "net_wlca_vs_bau_tCO2e": "net whole-life carbon versus business as usual",
    "carbon_payback_years": "carbon payback years",
}


def record_text(row: pd.Series, domain: str) -> str:
    parts = []
    for f in DOMAIN_FIELDS[domain]:
        v = row[f]
        if isinstance(v, float) and np.isinf(v):
            v = "never within horizon"
        parts.append(f"{_FIELD_WORDS[f]}: {v}")
    return " | ".join(parts)


def tokens(text: str) -> set:
    t = text.lower().replace("heat pump", "heatpump").replace("solar pv", "solarpv")
    return {w for w in _TOKEN_RE.findall(t) if w not in _STOP}


class LexicalRanker:
    """Deterministic configuration.  S = J + b*1[kappa(q) subset T(d)] + gamma*rho_o + alpha*rho_WLC."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.tok = {}
        for cid, r in df.set_index("case_id").iterrows():      # token sets only (memory-lean for 72,000 records)
            self.tok[cid] = tokens(" ".join(record_text(r, d) for d in DOMAIN_FIELDS))
        # percentile ranks (lower carbon -> higher rank score)
        self.rank_wlc = 1.0 - df["life_cycle_60y_tCO2e"].rank(pct=True)
        self.rank_obj = {
            "min_wlca": self.rank_wlc,
            "min_embodied": 1.0 - df["total_embodied_60y_tCO2e"].rank(pct=True),
            "min_operational": 1.0 - df["operational_carbon_60y_tCO2e"].rank(pct=True),
            "max_fabric": pd.Series(1.0, index=df.index),
        }
        self.rank_wlc.index = df["case_id"].values
        for k in self.rank_obj:
            self.rank_obj[k].index = df["case_id"].values

    def score(self, query_norm: str, cand: pd.DataFrame, objective: str | None) -> pd.Series:
        qt = tokens(query_norm)
        kq = qt & _KEY_TERMS
        b, g, a = R.HYPER["b_keyword_bonus"], R.HYPER["gamma_objective"], R.HYPER["alpha_wlc"]
        out = []
        for cid in cand["case_id"]:
            dt = self.tok[cid]
            j = len(qt & dt) / max(len(qt | dt), 1)
            bonus = b if kq and kq <= dt else 0.0
            obj = g * self.rank_obj[objective][cid] if objective else 0.0
            out.append(j + bonus + obj + a * self.rank_wlc[cid])
        return pd.Series(out, index=cand.index)
