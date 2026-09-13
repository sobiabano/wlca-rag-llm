"""Synthesiser Agent (Section 3.6.4): the only generative component in the pipeline. Composes a
narrative from the verified bundle and the three carbon agents' results; every value is copied
verbatim and cited, never calculated. Used only by the LLM-RAG configuration and only when an
OpenAI API key is present (see available)."""
from __future__ import annotations
import json
import os
import time

import numpy as np
import pandas as pd

from .. import registry as R
from ..retrieval.interface_agent import RetrievalConstraints

_SYS_PROMPT = (
    "You are the Synthesiser Agent of a whole-life carbon assessment system for Irish residential retrofit. "
    "Answer the stakeholder's question using ONLY the evidence records supplied. Rules: "
    "(1) Every carbon, energy, payback or rating value you state must be copied verbatim from the records; "
    "never calculate, average, convert or estimate a value. "
    "(2) Cite the case_id in square brackets immediately after every numerical value, e.g. '27.7 tCO2e [semi_D_C_nopv_A2]'. "
    "(3) EPC ratings must be written exactly as in the records (A0, A, B, C, D, E, F, G). "
    "(4) Do not mention years of replacement, service lives or physical constants as numbers. "
    "(5) If the records were retrieved after a constraint relaxation, say so in one sentence. "
    "(6) Keep the answer under 180 words, plain text, no markdown tables."
)


def _bundle_context(bundle: pd.DataFrame, n: int) -> str:
    cols = ["case_id", "building_type_label", "period", "pre_heating", "BER_rating", "BER_rating_after", "pathway_label",
            "delivered_EUI_kWh_m2", "primary_EUI_kWh_m2", "operational_carbon_60y_tCO2e",
            "fabric_embodied_60y_tCO2e", "system_embodied_60y_tCO2e", "total_embodied_60y_tCO2e",
            "life_cycle_60y_tCO2e", "life_cycle_60y_tCO2e_per_m2", "no_retrofit_operational_carbon_60y_tCO2e",
            "carbon_saving_60y_tCO2e", "carbon_payback_years"]
    recs = bundle.head(n)[cols].copy()
    recs["carbon_payback_years"] = recs["carbon_payback_years"].replace(np.inf, "never")
    return "\n".join(json.dumps({k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items()})
                     for r in recs.to_dict("records"))


class SynthesiserAgent:
    def __init__(self, model: str = R.HYPER["llm_model"]):
        self.model = model
        self._client = None

    @property
    def available(self) -> bool:
        return bool(os.environ.get("OPENAI_API_KEY"))

    def run(self, query: str, c: RetrievalConstraints, bundle: pd.DataFrame, agents: dict) -> tuple[str, float, int]:
        if not self.available:
            return "", 0.0, 0
        from openai import OpenAI
        if self._client is None:
            self._client = OpenAI()
        ctx = _bundle_context(bundle, R.HYPER["k_agent"])
        relax = f"Constraint relaxation applied: {c.relaxations}." if c.relaxations else "No relaxation applied."
        user = (f"Question: {query}\n\nInterpreted constraints: {json.dumps({k: v for k, v in c.to_dict().items() if k in ('categorical','retrofit','epc','objective','domains')})}\n"
                f"{relax}\n\nEvidence records (JSON, one per line):\n{ctx}\n\n"
                f"Agent summaries: operational={json.dumps(agents['operational'].result['per_pathway'])}, "
                f"embodied={json.dumps(agents['embodied'].result['per_pathway'])}, "
                f"whole-life={json.dumps(agents['wlca'].result['per_pathway'])}, consistency={agents['wlca'].result['consistency_check']}."
                + (f" Scenarios that reach the requested band {agents['wlca'].result['target_band']} or better: {agents['wlca'].result['scenarios_achieving_target']} - state this explicitly." if agents['wlca'].result.get('target_band') else ""))
        t0 = time.perf_counter()
        resp = self._client.chat.completions.create(
            model=self.model, temperature=0.0, max_tokens=400,
            messages=[{"role": "system", "content": _SYS_PROMPT}, {"role": "user", "content": user}])
        ms = (time.perf_counter() - t0) * 1000
        return resp.choices[0].message.content.strip(), ms, resp.usage.total_tokens
