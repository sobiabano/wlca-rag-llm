"""Embodied Carbon Agent (Section 3.6.3, Eq. emb_carbon), by component. Rule based -- reads
verified records copied by the pipeline; no value is computed by a language model."""
from __future__ import annotations
import time
import pandas as pd

from .base import AgentOutput


class EmbodiedCarbonAgent:
    name = "embodied"

    def run(self, bundle: pd.DataFrame) -> AgentOutput:
        t0 = time.perf_counter()
        top = bundle.iloc[0]
        res = dict(
            case_id=top["case_id"],
            components=dict(wall=top["wall_embodied_tCO2e"], roof=top["roof_embodied_tCO2e"],
                            glazing_floor=top["glazing_floor_embodied_tCO2e"], heatpump=top["heatpump_embodied_60y_tCO2e"],
                            pv=top["pv_embodied_60y_tCO2e"], boiler=top["boiler_embodied_60y_tCO2e"]),
            fabric_embodied_60y_tCO2e=top["fabric_embodied_60y_tCO2e"], system_embodied_60y_tCO2e=top["system_embodied_60y_tCO2e"],
            total_embodied_60y_tCO2e=top["total_embodied_60y_tCO2e"], embodied_per_m2=top["total_embodied_60y_tCO2e_per_m2"],
            inventory_source=top["inventory_source"],
            per_pathway={r["pathway"]: r["total_embodied_60y_tCO2e"] for _, r in bundle.drop_duplicates("pathway").iterrows()},
        )
        return AgentOutput(self.name, res, bundle["case_id"].tolist(), (time.perf_counter() - t0) * 1000)
