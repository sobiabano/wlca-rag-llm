"""Operational Carbon Agent (Section 3.6.3, Eq. op_carbon). Rule based -- reads verified records
copied by the pipeline; no value is computed by a language model."""
from __future__ import annotations
import time
import pandas as pd

from .. import registry as R
from .base import AgentOutput


class OperationalCarbonAgent:
    name = "operational"

    def run(self, bundle: pd.DataFrame) -> AgentOutput:
        t0 = time.perf_counter()
        top = bundle.iloc[0]
        res = dict(
            case_id=top["case_id"], BER_rating_pre=top["BER_rating"], BER_rating_after=top["BER_rating_after"],
            primary_EUI_pre_kWh_m2=top["primary_EUI_pre_kWh_m2"], primary_EUI_after_kWh_m2=top["primary_EUI_kWh_m2"],
            delivered_EUI_after_kWh_m2=top["delivered_EUI_kWh_m2"],
            delivered_kwh_yr=top["delivered_kwh_yr"], emission_factor_kg_per_kwh=R.GRID_MEAN_KG_PER_KWH,
            operational_carbon_60y_tCO2e=top["operational_carbon_60y_tCO2e"],
            operational_carbon_per_m2=top["operational_carbon_60y_tCO2e_per_m2"],
            no_retrofit_operational_carbon_60y_tCO2e=top["no_retrofit_operational_carbon_60y_tCO2e"],
            per_pathway={r["pathway"]: r["operational_carbon_60y_tCO2e"] for _, r in bundle.drop_duplicates("pathway").iterrows()},
        )
        return AgentOutput(self.name, res, bundle["case_id"].tolist(), (time.perf_counter() - t0) * 1000)
