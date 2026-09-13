"""Whole-Life Carbon Agent (Section 3.6.3, Eq. wlc_total, Eq. wlc_check). Rule based -- combines
the operational and embodied agent results, checks internal consistency against the stored total,
and reports the trade-off metrics (net saving, payback, best pathway)."""
from __future__ import annotations
import time
import pandas as pd

from .. import registry as R
from .base import AgentOutput

H = R.ASSESSMENT_HORIZON_YEARS


class WholeLifeCarbonAgent:
    name = "wlca"

    def run(self, bundle: pd.DataFrame, op: AgentOutput, emb: AgentOutput) -> AgentOutput:
        t0 = time.perf_counter()
        top = bundle.iloc[0]
        wlc = op.result["operational_carbon_60y_tCO2e"] + emb.result["total_embodied_60y_tCO2e"]   # Eq. wlc_total
        dev = abs(wlc - top["life_cycle_60y_tCO2e"])                                               # Eq. wlc_check
        bau = top["no_retrofit_operational_carbon_60y_tCO2e"]
        saving = bau - op.result["operational_carbon_60y_tCO2e"]
        payback = emb.result["total_embodied_60y_tCO2e"] / (saving / H) if saving > 0 else None
        res = dict(
            case_id=top["case_id"], wlca_60y_tCO2e=round(wlc, 4), wlca_per_m2=round(wlc / top["building_area_m2"], 4),
            provisional_wlca=top["life_cycle_60y_tCO2e"], consistency_deviation=round(dev, 6),
            consistency_check="pass" if dev <= R.HYPER["tau_wlc_tco2e"] else "fail",
            net_carbon_saving_vs_bau_tCO2e=round(saving, 4), carbon_payback_years=round(payback, 2) if payback else None,
            op_to_emb_ratio=round(op.result["operational_carbon_60y_tCO2e"] / max(emb.result["total_embodied_60y_tCO2e"], 1e-9), 3),
            per_pathway={r["pathway"]: r["life_cycle_60y_tCO2e"] for _, r in bundle.drop_duplicates("pathway").iterrows()},
            best_pathway=min(((r["pathway"], r["life_cycle_60y_tCO2e"]) for _, r in bundle.iterrows()), key=lambda x: x[1])[0],
        )
        return AgentOutput(self.name, res, bundle["case_id"].tolist(), (time.perf_counter() - t0) * 1000)
