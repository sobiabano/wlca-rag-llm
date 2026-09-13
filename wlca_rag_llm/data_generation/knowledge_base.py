"""
WLCA-KBS generation: enumerates the parametric design space once and builds the linked
ArchetypeData / OperationalData / EmbodiedData / case tables in memory. Every value is derived
from the Parameter Registry by the equations of the paper (Eq. op_carbon, emb_carbon, wlc_total,
bau_carbon). Reads the 24,000-run design of experiments from DuckDB (data/wlca_kbs.duckdb);
no CSV is read or written by this module.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

from .. import registry as R
from .parametric_ensemble import load_runs

H = R.ASSESSMENT_HORIZON_YEARS


# ---------------------------------------------------------------------------
# Post-retrofit outcome per (archetype, pathway) -- Eq. op_carbon / emb_carbon
# ---------------------------------------------------------------------------
def post_retrofit_outcome(arch: str, pw: str) -> dict:
    s = R.ARCHETYPE_SIM[arch]
    A = s["area_m2"]
    fab = R.FABRIC_EMBODIED[arch]
    fabric_emb = fab["wall"] + fab["roof"] + fab["glazing_floor"]

    if pw == "S3":
        gas, elec, pv = s["gas_kwh"], s["boiler_elec_kwh"], 0
        delivered = gas + elec
        primary = gas * R.PEF_GAS + elec * R.PEF_ELECTRICITY
        op_gas = gas * R.GAS_KG_PER_KWH * H / 1000.0
        op_elec = elec * R.GRID_MEAN_KG_PER_KWH * H / 1000.0
        systems = {"boiler": R.SYSTEM_EMBODIED["boiler"]}
    else:
        gas = 0
        elec = s["hp_elec_kwh"]
        pv = s["pv_credit_kwh"] if pw == "S1" else 0
        delivered = elec                          # gross delivered demand
        net_elec = elec - pv                      # self-consumed PV credited
        primary = net_elec * R.PEF_ELECTRICITY
        op_gas = 0.0
        op_elec = net_elec * R.GRID_MEAN_KG_PER_KWH * H / 1000.0
        systems = {"heatpump": R.SYSTEM_EMBODIED["heatpump"]}
        if pw == "S1":
            systems["pv"] = R.SYSTEM_EMBODIED["pv"]

    sys_emb = {k: v["per_install"] * len(R.installs_within_horizon(v["life"])) for k, v in systems.items()}
    system_emb = sum(sys_emb.values())
    op = op_gas + op_elec
    emb = fabric_emb + system_emb
    prim_eui = primary / A
    return dict(
        building_area_m2=A,
        delivered_kwh_yr=delivered, gas_kwh_yr=gas, electricity_kwh_yr=elec,
        pv_credit_kwh_yr=pv, primary_kwh_yr=primary,
        delivered_EUI_kWh_m2=round(delivered / A, 2),
        primary_EUI_kWh_m2=round(prim_eui, 2),
        BER_rating_after=R.ber_new_from_eui(prim_eui, all_electric=(pw != "S3")),
        BER_rating_after_old=R.ber_old_from_eui(prim_eui),
        operational_carbon_60y_tCO2e=round(op, 4),
        operational_gas_60y_tCO2e=round(op_gas, 4),
        operational_elec_60y_tCO2e=round(op_elec, 4),
        wall_embodied_tCO2e=fab["wall"], roof_embodied_tCO2e=fab["roof"],
        glazing_floor_embodied_tCO2e=fab["glazing_floor"],
        fabric_embodied_60y_tCO2e=round(fabric_emb, 4),
        boiler_embodied_60y_tCO2e=round(sys_emb.get("boiler", 0.0), 4),
        heatpump_embodied_60y_tCO2e=round(sys_emb.get("heatpump", 0.0), 4),
        pv_embodied_60y_tCO2e=round(sys_emb.get("pv", 0.0), 4),
        system_embodied_60y_tCO2e=round(system_emb, 4),
        total_embodied_60y_tCO2e=round(emb, 4),
        life_cycle_60y_tCO2e=round(op + emb, 4),          # provisional total, Eq. wlc_total
        life_cycle_60y_tCO2e_per_m2=round((op + emb) / A, 4),
        operational_carbon_60y_tCO2e_per_m2=round(op / A, 4),
        total_embodied_60y_tCO2e_per_m2=round(emb / A, 4),
        emission_factor_source="Grid: SEAI/EirGrid CAP24 trajectory (mean 63.4 g/kWh); Gas: SEAI 202.9 g/kWh",
        inventory_source="ICE v3.0; Ecoinvent v3; CIBSE TM65 (heat pump)",
    )


# ---------------------------------------------------------------------------
# KBS from the simulation ensemble: one case per simulated dwelling, three pathway records each.
# What the ensemble supplies: archetype, period, sampled envelope / airtightness / system / PV,
#   pre-retrofit delivered and primary energy, EPC band, and the no-retrofit (business-as-usual)
#   operational carbon over the horizon (Eq. op_carbon with the registry factors).
# What it cannot supply: the post-retrofit outcome, because every pathway fixes the fabric at the
#   NZEB specification, a single point at the efficient edge of the sampled space that no run occupies;
#   the twelve archetype x pathway outcomes therefore come from the pathway simulations (registry).
# ---------------------------------------------------------------------------
def build_kbs(db_path: Path) -> dict[str, pd.DataFrame]:
    doe = load_runs(db_path)
    doe["period"] = doe["age_band"].map(R.DEAP_TO_GROUP)
    doe["pre_pv"] = np.where(doe["solar"].astype(bool), "pv", "nopv")
    doe["pre_heating"] = doe["heating_system"].map({"boiler": "boiler", "heat_pump": "heat_pump"})
    doe["gas_kwh_yr"] = doe["gas_kwh"].fillna(0.0)
    doe["grid_kwh_yr"] = doe["net_electricity_from_utility_kwh"].fillna(doe["electricity_kwh"]).clip(lower=0)
    doe["bau"] = (doe["gas_kwh_yr"] * R.GAS_KG_PER_KWH + doe["grid_kwh_yr"] * R.GRID_MEAN_KG_PER_KWH) * H / 1000
    doe["pre_prim_eui"] = doe["primary_EUI_kWh_m2"]
    doe["pre_del_eui"] = doe["delivered_EUI_kWh_m2"]
    doe["BER_pre"] = [R.ber_new_from_eui(e, all_electric=(h == "heat_pump")) for e, h in zip(doe["pre_prim_eui"], doe["pre_heating"])]
    doe["BER_pre_old"] = doe["pre_prim_eui"].map(R.ber_old_from_eui)

    # --- ArchetypeData: archetype x period (descriptor ranges of the ensemble) ----
    g = doe.groupby(["building_type", "period"])
    arch = g.agg(n_dwellings=("case", "size"), wall_U_min=("wall_u", "min"), wall_U_max=("wall_u", "max"),
                 roof_U_min=("roof_u", "min"), roof_U_max=("roof_u", "max"), floor_U_min=("floor_u", "min"), floor_U_max=("floor_u", "max"),
                 window_U_min=("window_u", "min"), window_U_max=("window_u", "max"), ach_min=("ach", "min"), ach_max=("ach", "max")).reset_index()
    arch["archetype_id"] = arch.building_type + "_" + arch.period.map(R.PERIOD_GROUP_CODE)
    arch["building_type_label"] = arch.building_type.map(R.ARCHETYPE_LABEL)
    arch["building_area_m2"] = arch.building_type.map(lambda a: R.ARCHETYPE_SIM[a]["area_m2"])
    for k, v in R.POST_RETROFIT_UVALUES.items():
        arch[f"{k}_U_post"] = v
    arch["stock_share"] = arch.building_type.map(R.ARCHETYPE_STOCK_SHARE)
    arch["dataset_version"] = R.DATASET_VERSION
    archetype_df = arch.round(3)

    # --- EmbodiedData: 12 archetype x pathway outcomes ------------------------------
    post = {(a, p): post_retrofit_outcome(a, p) for a in R.ARCHETYPES for p in R.PATHWAYS}
    embodied_df = pd.DataFrame([dict(scenario_id=f"{a}_{p}", building_type=a, pathway=p,
                                     **{k: o[k] for k in o if "embodied" in k or k == "inventory_source"}) for (a, p), o in post.items()])

    # --- Cases: dwelling x pathway ------------------------------------------------
    base_cols = ["case", "building_type", "period", "age_band", "pre_heating", "pre_pv", "wall_u", "roof_u", "floor_u", "window_u", "ach",
                 "boiler_eff", "heatpump_cop", "pv_scale", "building_area_m2", "pre_del_eui", "pre_prim_eui", "BER_pre", "BER_pre_old", "bau",
                 "gas_kwh_yr", "grid_kwh_yr"]
    d = doe[base_cols].rename(columns={"case": "dwelling_id", "BER_pre": "BER_rating", "BER_pre_old": "BER_rating_old",
                                       "pre_pv": "pre_retrofit_baseline", "bau": "no_retrofit_operational_carbon_60y_tCO2e",
                                       "pre_del_eui": "delivered_EUI_pre_kWh_m2", "pre_prim_eui": "primary_EUI_pre_kWh_m2"})
    frames = []
    for pw in R.PATHWAYS:
        m = R.PATHWAY_META[pw]
        f = d.copy()
        f["pathway"] = pw
        f["renovation_type"] = m["renovation_type"]
        f["retrofit_scenario"] = m["scenario"]
        f["pathway_label"] = m["label"]
        f["heating_system"] = m["heating"]
        f["pv_option"] = "PV on" if m["pv"] else "PV off"
        # Applicability: gas retention is a retrofit only for a gas-heated dwelling; replacing a heat pump with a
        # boiler is a system downgrade and is flagged rather than ranked.
        f["pathway_applicable"] = ~((pw == "S3") & (f.pre_heating == "heat_pump"))
        for a in R.ARCHETYPES:
            o = post[(a, pw)]
            idx = f.building_type == a
            for k, v in o.items():
                if k == "building_area_m2":
                    continue
                f.loc[idx, k] = v
        frames.append(f)
    cases = pd.concat(frames, ignore_index=True)
    cases["case_id"] = cases.dwelling_id + "_" + cases.pathway
    cases["archetype_id"] = cases.building_type + "_" + cases.period.map(R.PERIOD_GROUP_CODE)
    cases["scenario_id"] = cases.building_type + "_" + cases.pathway
    cases["building_type_label"] = cases.building_type.map(R.ARCHETYPE_LABEL)
    bau = cases.no_retrofit_operational_carbon_60y_tCO2e
    cases["carbon_saving_60y_tCO2e"] = (bau - cases.operational_carbon_60y_tCO2e).round(4)
    cases["net_wlca_vs_bau_tCO2e"] = (cases.life_cycle_60y_tCO2e - bau).round(4)
    sav = cases.carbon_saving_60y_tCO2e
    cases["carbon_payback_years"] = np.where(sav > 0, cases.total_embodied_60y_tCO2e / (sav / H), np.inf)
    cases["carbon_payback_years"] = cases["carbon_payback_years"].round(3)
    cases["repays_within_horizon"] = (sav > 0) & (cases.carbon_payback_years <= H)
    cases["ber_improvement"] = cases.BER_rating_after.map(R.ber_rank) < cases.BER_rating.map(R.ber_rank)
    cases["dataset_version"] = R.DATASET_VERSION
    cases["simulation_id"] = "EP-" + cases.dwelling_id + "|EP-" + cases.building_type + "-" + cases.retrofit_scenario
    cases["bau_source"] = "simulated (own run)"
    for c in ("boiler_eff", "heatpump_cop", "pv_scale"):
        cases[c] = cases[c].round(3)
    op_cols = ["case_id", "dwelling_id", "archetype_id", "scenario_id", "BER_rating", "BER_rating_after", "BER_rating_old", "BER_rating_after_old",
               "delivered_EUI_pre_kWh_m2", "primary_EUI_pre_kWh_m2", "delivered_kwh_yr", "primary_kwh_yr", "delivered_EUI_kWh_m2", "primary_EUI_kWh_m2",
               "operational_carbon_60y_tCO2e", "no_retrofit_operational_carbon_60y_tCO2e", "simulation_id", "emission_factor_source"]
    return dict(cases=cases, archetype=archetype_df, operational=cases[op_cols].copy(), embodied=embodied_df, dwellings=doe)
