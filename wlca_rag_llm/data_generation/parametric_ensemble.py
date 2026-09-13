"""
Parametric simulation evidence: the 24,000-run Latin-hypercube design over fabric, airtightness,
heating-system and PV parameters (EnergyPlus + LCA, scripted workflow).

Three uses in the case study:
  1. sensitivity ranking of the free parameters (which justifies bundling fabric at the NZEB
     standard and treating heating / PV as named pathways);
  2. the simulated no-retrofit counterfactual per archetype x pre-retrofit EPC band x
     existing-PV state (Eq. bau_carbon), taken from the boiler runs;
  3. coverage of the eight-band EPC scale by unretrofitted (boiler) dwellings.

Only ENERGY outputs are taken from the runs; carbon is re-evaluated with the registry factors so
that every carbon value in the knowledge base shares one emission-factor set.

Storage: the run table and the sensitivity ranking live in the DuckDB file data/wlca_kbs.duckdb
(tables parametric_runs, sensitivity_ranking) -- no CSV in this repository.
"""
from __future__ import annotations
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .. import registry as R

H = R.ASSESSMENT_HORIZON_YEARS
PARAMS = ["wall_u", "roof_u", "floor_u", "window_u", "ach", "boiler_eff", "heatpump_cop", "pv_scale"]
PARAM_LABEL = {"wall_u": "Wall U-value", "roof_u": "Roof U-value", "floor_u": "Floor U-value", "window_u": "Window U-value",
               "ach": "Air change rate", "boiler_eff": "Boiler efficiency", "heatpump_cop": "Heat pump COP", "pv_scale": "PV capacity (kWp)"}
MIN_RUNS = 10
G_BAND_CAP_EUI = 525.0   # open-ended G band truncated two band-widths above the F/G threshold (375 + 2 x 75)


def load_runs(db_path: Path) -> pd.DataFrame:
    """The 24,000 parametric runs, read from DuckDB (table parametric_runs)."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        return con.execute("SELECT * FROM parametric_runs").df()
    finally:
        con.close()


def load_sensitivity_ranking(db_path: Path) -> pd.DataFrame:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        return con.execute("SELECT * FROM sensitivity_ranking").df()
    finally:
        con.close()


def load_doe(db_path: Path) -> pd.DataFrame:
    """Parametric runs with the eight-band EPC scale and registry-consistent carbon added."""
    df = load_runs(db_path)
    df["BER_new"] = [R.ber_new_from_eui(e, all_electric=(h == "heat_pump")) for e, h in zip(df["primary_EUI_kWh_m2"], df["heating_system"])]
    df["BER_old"] = df["primary_EUI_kWh_m2"].map(R.ber_old_from_eui)
    df["pv_state"] = np.where(df["solar"], "pv", "nopv")
    # carbon re-evaluated with the registry factors (Eq. op_carbon), from the simulated energy
    df["gas_kwh_yr"] = df["gas_kwh"].fillna(0.0)
    df["grid_kwh_yr"] = df["net_electricity_from_utility_kwh"].fillna(df["electricity_kwh"]).clip(lower=0)
    df["op_carbon_60y_registry_tCO2e"] = (df["gas_kwh_yr"] * R.GAS_KG_PER_KWH + df["grid_kwh_yr"] * R.GRID_MEAN_KG_PER_KWH) * H / 1000
    return df
