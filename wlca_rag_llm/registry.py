"""
Parameter Registry for the WLCA-KBS (Section 3.5 of the paper).

Every constant that enters a reported carbon value lives here, under a single
dataset version, so that each result is traceable to the assumptions that
produced it.  Nothing in this module is computed; it is the declared evidence
base of the case study (Appendix: Key Assumptions and System Boundary).
"""
from __future__ import annotations
import json
from pathlib import Path

DATASET_VERSION = "WLCA-KBS-IRL v4.3"
ASSESSMENT_HORIZON_YEARS = 60
START_YEAR = 2026

# ---------------------------------------------------------------------------
# BER rating scales  (kWh primary energy / m2 / yr)
# ---------------------------------------------------------------------------
# NEW 8-band scale (SEAI 2025 label, image supplied by the user):
#   A0 <= 42 (ZEB criteria), A <= 75, B <= 150, C <= 225, D <= 275,
#   E <= 325, F <= 375, G > 375.
BER_NEW_UPPER = [("A0", 42), ("A", 75), ("B", 150), ("C", 225),
                 ("D", 275), ("E", 325), ("F", 375), ("G", float("inf"))]
BER_NEW_ORDER = [b for b, _ in BER_NEW_UPPER]           # best -> worst

# Band midpoints used for the business-as-usual counterfactual (Eq. bau_carbon).
# G is open-ended; 450 kWh/m2/yr is adopted (375 + one band-width of 75),
# consistent with the F/G boundary of the legacy 15-band scale.
BER_NEW_MIDPOINT = {"A0": 21.0, "A": 58.5, "B": 112.5, "C": 187.5,
                    "D": 250.0, "E": 300.0, "F": 350.0, "G": 450.0}

# LEGACY 15-band scale (DEAP 4.2), retained for record purposes only.
BER_OLD_UPPER = [("A1", 25), ("A2", 50), ("A3", 75), ("B1", 100), ("B2", 125),
                 ("B3", 150), ("C1", 175), ("C2", 200), ("C3", 225), ("D1", 260),
                 ("D2", 300), ("E1", 340), ("E2", 380), ("F", 450), ("G", float("inf"))]
BER_OLD_ORDER = [b for b, _ in BER_OLD_UPPER]

# Mapping from a legacy sub-band named in a query to the new band it falls in.
OLD_TO_NEW = {"A1": "A0", "A2": "A", "A3": "A", "B1": "B", "B2": "B", "B3": "B",
              "C1": "C", "C2": "C", "C3": "C", "D1": "D", "D2": "D",
              "E1": "E", "E2": "E", "F": "F", "G": "G"}


def ber_new_from_eui(eui: float, all_electric: bool = True) -> str:
    """Eight-band label from primary EUI.  A0 (zero-emission building) additionally requires that no fossil fuel
    is used on site: a dwelling within the A0 energy threshold that still burns gas or oil is rated A."""
    for band, upper in BER_NEW_UPPER:
        if eui <= upper:
            return "A" if (band == "A0" and not all_electric) else band
    return "G"


def ber_old_from_eui(eui: float) -> str:
    for band, upper in BER_OLD_UPPER:
        if eui <= upper:
            return band
    return "G"


def ber_rank(band: str) -> int:
    """0 = best (A0).  Used for the ordinal post-retrofit constraint."""
    return BER_NEW_ORDER.index(band)


# ---------------------------------------------------------------------------
# Design-space dimensions
# ---------------------------------------------------------------------------
ARCHETYPES = ["bungalow", "detached", "semi", "terraced"]
ARCHETYPE_LABEL = {"bungalow": "Bungalow", "detached": "Detached",
                   "semi": "Semi-detached", "terraced": "Terraced"}
# CSO 2022 share of national stock covered by the four archetypes
ARCHETYPE_STOCK_SHARE = {"semi": 0.30, "detached": 0.30, "terraced": 0.25, "bungalow": 0.10}

DEAP_AGE_BANDS = ["Before 1900", "1900-1929", "1930-1949", "1950-1966", "1967-1977",
                  "1978-1982", "1983-1993", "1994-1999", "2000-2004", "2005-2009",
                  "2010-2013", "2014-Onwards"]
AGE_BAND_CODE = dict(zip(DEAP_AGE_BANDS, "ABCDEFGHIJKL"))
# Representative pre-retrofit U-values (roof, wall, floor) W/m2K per DEAP band
AGE_BAND_UVALUES = {
    "Before 1900": (2.30, 2.10, 1.20), "1900-1929": (2.30, 2.10, 1.20),
    "1930-1949": (2.30, 2.10, 1.20), "1950-1966": (2.30, 2.10, 1.20),
    "1967-1977": (0.40, 1.10, 0.60), "1978-1982": (0.40, 0.60, 0.60),
    "1983-1993": (0.35, 0.55, 0.45), "1994-1999": (0.35, 0.55, 0.45),
    "2000-2004": (0.25, 0.37, 0.37), "2005-2009": (0.22, 0.27, 0.25),
    "2010-2013": (0.20, 0.21, 0.21), "2014-Onwards": (0.18, 0.20, 0.20),
}
POST_RETROFIT_UVALUES = {"wall": 0.15, "roof": 0.10, "floor": 0.15, "window": 0.80}
# Specification of the twelve archetype x scenario post-retrofit simulations whose energy outputs are held in
# ARCHETYPE_SIM (Appendix table).  Fields marked FILL must be completed from the simulation log before submission.
SCENARIO_RUN_SPEC = dict(
    engine="EnergyPlus (FILL: version), runs generated and executed by a Python scripted workflow", weather="Met Eireann Design Reference Year, Dublin (EPW)",
    envelope="NZEB deep-retrofit specification: U = 0.15 / 0.10 / 0.15 / 0.80 W/m2K (wall / roof / floor / window)",
    airtightness="3 m3/(h m2) at 50 Pa (equivalent infiltration used in the model: FILL ach)",
    heating={"S1": "ASHP, seasonal COP 4.0", "S2": "ASHP, seasonal COP 4.0", "S3": "gas condensing boiler, seasonal efficiency 0.95"},
    pv={"S1": "4 kWp roof-mounted, south-facing; credited generation = hourly self-consumed fraction from the simulation (coincidence of array output and dwelling demand); export not credited", "S2": "none", "S3": "none"},
    occupancy="DEAP standard occupancy and internal gains", geometry="archetype floor areas 85.9 / 91.7 / 107.7 / 130.8 m2 (median of the BER segment)",
    run_ids={a: {pw: f"EP-{a}-{PATHWAY_META[pw]['scenario']}" for pw in PATHWAYS} for a in ARCHETYPES} if False else "EP-<archetype>-<scenario id>",
    same_model_as_ensemble="FILL: yes/no - state whether the twelve runs were executed with the same model files and version as the 24,000-run ensemble",
)
POST_RETROFIT_AIR_PERMEABILITY = 3.0   # m3/h/m2 @ 50 Pa

PRE_PV_STATES = ["nopv", "pv"]
PRE_HEATING = ["boiler", "heat_pump"]

# Construction-period groups of the simulation ensemble.  The ensemble labels each sampled dwelling with
# one of seven DEAP periods; DEAP periods sharing a regulatory envelope regime are merged into one group
# because envelope properties are sampled across the full regulatory range within every group.
PERIOD_GROUPS = ["Before 1967", "1967-1982", "1983-1993", "1994-2004", "2005-2009", "2010-2013", "2014-Onwards"]
DEAP_TO_GROUP = {"Before 1900": "Before 1967", "1900-1929": "Before 1967", "1930-1949": "Before 1967", "1950-1966": "Before 1967",
                 "1967-1977": "1967-1982", "1978-1982": "1967-1982", "1983-1993": "1983-1993", "1994-1999": "1994-2004",
                 "2000-2004": "1994-2004", "2005-2009": "2005-2009", "2010-2013": "2010-2013", "2014-Onwards": "2014-Onwards"}
PERIOD_GROUP_CODE = dict(zip(PERIOD_GROUPS, "ABCDEFG"))

# Pathway codes S1-S3 name the measure package (ordered by increasing mean whole-life carbon); they are not rating bands.
PATHWAYS = ["S3", "S2", "S1"]
PATHWAY_META = {
    "S3": dict(renovation_type="fabric_boiler",      heating="Gas boiler (95% seasonal)",
               pv=False, label="S3 (boiler upgrade)",     scenario="retrofit_s3_boiler_0p95_pv_off"),
    "S2": dict(renovation_type="fabric_heatpump",    heating="Air-source heat pump (COP 4.0)",
               pv=False, label="S2 (heat pump)",      scenario="retrofit_s2_heatpump_cop4_pv_off"),
    "S1": dict(renovation_type="fabric_heatpump_pv", heating="Air-source heat pump (COP 4.0)",
               pv=True,  label="S1 (heat pump + PV)", scenario="retrofit_s1_heatpump_cop4_pv_on"),
}
RENOVATION_TO_PATHWAY = {v["renovation_type"]: k for k, v in PATHWAY_META.items()}
PV_ARRAY_KWP = 4.0
HEAT_PUMP_COP = 4.0
BOILER_EFFICIENCY = 0.95

# ---------------------------------------------------------------------------
# Simulation-derived archetype quantities (EnergyPlus, Dublin DRY, Python scripted workflow)
# Annual delivered energy per dwelling in kWh/yr, post-retrofit fabric.
# ---------------------------------------------------------------------------
ARCHETYPE_SIM = {
    #            floor area | HP elec | boiler gas | boiler elec residual | PV credited (self-consumed)
    "bungalow": dict(area_m2=85.9,  hp_elec_kwh=5681, gas_kwh=6476, boiler_elec_kwh=1668, pv_credit_kwh=3661),
    "detached": dict(area_m2=130.8, hp_elec_kwh=4827, gas_kwh=8190, boiler_elec_kwh=2187, pv_credit_kwh=1851),
    "semi":     dict(area_m2=107.7, hp_elec_kwh=4098, gas_kwh=6723, boiler_elec_kwh=1911, pv_credit_kwh=1298),
    "terraced": dict(area_m2=91.7,  hp_elec_kwh=3813, gas_kwh=5176, boiler_elec_kwh=1805, pv_credit_kwh=1826),
}

# ---------------------------------------------------------------------------
# Emission and primary-energy factors
# ---------------------------------------------------------------------------
# SEAI / EirGrid grid carbon intensity trajectory 2026-2085 (gCO2e/kWh)
GRID_CARBON_G_PER_KWH = [
    256, 239, 191, 159, 134, 112, 98, 95, 80, 79, 82, 82, 69, 65, 71, 67, 58, 57, 56, 60,
    58, 64, 51, 54, 53, 54, 53, 52, 52, 51, 50, 49, 48, 48, 47, 46, 45, 44, 44, 43,
    42, 41, 40, 40, 39, 38, 37, 36, 36, 35, 34, 33, 32, 32, 31, 30, 29, 28, 28, 27,
]
assert len(GRID_CARBON_G_PER_KWH) == ASSESSMENT_HORIZON_YEARS
GRID_MEAN_KG_PER_KWH = sum(GRID_CARBON_G_PER_KWH) / len(GRID_CARBON_G_PER_KWH) / 1000.0   # 0.0634
GAS_KG_PER_KWH = 0.2029           # SEAI conversion factors, fixed over horizon
PEF_ELECTRICITY = 1.75            # DEAP primary energy factor, grid electricity
PEF_GAS = 1.10                    # DEAP primary energy factor, natural gas

# ---------------------------------------------------------------------------
# Embodied carbon (tCO2e per dwelling, 60-yr, A1-A5 + B4; C1-C4 landfill = 0)
# ---------------------------------------------------------------------------
# Fabric package is identical across pathways -> archetype-specific only.
# Sources: ICE v3.0 (A1-A3), Ecoinvent v3 (where no ICE factor), quantity take-off
# from the reconstructed period assemblies.  Glazing includes one replacement (yr 30).
FABRIC_EMBODIED = {
    #             wall   roof   glazing+floor (combined inventory line)
    "bungalow": dict(wall=0.7824, roof=0.5332, glazing_floor=3.9410),
    "detached": dict(wall=0.2248, roof=0.2807, glazing_floor=2.9482),
    "semi":     dict(wall=0.1004, roof=0.1378, glazing_floor=2.3251),
    "terraced": dict(wall=0.2248, roof=0.2807, glazing_floor=2.9482),
}
# Mechanical / renewable systems: per-install embodied (tCO2e) and service life (yr)
SYSTEM_EMBODIED = {
    "boiler":   dict(per_install=0.90, life=20, source="ICE v3.0 / Ecoinvent"),
    "heatpump": dict(per_install=2.20, life=15, source="CIBSE TM65"),
    "pv":       dict(per_install=2.00, life=25, source="Ecoinvent v3 (4 kWp mono-Si)"),
}
COMPONENT_LIFE = {"wall": 60, "roof": 60, "floor": 60, "glazing": 30,
                  "boiler": 20, "heatpump": 15, "pv": 25}


def installs_within_horizon(life: int, horizon: int = ASSESSMENT_HORIZON_YEARS) -> list[int]:
    """Years (0-indexed from retrofit) at which a component is installed/replaced.
    A replacement falling exactly on the horizon boundary is not counted."""
    return [y for y in range(0, horizon, life)]


# ---------------------------------------------------------------------------
# Business-as-usual calibration factors kappa_arch (Eq. arch_calibration)
# ---------------------------------------------------------------------------
KAPPA_ARCH = {"bungalow": 0.1247, "detached": 0.1230, "semi": 0.1197, "terraced": 0.1118}

# ---------------------------------------------------------------------------
# Retrieval hyper-parameters (Table hyperparams)
# ---------------------------------------------------------------------------
HYPER = dict(
    b_keyword_bonus=0.25, gamma_objective=0.80, alpha_wlc=0.25,
    k_stage1=150, k_min=10, k_agent=10, k_final=100,   # k_min = k_agent: relax only when the agent bundle cannot be filled
    tau_wlc_tco2e=0.010, tau_cite_tco2e=0.5,
    embed_model="all-MiniLM-L6-v2", llm_model="gpt-4o-mini",
)


def registry_dict() -> dict:
    return dict(
        dataset_version=DATASET_VERSION, horizon_years=ASSESSMENT_HORIZON_YEARS,
        start_year=START_YEAR, ber_new_upper=BER_NEW_UPPER, ber_new_midpoint=BER_NEW_MIDPOINT,
        ber_old_upper=BER_OLD_UPPER, archetypes=ARCHETYPES, age_bands=DEAP_AGE_BANDS,
        age_band_uvalues=AGE_BAND_UVALUES, post_retrofit_uvalues=POST_RETROFIT_UVALUES, scenario_run_spec=SCENARIO_RUN_SPEC,
        pathways=PATHWAY_META, archetype_sim=ARCHETYPE_SIM,
        grid_carbon_g_per_kwh=GRID_CARBON_G_PER_KWH, grid_mean_kg_per_kwh=GRID_MEAN_KG_PER_KWH,
        gas_kg_per_kwh=GAS_KG_PER_KWH, pef_electricity=PEF_ELECTRICITY, pef_gas=PEF_GAS,
        fabric_embodied=FABRIC_EMBODIED, system_embodied=SYSTEM_EMBODIED,
        component_life=COMPONENT_LIFE, kappa_arch=KAPPA_ARCH, hyper=HYPER,
    )


def write_registry(path: Path) -> None:
    path.write_text(json.dumps(registry_dict(), indent=2, default=str))
