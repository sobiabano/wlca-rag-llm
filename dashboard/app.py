"""
RetroFit IRL v4 -- Whole-Life Carbon Dashboard
Stakeholder-facing decision-support interface over the WLCA-KBS (24,000 simulated dwellings x
3 retrofit scenarios, eight-band BER/EPC scale) and the multi-agent RAG model of the paper.
Same pipeline code as wlca_rag_llm/; no carbon value is ever computed by the language model.

Entry point: run via the root-level app.py (streamlit run app.py), which imports this module.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wlca_rag_llm import registry as R
from dashboard.loader import load_system, synthesiser_available, load_parameter_registry
from dashboard.query_logger import log_query, log_error, load_query_log, load_error_log, query_log_stats
from dashboard.sample_queries import SAMPLE_QUERIES, get_labels as _sample_labels
from dashboard.paper_test_queries import PAPER_TEST_QUERIES

PW = ["S1", "S2", "S3"]
PWC = {"S1": "#4C9ED9", "S2": "#3FBFB2", "S3": "#F4A259"}
PWL = {p: R.PATHWAY_META[p]["label"] for p in PW}
AL = R.ARCHETYPE_LABEL

# ══════════════════════════════════════════════════════════
# PAGE CONFIG  (must be FIRST Streamlit call)
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="RetroFit IRL v4 · Whole-Life Carbon",
    page_icon="🍀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════
# STAKEHOLDER LOGIN GATE
# ══════════════════════════════════════════════════════════
def _check_login() -> bool:
    """
    Returns True if the user is already authenticated this session.
    Shows a login screen and returns False if not yet authenticated.
    Codes are stored in st.secrets["stakeholders"] as  name = "CODE".
    Falls back to open access if no secrets are configured (local dev).
    """
    try:
        _valid_codes = dict(st.secrets.get("stakeholders", {}))
    except Exception:
        _valid_codes = {}

    if not _valid_codes:
        return True   # local / no secrets set -- open access

    if st.session_state.get("_authenticated"):
        return True

    st.markdown("""
        <div style='max-width:420px;margin:80px auto 0;padding:40px;
                    border:1px solid #E2E8F0;border-radius:16px;
                    box-shadow:0 4px 24px rgba(0,0,0,0.08);background:#fff;'>
          <div style='text-align:center;margin-bottom:28px;'>
            <div style='font-size:40px;'>🍀</div>
            <div style='font-size:20px;font-weight:700;color:#1E293B;margin-top:8px;'>
              RetroFit IRL Dashboard
            </div>
            <div style='font-size:13px;color:#64748B;margin-top:6px;'>
              Whole-Life Carbon Assessment · Stakeholder Access
            </div>
          </div>
        </div>
    """, unsafe_allow_html=True)

    with st.form("login_form"):
        st.markdown("#### Enter your access details")
        _name = st.text_input("Your name", placeholder="e.g. John Smith")
        _code = st.text_input("Access code", placeholder="RETRO-2026-XX", type="password")
        _submitted = st.form_submit_button("🔓  Access dashboard", use_container_width=True, type="primary")

    if _submitted:
        _code_clean = _code.strip().upper()
        _matched_user = next((name for name, code in _valid_codes.items() if code.strip().upper() == _code_clean), None)
        if _matched_user and _name.strip():
            st.session_state["_authenticated"] = True
            st.session_state["_user_name"] = _name.strip()
            st.session_state["_user_key"] = _matched_user
            try:
                log_query(user_name=_name.strip(), user_key=_matched_user, input_mode="login",
                          query_text=f"LOGIN: {_name.strip()} [{_matched_user}]")
            except Exception:
                pass
            st.rerun()
        else:
            st.error("Name and access code do not match a stakeholder record.")
    return False


if not _check_login():
    st.stop()

# ══════════════════════════════════════════════════════════
# LOAD THE KNOWLEDGE BASE AND THE MULTI-AGENT RAG PIPELINE
# ══════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading knowledge base, vector stores and parametric runs…")
def _load():
    return load_system(semantic=True)


sys_state = _load()
kbs, rag, doe, sens = sys_state["kbs"], sys_state["rag"], sys_state["doe"], sys_state["sens"]
df = kbs["cases"]
_semantic_mode = sys_state["semantic_mode"]

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("## 🍀 RetroFit IRL · v4")
    st.caption(f"{R.DATASET_VERSION} · {df.dwelling_id.nunique():,} simulated dwellings × 3 pathways · 60-yr WLCA")
    st.caption(f"Signed in as **{st.session_state.get('_user_name', 'guest')}**")

    key = st.text_input("OpenAI API key (optional, for the Synthesiser Agent)", type="password",
                        value=os.environ.get("OPENAI_API_KEY", ""))
    if key:
        os.environ["OPENAI_API_KEY"] = key

    _has_semantic = rag.semantic is not None
    config = st.radio(
        "Configuration", ["deterministic", "llm"],
        format_func=lambda c: {"deterministic": "Deterministic (lexical ranking, structured output)",
                                "llm": "LLM-RAG (semantic ranking + GPT synthesis + citation check)"}[c],
        index=(1 if st.query_params.get("config") == "llm" and _has_semantic else 0),
        disabled=not _has_semantic,
    )
    if not _has_semantic:
        st.warning("Semantic ranking unavailable — deterministic configuration only.")
    elif _semantic_mode == "lazy":
        st.caption("Semantic ranking: encoding candidates on the fly (no cached vector table found).")

    st.markdown("---")
    st.markdown("**Guided inputs** (the five stakeholder inputs)")
    g_arch = st.selectbox("Dwelling type", ["(any)"] + list(AL), format_func=lambda a: AL.get(a, a))
    g_start = st.selectbox("Starting point", ["(any)"] + [f"EPC {b}" for b in R.BER_NEW_ORDER] + R.PERIOD_GROUPS)
    g_heat = st.selectbox("Existing heating", ["(any)", "gas boiler", "heat pump"])
    g_target = st.selectbox("Target EPC band (at least)", ["(any)"] + R.BER_NEW_ORDER)
    g_pw = st.selectbox("Pathway", ["(compare all)"] + [f"{p} – {PWL[p]}" for p in PW])
    g_obj = st.selectbox("Priority", ["minimise whole-life carbon", "minimise embodied carbon",
                                      "minimise operational carbon", "maximise fabric performance"])

    st.markdown("---")
    _preset_labels = ["-- type your own query below --"] + _sample_labels()
    _preset_sel = st.selectbox("Practitioner-phrased preset", _preset_labels)
    _preset_query = ""
    if _preset_sel != "-- type your own query below --":
        _idx = _preset_labels.index(_preset_sel) - 1
        _preset_query = SAMPLE_QUERIES[_idx]["query"]

tabs = st.tabs(["🔎 Ask", "📊 Pathway comparison", "🧪 Parametric evidence (24,000 runs)",
                "🧾 Audit & methodology", "🔒 Admin log"])


# ------------------------------------------------------------------ helpers
def guided_query() -> str:
    parts = []
    if g_arch != "(any)":
        parts.append(AL[g_arch].lower() + " house")
    if g_start != "(any)":
        parts.append(f"currently rated {g_start.split()[-1]}" if g_start.startswith("EPC") else f"built {g_start}")
    if g_heat != "(any)":
        parts.append(f"{g_heat} at the moment" if g_heat == "gas boiler" else "already has a heat pump")
    if g_target != "(any)":
        parts.append(f"reach {g_target}")
    if g_pw != "(compare all)":
        p = g_pw.split()[0]
        parts.append({"S1": "with heat pump and solar PV", "S2": "with heat pump only", "S3": "with gas boiler"}[p])
    else:
        parts.append("compare all pathways")
    parts.append({"minimise whole-life carbon": "lowest whole-life carbon", "minimise embodied carbon": "lowest embodied carbon",
                  "minimise operational carbon": "lowest operational carbon", "maximise fabric performance": "best fabric performance"}[g_obj])
    return ", ".join(parts)


def kpi(cols, items):
    for c, (lab, val, unit) in zip(cols, items):
        c.metric(lab, f"{val}{unit}")


def profile(row) -> pd.DataFrame:
    """Year-indexed operational + embodied profile from the registry (display only; totals equal the KBS record)."""
    g = np.array(R.GRID_CARBON_G_PER_KWH) / 1e6
    s = R.ARCHETYPE_SIM[row.building_type]
    if row.pathway == "S3":
        op = s["gas_kwh"] * R.GAS_KG_PER_KWH / 1000 + s["boiler_elec_kwh"] * g
    else:
        op = (s["hp_elec_kwh"] - (s["pv_credit_kwh"] if row.pathway == "S1" else 0)) * g
    op = op * row.operational_carbon_60y_tCO2e / op.sum()
    emb = np.zeros(60)
    emb[0] += row.fabric_embodied_60y_tCO2e
    for comp, col in (("heatpump", "heatpump_embodied_60y_tCO2e"), ("pv", "pv_embodied_60y_tCO2e"), ("boiler", "boiler_embodied_60y_tCO2e")):
        if row[col] > 0:
            idx = R.installs_within_horizon(R.SYSTEM_EMBODIED[comp]["life"])
            for i in idx:
                emb[i] += row[col] / len(idx)
    yrs = np.arange(R.START_YEAR, R.START_YEAR + 60)
    return pd.DataFrame({"Year": yrs, "Operational": op, "Embodied": emb, "Cumulative": np.cumsum(op + emb)})


# ------------------------------------------------------------------ Ask
with tabs[0]:
    c1, c2 = st.columns([3, 1])
    default_q = _preset_query or guided_query()
    q = c1.text_input("Free-text query (or use the guided inputs / preset in the sidebar)",
                      value=st.query_params.get("q", default_q), key="ask_query")
    run = c2.button("Run analysis", type="primary", use_container_width=True) or bool(st.query_params.get("q"))

    if run and q.strip():
        try:
            out = rag.run(q, config=config)
        except Exception as exc:
            log_error(exc, context="rag.run")
            st.error(f"The pipeline could not answer this query: {exc}")
            out = None

        if out is not None:
            c = out.constraints
            st.markdown("#### Interpreted constraints")
            st.json({k: c[k] for k in ("categorical", "retrofit", "epc", "objective", "domains", "relaxations")}, expanded=False)
            if c["relaxations"]:
                st.warning(f"Constraint relaxation applied and recorded: {c['relaxations']} — the answer addresses a broader question than asked.")

            top = pd.DataFrame()
            if out.retrieved:
                top = df[df.case_id.isin(out.retrieved)].set_index("case_id").loc[out.retrieved].reset_index()
                st.markdown(f"#### Verified evidence bundle ({len(top)} records, dataset {out.dataset_version})")
                st.dataframe(top[["case_id", "building_type_label", "period", "pre_heating", "BER_rating",
                                  "pre_retrofit_baseline", "pathway", "pathway_applicable", "BER_rating_after",
                                  "primary_EUI_kWh_m2", "operational_carbon_60y_tCO2e", "total_embodied_60y_tCO2e",
                                  "life_cycle_60y_tCO2e", "no_retrofit_operational_carbon_60y_tCO2e",
                                  "carbon_payback_years"]].round(2), use_container_width=True, hide_index=True)
            else:
                st.warning("No case in the knowledge base satisfies the requested dwelling type and rating; no figure is reported.")

            if out.wlca.get("status") != "no_matching_records":
                w, o, e = out.wlca, out.operational, out.embodied
                kpi(st.columns(5), [
                    ("Whole-life carbon", f"{w['wlca_60y_tCO2e']:.1f}", " t"),
                    ("Operational (60 yr)", f"{o['operational_carbon_60y_tCO2e']:.1f}", " t"),
                    ("Embodied (60 yr)", f"{e['total_embodied_60y_tCO2e']:.1f}", " t"),
                    ("Net saving vs no-retrofit", f"{w['net_carbon_saving_vs_bau_tCO2e']:.1f}", " t"),
                    ("Carbon payback", f"{w['carbon_payback_years'] if w['carbon_payback_years'] else '—'}", " yr"),
                ])
                st.caption(f"Consistency check (Eq. 8): {w['consistency_check']} (deviation {w['consistency_deviation']:.4f} t ≤ τ = {R.HYPER['tau_wlc_tco2e']} t) · "
                          f"Best pathway in bundle: {w['best_pathway']} · Latency: {out.latency_ms['total_ms']:.0f} ms")

            if out.response:
                st.markdown("#### Synthesiser response (GPT-4o-mini, values copied from the records)")
                st.info(out.response)
                cit = out.citation
                st.markdown(f"**CitationChecker** — accuracy {cit['citation_accuracy']:.0%} · verified {cit['verified']} · "
                            f"mismatched {cit['unverified']} · uncited {cit['uncited']} · EPC codes {cit['ber_ok']}/{cit['ber_checked']} · "
                            + ("✅ fully verified" if cit["full_pass"] else "⚠️ flagged and retained for audit"))
            elif config == "llm" and not synthesiser_available():
                st.info("No API key: retrieval, verification and the three carbon agents ran; narrative synthesis skipped.")

            if len(top):
                st.markdown("#### 60-year profile of the top record")
                pr = profile(top.iloc[0])
                fig = go.Figure()
                fig.add_bar(x=pr.Year, y=pr.Operational, name="Operational", marker_color="#4C9ED9")
                fig.add_bar(x=pr.Year, y=pr.Embodied, name="Embodied", marker_color="#F2B134")
                fig.add_scatter(x=pr.Year, y=pr.Cumulative, name="Cumulative", yaxis="y2", line=dict(color="#111", width=2))
                fig.update_layout(barmode="stack", height=320, yaxis_title="tCO₂e / yr",
                                  yaxis2=dict(overlaying="y", side="right", title="cumulative tCO₂e"), margin=dict(t=20))
                st.plotly_chart(fig, use_container_width=True)

            with st.expander("Audit trail (AuditableOutputs record)"):
                st.json(out.to_dict(), expanded=False)

            try:
                log_query(user_name=st.session_state.get("_user_name", ""), user_key=st.session_state.get("_user_key", ""),
                          input_mode=("preset" if q.strip() == _preset_query and _preset_query else "guided" if q.strip() == guided_query() else "free_text"),
                          config=config, query_text=q, output=out, top_row=(top.iloc[0] if len(top) else None),
                          preset_label=(_preset_sel if _preset_sel != "-- type your own query below --" else ""))
            except Exception as exc:
                log_error(exc, context="log_query")

    with st.expander("Free-text queries from the paper's evaluation set (Supplementary Table S14)"):
        for pq in PAPER_TEST_QUERIES:
            st.markdown(f"**{pq['id']}** ({pq['category']}) — {pq['query']}")

# ------------------------------------------------------------------ Pathway comparison
with tabs[1]:
    c1, c2, c3, c4 = st.columns(4)
    arch = c1.selectbox("Archetype", list(AL), format_func=lambda a: AL[a], key="cmp_arch")
    band = c2.selectbox("Pre-retrofit EPC band", R.BER_NEW_ORDER, index=5, key="cmp_band")
    heat = c3.radio("Existing heating", ["boiler", "heat_pump"], horizontal=True,
                    format_func=lambda x: {"boiler": "Gas boiler", "heat_pump": "Heat pump"}[x])
    pvs = c4.radio("Existing PV", ["nopv", "pv"], horizontal=True, format_func=lambda x: {"nopv": "No", "pv": "Yes"}[x])
    sel = df[(df.building_type == arch) & (df.BER_rating == band) & (df.pre_heating == heat) & (df.pre_retrofit_baseline == pvs)]

    if len(sel) == 0:
        st.warning("No simulated dwelling in this cell (A0/A occur only among heat-pump dwellings).")
    else:
        n_dw = sel.dwelling_id.nunique()
        st.caption(f"{n_dw} simulated dwellings in this cell; no-retrofit carbon median "
                  f"{sel.no_retrofit_operational_carbon_60y_tCO2e.median():.1f} tCO₂e "
                  f"(IQR {sel.no_retrofit_operational_carbon_60y_tCO2e.quantile(.25):.1f}–{sel.no_retrofit_operational_carbon_60y_tCO2e.quantile(.75):.1f}). "
                  "Post-retrofit outcomes are invariant to the pre-retrofit state; savings and payback are per dwelling.")
        cols = st.columns(3)
        for c, p in zip(cols, PW):
            r = sel[sel.pathway == p]
            c.markdown(f"**{PWL[p]}**")
            if not r.pathway_applicable.iloc[0]:
                c.info("Not applicable: gas retention is not a retrofit for a heat-pump dwelling.")
                continue
            c.metric("Whole-life carbon", f"{r.life_cycle_60y_tCO2e.iloc[0]:.1f} t",
                     delta=f"{r.net_wlca_vs_bau_tCO2e.median():+.1f} t vs no-retrofit (median)", delta_color="inverse")
            c.metric("EPC achieved", r.BER_rating_after.iloc[0], delta=(f"{r.ber_improvement.mean():.0%} improve"), delta_color="off")
            fin = r.carbon_payback_years[np.isfinite(r.carbon_payback_years)]
            c.metric("Carbon payback (median)", ("never" if len(fin) == 0 else f"{fin.median():.1f} yr"),
                     delta=f"{r.repays_within_horizon.mean():.0%} repay within 60 yr", delta_color="off")
            c.write(f"Operational {r.operational_carbon_60y_tCO2e.iloc[0]:.1f} t · Embodied {r.total_embodied_60y_tCO2e.iloc[0]:.1f} t · "
                   f"Primary EUI {r.primary_EUI_kWh_m2.iloc[0]:.0f} kWh/m²·yr")

        fig = go.Figure()
        for p in PW:
            r = sel[(sel.pathway == p) & sel.pathway_applicable]
            if len(r):
                fig.add_box(y=r.net_wlca_vs_bau_tCO2e, name=PWL[p], marker_color=PWC[p])
        fig.update_layout(height=340, yaxis_title="Net whole-life carbon vs no-retrofit (tCO₂e, per dwelling)", margin=dict(t=30))
        fig.add_hline(y=0, line_dash="dot")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("##### Post-retrofit outcome by archetype (invariant to the pre-retrofit state)")
        piv = df.drop_duplicates(["building_type", "pathway"]).pivot(index="building_type", columns="pathway", values="life_cycle_60y_tCO2e")[PW]
        piv.index = [AL[a] for a in piv.index]
        st.dataframe(piv.round(1).style.highlight_min(axis=1, color="#E2EFDA"), use_container_width=True)

        with st.expander("Component embodied carbon"):
            comp = sel.drop_duplicates("pathway").set_index("pathway")[
                ["wall_embodied_tCO2e", "roof_embodied_tCO2e", "glazing_floor_embodied_tCO2e",
                 "boiler_embodied_60y_tCO2e", "heatpump_embodied_60y_tCO2e", "pv_embodied_60y_tCO2e"]]
            comp.columns = ["Walls", "Roof", "Glazing + floor", "Gas boiler", "Heat pump", "PV"]
            st.plotly_chart(px.bar(comp, barmode="stack", height=300, labels={"value": "tCO₂e", "pathway": "Pathway"}),
                            use_container_width=True)

# ------------------------------------------------------------------ Parametric evidence
with tabs[2]:
    st.markdown("The 24,000 EnergyPlus + LCA runs (Latin-hypercube design over fabric, airtightness, heating and PV) "
               "generate the evidence behind the knowledge base: the parameter ranking that justifies bundling fabric at "
               "the NZEB standard, and the simulated no-retrofit counterfactual per archetype × band × existing-PV state.")
    from wlca_rag_llm.data_generation.parametric_ensemble import PARAM_LABEL, PARAMS

    c1, c2 = st.columns(2)
    s = sens.sort_values("spearman_whole_life", key=abs)
    f1 = go.Figure()
    f1.add_bar(y=s.parameter, x=s.spearman_whole_life, orientation="h", name="whole-life carbon",
              marker_color=["#1565C0" if v < 0 else "#E65100" for v in s.spearman_whole_life])
    f1.add_bar(y=s.parameter, x=s.spearman_embodied, orientation="h", name="embodied carbon", marker_color="grey", opacity=0.5)
    f1.update_layout(title="Spearman rank correlation with 60-yr carbon", height=360, barmode="overlay", margin=dict(t=40))
    c1.plotly_chart(f1, use_container_width=True)

    a2 = c2.selectbox("Archetype", list(AL), format_func=lambda a: AL[a], key="doe_arch")
    d = doe[doe.building_type == a2]
    f2 = px.histogram(d, x=d.primary_EUI_kWh_m2.clip(upper=600), color="heating_system", nbins=60, opacity=0.6,
                      barmode="overlay", labels={"x": "Primary EUI kWh/m²·yr"}, title="Design-space coverage of the EPC scale")
    for b, u in R.BER_NEW_UPPER[:-1]:
        f2.add_vline(x=u, line_dash="dot", annotation_text=b, annotation_position="top")
    for p in PW:
        v = df[(df.building_type == a2) & (df.pathway == p)].primary_EUI_kWh_m2.iloc[0]
        f2.add_vline(x=v, line_color=PWC[p], annotation_text=p, annotation_position="bottom")
    f2.update_layout(height=360, margin=dict(t=40))
    c2.plotly_chart(f2, use_container_width=True)

    xcol = st.selectbox("Explore a parameter", PARAMS, format_func=lambda p: PARAM_LABEL[p])
    dd = d.dropna(subset=[xcol]).sample(min(3000, len(d)), random_state=1)
    f3 = px.scatter(dd, x=xcol, y="op_carbon_60y_registry_tCO2e", color="heating_system", opacity=0.45,
                    labels={xcol: PARAM_LABEL[xcol], "op_carbon_60y_registry_tCO2e": "Operational carbon 60 yr (tCO₂e, registry factors)"}, height=380)
    st.plotly_chart(f3, use_container_width=True)

    st.markdown("##### No-retrofit carbon by pre-retrofit band and existing system (this archetype)")
    dwa = df[(df.pathway == "S2") & (df.building_type == a2)]
    bt = dwa.groupby(["BER_rating", "pre_heating"]).no_retrofit_operational_carbon_60y_tCO2e.agg(["count", "median", "mean"]).unstack().reindex(R.BER_NEW_ORDER)
    st.dataframe(bt.round(1), use_container_width=True)

# ------------------------------------------------------------------ Audit & methodology
with tabs[3]:
    reg = load_parameter_registry()
    st.markdown(f"**Dataset version** {R.DATASET_VERSION} · **Cases** {df.dwelling_id.nunique():,} simulated dwellings "
               "= 4 archetypes × 2 existing heating × 2 existing PV × 1,500 Latin-hypercube samples · "
               f"**Records** {len(df):,} (× 3 pathways)")
    st.markdown("**Emission factors** grid: SEAI/EirGrid CAP24 trajectory, horizon mean {:.1f} gCO₂e/kWh; "
               "gas: {:.1f} gCO₂e/kWh fixed · **Inventory** ICE v3.0, Ecoinvent v3, CIBSE TM65 · "
               "**Boundary** A1–A5, B4, B6; C1–C4 landfill".format(R.GRID_MEAN_KG_PER_KWH * 1000, R.GAS_KG_PER_KWH * 1000))
    st.markdown("**EPC scale (eight-band)** " + " · ".join(
        f"{b} ≤ {u}" if u != float("inf") else "G > 375" for b, u in R.BER_NEW_UPPER) + " kWh/m²·yr")

    with st.expander("Parameter registry (JSON)"):
        st.json(reg, expanded=False)
    with st.expander("Post-retrofit outcomes (12 archetype × pathway combinations)"):
        st.dataframe(df.groupby(["building_type", "pathway"])[
            ["delivered_EUI_kWh_m2", "primary_EUI_kWh_m2", "BER_rating_after", "operational_carbon_60y_tCO2e",
             "fabric_embodied_60y_tCO2e", "system_embodied_60y_tCO2e", "life_cycle_60y_tCO2e"]].first().round(2),
            use_container_width=True)
    with st.expander("Practitioner-phrased presets"):
        for pq in SAMPLE_QUERIES:
            st.markdown(f"**{pq['label']}** — {pq['query']}  \n*{pq['notes']}*")

# ------------------------------------------------------------------ Admin log
with tabs[4]:
    st.markdown("### 🔒 Admin — query & error log")
    _admin_pw = st.text_input("Enter admin password to view logs", type="password", key="admin_pw_input")
    _ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "retrofit2025")

    if _admin_pw == _ADMIN_PASSWORD:
        st.success("Access granted")
        _stats = query_log_stats()
        if _stats:
            _c1, _c2, _c3, _c4, _c5 = st.columns(5)
            _c1.metric("Total queries", _stats.get("total_queries", 0))
            _c2.metric("Unique queries", _stats.get("unique_queries", 0))
            _c3.metric("Share LLM-RAG", _stats.get("share_llm", "—"))
            _c4.metric("Citation pass rate", _stats.get("citation_pass_rate", "—"))
            _c5.metric("Avg latency (ms)", _stats.get("avg_latency_ms", "—"))
        st.divider()

        st.markdown("#### 📋 Query log")
        _qlog = load_query_log()
        if _qlog.empty:
            st.info("No queries logged yet.")
        else:
            st.caption(f"{len(_qlog)} queries recorded (stored in DuckDB: logs/dashboard_logs.duckdb)")
            st.dataframe(_qlog, use_container_width=True, height=400)
            st.download_button("⬇️  Download query_log.csv", data=_qlog.to_csv(index=False).encode("utf-8"),
                               file_name="query_log.csv", mime="text/csv")
        st.divider()

        st.markdown("#### ⚠️ Error log")
        _elog = load_error_log()
        if _elog.empty:
            st.success("No errors recorded.")
        else:
            st.warning(f"{len(_elog)} error(s) recorded")
            st.dataframe(_elog, use_container_width=True, height=300)
            st.download_button("⬇️  Download error_log.csv", data=_elog.to_csv(index=False).encode("utf-8"),
                               file_name="error_log.csv", mime="text/csv")
    elif _admin_pw:
        st.error("Incorrect password.")
