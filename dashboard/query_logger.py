"""
query_logger.py -- Stakeholder query & error logging for app.py (RetroFit IRL v4)

Backed by DuckDB (logs/dashboard_logs.duckdb), one file holding two tables:
  query_log   -- one row per query run (stakeholder activity)
  error_log   -- one row per exception / warning caught

A single embedded file is simpler to query (ad hoc SQL from the Admin tab) and to ship than a
folder of growing CSVs, and avoids the header-drift issues a hand-appended CSV can develop.
The database file is created automatically on first run.
"""

from __future__ import annotations

import os
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

# ── Log location ──────────────────────────────────────────────────────────────
# Streamlit Community Cloud runs in a read-only app directory.
# Use /tmp/ when on Cloud (STREAMLIT_SHARING_MODE is set), local logs/ otherwise.
if os.environ.get("STREAMLIT_SHARING_MODE") or os.environ.get("HOME", "").startswith("/home/appuser"):
    LOG_DIR = Path("/tmp/retrofit_logs")
else:
    LOG_DIR = Path(__file__).resolve().parents[1] / "logs"

DB_PATH = LOG_DIR / "dashboard_logs.duckdb"

QUERY_COLS = [
    "timestamp_utc", "user_name", "user_key", "input_mode", "config", "query_text",
    "building_type", "pathway", "epc_pre", "epc_post", "objective", "relaxations",
    "n_retrieved", "top_result_case", "top_result_pathway", "top_result_epc_after",
    "top_result_wlca_t", "consistency_check", "citation_status", "latency_ms",
    "dataset_version", "preset_label",
]
ERROR_COLS = ["timestamp_utc", "error_type", "error_message", "traceback_snippet", "context"]


def _connect() -> duckdb.DuckDBPyConnection:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))
    con.execute("CREATE TABLE IF NOT EXISTS query_log ("
                "timestamp_utc TIMESTAMP, user_name VARCHAR, user_key VARCHAR, input_mode VARCHAR, "
                "config VARCHAR, query_text VARCHAR, building_type VARCHAR, pathway VARCHAR, "
                "epc_pre VARCHAR, epc_post VARCHAR, objective VARCHAR, relaxations VARCHAR, "
                "n_retrieved INTEGER, top_result_case VARCHAR, top_result_pathway VARCHAR, "
                "top_result_epc_after VARCHAR, top_result_wlca_t DOUBLE, consistency_check VARCHAR, "
                "citation_status VARCHAR, latency_ms DOUBLE, dataset_version VARCHAR, preset_label VARCHAR)")
    con.execute("CREATE TABLE IF NOT EXISTS error_log ("
                "timestamp_utc TIMESTAMP, error_type VARCHAR, error_message VARCHAR, "
                "traceback_snippet VARCHAR, context VARCHAR)")
    return con


def _now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


# ── Public API ────────────────────────────────────────────────────────────────

def log_query(
    *,
    user_name: str = "",
    user_key: str = "",
    input_mode: str = "",
    config: str = "",
    query_text: str = "",
    output: Any = None,          # wlca_rag.agents.AuditableOutput (or None for login events)
    top_row: Any = None,         # pandas Series of the top retrieved record
    preset_label: str = "",
) -> None:
    row: dict = {c: None for c in QUERY_COLS}
    row.update({
        "timestamp_utc": _now_utc(), "user_name": user_name, "user_key": user_key,
        "input_mode": input_mode, "config": config,
        "query_text": str(query_text).replace("\n", " ").strip(), "preset_label": preset_label,
        "n_retrieved": 0,
    })

    if output is not None:
        c = output.constraints
        row["building_type"] = c.get("categorical", {}).get("building_type", "")
        pw = c.get("retrofit", {}).get("pathway", "")
        row["pathway"] = "|".join(pw) if isinstance(pw, list) else (pw or "")
        row["epc_pre"] = c.get("epc", {}).get("pre", "")
        row["epc_post"] = c.get("epc", {}).get("post", "")
        row["objective"] = c.get("objective", "") or ""
        row["relaxations"] = "|".join(c.get("relaxations", []))
        row["n_retrieved"] = len(output.retrieved)
        w = output.wlca or {}
        row["consistency_check"] = w.get("consistency_check", "")
        row["citation_status"] = ("pass" if output.citation["full_pass"] else "fail") if output.citation else "n/a"
        row["latency_ms"] = round(float(output.latency_ms.get("total_ms", 0.0)), 1)
        row["dataset_version"] = output.dataset_version
        if top_row is not None:
            row["top_result_case"] = top_row.get("case_id", "")
            row["top_result_pathway"] = top_row.get("pathway", "")
            row["top_result_epc_after"] = top_row.get("BER_rating_after", "")
            wl = top_row.get("life_cycle_60y_tCO2e", "")
            row["top_result_wlca_t"] = round(float(wl), 2) if wl != "" else None

    con = _connect()
    try:
        con.execute(f"INSERT INTO query_log VALUES ({', '.join(['?'] * len(QUERY_COLS))})",
                    [row[c] for c in QUERY_COLS])
    finally:
        con.close()


def log_error(error, context: str = "") -> None:
    if isinstance(error, Exception):
        etype, emsg, tb = type(error).__name__, str(error), traceback.format_exc()[:400]
    else:
        etype, emsg, tb = "Warning", str(error), ""

    con = _connect()
    try:
        con.execute("INSERT INTO error_log VALUES (?, ?, ?, ?, ?)",
                    [_now_utc(), etype, emsg.replace("\n", " "), tb.replace("\n", " | "), context])
    finally:
        con.close()


def load_query_log() -> pd.DataFrame:
    con = _connect()
    try:
        return con.execute("SELECT * FROM query_log ORDER BY timestamp_utc DESC").df()
    finally:
        con.close()


def load_error_log() -> pd.DataFrame:
    con = _connect()
    try:
        return con.execute("SELECT * FROM error_log ORDER BY timestamp_utc DESC").df()
    finally:
        con.close()


def query_log_stats() -> dict:
    con = _connect()
    try:
        n = con.execute("SELECT count(*) FROM query_log WHERE input_mode != 'login'").fetchone()[0]
        if not n:
            return {"total_queries": 0, "unique_queries": 0, "share_llm": "—",
                    "last_query_at": "—", "citation_pass_rate": "—", "avg_latency_ms": "—"}
        uniq, share_llm, last, avg_lat = con.execute(
            "SELECT count(DISTINCT query_text), avg(CAST(config = 'llm' AS DOUBLE)), max(timestamp_utc), avg(latency_ms) "
            "FROM query_log WHERE input_mode != 'login'").fetchone()
        scored = con.execute(
            "SELECT avg(CAST(citation_status = 'pass' AS DOUBLE)) FROM query_log "
            "WHERE citation_status IN ('pass', 'fail')").fetchone()[0]
        return {
            "total_queries": n, "unique_queries": uniq,
            "share_llm": f"{share_llm:.0%}" if share_llm is not None else "—",
            "last_query_at": str(last), "citation_pass_rate": f"{scored:.0%}" if scored is not None else "—",
            "avg_latency_ms": round(avg_lat, 1) if avg_lat is not None else "—",
        }
    finally:
        con.close()
