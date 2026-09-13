"""
dashboard/loader.py -- Loads the whole-life carbon knowledge base and the multi-agent RAG
pipeline (wlca_rag_llm/) for the Streamlit dashboard.

The knowledge base lives in one DuckDB file (data/wlca_kbs.duckdb): the 24,000-run parametric
ensemble, the sensitivity ranking, the parameter registry and, once fetched, the semantic
vectors. No CSV is read or written by the deployed app.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import duckdb
import pandas as pd

from wlca_rag_llm import registry as R
from wlca_rag_llm.agents.pipeline import MultiAgentRAG
from wlca_rag_llm.data_generation.knowledge_base import build_kbs
from wlca_rag_llm.data_generation.parametric_ensemble import PARAM_LABEL, PARAMS, load_doe

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DB_PATH = DATA_DIR / "wlca_kbs.duckdb"

DATASET_VERSION = R.DATASET_VERSION
PATHWAYS = ["S1", "S2", "S3"]
PATHWAY_LABEL = {p: R.PATHWAY_META[p]["label"] for p in PATHWAYS}
PATHWAY_COLOR = {"S1": "#4C9ED9", "S2": "#3FBFB2", "S3": "#F4A259"}
ARCHETYPE_LABEL = R.ARCHETYPE_LABEL


def load_parameter_registry() -> dict:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        return json.loads(con.execute("SELECT registry_json FROM parameter_registry").fetchone()[0])
    finally:
        con.close()


def load_system(semantic: bool = True) -> dict:
    """Build the knowledge base and the pipeline. Returns a dict so the app can unpack what it needs."""
    kbs = build_kbs(DB_PATH)
    rag = MultiAgentRAG(kbs, build_semantic=False)
    semantic_mode = "off"
    if semantic:
        try:
            from wlca_rag_llm.deploy_vector_store import DeployVectorStore
            rag.semantic = DeployVectorStore(kbs["cases"], DB_PATH)
            semantic_mode = rag.semantic.mode          # "duckdb" (stored embeddings) or "lazy" (encode candidates)
        except Exception as exc:                        # sentence-transformers not installed, download failed, ...
            rag.semantic = None
            semantic_mode = f"off ({type(exc).__name__})"
    doe = load_doe(DB_PATH)
    from wlca_rag_llm.data_generation.parametric_ensemble import load_sensitivity_ranking
    sens = load_sensitivity_ranking(DB_PATH)
    return dict(kbs=kbs, rag=rag, doe=doe, sens=sens, semantic_mode=semantic_mode)


def synthesiser_available() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


__all__ = ["load_system", "synthesiser_available", "load_parameter_registry", "DATASET_VERSION",
           "PATHWAYS", "PATHWAY_LABEL", "PATHWAY_COLOR", "ARCHETYPE_LABEL", "PARAMS", "PARAM_LABEL", "R", "DB_PATH"]
