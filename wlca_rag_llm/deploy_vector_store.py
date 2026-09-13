"""
deploy_vector_store.py -- Semantic record embeddings for the deployed dashboard (LLM-RAG configuration)

The paper pipeline keeps one FAISS index per carbon domain (105 MB each, float32). Those files
exceed the GitHub file limit and the Streamlit Community Cloud memory budget, so the deployed
dashboard uses the SAME embeddings stored in the DuckDB knowledge-base file as float16 BLOBs
(table `vectors`, one row per domain) -- no separate FAISS or CSV files ship with this repository.
Scoring is identical to retrieval.semantic_ranker.SemanticStores.scores(): cosine similarity =
normalised record vector . normalised query vector, averaged over routed domains.

Resolution order
----------------
1. a `vectors` table already present in data/wlca_kbs.duckdb (committed, or fetched once and
   cached there by fetch_and_cache_vectors());
2. an archive downloaded from st.secrets["VECTOR_URL"] / env VECTOR_URL, then cached into the
   same DuckDB file so future runs skip the download;
3. lazy encoding of the pre-filtered candidates with the same sentence-transformer (exact, but
   slower when a query carries no archetype constraint -- capped at LAZY_MAX rows).

Build the archive locally with:  python scripts/build_vectors.py   (reads notebooks4/faiss_indexes)
"""
from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path
from urllib.request import urlopen

import duckdb
import numpy as np
import pandas as pd

from . import registry as R
from .retrieval.lexical_ranker import DOMAIN_FIELDS, record_text

DOMAINS = list(DOMAIN_FIELDS) + ["shared"]
LAZY_MAX = 4000        # candidates encoded on the fly before falling back to the deterministic ranker


def _vector_url() -> str:
    url = os.environ.get("VECTOR_URL", "")
    if not url:
        try:
            import streamlit as st
            url = str(st.secrets.get("VECTOR_URL", ""))
        except Exception:
            url = ""
    return url


def _has_vectors_table(db_path: Path) -> bool:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        n = con.execute("SELECT count(*) FROM information_schema.tables WHERE table_name = 'vectors'").fetchone()[0]
        if not n:
            return False
        return con.execute("SELECT count(DISTINCT domain) FROM vectors").fetchone()[0] == len(DOMAINS)
    except Exception:
        return False
    finally:
        con.close()


def fetch_and_cache_vectors(db_path: Path) -> bool:
    """Ensure data/wlca_kbs.duckdb holds a `vectors` table; download the archive named by
    VECTOR_URL and cache it there on first use. Returns True if vectors are available."""
    if _has_vectors_table(db_path):
        return True
    url = _vector_url()
    if not url:
        return False
    with urlopen(url, timeout=600) as r:
        blob = r.read()
    con = duckdb.connect(str(db_path))
    try:
        con.execute("CREATE TABLE IF NOT EXISTS vectors (domain VARCHAR, case_id VARCHAR, vec BLOB)")
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            for domain in DOMAINS:
                arr = np.load(io.BytesIO(z.read(f"{domain}.npy"))).astype("float16")
                meta = pd.read_parquet(io.BytesIO(z.read(f"{domain}_case_ids.parquet"))) if f"{domain}_case_ids.parquet" in z.namelist() else None
                con.execute("DELETE FROM vectors WHERE domain = ?", [domain])
                con.execute("INSERT INTO vectors SELECT ?, unnest(?), unnest(?)",
                            [domain, meta["case_id"].tolist() if meta is not None else list(range(len(arr))),
                             [v.tobytes() for v in arr]])
        return True
    finally:
        con.close()


class DeployVectorStore:
    """Semantic ranker backed by the DuckDB `vectors` table (float16), or lazy on-the-fly encoding."""

    def __init__(self, df: pd.DataFrame, db_path: Path, model_name: str = R.HYPER["embed_model"]):
        from sentence_transformers import SentenceTransformer
        self.df = df.reset_index(drop=True)
        self.db_path = db_path
        self.model = SentenceTransformer(model_name)
        self.pos = {cid: i for i, cid in enumerate(self.df["case_id"])}
        self._vec: dict[str, np.ndarray] = {}
        self._lazy: dict[str, dict[str, np.ndarray]] = {d: {} for d in DOMAINS}
        self.mode = "duckdb" if fetch_and_cache_vectors(db_path) else "lazy"

    # -- exact path: stored embeddings ---------------------------------------
    def _vectors(self, domain: str) -> np.ndarray:
        if domain not in self._vec:
            con = duckdb.connect(str(self.db_path), read_only=True)
            try:
                rows = con.execute("SELECT case_id, vec FROM vectors WHERE domain = ?", [domain]).fetchall()
            finally:
                con.close()
            order = {cid: i for i, (cid, _) in enumerate(rows)}
            mat = np.zeros((len(self.df), len(np.frombuffer(rows[0][1], dtype="float16"))), dtype="float32")
            for cid, blob in rows:
                if cid in self.pos:
                    mat[self.pos[cid]] = np.frombuffer(blob, dtype="float16").astype("float32")
            self._vec[domain] = mat
        return self._vec[domain]

    def scores(self, query: str, domains: list[str]) -> np.ndarray:
        q = self.model.encode([query], normalize_embeddings=True).astype("float32")[0]
        acc = np.zeros(len(self.df), dtype="float32")
        for d in domains:
            acc += self._vectors(d) @ q
        return acc / max(len(domains), 1)

    # -- fallback path: encode only the candidates ---------------------------
    def _lazy_scores(self, query: str, cand: pd.DataFrame, domains: list[str]) -> pd.Series:
        q = self.model.encode([query], normalize_embeddings=True).astype("float32")[0]
        acc = np.zeros(len(cand), dtype="float32")
        for d in domains:
            cache = self._lazy[d]
            todo = [(i, cid) for i, cid in enumerate(cand["case_id"]) if cid not in cache]
            if todo:
                if d == "shared":
                    texts = [" | ".join(record_text(cand.iloc[i], dd) for dd in DOMAIN_FIELDS) for i, _ in todo]
                else:
                    texts = [record_text(cand.iloc[i], d) for i, _ in todo]
                vec = self.model.encode(texts, batch_size=128, normalize_embeddings=True, show_progress_bar=False).astype("float32")
                for (_, cid), v in zip(todo, vec):
                    cache[cid] = v
            acc += np.array([cache[cid] @ q for cid in cand["case_id"]], dtype="float32")
        return pd.Series(acc / max(len(domains), 1), index=cand.index)

    def score(self, query: str, cand: pd.DataFrame, domains: list[str]) -> pd.Series:
        if self.mode == "duckdb":
            full = self.scores(query, domains)
            return pd.Series([float(full[self.pos[c]]) for c in cand["case_id"]], index=cand.index)
        if len(cand) > LAZY_MAX:
            raise RuntimeError(f"lazy semantic scoring capped at {LAZY_MAX} candidates ({len(cand)} requested)")
        return self._lazy_scores(query, cand, domains)
