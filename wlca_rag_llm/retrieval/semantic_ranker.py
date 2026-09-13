"""
Semantic ranker (Section 3.6.1): one sentence-embedding index per carbon domain, cosine
similarity (Eq. cos_sim). LLM-RAG configuration.

This module holds the FAISS-backed form used to build the paper's evidence (SemanticStores).
The deployed dashboard uses a lighter, FAISS-free backend for the same scoring equation --
see wlca_rag_llm/deploy_vector_store.py -- so this class is used only by scripts/build_vectors.py.
"""
from __future__ import annotations
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .. import registry as R
from .lexical_ranker import DOMAIN_FIELDS, record_text


class SemanticStores:
    """One FAISS index per carbon domain (inner product on normalised embeddings = cosine)."""

    def __init__(self, df: pd.DataFrame, index_dir: Path, model_name: str = R.HYPER["embed_model"]):
        from sentence_transformers import SentenceTransformer
        import faiss
        self.faiss = faiss
        self.df = df.reset_index(drop=True)
        self.index_dir = Path(index_dir)
        self.model = SentenceTransformer(model_name)
        self.index: dict[str, object] = {}
        self.pos = {cid: i for i, cid in enumerate(self.df["case_id"])}

    def build(self, save: bool = True) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        for domain in list(DOMAIN_FIELDS) + ["shared"]:
            if domain == "shared":   # ablation baseline: one undifferentiated index over all domain descriptions
                texts = [" | ".join(record_text(r, d) for d in DOMAIN_FIELDS) for _, r in self.df.iterrows()]
            else:
                texts = [record_text(r, domain) for _, r in self.df.iterrows()]
            idx = None
            for i0 in range(0, len(texts), 8192):               # chunked encoding keeps peak memory low
                vec = self.model.encode(texts[i0:i0 + 8192], batch_size=128, normalize_embeddings=True,
                                        show_progress_bar=False).astype("float32")
                if idx is None:
                    idx = self.faiss.IndexFlatIP(vec.shape[1])
                idx.add(vec)
                del vec
            self.index[domain] = idx
            del texts
            if save:
                self.faiss.write_index(idx, str(self.index_dir / f"{domain}.faiss"))
                with open(self.index_dir / f"{domain}_meta.pkl", "wb") as f:
                    pickle.dump({"case_ids": self.df["case_id"].tolist(), "dataset_version": R.DATASET_VERSION}, f)

    def load(self) -> bool:
        ok = True
        for domain in list(DOMAIN_FIELDS) + ["shared"]:
            p = self.index_dir / f"{domain}.faiss"
            if p.exists():
                self.index[domain] = self.faiss.read_index(str(p))
            else:
                ok = False
        return ok

    def _vectors(self, domain: str) -> np.ndarray:
        """Normalised record embeddings of one domain index (materialised once; exhaustive cosine
        scoring is then a matrix-vector product, which is exact for IndexFlatIP)."""
        if not hasattr(self, "_vec"):
            self._vec = {}
        if domain not in self._vec:
            idx = self.index[domain]
            self._vec[domain] = idx.reconstruct_n(0, idx.ntotal).astype("float32")
        return self._vec[domain]

    def scores(self, query: str, domains: list[str]) -> np.ndarray:
        """Cosine similarity of the query to every record, averaged over routed domains (Eq. cos_sim)."""
        q = self.model.encode([query], normalize_embeddings=True).astype("float32")[0]
        acc = np.zeros(len(self.df), dtype="float32")
        for d in domains:
            acc += self._vectors(d) @ q
        return acc / max(len(domains), 1)

    def score(self, query: str, cand: pd.DataFrame, domains: list[str]) -> pd.Series:
        full = self.scores(query, domains)
        return pd.Series([full[self.pos[c]] for c in cand["case_id"]], index=cand.index)
