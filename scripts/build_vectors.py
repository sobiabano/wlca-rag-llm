"""
build_vectors.py -- Convert the paper's FAISS indexes to the archive wlca_rag_llm.deploy_vector_store expects

Run locally once, after the notebooks4 pipeline has built notebooks4/faiss_indexes/:
    python scripts/build_vectors.py [--faiss ../notebooks4/faiss_indexes] [--out vectors.zip]

For each domain writes <domain>.npy (float16, N x 384) and <domain>_case_ids.parquet (the
case_id each row corresponds to) into one zip archive. Upload the zip to a release / Drive /
Zenodo link and point st.secrets["VECTOR_URL"] (or env VECTOR_URL) at it; on first start the
deployed app downloads it once and caches the vectors into data/wlca_kbs.duckdb, so no CSV or
FAISS file is ever committed to this repository.
"""
from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from wlca_rag_llm import registry as R
from wlca_rag_llm.retrieval.lexical_ranker import DOMAIN_FIELDS

HERE = Path(__file__).resolve().parent.parent
DOMAINS = list(DOMAIN_FIELDS) + ["shared"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--faiss", default=str(HERE.parent / "notebooks4" / "faiss_indexes"))
    ap.add_argument("--out", default=str(HERE / "vectors.zip"))
    a = ap.parse_args()
    import faiss
    src, out = Path(a.faiss), Path(a.out)

    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for d in DOMAINS:
            idx = faiss.read_index(str(src / f"{d}.faiss"))
            vec = idx.reconstruct_n(0, idx.ntotal).astype("float16")
            import pickle
            meta = pickle.loads((src / f"{d}_meta.pkl").read_bytes())
            case_ids = pd.DataFrame({"case_id": meta["case_ids"]})

            buf = io.BytesIO()
            np.save(buf, vec)
            z.writestr(f"{d}.npy", buf.getvalue())

            buf2 = io.BytesIO()
            case_ids.to_parquet(buf2)
            z.writestr(f"{d}_case_ids.parquet", buf2.getvalue())
            print(f"{d}: {vec.shape}")
    print("archive:", out, f"({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
