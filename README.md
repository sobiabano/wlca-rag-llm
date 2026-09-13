# wlca-rag-llm — RetroFit IRL v4 dashboard

Stakeholder-facing decision-support interface over the WLCA-KBS (24,000 simulated dwellings × 3 retrofit scenarios, eight-band EPC/BER scale) and the multi-agent RAG model of the paper. — the knowledge base, the query logs and (once fetched) the semantic vectors all live in DuckDB.

## Layout
```
app.py                       thin Streamlit Cloud entry point (imports dashboard.app)
dashboard/                   the dashboard application
  app.py                       login gate, sidebar, five tabs (Ask, Pathway comparison, Parametric evidence, Audit, Admin log)
  loader.py                    loads the knowledge base and the MultiAgentRAG pipeline (cached resource)
  query_logger.py               query / error logging, DuckDB-backed (logs/dashboard_logs.duckdb)
  sample_queries.py             10 practitioner-phrased presets (the paper's practitioner queries)
  paper_test_queries.py         10 free-text queries from the evaluation set
wlca_rag_llm/                 the model, one file per agent
  registry.py                   constants, EPC scale, hyper-parameters, emission factors
  data_generation/
    parametric_ensemble.py        loads the 24,000-run design of experiments from DuckDB
    knowledge_base.py             builds the 72,000-record case set (ArchetypeData / OperationalData / EmbodiedData)
  retrieval/
    interface_agent.py            rule-based query interpretation + structured pre-filter
    lexical_ranker.py             deterministic-configuration ranker
    semantic_ranker.py            FAISS ranker used to build the paper's evidence (scripts/build_vectors.py only)
  agents/
    grounding.py                  completeness / consistency / provenance checks
    operational_agent.py, embodied_agent.py, wholelife_agent.py   rule-based carbon agents
    synthesiser_agent.py          the only generative step (GPT-4o-mini); LLM-RAG configuration only
    citation_checker.py           verifies every value the Synthesiser states
    pipeline.py                   MultiAgentRAG orchestrator + AuditableOutput
  deploy_vector_store.py        DuckDB-backed semantic embeddings for deployment (no FAISS on Cloud)
scripts/
  build_vectors.py               builds the vector archive from notebooks4/faiss_indexes (run locally)
data/
  wlca_kbs.duckdb                24,000 runs + sensitivity ranking + parameter registry (+ cached vectors)
logs/
  dashboard_logs.duckdb          query_log / error_log tables, created at first run
```

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
The 72,000-record knowledge base is rebuilt from `data/wlca_kbs.duckdb` at start-up (about 10 s).

## Configurations
| | Deterministic | LLM-RAG |
|---|---|---|
| Ranking | lexical (Eq. 2–3) | domain-routed sentence embeddings (Eq. 4) |
| Needs | nothing | cached vectors or `VECTOR_URL`; `OPENAI_API_KEY` for the Synthesiser |
| Output | structured tables | narrative + CitationChecker, plus the same tables |

Without vectors the LLM-RAG ranker encodes the pre-filtered candidates on the fly (exact, slower; capped at 4,000 candidates). Without an API key the narrative step is skipped and everything else runs.

## Semantic vectors
Build once from the paper's FAISS indexes and upload the result somewhere reachable by URL:
```bash
python scripts/build_vectors.py --faiss ../notebooks4/faiss_indexes --out vectors.zip
```
Point `VECTOR_URL` (env var or Streamlit secret) at the uploaded `vectors.zip`. On first start the app downloads it once and caches the embeddings into `data/wlca_kbs.duckdb`, so no separate vector file is ever committed to this repository.

## Secrets (Streamlit Cloud → App settings → Secrets)
```toml
OPENAI_API_KEY = "sk-..."
VECTOR_URL = "https://.../vectors.zip"
ADMIN_PASSWORD = "..."

[stakeholders]                                 # omit to allow open access
john_seai = "RETRO-2026-01"
```

## Deploy
Streamlit Community Cloud, main file path `app.py`. Access restricted to invited stakeholders when `[stakeholders]` is set. Python 3.11 or later.
