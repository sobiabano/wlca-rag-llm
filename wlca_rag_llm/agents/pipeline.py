"""
Orchestrator (Section 3.6): Interface Agent -> Structured Pre-Filter -> Ranking (lexical |
semantic) -> Grounding & Verification -> Operational / Embodied / Whole-Life Carbon Agents ->
[Synthesiser Agent (GPT) + CitationChecker] -> AuditableOutput.

Two configurations share everything up to ranking:
  config="deterministic": lexical ranking, structured-record output, no LLM.
  config="llm":           domain-routed semantic ranking, GPT narrative, citation check.
No carbon value is ever computed by the language model.
"""
from __future__ import annotations
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd

from .. import registry as R
from ..retrieval.interface_agent import RetrievalConstraints, interpret, prefilter
from ..retrieval.lexical_ranker import LexicalRanker
from .base import AgentOutput
from .grounding import ground_and_verify
from .operational_agent import OperationalCarbonAgent
from .embodied_agent import EmbodiedCarbonAgent
from .wholelife_agent import WholeLifeCarbonAgent
from .synthesiser_agent import SynthesiserAgent
from .citation_checker import check_citations


def _load_env_file() -> None:
    """Load OPENAI_API_KEY from a .env file at the repo root (KEY=value lines), if not already set."""
    env = Path(__file__).resolve().parents[2] / ".env"
    if env.exists() and not os.environ.get("OPENAI_API_KEY"):
        for line in env.read_text().splitlines():
            if line.strip().startswith("OPENAI_API_KEY="):
                os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip().strip('"').strip("'")


_load_env_file()


@dataclass
class AuditableOutput:
    query: str
    config: str
    constraints: dict
    retrieved: list
    grounding: list
    operational: dict
    embodied: dict
    wlca: dict
    response: str
    citation: dict
    latency_ms: dict
    tokens: int
    dataset_version: str = R.DATASET_VERSION
    llm_model: str = R.HYPER["llm_model"]
    embed_model: str = R.HYPER["embed_model"]

    def to_dict(self):
        return asdict(self)


class MultiAgentRAG:
    def __init__(self, kbs: dict, index_dir: Path | None = None, build_semantic: bool = True):
        self.kbs = kbs
        self.df = kbs["cases"].reset_index(drop=True)
        self.lexical = LexicalRanker(self.df)
        self.semantic = None
        if build_semantic:
            from ..retrieval.semantic_ranker import SemanticStores
            self.semantic = SemanticStores(self.df, index_dir or Path("faiss_indexes"))
            if not self.semantic.load():
                self.semantic.build(save=True)
        self.synth = SynthesiserAgent()
        self.op_agent, self.emb_agent, self.wlc_agent = OperationalCarbonAgent(), EmbodiedCarbonAgent(), WholeLifeCarbonAgent()

    # -- retrieval only (used by the evaluation) ---------------------------
    def retrieve(self, query: str, config: str = "llm", k: int | None = None,
                 force_domains: list | None = None, timings: dict | None = None,
                 use_prefilter: bool = True) -> tuple[pd.DataFrame, RetrievalConstraints]:
        t = timings if timings is not None else {}
        t0 = time.perf_counter()
        c = interpret(query)
        cand = prefilter(self.df, c, self.kbs["archetype"]) if use_prefilter else self.df.copy()
        t["constraint_extraction_prefilter_ms"] = (time.perf_counter() - t0) * 1000
        if len(cand) == 0:            # nothing satisfies archetype + EPC even after relaxation: report, never substitute
            c.relaxations.append("no_matching_records")
            return cand, c
        t1 = time.perf_counter()
        if config == "deterministic":
            cand["_score"] = self.lexical.score(c.query_norm, cand, c.objective)
            t["lexical_scoring_ms"] = (time.perf_counter() - t1) * 1000
        else:
            domains = force_domains or c.domains
            cand["_score"] = self.semantic.score(query, cand, domains)
            t["vector_retrieval_ms"] = (time.perf_counter() - t1) * 1000
            t["n_domains"] = len(domains)
        cand = cand.sort_values("_score", ascending=False, kind="mergesort").head(R.HYPER["k_stage1"])
        k = k or R.HYPER["k_final"]
        return cand.head(k), c

    # -- full pipeline -----------------------------------------------------
    def run(self, query: str, config: str = "llm") -> AuditableOutput:
        T0 = time.perf_counter()
        t = {}
        ranked, c = self.retrieve(query, config=config, timings=t)
        tg = time.perf_counter()
        if len(ranked) == 0:          # archetype + EPC constraints unsatisfiable: say so, retrieve nothing
            t["grounding_ms"] = 0.0
            t["total_ms"] = (time.perf_counter() - T0) * 1000
            empty = {"status": "no_matching_records", "constraints": c.to_dict()}
            return AuditableOutput(query=query, config=config, constraints=c.to_dict(), retrieved=[], grounding=[],
                                   operational=empty, embodied=empty, wlca=empty,
                                   response="No case in the knowledge base satisfies the requested dwelling type and rating; no figure is reported.",
                                   citation=None, latency_ms=t, dataset_version=R.DATASET_VERSION)
        bundle, glog = ground_and_verify(ranked, c, self.kbs)
        if len(bundle) == 0:
            bundle = ranked
        bundle = bundle.head(R.HYPER["k_agent"])
        t["grounding_ms"] = (time.perf_counter() - tg) * 1000
        op = self.op_agent.run(bundle)
        emb = self.emb_agent.run(bundle)
        wlc = self.wlc_agent.run(bundle, op, emb)
        if c.epc.get("post"):        # which scenarios reach the requested band (ordinal, Eq. epc_extract)
            tgt = R.ber_rank(c.epc["post"])
            ach = sorted({r["pathway"] for _, r in bundle.iterrows() if R.ber_rank(r["BER_rating_after"]) <= tgt}, key=lambda x: R.PATHWAYS[::-1].index(x))
            wlc.result["target_band"] = c.epc["post"]
            wlc.result["scenarios_achieving_target"] = [f"{p_} ({R.PATHWAY_META[p_]['label']})" for p_ in ach]
        t["carbon_agents_ms"] = op.latency_ms + emb.latency_ms + wlc.latency_ms
        response, cit, tokens = "", {}, 0
        if config == "llm":
            response, gen_ms, tokens = self.synth.run(query, c, bundle, {"operational": op, "embodied": emb, "wlca": wlc})
            t["response_generation_ms"] = gen_ms
            tc = time.perf_counter()
            cit = check_citations(response, bundle)
            t["citation_verification_ms"] = (time.perf_counter() - tc) * 1000
        t["total_ms"] = (time.perf_counter() - T0) * 1000
        t["orchestration_ms"] = t["total_ms"] - sum(v for k_, v in t.items() if k_.endswith("_ms") and k_ not in ("total_ms",))
        return AuditableOutput(query=query, config=config, constraints=c.to_dict(),
                               retrieved=bundle["case_id"].tolist(), grounding=glog[:R.HYPER["k_agent"]],
                               operational=op.result, embodied=emb.result, wlca=wlc.result,
                               response=response, citation={k: v for k, v in cit.items() if k != "details"},
                               latency_ms=t, tokens=tokens)
