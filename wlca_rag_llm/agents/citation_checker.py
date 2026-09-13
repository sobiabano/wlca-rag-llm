"""CitationChecker (Section 3.6.4, Eq. cite_check). Verifies that every numerical value in a
Synthesiser response is cited to a case_id in the verified bundle and lies within tolerance of a
value actually held in that record; EPC codes are checked by exact match. LLM-RAG configuration only."""
from __future__ import annotations
import re

import numpy as np
import pandas as pd

from .. import registry as R

_NUM_RE = re.compile(r"(?<![A-Za-z0-9_])(-?\d+(?:,\d{3})*(?:\.\d+)?)(?![0-9_])")
_CITE_RE = re.compile(r"\[([A-Za-z0-9_]+)\]")
_BAND_TOKEN_RE = re.compile(r"\b(A0|A|B|C|D|E|F|G)\b(?=\s*(?:rating|band|BER|EPC|\)|,|\.|;|$|\s))")


def check_citations(response: str, bundle: pd.DataFrame, tol: float = R.HYPER["tau_cite_tco2e"]) -> dict:
    """Eq. cite_check.  Every numerical value must (a) carry a case_id citation and (b) lie within
    tau_cite of a value held in that record.  Year labels, service lives and constants are excluded
    by the prompt; residual 4-digit years are excluded here.  EPC codes: exact match."""
    if not response:
        return dict(citation_accuracy=None, verified=0, unverified=0, uncited=0, ber_checked=0, ber_ok=0, full_pass=None, details=[])
    recs = bundle.set_index("case_id")
    numeric_cols = [c for c in recs.columns if recs[c].dtype.kind in "fi"]
    details, verified, unverified, uncited = [], 0, 0, 0
    # list enumerators ("1.", "2)") and "Case ID: <id>" tokens are not reported quantities
    body = re.sub(r"(?m)^\s*\d{1,2}[.)]\s+", "", response)
    body = re.sub(r"Case ID:?\s*[A-Za-z0-9_]+", "", body)
    sentences = re.split(r"(?<=[.;])(?=\s)|\n", body)     # sentence ends: ". " / "; " / newline (not decimal points)
    for sent in sentences:
        cites = _CITE_RE.findall(sent)
        for m in _NUM_RE.finditer(sent):
            raw = m.group(1).replace(",", "")
            try:
                v = float(raw)
            except ValueError:
                continue
            if re.fullmatch(r"(19|20)\d\d", raw) or v in (60.0, 15.0, 25.0, 30.0, 20.0, 4.0, 0.95):
                continue          # year label, service life or physical constant
            if not cites:
                uncited += 1
                details.append((raw, None, "uncited"))
                continue
            ok = False
            for cid in cites:
                if cid in recs.index:
                    vals = recs.loc[cid, numeric_cols].astype(float).values
                    if np.any(np.abs(vals - v) <= tol) or np.any(np.abs(vals * 1000 - v) <= tol * 1000):
                        ok = True
                        break
            verified += ok
            unverified += (not ok)
            details.append((raw, cites, "ok" if ok else "mismatch"))
    ber_ok = ber_checked = 0
    for sent in sentences:
        cites = [c for c in _CITE_RE.findall(sent) if c in recs.index]
        for b in _BAND_TOKEN_RE.findall(sent):
            ber_checked += 1
            if cites and any(b in (recs.loc[c, "BER_rating"], recs.loc[c, "BER_rating_after"]) for c in cites):
                ber_ok += 1
    n = verified + unverified + uncited
    acc = verified / n if n else 1.0
    return dict(citation_accuracy=acc, verified=verified, unverified=unverified, uncited=uncited,
                ber_checked=ber_checked, ber_ok=ber_ok, full_pass=bool(unverified == 0 and uncited == 0 and ber_ok == ber_checked),
                tolerance=tol, details=details)
