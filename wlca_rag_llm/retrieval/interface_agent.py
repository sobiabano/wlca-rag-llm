"""
Interface Agent (rule based; shared by both configurations): query normalisation, query
interpretation (RetrievalConstraints), EPC rating constraint extraction (Eq. epc_extract) and
structured pre-filtering with fixed-precedence relaxation.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field, asdict
import pandas as pd
from .. import registry as R

_NEW_BANDS = "A0|A|B|C|D|E|F|G"
_OLD_BANDS = "A1|A2|A3|B1|B2|B3|C1|C2|C3|D1|D2|E1|E2"
_BAND_RE = rf"({_OLD_BANDS}|{_NEW_BANDS})"

DOMAINS = ("operational", "embodied", "wlca", "archetype")

_DOMAIN_TERMS = {
    "operational": ["operational", "eui", "energy use", "primary energy", "delivered", "kwh",
                    "energy performance", "operating", "heating demand", "bill", "running", "in-use", "in use"],
    "embodied":    ["embodied", "material", "insulation", "fabric", "panel", "walls", "wall", "roof",
                    "glazing", "window", "component", "breakdown", "installation carbon", "upfront", "embedded"],
    "wlca":        ["whole-life", "whole life", "wlca", "life cycle", "lifecycle", "60-year", "60 year",
                    "payback", "net carbon", "saving", "savings", "business as usual", "bau", "total carbon",
                    "footprint", "lifetime", "over its life", "pay back", "pays back"],
    "archetype":   ["reach", "achieve", "attain", "rating", "ber", "band", "which cases", "all cases",
                    "target", "upgrade to", "improve to", "grade", "label", "zero-emission", "zeb"],
}
_COMPARE_TERMS = ["compare", "versus", " vs", "against", "better", "best", "lowest", "which pathway",
                  "which of the", "or gas", "or heat pump", "shortest", "minimis"]

_OBJECTIVES = {
    "min_embodied":    ["lower embodied", "lowest embodied", "minimise embodied", "minimize embodied", "least embodied"],
    "min_operational": ["lower operational", "lowest operational", "minimise operational", "minimize operational"],
    "min_wlca":        ["lowest wlca", "lowest whole", "minimise whole", "minimize whole", "lowest carbon",
                        "lowest total", "best pathway", "performs best"],
    "max_fabric":      ["fabric performance", "best fabric", "u-value"],
}

_AGE_PATTERNS = [   # year / decade / DEAP-period vocabulary -> construction-period group of the KBS
    (r"before\s*19[0-6]\d|pre[- ]19[0-6]\d|19th century|victorian|edwardian|\b18\d\d\b|\b19[0-5]\d\b|\b196[0-6]\b|\b19[0-5]0s\b|\b1960s\b|pre-regulation", "Before 1967"),
    (r"\b196[7-9]\b|\b197\d\b|\b198[0-2]\b|\b1970s\b", "1967-1982"),
    (r"\b198[3-9]\b|\b199[0-3]\b|\b1980s\b", "1983-1993"),
    (r"\b199[4-9]\b|\b200[0-4]\b|\b1990s\b|\b2000s\b", "1994-2004"),
    (r"\b200[5-9]\b", "2005-2009"),
    (r"\b201[0-3]\b", "2010-2013"),
    (r"\b201[4-9]\b|\b202\d\b|nzeb|new build|2014[- ]onwards", "2014-Onwards"),
]


@dataclass
class RetrievalConstraints:
    query_raw: str
    query_norm: str
    categorical: dict = field(default_factory=dict)     # building_type, age_band, BER_rating
    retrofit: dict = field(default_factory=dict)        # pathway (str or list)
    epc: dict = field(default_factory=dict)             # pre: band, post: band (ordinal >=)
    numerical: dict = field(default_factory=dict)       # area, U-values resolved from ArchetypeData
    envelope: dict = field(default_factory=dict)
    objective: str | None = None
    domains: list = field(default_factory=list)
    compare: bool = False
    relaxations: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
def normalise(query: str) -> str:
    q = query.lower()
    q = q.replace("semi detached", "semi-detached").replace("semidetached", "semi-detached")
    q = re.sub(r"heat[\s-]*pumps?", "heatpump", q)
    q = re.sub(r"solar\s*(pv|panels?|photovoltaics?)|photovoltaics?|\bpv\b|\bsolar\b", "solarpv", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q


def normalise_band(token: str) -> str:
    t = token.upper()
    return R.OLD_TO_NEW.get(t, t)


def extract_band_constraints(query: str) -> tuple[str | None, str | None]:
    """Eq. epc_extract: returns (b_pre, b_post) on the eight-band scale, either may be None."""
    q = query
    pre = post = None
    # "from X to Y"
    m = re.search(rf"from\s+(?:a\s+|an\s+|ber\s+)?{_BAND_RE}\b\s+(?:rating\s+|band\s+)?to\s+(?:a\s+|an\s+|ber\s+)?{_BAND_RE}\b", q, re.I)
    if m:
        return normalise_band(m.group(1)), normalise_band(m.group(2))
    post_verbs = r"(?:reach|reaching|reaches|achieve|achieving|achieves|attain|attaining|target|targeting|" \
                 r"get\s+to|upgrade\s+to|upgraded\s+to|improve\s+to|improved\s+to|bring\s+to|to\s+an?\s+(?=[A-G]))"
    m = re.search(rf"{post_verbs}\s*(?:a|an|the|ber|a\s+ber|an\s+ber)?\s*(?:zero-emission\s+)?{_BAND_RE}\b(?!\s*pathway)(?!\s*\()", q, re.I)
    if not m:   # "get/bring/take <object> to (the) <band>", "get an <band> label/grade"
        m = re.search(rf"(?:get|bring|take)\s+(?:\w+\s+){{0,3}}to\s+(?:the\s+|a\s+|an\s+)?(?:zero-emission\s+)?{_BAND_RE}\b(?!\s*pathway)", q, re.I) \
            or re.search(rf"\b(?:get|earn|obtain)\s+(?:a|an)\s+{_BAND_RE}\s+(?:energy\s+)?(?:label|grade|rating|band)", q, re.I)
    if not m:   # outcome phrasings: "<band> rated cases/records/results/options" (cases that achieve the band)
        m = re.search(rf"\b{_BAND_RE}[\s-]*rated\s+(?:cases?|records?|results?|options?|outcomes?|retrofits?)\b", q, re.I)
    if not m:   # "rated X to Y" / "X to Y" transitions: Y is the target
        m = re.search(rf"(?:rated|from|at)\s+{_BAND_RE}\s+(?:up\s+)?to\s+(?:a\s+|an\s+)?{_BAND_RE}\b", q, re.I)
        if m:
            m = re.search(rf"to\s+(?:a\s+|an\s+)?({_BAND_RE.strip('()')})\b", m.group(0), re.I)
    if not m:   # wish phrasings: "I would like (to reach / to be / to get) an A0", "want an A", "aiming for A0", "looking for a B"
        m = re.search(rf"(?:would\s+like|want|wants|aiming|aim|looking|hoping|hope|prefer|need)\s+(?:to\s+(?:reach|be|get|achieve|have)\s+|for\s+)?(?:a\s+|an\s+|the\s+)?{_BAND_RE}\b(?:\s+(?:rating|band|grade|label|ber|epc))?(?!\s*pathway)", q, re.I)
    if not m:   # threshold phrasings: "into the A band", "at least a B", "B2 or better", "to a B"
        m = re.search(rf"\binto\s+(?:the\s+|a\s+|an\s+)?{_BAND_RE}\s+(?:band|rating|grade)", q, re.I) \
            or re.search(rf"\bat\s+least\s+(?:a\s+|an\s+)?{_BAND_RE}\b", q, re.I) \
            or re.search(rf"\b{_BAND_RE}\s+or\s+better", q, re.I)
    if m:
        post = normalise_band(m.group(1))
    m = re.search(rf"(?:rated|starting\s+at|currently|current|existing|with\s+(?:a\s+)?ber|ber\s+rating\s+of|pre-retrofit\s+ber|"
                  rf"ber|rating\s+of|band)\s*(?:of\s+|is\s+|rating\s+|rated\s+)?{_BAND_RE}\b(?!\s*pathway)(?!\s*\()", q, re.I)
    if not m:   # "<band> rated" / "<band>-rated" (state of the dwelling) / "<band> at present|today|now" / "cert says <band>"
        m = re.search(rf"\b{_BAND_RE}[\s-]*rated\b(?!\s+(?:cases?|records?|results?|options?|outcomes?|retrofits?)\b)", q, re.I) \
            or re.search(rf"\b{_BAND_RE}\s+(?:at\s+present|today|now|currently)\b", q, re.I) \
            or re.search(rf"(?:cert|certificate)\s+says\s+{_BAND_RE}\b", q, re.I)
    if m:
        cand = normalise_band(m.group(1))
        # avoid re-capturing the post target
        if cand != post or re.search(r"start|current|existing|rated\s+(?!cases?|records?|results?|options?|outcomes?)|pre-retrofit", q, re.I):
            pre = cand
    return pre, post


def interpret(query: str) -> RetrievalConstraints:
    qn = normalise(query)
    c = RetrievalConstraints(query_raw=query, query_norm=qn)
    # archetype
    if "semi-detached" in qn or re.search(r"\bsemi\b", qn):
        c.categorical["building_type"] = "semi"
    elif "detached" in qn:
        c.categorical["building_type"] = "detached"
    elif "bungalow" in qn:
        c.categorical["building_type"] = "bungalow"
    elif "terrace" in qn:
        c.categorical["building_type"] = "terraced"
    # age band
    for pat, band in _AGE_PATTERNS:
        if re.search(pat, qn):
            c.categorical["period"] = band
            break
    # existing heating / PV of the dwelling (pre-retrofit state)
    if re.search(r"(?:currently|at the moment|existing|already|now)\s+(?:has|have|on|with|heated by|heating)?\s*(?:a\s+|an\s+)?(?:gas|oil)?\s*boiler|oil heating|gas heating|gas boiler at (?:the moment|present)", qn):
        c.categorical["pre_heating"] = "boiler"
    if re.search(r"already\s+(?:has|have|with)\s+(?:\d+(?:\.\d+)?\s*kwp?\s+of\s+)?solarpv|existing\s+solarpv|solarpv\s+already|solarpv\s+(?:is\s+)?(?:already\s+)?(?:installed|fitted)", qn):
        c.categorical["pre_retrofit_baseline"] = "pv"
    # pathway
    has_hp, has_pv, has_gas = "heatpump" in qn, "solarpv" in qn, bool(re.search(r"boiler|\bgas\b", qn))
    pw_named = re.findall(r"\b(s1|s2|s3)\b(?=\s*(?:pathway|option|route|\(|\+|versus|vs|v|and|or|,|for|in|on|$))", qn)
    pws = set(p.upper() for p in pw_named)
    if has_hp and has_pv and not has_gas:
        pws.add("S1")
    elif has_hp and not has_pv and not has_gas:
        pws.add("S2")
    elif has_gas and not has_hp:
        pws.add("S3")
    elif has_hp and has_gas and not has_pv:
        pws |= {"S2", "S3"}
    elif has_hp and has_gas and has_pv:
        pws |= {"S1", "S3"} if ("solarpv" in qn and "only" not in qn) else {"S1", "S2", "S3"}
    if re.search(r"heatpump\s+only|only\s+heatpump|no\s+solarpv|without\s+solarpv", qn):
        pws.discard("S1"); pws.add("S2")
    if pws:
        c.retrofit["pathway"] = sorted(pws) if len(pws) > 1 else pws.pop()
    # EPC band
    pre, post = extract_band_constraints(query)
    if pre:
        c.epc["pre"] = pre
    if post:
        c.epc["post"] = post
    # objective
    for obj, terms in _OBJECTIVES.items():
        if any(t in qn for t in terms):
            c.objective = obj
            break
    # domains
    c.compare = any(t in qn for t in _COMPARE_TERMS) or isinstance(c.retrofit.get("pathway"), list)
    scores = {d: sum(t in qn for t in terms) for d, terms in _DOMAIN_TERMS.items()}
    if "post" in c.epc or re.search(r"which cases|all cases|cases that|achiev|reach", qn):
        scores["archetype"] += 2
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    top = [d for d, s in ranked if s > 0]
    if c.compare and "wlca" not in top:
        top.append("wlca")
    if not top:
        top = ["wlca"]
    if c.compare:
        c.domains = top[:2] if len(top) > 1 else [top[0], "wlca"] if top[0] != "wlca" else ["wlca", "operational"]
    else:
        c.domains = [top[0]]
    return c


# ---------------------------------------------------------------------------
def _mask(df: pd.DataFrame, c: RetrievalConstraints, use_objective=True, use_pathway=True) -> pd.Series:
    m = pd.Series(True, index=df.index)
    for col, val in c.categorical.items():
        m &= df[col] == val
    if use_pathway and "pathway" in c.retrofit:
        pw = c.retrofit["pathway"]
        m &= df["pathway"].isin(pw if isinstance(pw, list) else [pw])
    if "pre" in c.epc:
        m &= df["BER_rating"] == c.epc["pre"]
    if "post" in c.epc:
        target = R.ber_rank(c.epc["post"])
        m &= df["BER_rating_after"].map(R.ber_rank) <= target
    return m


def prefilter(df: pd.DataFrame, c: RetrievalConstraints, archetype_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Structured pre-filter with fixed-precedence relaxation.
    Precedence: (1) release optimisation objective, (2) release retrofit pathway;
    archetype and EPC constraints are never released."""
    k_min, k_stage1 = R.HYPER["k_min"], R.HYPER["k_stage1"]
    # resolve envelope / numerical constraints from ArchetypeData
    if archetype_df is not None and "building_type" in c.categorical:
        sub = archetype_df[archetype_df.building_type == c.categorical["building_type"]]
        if "period" in c.categorical:
            sub = sub[sub.period == c.categorical["period"]]
        if len(sub):
            r = sub.iloc[0]
            c.numerical = {"building_area_m2": float(r.building_area_m2)}
            c.envelope = {"wall_U_pre_range": [float(sub.wall_U_min.min()), float(sub.wall_U_max.max())],
                          "roof_U_pre_range": [float(sub.roof_U_min.min()), float(sub.roof_U_max.max())],
                          "wall_U_post": float(r.wall_U_post), "roof_U_post": float(r.roof_U_post), "floor_U_post": float(r.floor_U_post)}
    m = _mask(df, c)
    if m.sum() < k_min and c.objective:
        c.relaxations.append("objective")            # objective only affects ranking; recorded for audit
    if m.sum() < k_min and "pathway" in c.retrofit:
        m2 = _mask(df, c, use_pathway=False)
        if m2.sum() > m.sum():
            c.relaxations.append("pathway")
            m = m2
    out = df[m].copy()
    out["_filtered_n"] = int(m.sum())
    return out   # truncation to k_stage1 is applied after ranking (best-scored candidates retained)


def record_satisfies(row: pd.Series, c: RetrievalConstraints) -> tuple[bool, bool]:
    """(pre ok, post ok) for constraint match rate (Eq. constraint_match)."""
    pre_ok = ("pre" not in c.epc) or (row["BER_rating"] == c.epc["pre"])
    post_ok = ("post" not in c.epc) or (R.ber_rank(row["BER_rating_after"]) <= R.ber_rank(c.epc["post"]))
    return pre_ok, post_ok
