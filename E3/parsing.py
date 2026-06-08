# -*- coding: utf-8 -*-
"""
Parse Step-4 verdicts from model output and compute the disconnect metrics
(SSC, Drop Rate, Add Rate in/out) 

These parsers give the AUTOMATIC SSC + label-clarity proxy. For V2/V3 the
Step-4 block is structured and parsing is reliable; for V0/V1 it is free-form
prose and parsing is necessarily heuristic -- which is exactly why the design
also collects HUMAN label-clarity annotations on 50 samples per variant
(analyze_ablation.py exports those sheets). The auto label-clarity column is a
proxy, not the authoritative measure.
"""
import re
import math
from statqa_core import ALL_METHODS, METHOD_ALIASES

_LOWER_TO_CANON = {m.lower(): m for m in ALL_METHODS}

# Exclude keywords are checked FIRST so that 'not applicable' is not read as
# 'applicable'. Order within each list does not matter.
_EXC_PAT = re.compile(
    r"(not applicable|inapplicable|exclud|should not|shouldn'?t|do not use|don'?t use|"
    r"not suitable|unsuitable|not included|not be included|violat|reject|"
    r"does not (satisfy|meet)|: ?no\b|- ?no\b|=\s*no\b|\bno\b)", re.I)
_INC_PAT = re.compile(
    r"(applicable|includ|keep|suitable|satisfi|can be used|appropriate|valid|accept|"
    r": ?yes\b|- ?yes\b|=\s*yes\b|\byes\b|use this)", re.I)


def canonicalize_method(name: str):
    """Map a model-emitted method string to its canonical name, else None."""
    if name is None:
        return None
    s = str(name).lower().strip()
    if s in _LOWER_TO_CANON:
        return _LOWER_TO_CANON[s]
    for canon, aliases in METHOD_ALIASES.items():
        for a in aliases:
            if a == s or a in s:
                return canon
    return None


def _verdict_from_text(seg: str):
    """Decide INCLUDE / EXCLUDE / None for a short text segment."""
    if _EXC_PAT.search(seg):
        return "exclude"
    if _INC_PAT.search(seg):
        return "include"
    return None


# Isolate the Step-4 region
_STEP4_START = re.compile(r"(?im)(^|\n)\s*(step\s*4\b|4\s*[\.\):])")
_STEP4_END = re.compile(r"(?im)(^|\n)\s*(step\s*5\b|5\s*[\.\):]|step\s*6\b|6\s*[\.\):]|final answer)")


def step4_region(text: str) -> str:
    if not isinstance(text, str):
        return ""
    m = _STEP4_START.search(text)
    start = m.start() if m else 0
    rest = text[start:]
    e = _STEP4_END.search(rest, pos=10)        # skip the start marker itself
    return rest[:e.start()] if e else rest


# Verdict parsers
def _parse_table(region: str):
    """Structured parse for V2 markdown tables: | Method | Decision | Reason |.
    Decision cell is taken left-to-right (it precedes the reason cell), so a
    'not applicable' inside a reason cell can't override an explicit decision."""
    verdicts = {}
    for raw in region.splitlines():
        line = raw.strip()
        if not line or "|" not in line:
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        method_cell = next((c for c in cells if canonicalize_method(c)), None)
        decision_cell = next((c for c in cells if _verdict_from_text(c)), None)
        if method_cell and decision_cell:
            canon = canonicalize_method(method_cell)
            v = _verdict_from_text(decision_cell)
            if canon and v:
                verdicts.setdefault(canon, v)
    return verdicts


def _parse_nl(region: str):
    """Heuristic per-method scan over sentences for free-form (V0/V1) Step 4."""
    verdicts = {}
    segments = re.split(r"\n+|(?<=[\.\;])\s+", region)
    for seg in segments:
        v = _verdict_from_text(seg)
        if v is None:
            continue
        seg_l = seg.lower()
        for canon, aliases in METHOD_ALIASES.items():
            if canon in verdicts:
                continue
            if any(a in seg_l for a in aliases):
                verdicts[canon] = v
    return verdicts


def parse_verdicts(text: str, variant: str):
    """
    Return {canonical_method: 'include'|'exclude'} parsed from the Step-4 block.
    NL scan first (recall), structured parse overrides it (precision).
    """
    region = step4_region(text)
    verdicts = _parse_nl(region)
    structured = _parse_table(region)
    verdicts.update(structured)                # structured wins on conflicts
    return verdicts, region


def methods_mentioned(region: str):
    region_l = region.lower()
    found = set()
    for canon, aliases in METHOD_ALIASES.items():
        if any(a in region_l for a in aliases):
            found.add(canon)
    return found


def label_clarity_auto(verdicts: dict, region: str):
    """Proxy for the human fully/partially/narrative annotation."""
    mentioned = methods_mentioned(region)
    n_v = len(verdicts)
    if n_v == 0:
        return "narrative_only"
    if mentioned and verdicts.keys() >= mentioned and n_v >= 2:
        return "fully_labeled"
    return "partially_labeled"


# Disconnect metrics
def compute_ssc_metrics(verdicts: dict, final_methods):
    """
    verdicts      : {canonical_method: 'include'|'exclude'}  (the "summary" S)
    final_methods : list of method strings from the final JSON answer

    Returns dict: ssc, drop_rate, add_rate_in, add_rate_out, plus the raw counts.
    Rates are NaN when their denominator is 0.
    """
    S = set(verdicts)
    S_app = {m for m, v in verdicts.items() if v == "include"}
    S_exc = {m for m, v in verdicts.items() if v == "exclude"}

    F_canon, F_unknown = set(), 0
    for m in (final_methods or []):
        c = canonicalize_method(m)
        if c:
            F_canon.add(c)
        else:
            F_unknown += 1                      # out-of-list selection
    F = F_canon

    # SSC: agreement between verdict and final membership, over S
    consistent = sum(1 for m in S if (m in F) == (verdicts[m] == "include"))
    ssc = consistent / len(S) if S else math.nan

    dropped = {m for m in S_app if m not in F}          # APPLICABLE but not selected
    added_in = {m for m in S_exc if m in F}             # NOT_APPLICABLE but selected
    added_out = (F - S)                                 # selected but never evaluated
    n_added_out = len(added_out) + F_unknown

    drop_rate = len(dropped) / len(S_app) if S_app else math.nan
    add_rate_in = len(added_in) / len(S_exc) if S_exc else math.nan
    add_rate_out = n_added_out / len(F | set()) if (len(F) + F_unknown) else math.nan
    # denominator for add_out is the size of the final selection
    n_final = len(F) + F_unknown
    add_rate_out = n_added_out / n_final if n_final else math.nan

    return {
        "ssc": ssc,
        "drop_rate": drop_rate,
        "add_rate_in": add_rate_in,
        "add_rate_out": add_rate_out,
        "n_summary": len(S),
        "n_summary_applicable": len(S_app),
        "n_summary_notapplicable": len(S_exc),
        "n_final_methods": n_final,
        "n_dropped": len(dropped),
        "n_added_in": len(added_in),
        "n_added_out": n_added_out,
        "n_consistent": consistent,
    }