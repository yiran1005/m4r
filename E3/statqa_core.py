# -*- coding: utf-8 -*-
"""
StatQA core utilities shared by inference and analysis.
"""
import ast
import json
import re

# The 27-method classification list, grouped by task (verbatim from StatQA).
TASKS_TO_METHODS = {
    "Correlation Analysis": [
        "Pearson Correlation Coefficient", "Spearman Correlation Coefficient",
        "Kendall Correlation Coefficient", "Partial Correlation Coefficient",
    ],
    "Distribution Compliance Test": [
        "Anderson-Darling Test", "Shapiro-Wilk Test of Normality",
        "Kolmogorov-Smirnov Test for Normality", "Lilliefors Test",
        "Kolmogorov-Smirnov Test",
        "Kolmogorov-Smirnov Test for Uniform distribution",
        "Kolmogorov-Smirnov Test for Gamma distribution",
        "Kolmogorov-Smirnov Test for Exponential distribution",
    ],
    "Contingency Table Test": [
        "Chi-square Independence Test", "Fisher Exact Test", "Mantel-Haenszel Test",
    ],
    "Descriptive Statistics": [
        "Mean", "Median", "Mode", "Range", "Quartile",
        "Standard Deviation", "Skewness", "Kurtosis",
    ],
    "Variance Test": [
        "Mood Variance Test", "Levene Test", "Bartlett Test", "F-Test for Variance",
    ],
}

ALL_METHODS = [m for ms in TASKS_TO_METHODS.values() for m in ms]
assert len(ALL_METHODS) == 27, len(ALL_METHODS)

TASK_ABBR = {
    "Correlation Analysis": "CA",
    "Distribution Compliance Test": "DCT",
    "Contingency Table Test": "CTT",
    "Descriptive Statistics": "DS",
    "Variance Test": "VT",
}

# The classification list exactly as shown to the model inside the prompt.
PROMPT_CLASSIFICATION = (
    "Correlation Analysis: Pearson Correlation Coefficient, Spearman Correlation Coefficient, "
    "Kendall Correlation Coefficient, Partial Correlation Coefficient;\n"
    "Distribution Compliance Test: Anderson-Darling Test, Shapiro-Wilk Test of Normality, "
    "Kolmogorov-Smirnov Test for Normality, Lilliefors Test, Kolmogorov-Smirnov Test, "
    "Kolmogorov-Smirnov Test for Uniform distribution, Kolmogorov-Smirnov Test for Gamma distribution, "
    "Kolmogorov-Smirnov Test for Exponential distribution;\n"
    "Contingency Table Test: Chi-square Independence Test, Fisher Exact Test, Mantel-Haenszel Test;\n"
    "Descriptive Statistics: Mean, Median, Mode, Range, Quartile, Standard Deviation, Skewness, Kurtosis;\n"
    "Variance Test: Mood Variance Test, Levene Test, Bartlett Test, F-Test for Variance."
)

# Distinctive aliases per method, for matching verdicts in free-form Step-4 prose.
METHOD_ALIASES = {
    "Pearson Correlation Coefficient": ["pearson"],
    "Spearman Correlation Coefficient": ["spearman"],
    "Kendall Correlation Coefficient": ["kendall"],
    "Partial Correlation Coefficient": ["partial correlation"],
    "Anderson-Darling Test": ["anderson-darling", "anderson darling", "anderson"],
    "Shapiro-Wilk Test of Normality": ["shapiro-wilk", "shapiro wilk", "shapiro"],
    "Kolmogorov-Smirnov Test for Normality": ["kolmogorov-smirnov test for normality",
                                              "ks test for normality", "k-s test for normality"],
    "Lilliefors Test": ["lilliefors"],
    "Kolmogorov-Smirnov Test": ["kolmogorov-smirnov test", "ks test", "k-s test"],
    "Kolmogorov-Smirnov Test for Uniform distribution": ["uniform distribution"],
    "Kolmogorov-Smirnov Test for Gamma distribution": ["gamma distribution"],
    "Kolmogorov-Smirnov Test for Exponential distribution": ["exponential distribution"],
    "Chi-square Independence Test": ["chi-square", "chi square", "chi2", "chisq"],
    "Fisher Exact Test": ["fisher"],
    "Mantel-Haenszel Test": ["mantel-haenszel", "mantel haenszel", "mantel"],
    "Mean": ["mean"],
    "Median": ["median"],
    "Mode": ["mode"],
    "Range": ["range"],
    "Quartile": ["quartile"],
    "Standard Deviation": ["standard deviation", "std dev", "std deviation"],
    "Skewness": ["skewness"],
    "Kurtosis": ["kurtosis"],
    "Mood Variance Test": ["mood variance", "mood's", "mood test", "mood"],
    "Levene Test": ["levene"],
    "Bartlett Test": ["bartlett"],
    "F-Test for Variance": ["f-test for variance", "f test for variance", "f-test", "f test"],
}



# Ground-truth extraction (official logic)
def _literal(cell: str):
    return ast.literal_eval(str(cell).replace("false", "False").replace("true", "True"))


def gt_methods(results_cell: str):
    """Ground-truth applicable methods: every method whose conclusion != 'Not applicable'."""
    try:
        results = _literal(results_cell)
        return [r["method"] for r in results if r.get("conclusion") != "Not applicable"]
    except Exception:
        return []


def gt_columns(relevant_column_cell: str):
    """Ground-truth relevant columns (column_header values)."""
    try:
        cols = _literal(relevant_column_cell)
        return [c["column_header"] for c in cols]
    except Exception:
        return []


# Robust final-answer JSON extraction
def _balanced_objects(text: str):
    """Yield every balanced {...} substring in text (handles nesting)."""
    stack, out = [], []
    for i, ch in enumerate(text):
        if ch == "{":
            stack.append(i)
        elif ch == "}" and stack:
            s = stack.pop()
            if not stack:                       # only top-level objects
                out.append(text[s:i + 1])
    return out


def extract_final_answer(text: str):
    """
    Return {"columns": [...], "methods": [...]} parsed from the LAST top-level
    JSON object that contains a 'methods' key. Returns None if none found.
    Tolerant to single quotes and trailing commas.
    """
    if not isinstance(text, str):
        return None
    good = None
    for cand in _balanced_objects(text):
        obj = _try_json(cand)
        if isinstance(obj, dict) and "methods" in obj:
            good = obj                          # keep overwriting -> last wins
    if good is None:
        return None
    cols = good.get("columns", []) or []
    meths = good.get("methods", []) or []
    if not isinstance(cols, list):
        cols = []
    if not isinstance(meths, list):
        meths = []
    return {"columns": [str(c) for c in cols], "methods": [str(m) for m in meths]}


def _try_json(s: str):
    try:
        return json.loads(s)
    except Exception:
        pass
    # second chance: normalise single quotes / trailing commas
    try:
        fixed = s.replace("'", '"')
        fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
        return json.loads(fixed)
    except Exception:
        return None



# Exact-match scoring (official Acc(C,M) policy)
def _norm_set(items):
    return {str(x).lower().strip() for x in items}


def score_selection(pred, gt_cols, gt_meths):
    """
    Replicates StatQA's exact-match scoring.

    Returns dict with:
      valid           : 1 if a final JSON answer was parsed, else 0
      methods_correct : exact-match (1/0) on the methods set
      columns_correct : exact-match (1/0) on the columns set
      overall_correct : 1 iff both methods AND columns are exact matches  (= Acc(C,M))
      m_correct/m_wrong/m_missed : counts for the methods set (for over/under analysis)
    """
    out = {"valid": 0, "methods_correct": 0, "columns_correct": 0, "overall_correct": 0,
           "m_correct": 0, "m_wrong": 0, "m_missed": 0}
    if pred is None:
        return out                              # invalid answer -> all zero
    out["valid"] = 1
    pm, gm = _norm_set(pred["methods"]), _norm_set(gt_meths)
    pc, gc = _norm_set(pred["columns"]), _norm_set(gt_cols)

    out["m_correct"] = len(pm & gm)
    out["m_wrong"] = len(pm - gm)
    out["m_missed"] = len(gm - pm)

    out["methods_correct"] = int(out["m_correct"] > 0 and out["m_wrong"] == 0 and out["m_missed"] == 0)
    c_correct, c_wrong, c_missed = len(pc & gc), len(pc - gc), len(gc - pc)
    out["columns_correct"] = int(c_correct > 0 and c_wrong == 0 and c_missed == 0)
    out["overall_correct"] = int(out["methods_correct"] == 1 and out["columns_correct"] == 1)
    return out


def context_from_prompt(prompt_cell: str) -> str:
    """
    Strip the trailing '### Response: ...' completion trigger from the original
    zero-shot-CoT prompt, leaving the shared context block:
        Task Description / Instruction / Classification List / Column Information /
        Statistical Question.
    The plan-specific instructions are appended by prompts.py.
    """
    txt = str(prompt_cell)
    idx = txt.find("### Response:")
    return (txt[:idx] if idx != -1 else txt).rstrip()