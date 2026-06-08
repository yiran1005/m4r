"""
Classification list + task-category candidate sets for StatQA.
"""

# --- Per-category method lists 
CORRELATION_ANALYSIS = [
    "Pearson Correlation Coefficient",
    "Spearman Correlation Coefficient",
    "Kendall Correlation Coefficient",
    "Partial Correlation Coefficient",
]

DISTRIBUTION_COMPLIANCE_TEST = [
    "Anderson-Darling Test",
    "Shapiro-Wilk Test of Normality",
    "Kolmogorov-Smirnov Test for Normality",
    "Lilliefors Test",
    "Kolmogorov-Smirnov Test",
    "Kolmogorov-Smirnov Test for Uniform distribution",
    "Kolmogorov-Smirnov Test for Gamma distribution",
    "Kolmogorov-Smirnov Test for Exponential distribution",
]

CONTINGENCY_TABLE_TEST = [
    "Chi-square Independence Test",
    "Fisher Exact Test",
    "Mantel-Haenszel Test",
]

DESCRIPTIVE_STATISTICS = [
    "Mean",
    "Median",
    "Mode",
    "Range",
    "Quartile",
    "Standard Deviation",
    "Skewness",
    "Kurtosis",
]

VARIANCE_TEST = [
    "Mood Variance Test",
    "Levene Test",
    "Bartlett Test",
    "F-Test for Variance",
]

# Flat list of all 27 (used for validation and out-of-summary detection).
CLASSIFICATION_LIST = (
    CORRELATION_ANALYSIS
    + DISTRIBUTION_COMPLIANCE_TEST
    + CONTINGENCY_TABLE_TEST
    + DESCRIPTIVE_STATISTICS
    + VARIANCE_TEST
)
assert len(CLASSIFICATION_LIST) == 27, len(CLASSIFICATION_LIST)

#Task category  
TASK_CANDIDATES = {
    "Correlation Analysis": CORRELATION_ANALYSIS,
    "Distribution Compliance Test": DISTRIBUTION_COMPLIANCE_TEST,
    "Contingency Table Test": CONTINGENCY_TABLE_TEST,
    "Descriptive Statistics": DESCRIPTIVE_STATISTICS,
    "Variance Test": VARIANCE_TEST,
}

# Candidate-set sizes documented in 5.2.2
_EXPECTED_SIZES = {
    "Correlation Analysis": 4,
    "Distribution Compliance Test": 8,
    "Contingency Table Test": 3,
    "Descriptive Statistics": 8,
    "Variance Test": 4,
}
for _task, _methods in TASK_CANDIDATES.items():
    assert len(_methods) == _EXPECTED_SIZES[_task], (
        f"{_task}: expected {_EXPECTED_SIZES[_task]} candidates, "
        f"got {len(_methods)}"
    )


# Helpers 
# Map any method name to its task category (for Step-3 correctness checks).
METHOD_TO_TASK = {}
for _task, _methods in TASK_CANDIDATES.items():
    for _m in _methods:
        METHOD_TO_TASK[_m] = _task


def candidates_for_task(task: str) -> list[str]:
    """Return the candidate method list for a task category.

    Raises KeyError with a helpful message if the task string is unknown,
    so a typo in the CSV surfaces immediately rather than silently yielding
    an empty candidate set.
    """
    if task not in TASK_CANDIDATES:
        raise KeyError(
            f"Unknown task category {task!r}. "
            f"Known: {sorted(TASK_CANDIDATES)}"
        )
    return TASK_CANDIDATES[task]


# Keyword-based task normalisation.
#
# The model rarely echoes the canonical task name in Step 3; it paraphrases
# ("Testing Normality", "Comparing Variances", "Association between two
# categorical variables", etc.). A fixed alias table mis-classified ~72% of
# these as Step-3 errors, contradicting the C-Open finding of ~0 task-level
# conflict (Section 5.2.5.2). We instead scan for characteristic keywords.
#
# RULE ORDER MATTERS — more specific categories are checked first:
#   1. Contingency Table Test  — "categorical", "chi-square", "independence",
#      "contingency", method names (Fisher/Mantel), and the StatQA phrasing
#      "control variable". Checked before Correlation so that "association
#      between categorical variables" routes to CTT, not Correlation.
#   2. Correlation Analysis    — "correlation", "relationship between",
#      "monotonic", "linear association".
#   3. Variance Test           — variance/variability/dispersion/spread,
#      "homogeneity", "comparing means" (a t-test framing seen in S50).
#      Checked before Distribution so "comparing two means from normally
#      distributed samples" routes to VT, not DCT.
#   4. Distribution Compliance — normal/distribution/goodness-of-fit and the
#      DCT method names (Shapiro/Anderson/Kolmogorov/Lilliefors) and target
#      distributions (gamma/exponential/uniform).
#   5. Descriptive Statistics  — median/mode/quartile/skewness/kurtosis.
TASK_KEYWORD_RULES = [
    (["contingency", "chi-square", "chi square", "chisquare",
      "independence", "categorical", "fisher exact", "mantel",
      "control variable"], "Contingency Table Test"),
    (["correlation", "relationship between", "monotonic",
      "linear association"], "Correlation Analysis"),
    (["variance", "variances", "variability", "variation", "dispersion",
      "fluctuation", "homogeneity", "comparing two means", "spread",
      "comparing means"], "Variance Test"),
    (["normal", "distribution", "goodness of fit", "goodness-of-fit",
      "shapiro", "anderson", "kolmogorov", "lilliefors", "gamma",
      "exponential", "uniform"], "Distribution Compliance Test"),
    (["descriptive", "median", "mode", "quartile", "skewness", "kurtosis",
      "range of"], "Descriptive Statistics"),
]


def normalise_task(raw: str) -> str | None:
    """Map a Step-3 task string to a canonical category, or None.

    With the closed-set Step-3 prompt the model should copy one of the five
    canonical names verbatim, so an exact (case-insensitive) match handles the
    overwhelming majority. The keyword scan below is a fallback for the rare
    case where the model paraphrases despite the instruction.
    """
    if not raw:
        return None
    key = raw.strip().lower()

    # 1. Exact match against the five canonical task names.
    for canonical in TASK_CANDIDATES:
        if key == canonical.lower():
            return canonical

    # 2. Keyword fallback (priority order; see TASK_KEYWORD_RULES).
    for keywords, task in TASK_KEYWORD_RULES:
        if any(kw in key for kw in keywords):
            return task
    return None