"""
Classification list + task-category candidate sets for StatQA.
"""

# --- Per-category method lists (members identical to Exp 1's 27-method list) -
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

# --- Task category 
TASK_CANDIDATES = {
    "Correlation Analysis": CORRELATION_ANALYSIS,
    "Distribution Compliance Test": DISTRIBUTION_COMPLIANCE_TEST,
    "Contingency Table Test": CONTINGENCY_TABLE_TEST,
    "Descriptive Statistics": DESCRIPTIVE_STATISTICS,
    "Variance Test": VARIANCE_TEST,
}

# Candidate-set sizes documented in 5.2.2: CA 4, CTT 3, VT 4, DCT 8, DS 8.
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


#Helpers 

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


# Normalise common task-name variants the model might emit in Step 3.
TASK_ALIASES = {
    "correlation": "Correlation Analysis",
    "correlation analysis": "Correlation Analysis",
    "distribution compliance": "Distribution Compliance Test",
    "distribution compliance test": "Distribution Compliance Test",
    "distribution test": "Distribution Compliance Test",
    "normality test": "Distribution Compliance Test",
    "contingency": "Contingency Table Test",
    "contingency table": "Contingency Table Test",
    "contingency table test": "Contingency Table Test",
    "descriptive": "Descriptive Statistics",
    "descriptive statistics": "Descriptive Statistics",
    "variance": "Variance Test",
    "variance test": "Variance Test",
}


def normalise_task(raw: str) -> str | None:
    """Map a free-text Step-3 task string to a canonical category, or None."""
    if not raw:
        return None
    key = raw.strip().lower()
    if key in TASK_ALIASES:
        return TASK_ALIASES[key]
    # Substring fallback: find the first canonical name mentioned.
    for alias, canonical in TASK_ALIASES.items():
        if alias in key:
            return canonical
    return None