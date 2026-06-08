"""
Classification list of 27 statistical methods used in StatQA.
Copied from Exp 1 Part 1.
"""

CLASSIFICATION_LIST = [
    # --- Correlation Analysis (4) ---
    "Pearson Correlation Coefficient",
    "Spearman Correlation Coefficient",
    "Kendall Correlation Coefficient",
    "Partial Correlation Coefficient",

    # --- Distribution Compliance Test (8) ---
    "Anderson-Darling Test",
    "Shapiro-Wilk Test of Normality",
    "Kolmogorov-Smirnov Test for Normality",
    "Lilliefors Test",
    "Kolmogorov-Smirnov Test",
    "Kolmogorov-Smirnov Test for Uniform distribution",
    "Kolmogorov-Smirnov Test for Gamma distribution",
    "Kolmogorov-Smirnov Test for Exponential distribution",

    # --- Contingency Table Test (3) ---
    "Chi-square Independence Test",
    "Fisher Exact Test",
    "Mantel-Haenszel Test",

    # --- Descriptive Statistics (8) ---
    "Mean",
    "Median",
    "Mode",
    "Range",
    "Quartile",
    "Standard Deviation",
    "Skewness",
    "Kurtosis",

    # --- Variance Test (4) ---
    "Mood Variance Test",
    "Levene Test",
    "Bartlett Test",
    "F-Test for Variance",
]

assert len(CLASSIFICATION_LIST) == 27