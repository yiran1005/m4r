# -*- coding: utf-8 -*-

import os
import json
import pandas as pd

# =========================
# 路径设置
# =========================
input_file = "/rds/general/user/yx3422/home/m4r_project/Model Answer/Processed Answer/qwen_PS_AE_checkqwen2.5_14b_PS_AE_check.csv" 
output_file = "/rds/general/user/yx3422/home/m4r_project/Model Answer/Processed Answer/qwen2.5_14b_AE_check_component.csv"

# 随机种子，保证可复现
random_seed = 42

# =========================
# Method list
# =========================
methods_list = [
    # Correlation Analysis
    "Pearson Correlation Coefficient",
    "Spearman Correlation Coefficient",
    "Kendall Correlation Coefficient",
    "Partial Correlation Coefficient",

    # Distribution Compliance Test
    "Anderson-Darling Test",
    "Shapiro-Wilk Test of Normality",
    "Kolmogorov-Smirnov Test for Normality",
    "Lilliefors Test",
    "Kolmogorov-Smirnov Test",
    "Kolmogorov-Smirnov Test for Uniform distribution",
    "Kolmogorov-Smirnov Test for Gamma distribution",
    "Kolmogorov-Smirnov Test for Exponential distribution",

    # Contingency Table Test
    "Chi-square Independence Test",
    "Fisher Exact Test",
    "Mantel-Haenszel Test",

    # Descriptive Statistics
    "Mean",
    "Median",
    "Mode",
    "Range",
    "Quartile",
    "Standard Deviation",
    "Skewness",
    "Kurtosis",

    # Variance Test
    "Mood Variance Test",
    "Levene Test",
    "Bartlett Test",
    "F-Test for Variance"
]

# 全部转小写，便于匹配
methods_list = [m.lower() for m in methods_list]


def is_invalid_answer(extracted_answer, methods_list):
    """
    按你原来的逻辑：
    如果 extracted_answer 中没有出现任何 method 名称，就算 invalid answer
    """
    if pd.isna(extracted_answer):
        return True

    extracted_answer = str(extracted_answer).lower()
    return not any(method in extracted_answer for method in methods_list)


def classify_error(row, methods_list):
    """
    对单行进行错误分类，返回字典：
    {
        'invalid_answer': bool,
        'column_error': bool,
        'statistical_confusion': bool,
        'applicability_error': bool
    }
    """
    result = {
        'invalid_answer': False,
        'column_error': False,
        'statistical_confusion': False,
        'applicability_error': False
    }

    # 1. invalid answer
    if is_invalid_answer(row.get("extracted_answer", None), methods_list):
        result["invalid_answer"] = True
        return result

    # 2. column error
    try:
        result["column_error"] = (int(row["columns_score"]) == 0)
    except Exception:
        result["column_error"] = False

    # 3. statistical confusion / applicability error
    methods_raw = row.get("methods_comparison_result", "")

    try:
        methods_result = json.loads(methods_raw)

        correct = methods_result.get("Correct", 0)
        wrong = methods_result.get("Wrong", 0)
        missed = methods_result.get("Missed", 0)
        wrong_missed = wrong + missed

        if correct == 0 and wrong_missed > 0:
            result["statistical_confusion"] = True

        if correct > 0 and wrong_missed > 0:
            result["applicability_error"] = True

    except json.JSONDecodeError:
        # 按你原来的逻辑，JSON decode 失败记为 statistical confusion
        result["statistical_confusion"] = True
    except Exception:
        result["statistical_confusion"] = True

    return result


def main():
    # 检查输入文件
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # 读取数据
    df = pd.read_csv(input_file)

    # 对每一行做 error classification
    error_records = []
    for idx, row in df.iterrows():
        error_info = classify_error(row, methods_list)
        error_info["row_index"] = idx
        error_records.append(error_info)

    error_df = pd.DataFrame(error_records)

    # 合并 error labels 到原始 dataframe
    df = pd.concat(
        [df.reset_index(drop=True), error_df.drop(columns=["row_index"])],
        axis=1
    )

    # =========================
    # 抽取所有带 AE 成分的样本
    # 即：
    # applicability_error = True
    # invalid_answer = False
    # =========================
    ae_component_df = df[
        (df["applicability_error"] == True) &
        (df["invalid_answer"] == False)
    ].copy()

    print(f"Total rows: {len(df)}")
    print(f"Rows with AE component (excluding invalid answers): {len(ae_component_df)}")

    # 随机抽取 50 条；如果不足 50 条则全部保留
    sample_n = min(50, len(ae_component_df))
    sampled_df = ae_component_df.sample(n=sample_n, random_state=random_seed)

    # 保存
    sampled_df.to_csv(output_file, index=False)

    print(f"Sampled {sample_n} rows with AE component.")
    print(f"Saved to: {output_file}")

    # 顺手打印一下其中各种子类型数量，方便你检查
    pure_ae = ae_component_df[
        (ae_component_df["applicability_error"] == True) &
        (ae_component_df["column_error"] == False) &
        (ae_component_df["statistical_confusion"] == False)
    ]

    cse_ae = ae_component_df[
        (ae_component_df["applicability_error"] == True) &
        (ae_component_df["column_error"] == True) &
        (ae_component_df["statistical_confusion"] == False)
    ]

    stc_ae = ae_component_df[
        (ae_component_df["applicability_error"] == True) &
        (ae_component_df["column_error"] == False) &
        (ae_component_df["statistical_confusion"] == True)
    ]

    all_three = ae_component_df[
        (ae_component_df["applicability_error"] == True) &
        (ae_component_df["column_error"] == True) &
        (ae_component_df["statistical_confusion"] == True)
    ]

    print("\nBreakdown of AE-component rows:")
    print(f"Pure AE: {len(pure_ae)}")
    print(f"CSE + AE: {len(cse_ae)}")
    print(f"STC + AE: {len(stc_ae)}")
    print(f"CSE + STC + AE: {len(all_three)}")


if __name__ == "__main__":
    main()