# -*- coding: utf-8 -*-

"""
从 qwen2.5_72b_PS_AE_check.csv 中随机抽取 20 条 IA 样本,
用于 4.2.1 节关于 72B-plan IA 异常 (~25%) 的来源核查。

工作流:
1. 沿用原脚本的 error classification 逻辑,识别出所有 invalid_answer 行。
2. 在 IA 子集上,基于 response/原始输出的内容做一个启发式 IA 子类标签
   (parsing_recoverable / json_format_issue / empty / refusal /
    plan_derailed_or_other),为人工标注提供起点。
3. 随机抽取 20 条,保存全部列以便人工读取 response 全文。
"""

import os
import re
import json
import pandas as pd

# =========================
# 路径设置
# =========================
input_file  = "/rds/general/user/yx3422/home/m4r_project/Model Answer/Processed Answer/qwen2.5_72b_PS_AE_check.csv"
output_file = "/rds/general/user/yx3422/home/m4r_project/Model Answer/Processed Answer/qwen2.5_72b_IA_sample20.csv"

random_seed = 42
sample_size = 20


RESPONSE_COL_CANDIDATES = [
    "model_answer"
]

# =========================
# Method list (与原脚本一致)
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
    "Mean", "Median", "Mode", "Range", "Quartile",
    "Standard Deviation", "Skewness", "Kurtosis",

    # Variance Test
    "Mood Variance Test", "Levene Test",
    "Bartlett Test", "F-Test for Variance",
]
methods_list = [m.lower() for m in methods_list]


# =========================
# 错误分类逻辑 (与原脚本一致,仅保留 IA 部分需要的判定)
# =========================
def is_invalid_answer(extracted_answer, methods_list):
    """与原脚本一致:extracted_answer 中不包含任何 method 名 -> IA"""
    if pd.isna(extracted_answer):
        return True
    extracted_answer = str(extracted_answer).lower()
    return not any(method in extracted_answer for method in methods_list)


# =========================
# IA 子类启发式分类
# 仅作为人工核查的起点,不替代人工判断
# =========================
def classify_ia_subclass(resp: str, methods_list) -> str:
    """
    给一条 IA 的 response 打一个粗粒度子类标签:

    - empty            : 空字符串、纯空白、占位
    - refusal          : 模型表示无法/拒绝回答
    - json_format_issue: 出现明显的 JSON 格式破损 (括号不匹配、markdown 代码块包裹等)
                         但实际生成了内容
    - parsing_recoverable :
                         response 中包含至少一个 method list 里的方法名,
                         说明模型其实选了方法,只是 extracted_answer 没抽出来
                         -> 这正是"被严格 JSON 评估口径误伤"的核心情形
    - plan_derailed_or_other : 以上都不是;通常是叙述性输出、复读 prompt、
                              中途断掉等需要人工读的情况
    """
    if resp is None or (isinstance(resp, float) and pd.isna(resp)):
        return "empty"
    resp_str = str(resp).strip()
    if resp_str == "":
        return "empty"

    low = resp_str.lower()

    # 拒绝/无法回答类
    refusal_patterns = [
        "i cannot", "i can't", "i am unable", "i'm unable",
        "i do not", "i don't have", "sorry, i", "as an ai",
        "no method is applicable", "none of the methods",
    ]
    if any(p in low for p in refusal_patterns):
        return "refusal"

    # 如果 response 中出现了 method list 里的方法名,
    # 这很可能是格式问题导致 extracted_answer 抽不出来,人工读一眼就能恢复
    if any(m in low for m in methods_list):
        # 进一步看是否伴随明显 JSON 破损
        if "```" in resp_str or resp_str.count("{") != resp_str.count("}"):
            return "json_format_issue"
        return "parsing_recoverable"

    # 没有方法名,但 JSON 结构看起来有意图但写坏了
    if "```" in resp_str or "{" in resp_str or "}" in resp_str:
        return "json_format_issue"

    return "plan_derailed_or_other"


# =========================
# 主流程
# =========================
def main():
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    df = pd.read_csv(input_file)
    print(f"Total rows: {len(df)}")

    # 1. 标 IA
    df["invalid_answer"] = df["extracted_answer"].apply(
        lambda x: is_invalid_answer(x, methods_list)
    )

    ia_df = df[df["invalid_answer"] == True].copy()
    ia_rate = len(ia_df) / max(len(df), 1)
    print(f"IA rows: {len(ia_df)}  ({ia_rate:.3%} of total)")

    if len(ia_df) == 0:
        print("No IA rows found. Nothing to sample.")
        return

    # 2. 选定 response 列 (用于子类自动分类与人工预览)
    response_col = next(
        (c for c in RESPONSE_COL_CANDIDATES if c in ia_df.columns),
        None
    )
    if response_col is None:
        print(
            "WARNING: no response-like column found among "
            f"{RESPONSE_COL_CANDIDATES}. Skipping auto subclass tagging."
        )
        ia_df["ia_subclass"] = "unknown"
    else:
        print(f"Using '{response_col}' as the response column for subclass tagging.")
        ia_df["ia_subclass"] = ia_df[response_col].apply(
            lambda r: classify_ia_subclass(r, methods_list)
        )

    # 3. 自动子类分布 (在全部 IA 上,不只是 20 条抽样)
    print("\nAuto IA-subclass distribution over ALL IA rows (heuristic):")
    print(ia_df["ia_subclass"].value_counts().to_string())
    print(
        "\nNote: these labels are heuristic starting points;\n"
        "      please manually verify on the 20 sampled rows below.\n"
    )

    # 4. 随机抽取 20 条 (不足 20 则全取)
    n = min(sample_size, len(ia_df))
    sampled = ia_df.sample(n=n, random_state=random_seed).reset_index(drop=True)

    sampled.to_csv(output_file, index=False)
    print(f"Sampled {n} IA rows.\nSaved to: {output_file}\n")

    # 5. 控制台预览,便于直接开始读
    if response_col is not None:
        print("=" * 80)
        print(f"Preview of {n} sampled IA responses (truncated to 800 chars each):")
        print("=" * 80)
        for i, row in sampled.iterrows():
            print(f"\n--- [{i+1}/{n}] row_index={row.name}  "
                  f"task={row.get('task', 'N/A')}  "
                  f"ia_subclass={row['ia_subclass']} ---")
            resp_text = str(row[response_col])
            print(resp_text[:800] + ("..." if len(resp_text) > 800 else ""))


if __name__ == "__main__":
    main()