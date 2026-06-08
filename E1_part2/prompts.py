"""
Prompt templates for Experiment 1, Part 2.

"""

from classification_list import CLASSIFICATION_LIST

# Same comma-separated rendering used in StatQA's PROMPT_CLASSIFICATION, but
# flat — Part 2 does not need category headings.
CLASSIFICATION_LIST_STRING = ", ".join(CLASSIFICATION_LIST)


# Main task: per-method instance judgement
JUDGEMENT_SYSTEM_MESSAGE = (
    "You are a careful assistant that evaluates the applicability of "
    "statistical methods given concrete data features. Be precise. "
    "Always state APPLICABLE or NOT APPLICABLE in capital letters as the "
    "first line of your answer."
)


JUDGEMENT_USER_TEMPLATE = """Problem: {question}

Column information:
{column_info}

Question: Is the method "{method_name}" applicable to this problem?
Please:
1. State your conclusion: APPLICABLE or NOT APPLICABLE.
2. Cite specific data features (e.g., data type, normality, sample size) as evidence.
3. For each known applicability condition of this method, explicitly state whether the data MEETS or DOES NOT MEET that condition."""


def build_judgement_messages(question: str, column_info: str,
                             method_name: str) -> list[dict]:
    return [
        {"role": "system", "content": JUDGEMENT_SYSTEM_MESSAGE},
        {"role": "user", "content": JUDGEMENT_USER_TEMPLATE.format(
            question=question,
            column_info=column_info,
            method_name=method_name,
        )},
    ]


