"""
prompts.py — Turn 1 (Judgement) and Turn 2 (Selection) templates.
"""

from __future__ import annotations

import json

from classification_list import CLASSIFICATION_LIST

CLASSIFICATION_LIST_STRING = ", ".join(CLASSIFICATION_LIST)


# Turn 1: Applicability Summary Generation
TURN1_SYSTEM_MESSAGE = (
    "You are a careful statistical assistant. You solve problems in stages. "
    "In this stage you only produce an applicability summary as a JSON object; "
    "you do NOT output a final method list yet."
)


TURN1_USER_TEMPLATE = """Let's solve this problem in stages. In this stage, you will only produce an applicability summary; you will NOT yet output the final answer.

Problem: {question}

Column information:
{column_info}

Classification list (for reference):
{classification_list}

Plan:
1. Identify the variables and their data types.
2. Determine the statistical task type.
3. From the classification list, identify ONLY the methods relevant to the task type determined in Step 2 (typically 3-8 methods).
4. For EACH of these candidate methods, evaluate whether it satisfies the data type, sample size, and statistical assumptions of this problem.

Output a JSON object with this exact structure:
{{
  "task_type": "<the task type you determined in Step 2>",
  "summary": {{
    "<method name>": {{
      "verdict": "APPLICABLE" or "NOT_APPLICABLE",
      "reason": "<brief explanation citing specific data features>",
      "conditions_checked": ["<condition>: met/not_met", ...]
    }}
  }}
}}

Use method names exactly as written in the classification list. Do not output a final method list yet. Output only the JSON object."""


def build_turn1_messages(question: str, column_info: str) -> list[dict]:
    return [
        {"role": "system", "content": TURN1_SYSTEM_MESSAGE},
        {"role": "user", "content": TURN1_USER_TEMPLATE.format(
            question=question,
            column_info=column_info,
            classification_list=CLASSIFICATION_LIST_STRING,
        )},
    ]


# Turn 2: Selection Conditioned on Summary
TURN2_SYSTEM_MESSAGE = (
    "You are a careful statistical assistant making a final method selection "
    "based strictly on a provided applicability table. You do not re-evaluate."
)

TURN2_USER_TEMPLATE = """An applicability analysis has been completed for this problem. The results are:

{verdict_table}

Problem: {question}

Column information:
{column_info}

Based STRICTLY on the table above, output the final answer as a JSON object:
{{
  "selected_methods": [ ... ]
}}

Rules:
- Include ONLY methods marked APPLICABLE in the table.
- Do not re-evaluate any method.
- Do not add methods not marked APPLICABLE.
- Do not omit methods marked APPLICABLE.
Output only the JSON object."""


def render_verdict_table(verdict_map: dict[str, str]) -> str:
    """Render the clean verdict table (method -> APPLICABLE/NOT_APPLICABLE).

    A simple, unambiguous one-line-per-method format. We avoid raw JSON here
    so the model reads it as a table of conclusions, not as something to
    re-parse and reason over.
    """
    lines = []
    for method, verdict in verdict_map.items():
        lines.append(f"- {method}: {verdict}")
    return "\n".join(lines)


def build_turn2_messages(question: str, column_info: str,
                         verdict_map: dict[str, str]) -> list[dict]:
    """Fresh message list — NO Turn 1 history. This is the context reset."""
    return [
        {"role": "system", "content": TURN2_SYSTEM_MESSAGE},
        {"role": "user", "content": TURN2_USER_TEMPLATE.format(
            verdict_table=render_verdict_table(verdict_map),
            question=question,
            column_info=column_info,
        )},
    ]