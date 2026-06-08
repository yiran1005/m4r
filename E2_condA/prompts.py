"""
prompts.py — Turn 1 (Judgement) and Turn 2 (Selection) templates.
=================================================================

Turn 1 reproduces the Section 5.2.2 plan: identify variables, determine task
type, prune to candidate methods, then per-method APPLICABLE/NOT_APPLICABLE
verdicts as a JSON object.

Turn 2 reproduces the Section 5.2.3 instruction: select STRICTLY from the
clean verdict table, output only methods marked APPLICABLE, no re-evaluation,
no additions, no omissions.

Crucially, the Turn 2 prompt is built from a fresh message list with NO Turn 1
history — that is the context reset described in 5.2.1.1.
"""

from __future__ import annotations

import json

from classification_list import CLASSIFICATION_LIST, TASK_CANDIDATES

CLASSIFICATION_LIST_STRING = ", ".join(CLASSIFICATION_LIST)

# The five task categories offered as closed-set choices in Step 3, rendered
# one per line. Built from TASK_CANDIDATES so prompt and code never drift.
TASK_OPTIONS_STRING = "\n   ".join(f"- {t}" for t in TASK_CANDIDATES)


# --------------------------------------------------------------------------- #
# Turn 1: Applicability Summary Generation
# --------------------------------------------------------------------------- #
TURN1_SYSTEM_MESSAGE = (
    "You are a careful statistical assistant. You solve problems in stages. "
    "In this stage you only produce an applicability summary as a JSON object; "
    "you do NOT output a final method list yet."
)

# The plan follows 5.2.2. Step 3 is a CLOSED-SET choice: the model must pick
# exactly one of the five StatQA task categories, rather than free-text a task
# name. This removes the downstream normalisation ambiguity that otherwise
# mis-flagged paraphrases ("Testing Normality", "Comparing Variances") as
# Step-3 errors. The five categories are injected from TASK_CANDIDATES so the
# prompt and the code share one source of truth.
TURN1_USER_TEMPLATE = """Let's solve this problem in stages. In this stage, you will only produce an applicability summary; you will NOT yet output the final answer.

Problem: {question}

Column information:
{column_info}

Classification list (for reference):
{classification_list}

Plan:
1. Identify the variables and their data types.
2. Determine the statistical task type. Choose EXACTLY ONE from this list, copying its name verbatim:
   {task_options}
3. From the classification list, identify ONLY the methods relevant to the task type determined in Step 2 (typically 3-8 methods).
4. For EACH of these candidate methods, evaluate whether it satisfies the data type, sample size, and statistical assumptions of this problem.

Output a JSON object with this exact structure:
{{
  "task_type": "<one of the five task types listed in Step 2, copied verbatim>",
  "summary": {{
    "<method name>": {{
      "verdict": "APPLICABLE" or "NOT_APPLICABLE",
      "reason": "<brief explanation citing specific data features>",
      "conditions_checked": ["<condition>: met/not_met", ...]
    }}
  }}
}}

Use method names exactly as written in the classification list, and the task_type exactly as written in Step 2. Do not output a final method list yet. Output only the JSON object."""


def build_turn1_messages(question: str, column_info: str) -> list[dict]:
    return [
        {"role": "system", "content": TURN1_SYSTEM_MESSAGE},
        {"role": "user", "content": TURN1_USER_TEMPLATE.format(
            question=question,
            column_info=column_info,
            classification_list=CLASSIFICATION_LIST_STRING,
            task_options=TASK_OPTIONS_STRING,
        )},
    ]


# --------------------------------------------------------------------------- #
# Turn 2: Selection Conditioned on Summary
# --------------------------------------------------------------------------- #
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