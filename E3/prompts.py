# -*- coding: utf-8 -*-
"""
Prompt construction for the five plan variants 
"""
from statqa_core import context_from_prompt

# Shared opening + closing fragments 
_PLAN_HEAD = "Let's first understand the problem and follow this plan:"

_STEPS_123 = (
    "1. Identify the variables involved.\n"
    "2. Determine the statistical task type.\n"
    "3. Select the appropriate method category."
)

# Final-answer instruction. The literal phrase 'FINAL ANSWER (JSON):' is the
# anchor; extract_final_answer takes the last {...} with a 'methods' key anyway.
_FINAL_JSON = (
    'Then, on a new line beginning with "FINAL ANSWER (JSON):", output one JSON '
    'object with exactly two keys, "columns" and "methods". '
    "Only include methods from the classification list. Output the JSON only once."
)

_CLOSING = "Now solve the problem step by step following the plan."


# ---- Step 4 wordings ---------------------------------------------------------------
_S4_NL = (
    "4. From the candidate methods in the provided classification list, keep only those "
    "that satisfy the data type, sample size and assumptions. Evaluate each method "
    "individually and state whether it is included or excluded."
)

_S4_TABULAR = (
    "4. Evaluate the candidate methods. Output a Markdown table with EXACTLY three "
    "columns and one row per candidate method:\n"
    "   | Method | Decision | Reason |\n"
    "   The Decision cell must be exactly INCLUDE or EXCLUDE for every method "
    "(no blanks, no narrative prose, no 'see above'). The Reason cell is a brief "
    "justification citing the relevant data feature."
)

_S4_YESNO = (
    "4. Evaluate the candidate methods. For every candidate method output one line of "
    "the exact form '<Method Name>: INCLUDE' or '<Method Name>: EXCLUDE'. Give a "
    "decision for every candidate method. Do NOT write any reason or explanation in "
    "this step -- the decision word only."
)

# ---- Step 5 (self-check) wording ---------------------------------------------------
_S5_SELFCHECK = (
    "5. Self-check: list only the methods marked INCLUDE in Step 4. For each one, "
    "confirm it does not contradict any condition identified in Step 4. Remove any "
    "method that does. Do not add any new method at this stage."
)


def _assemble(context: str, body: str) -> str:
    """Glue the shared context block to a variant-specific plan body."""
    return f"{context}\n\n{_PLAN_HEAD}\n{body}\n\n{_CLOSING}"


# Single-turn variants (V0, V1, V2, V3)  ->  return chat-message list
def _v0(context):
    body = f"{_STEPS_123}\n{_S4_NL}\n{_S5_SELFCHECK}\n6. {_FINAL_JSON}"
    return _assemble(context, body)


def _v1(context):  # remove self-check; final JSON references Step 4 directly
    final = ('5. ' + _FINAL_JSON.replace("Then, on a new line",
                                         "Using only the methods marked INCLUDE in Step 4, "
                                         "on a new line"))
    body = f"{_STEPS_123}\n{_S4_NL}\n{final}"
    return _assemble(context, body)


def _v2(context):
    body = f"{_STEPS_123}\n{_S4_TABULAR}\n{_S5_SELFCHECK}\n6. {_FINAL_JSON}"
    return _assemble(context, body)


def _v3(context):
    body = f"{_STEPS_123}\n{_S4_YESNO}\n{_S5_SELFCHECK}\n6. {_FINAL_JSON}"
    return _assemble(context, body)


_SINGLE_TURN_BUILDERS = {"V0": _v0, "V1": _v1, "V2": _v2, "V3": _v3}


def build_messages(row, variant):
    """Chat messages for a single-turn variant. `row` is a pandas Series."""
    context = context_from_prompt(row["prompt"])
    user = _SINGLE_TURN_BUILDERS[variant](context)
    return [{"role": "user", "content": user}]


# V4: two-turn (zero-shot CoT, then post-hoc self-check)
def build_v4_turn1_messages(row):
    """
    Turn 1 = the ORIGINAL zero-shot-CoT prompt, used verbatim (the full `prompt`
    column, which already ends with the CoT trigger and response cue).
    """
    return [{"role": "user", "content": str(row["prompt"]).rstrip()}]


_V4_SELFCHECK = (
    "Now perform a self-check on the answer you just gave. For EACH method in your "
    "selected 'methods' list, verify it does not violate the data type, sample size, "
    "or statistical assumptions of this problem. Remove any method that does. Do NOT "
    "add any method that was not already selected. " + _FINAL_JSON
)


def build_v4_turn2_messages(row, turn1_output):
    """
    Turn 2 = Turn-1 conversation + the model's Turn-1 reply + a self-check request.
    """
    msgs = build_v4_turn1_messages(row)
    msgs.append({"role": "assistant", "content": str(turn1_output)})
    msgs.append({"role": "user", "content": _V4_SELFCHECK})
    return msgs