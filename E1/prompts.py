"""
Prompt templates for Experiment 1, Part 1.
==========================================
"""

# Light system message. Keep it minimal and neutral: the goal of Part 1 is to
# probe what the model knows when asked plainly, not to coach it.
SYSTEM_MESSAGE = (
    "You are a careful assistant answering questions about statistical "
    "methods. Be precise and concise. Do not invent conditions you are "
    "unsure about; if a method has no requirement on a given dimension, "
    "say so explicitly."
)

# Verbatim from the outline. The single placeholder is {method_name}.
USER_PROMPT_TEMPLATE = """For the statistical method "{method_name}", please describe:
1. Required data type(s) for input variables.
2. Sample size requirements (if any).
3. Key statistical assumptions (e.g., normality, independence, equal variance).
4. Conditions under which this method should NOT be used.
Provide your answer as structured bullet points."""


def build_messages(method_name: str) -> list[dict]:
    """Return a `messages`-style list compatible with chat templates."""
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(method_name=method_name)},
    ]