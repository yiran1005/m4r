"""
parse_outputs.py — Turn free-text model output into a structured verdict.

We prompted the model to put "APPLICABLE" or "NOT APPLICABLE" on the first
line (per JUDGEMENT_SYSTEM_MESSAGE in prompts.py). In practice models do
not always comply: they may bury the verdict in the middle, prefix it with
"Conclusion:", paraphrase it as "is applicable", or contradict themselves
across the response. This module is the central place that handles those
variations.

Verdict precedence:
  1. If the FIRST 200 chars contain a clear NOT-APPLICABLE phrase → NOT_APPLICABLE
  2. Else if the FIRST 200 chars contain a clear APPLICABLE phrase → APPLICABLE
  3. Else search the entire response for the same phrases.
  4. Else return PARSE_ERROR.

The "NOT" check runs first because "APPLICABLE" is a substring of
"NOT APPLICABLE" — if you check APPLICABLE first you mis-classify every
negative verdict as positive.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import config


# Patterns ordered from most-specific to least-specific within each polarity.
NOT_APPLICABLE_PATTERNS = [
    r"\bNOT\s+APPLICABLE\b",
    r"\bNOT\s+APPROPRIATE\b",
    r"\bis\s+not\s+applicable\b",
    r"\bcannot\s+be\s+(?:applied|used)\b",
    r"\bshould\s+not\s+be\s+(?:applied|used)\b",
    r"\bdoes\s+not\s+apply\b",
    r"\bINAPPLICABLE\b",
]

APPLICABLE_PATTERNS = [
    r"\bAPPLICABLE\b",
    r"\bAPPROPRIATE\b",
    r"\bis\s+applicable\b",
    r"\bcan\s+be\s+(?:applied|used)\b",
    r"\bshould\s+be\s+(?:applied|used)\b",
]

NOT_RE = [re.compile(p, re.IGNORECASE) for p in NOT_APPLICABLE_PATTERNS]
APP_RE = [re.compile(p, re.IGNORECASE) for p in APPLICABLE_PATTERNS]


def parse_verdict(response: str) -> tuple[str, str]:
    """Return (verdict, reason) where verdict is APPLICABLE / NOT_APPLICABLE /
    PARSE_ERROR. `reason` is a short tag for which rule fired (useful when
    debugging parse failures).
    """
    if not response or not response.strip():
        return "PARSE_ERROR", "empty_response"

    head = response[:200]

    # Stage 1: NOT first, in the head.
    for rx in NOT_RE:
        if rx.search(head):
            return "NOT_APPLICABLE", f"head:{rx.pattern}"
    # Stage 2: APPLICABLE in the head.
    for rx in APP_RE:
        if rx.search(head):
            return "APPLICABLE", f"head:{rx.pattern}"

    # Stage 3: full-body fallback. NOT first again.
    for rx in NOT_RE:
        if rx.search(response):
            return "NOT_APPLICABLE", f"body:{rx.pattern}"
    for rx in APP_RE:
        if rx.search(response):
            return "APPLICABLE", f"body:{rx.pattern}"

    return "PARSE_ERROR", "no_pattern_matched"


def parse_jsonl(input_path: Path, output_path: Path) -> dict:
    """Read raw_judgements.jsonl, parse each verdict, write a parsed file.

    Returns a stats dict {APPLICABLE, NOT_APPLICABLE, PARSE_ERROR} for the
    console summary.
    """
    stats = {"APPLICABLE": 0, "NOT_APPLICABLE": 0, "PARSE_ERROR": 0}
    n_total = 0

    with input_path.open("r", encoding="utf-8") as fin, \
         output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            verdict, parse_reason = parse_verdict(rec.get("response", ""))
            rec["verdict"] = verdict
            rec["parse_reason"] = parse_reason
            stats[verdict] += 1
            n_total += 1
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    stats["TOTAL"] = n_total
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=config.RAW_JUDGEMENTS_PATH)
    parser.add_argument("--output", type=Path, default=config.PARSED_JUDGEMENTS_PATH)
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")

    stats = parse_jsonl(args.input, args.output)
    print(f"Parsed {stats['TOTAL']} records → {args.output}")
    for k in ("APPLICABLE", "NOT_APPLICABLE", "PARSE_ERROR"):
        n = stats[k]
        pct = n / stats["TOTAL"] * 100 if stats["TOTAL"] else 0
        print(f"  {k:16s} {n:4d}  ({pct:5.1f}%)")

    if stats["PARSE_ERROR"] > 0:
        print()
        print(f"WARNING: {stats['PARSE_ERROR']} records could not be parsed. "
              f"Inspect them with:")
        print(f"  grep '\"verdict\": \"PARSE_ERROR\"' {args.output} | head -3")


if __name__ == "__main__":
    main()