from __future__ import annotations

import re
from typing import Any

_AGENT_EVALUATION_RE = re.compile(
    r"^\s*(?:```(?:json)?\s*)?<agent_evaluation>(.*?)</agent_evaluation>\s*(?:```)?\s*",
    flags=re.DOTALL | re.IGNORECASE,
)
_TRAILING_FENCE_RE = re.compile(r"(?:```(?:json)?\s*)$", flags=re.IGNORECASE)
_PARTIAL_CLOSE_RE = re.compile(r"</agent(?:_evaluation)?[^>]*$", flags=re.IGNORECASE)


def _clean_visible_prefix(text: str) -> str:
    return _TRAILING_FENCE_RE.sub("", text).rstrip()


def _clean_partial_planning_suffix(text: str) -> str:
    return _PARTIAL_CLOSE_RE.sub("", text).rstrip()


def parse_agent_evaluation(text: str) -> dict[str, Any]:
    text = text or ""
    open_tag = "<agent_evaluation>"
    close_tag = "</agent_evaluation>"

    # 1. Check for a complete <agent_evaluation> block at the start of text
    match = _AGENT_EVALUATION_RE.match(text)
    if match:
        planning_content = match.group(1).strip()
        visible_content = text[match.end() :].strip()
        return {
            "visible_content": visible_content,
            "planning_content": planning_content,
            "has_incomplete_tag": False,
        }

    # 2. Check for incomplete block at the start (open tag present but no close tag yet)
    normalized_start = text.lstrip()
    for prefix in ("```json", "```"):
        if normalized_start.startswith(prefix):
            normalized_start = normalized_start[len(prefix) :].lstrip()

    if normalized_start.startswith(open_tag) and close_tag not in text:
        open_idx = text.find(open_tag)
        return {
            "visible_content": _clean_visible_prefix(text[:open_idx]),
            "planning_content": _clean_partial_planning_suffix(
                text[open_idx + len(open_tag) :].strip()
            ),
            "has_incomplete_tag": True,
        }

    # 3. Check for partial tag being typed at the start (e.g., "<agen...")
    if (
        normalized_start.startswith("<agent")
        and open_tag not in text
        and close_tag not in text
    ):
        partial_open_idx = text.find("<agent")
        return {
            "visible_content": _clean_visible_prefix(text[:partial_open_idx]),
            "planning_content": "",
            "has_incomplete_tag": True,
        }

    # 4. No agent_evaluation block at the beginning — everything is visible content
    return {
        "visible_content": text,
        "planning_content": "",
        "has_incomplete_tag": False,
    }


def remove_agent_evaluation(text: str) -> str:
    return str(parse_agent_evaluation(text)["visible_content"])
