from __future__ import annotations

import re
from typing import Any

_AGENT_EVALUATION_RE = re.compile(
    r"(?:```(?:json)?\s*)?<agent_evaluation>(.*?)</agent_evaluation>\s*(?:```)?",
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
    has_open_tag = open_tag in text
    has_close_tag = close_tag in text

    if has_open_tag and not has_close_tag:
        open_idx = text.find(open_tag)
        return {
            "visible_content": _clean_visible_prefix(text[:open_idx]),
            "planning_content": _clean_partial_planning_suffix(
                text[open_idx + len(open_tag) :].strip()
            ),
            "has_incomplete_tag": True,
        }

    partial_open_idx = text.find("<agent")
    if partial_open_idx != -1 and not has_open_tag and not has_close_tag:
        return {
            "visible_content": _clean_visible_prefix(text[:partial_open_idx]),
            "planning_content": "",
            "has_incomplete_tag": True,
        }

    matches = list(_AGENT_EVALUATION_RE.finditer(text))
    planning_parts = [
        match.group(1).strip() for match in matches if match.group(1).strip()
    ]
    visible_content = _AGENT_EVALUATION_RE.sub("", text).strip()
    planning_content = "\n\n".join(planning_parts).strip()
    return {
        "visible_content": visible_content,
        "planning_content": planning_content,
        "has_incomplete_tag": False,
    }


def remove_agent_evaluation(text: str) -> str:
    return str(parse_agent_evaluation(text)["visible_content"])
