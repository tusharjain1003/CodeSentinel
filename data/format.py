from __future__ import annotations

import json
from typing import Any


def infer_severity(comment: str) -> str:
    lowered = comment.lower()
    if any(
        term in lowered
        for term in ("critical", "exploit", "data loss", "injection", "vulnerability")
    ):
        return "critical"
    if any(
        term in lowered
        for term in ("bug", "incorrect", "breaks", "race", "crash", "deadlock", "corrupt")
    ):
        return "major"
    if lowered.startswith("nit"):
        return "nit"
    return "minor"


def infer_category(comment: str) -> str:
    lowered = comment.lower()
    if any(term in lowered for term in (
        "injection", "xss", "csrf", "secret", "auth",
        "vulnerability", "exploit", "ssl", "tls",
        "encrypt", "permission", "access control",
        "sanitize", "escape", "sql", "command injection",
    )):
        return "security"
    if any(term in lowered for term in (
        "bug", "null", "error", "race", "incorrect",
        "crash", "deadlock", "corrupt", "leak",
        "off-by-one", "index", "overflow",
        "undefined", "exception", "fail",
    )):
        return "bug"
    if any(term in lowered for term in ("rename", "format", "style", "typo")):
        return "style"
    return "maintainability"


def extract_line_range(comment: dict[str, Any]) -> tuple[int, int]:
    start = (
        comment.get("start_line")
        or comment.get("original_start_line")
        or comment.get("line")
        or comment.get("original_line")
        or 1
    )
    end = comment.get("line") or comment.get("original_line") or start
    start_int = max(1, int(start))
    end_int = max(start_int, int(end))
    return start_int, end_int


def extract_comment_body(comment: str | dict[str, Any]) -> str:
    if isinstance(comment, str):
        return comment
    return str(comment.get("body") or "")


def format_sample(
    diff_hunk: str,
    file_path: str,
    comment: str | dict[str, Any],
    category: str | None = None,
) -> dict:
    comment_body = extract_comment_body(comment)
    if isinstance(comment, dict):
        line_start, line_end = extract_line_range(comment)
        file_path = comment.get("path") or file_path
    else:
        line_start, line_end = 1, 1

    system = (
        "You are an expert code reviewer. Analyze the provided code diff and identify issues. "
        "Respond only with a valid JSON object matching the ReviewComment schema."
    )
    user = f"""Review this code change in {file_path}:

```diff
{diff_hunk}
```

Identify any issues. Respond with a JSON ReviewComment object."""
    assistant = json.dumps(
        {
            "category": category or infer_category(comment_body),
            "severity": infer_severity(comment_body),
            "file_path": file_path,
            "line_start": line_start,
            "line_end": line_end,
            "message": comment_body,
            "suggestion": None,
            "confidence": 0.9,
        }
    )
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }
