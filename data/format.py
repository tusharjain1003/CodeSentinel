from __future__ import annotations

import json


def infer_severity(comment: str) -> str:
    lowered = comment.lower()
    if any(term in lowered for term in ("critical", "exploit", "data loss", "injection")):
        return "critical"
    if any(term in lowered for term in ("bug", "incorrect", "breaks", "race")):
        return "major"
    if lowered.startswith("nit"):
        return "nit"
    return "minor"


def infer_category(comment: str) -> str:
    lowered = comment.lower()
    if any(term in lowered for term in ("injection", "xss", "csrf", "secret", "auth")):
        return "security"
    if any(term in lowered for term in ("bug", "null", "error", "race", "incorrect")):
        return "bug"
    if any(term in lowered for term in ("rename", "format", "style")):
        return "style"
    return "maintainability"


def format_sample(
    diff_hunk: str,
    file_path: str,
    comment: str,
    category: str | None = None,
) -> dict:
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
            "category": category or infer_category(comment),
            "severity": infer_severity(comment),
            "file_path": file_path,
            "line_start": 1,
            "line_end": 1,
            "message": comment,
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
