from __future__ import annotations

import json
import time
from abc import ABC
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from serving.client import complete_with_function_call


class Category(str, Enum):
    bug = "bug"
    security = "security"
    style = "style"
    maintainability = "maintainability"


class Severity(str, Enum):
    critical = "critical"
    major = "major"
    minor = "minor"
    nit = "nit"


class ReviewComment(BaseModel):
    category: Category
    severity: Severity
    file_path: str
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    message: str
    suggestion: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)


class AgentReview(BaseModel):
    agent_name: str
    comments: List[ReviewComment]
    timing_ms: int
    token_usage: Dict[str, int] = Field(default_factory=dict)


REVIEW_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_review",
        "description": "Submit code review comments",
        "parameters": {
            "type": "object",
            "properties": {
                "comments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "category": {
                                "type": "string",
                                "enum": ["bug", "security", "style", "maintainability"],
                            },
                            "severity": {
                                "type": "string",
                                "enum": ["critical", "major", "minor", "nit"],
                            },
                            "file_path": {"type": "string"},
                            "line_start": {"type": "integer"},
                            "line_end": {"type": "integer"},
                            "message": {"type": "string"},
                            "suggestion": {"type": "string"},
                            "confidence": {"type": "number"},
                        },
                        "required": [
                            "category",
                            "severity",
                            "file_path",
                            "line_start",
                            "line_end",
                            "message",
                            "confidence",
                        ],
                    },
                }
            },
            "required": ["comments"],
        },
    },
}


class BaseAgent(ABC):
    name: str
    focus: str
    system_prompt: str

    async def review(self, diff: str, file_path: str) -> AgentReview:
        t0 = time.monotonic()
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self._build_prompt(diff, file_path)},
        ]
        try:
            result = await complete_with_function_call(messages, tools=[REVIEW_TOOL])
            comments = [ReviewComment(**c) for c in result.get("comments", [])]
        except Exception:
            comments = self._heuristic_review(diff, file_path)

        return AgentReview(
            agent_name=self.name,
            comments=comments,
            timing_ms=int((time.monotonic() - t0) * 1000),
            token_usage={},
        )

    def _build_prompt(self, diff: str, file_path: str) -> str:
        return f"""Review this code change in `{file_path}`. Focus only on {self.focus}.

```diff
{diff}
```

Use the submit_review function to return findings. Return an empty comments array when
there are no issues."""

    def _heuristic_review(self, diff: str, file_path: str) -> list[ReviewComment]:
        return []


def parse_tool_arguments(arguments: str | dict) -> dict:
    if isinstance(arguments, dict):
        return arguments
    return json.loads(arguments)
