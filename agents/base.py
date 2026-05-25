from __future__ import annotations

import json
import time
from abc import ABC
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError, model_validator

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
    suggestion: Optional[str] = None  # noqa: UP045
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_line_range(self) -> ReviewComment:
        if self.line_end < self.line_start:
            raise ValueError("line_end must be greater than or equal to line_start")
        return self


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
            comments = await self._model_review(messages, file_path)
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

    async def _model_review(
        self,
        messages: list[dict[str, str]],
        file_path: str,
        max_attempts: int = 2,
    ) -> list[ReviewComment]:
        last_error: Exception | None = None
        repair_messages = list(messages)
        for attempt in range(max_attempts):
            try:
                result = await complete_with_function_call(repair_messages, tools=[REVIEW_TOOL])
                return validate_review_result(result, fallback_file_path=file_path)
            except (KeyError, TypeError, ValueError, ValidationError) as exc:
                last_error = exc
                if attempt == max_attempts - 1:
                    break
                repair_messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous tool output did not match the schema. "
                            "Retry with valid category/severity values, positive line numbers, "
                            "line_end >= line_start, confidence between 0 and 1, and the "
                            f"file_path set to `{file_path}` when uncertain."
                        ),
                    }
                )
        raise last_error or ValueError("model review failed")

    def _heuristic_review(self, diff: str, file_path: str) -> list[ReviewComment]:
        return []


def parse_tool_arguments(arguments: str | dict) -> dict:
    if isinstance(arguments, dict):
        return arguments
    return json.loads(arguments)


def validate_review_result(result: dict, fallback_file_path: str) -> list[ReviewComment]:
    raw_comments = result.get("comments", [])
    if not isinstance(raw_comments, list):
        raise ValueError("comments must be a list")

    comments: list[ReviewComment] = []
    for raw in raw_comments:
        if not isinstance(raw, dict):
            raise ValueError("each comment must be an object")
        normalized = dict(raw)
        normalized["file_path"] = normalized.get("file_path") or fallback_file_path
        if normalized.get("line_end") is None and normalized.get("line_start") is not None:
            normalized["line_end"] = normalized["line_start"]
        comments.append(ReviewComment(**normalized))
    return comments
