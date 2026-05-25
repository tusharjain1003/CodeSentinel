from __future__ import annotations

from agents.base import BaseAgent, Category, ReviewComment, Severity


class StyleAgent(BaseAgent):
    name = "style_reviewer"
    focus = (
        "maintainability, complex functions, poor naming, missing type hints, unclear "
        "abstractions, duplicated logic, and missing error handling patterns"
    )
    system_prompt = """You are a senior engineer focused on code quality and maintainability.
Focus on changes that meaningfully improve readability or reduce future bugs.
Avoid trivial stylistic preferences."""

    def _heuristic_review(self, diff: str, file_path: str) -> list[ReviewComment]:
        comments: list[ReviewComment] = []
        for idx, line in enumerate(diff.splitlines(), start=1):
            added = line.startswith("+") and not line.startswith("+++")
            if added and len(line) > 140:
                comments.append(
                    ReviewComment(
                        category=Category.maintainability,
                        severity=Severity.nit,
                        file_path=file_path,
                        line_start=idx,
                        line_end=idx,
                        message=(
                            "This added line is long enough to be difficult to scan "
                            "during reviews."
                        ),
                        suggestion="Break it into named intermediate values or multiple lines.",
                        confidence=0.61,
                    )
                )
        return comments
