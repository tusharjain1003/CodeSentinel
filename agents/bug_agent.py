from __future__ import annotations

from agents.base import BaseAgent, Category, ReviewComment, Severity


class BugAgent(BaseAgent):
    name = "bug_detector"
    focus = (
        "logical errors, null dereferences, off-by-one errors, race conditions, "
        "incorrect error handling, resource leaks, and incorrect API usage"
    )
    system_prompt = """You are a senior software engineer specializing in bug detection.
Identify only genuine bugs, not style issues or subjective preferences.
For each bug, explain clearly why it is wrong and what the correct behavior should be."""

    def _heuristic_review(self, diff: str, file_path: str) -> list[ReviewComment]:
        comments: list[ReviewComment] = []
        for idx, line in enumerate(diff.splitlines(), start=1):
            added = line.startswith("+") and not line.startswith("+++")
            if added and "except:" in line:
                comments.append(
                    ReviewComment(
                        category=Category.bug,
                        severity=Severity.major,
                        file_path=file_path,
                        line_start=idx,
                        line_end=idx,
                        message=(
                            "Bare except blocks can hide real failures and make "
                            "review results unreliable."
                        ),
                        suggestion=(
                            "Catch the specific exception type and log or re-raise "
                            "unexpected failures."
                        ),
                        confidence=0.72,
                    )
                )
            if added and (".get(" in line and "None" not in line and "or" not in line):
                comments.append(
                    ReviewComment(
                        category=Category.bug,
                        severity=Severity.minor,
                        file_path=file_path,
                        line_start=idx,
                        line_end=idx,
                        message="Dictionary access via get may return None if the key is missing.",
                        suggestion=(
                            "Provide an explicit default or validate the value before "
                            "using it."
                        ),
                        confidence=0.55,
                    )
                )
        return comments
