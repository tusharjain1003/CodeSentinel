from __future__ import annotations

from agents.base import BaseAgent, Category, ReviewComment, Severity, iter_added_lines


class SecurityAgent(BaseAgent):
    name = "security_scanner"
    focus = (
        "SQL injection, XSS, CSRF, path traversal, insecure deserialization, "
        "hardcoded secrets, insufficient input validation, broken authentication, "
        "and insecure dependencies"
    )
    system_prompt = """You are a security engineer specializing in application security.
Follow OWASP Top 10 and CWE/SANS Top 25.
Only flag real vulnerabilities, not unlikely exploit chains."""

    def _heuristic_review(self, diff: str, file_path: str) -> list[ReviewComment]:
        comments: list[ReviewComment] = []
        risky_tokens = ("password=", "api_key=", "secret=", "token=")
        for added_line in iter_added_lines(diff):
            line = added_line.content
            normalized = line.lower().replace(" ", "")
            if any(token in normalized for token in risky_tokens):
                comments.append(
                    ReviewComment(
                        category=Category.security,
                        severity=Severity.critical,
                        file_path=file_path,
                        line_start=added_line.line_number,
                        line_end=added_line.line_number,
                        message="This change appears to introduce a hardcoded secret.",
                        suggestion=(
                            "Move the value into a secret manager or environment "
                            "variable and rotate the exposed credential."
                        ),
                        confidence=0.82,
                    )
                )
            interpolated_sql = "execute(" in normalized and (
                "+" in line or "f\"" in line or "f'" in line
            )
            if interpolated_sql:
                comments.append(
                    ReviewComment(
                        category=Category.security,
                        severity=Severity.critical,
                        file_path=file_path,
                        line_start=added_line.line_number,
                        line_end=added_line.line_number,
                        message=(
                            "SQL built through string interpolation or concatenation "
                            "can allow injection."
                        ),
                        suggestion=(
                            "Use parameterized queries instead of constructing SQL "
                            "strings directly."
                        ),
                        confidence=0.78,
                    )
                )
        return comments
