from __future__ import annotations

from github import Github

from agents.base import ReviewComment
from config import settings


def build_review_body(comments: list[ReviewComment]) -> str:
    counts = {severity: 0 for severity in ("critical", "major", "minor", "nit")}
    for comment in comments:
        counts[comment.severity.value] += 1
    return f"""## CodeSentinel Review

| Severity | Count |
|---|---:|
| Critical | {counts["critical"]} |
| Major | {counts["major"]} |
| Minor | {counts["minor"]} |
| Nit | {counts["nit"]} |

Reviewed by CodeSentinel multi-agent pipeline."""


async def post_pr_review(repo: str, pr_number: int, comments: list[ReviewComment]) -> None:
    if not settings.github_token:
        return

    github = Github(settings.github_token)
    repo_obj = github.get_repo(repo)
    pr = repo_obj.get_pull(pr_number)
    head_commit = repo_obj.get_commit(pr.head.sha)
    review_comments = []

    for comment in comments:
        body = f"**{comment.category.value} / {comment.severity.value}**\n\n{comment.message}"
        if comment.suggestion:
            body += f"\n\nSuggestion: {comment.suggestion}"
        body += f"\n\nConfidence: {comment.confidence:.0%}"
        review_comments.append(
            {
                "path": comment.file_path,
                "line": comment.line_end,
                "body": body,
            }
        )

    pr.create_review(
        commit=head_commit,
        body=build_review_body(comments),
        event="COMMENT",
        comments=review_comments,
    )
