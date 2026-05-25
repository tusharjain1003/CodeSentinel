from __future__ import annotations

import asyncio

from agents.base import AgentReview, ReviewComment, Severity
from agents.bug_agent import BugAgent
from agents.security_agent import SecurityAgent
from agents.style_agent import StyleAgent

SEVERITY_ORDER = {
    Severity.critical: 0,
    Severity.major: 1,
    Severity.minor: 2,
    Severity.nit: 3,
}


class ReviewCoordinator:
    def __init__(self) -> None:
        self.agents = [BugAgent(), SecurityAgent(), StyleAgent()]

    async def coordinate(self, diff: str, file_path: str) -> list[ReviewComment]:
        agent_reviews: list[AgentReview] = await asyncio.gather(
            *[agent.review(diff, file_path) for agent in self.agents]
        )
        all_comments = [comment for review in agent_reviews for comment in review.comments]
        deduped = self._deduplicate(all_comments)
        ranked = sorted(
            deduped,
            key=lambda comment: (SEVERITY_ORDER[comment.severity], -comment.confidence),
        )
        return ranked[:20]

    def _deduplicate(self, comments: list[ReviewComment]) -> list[ReviewComment]:
        seen: list[ReviewComment] = []
        for comment in comments:
            is_duplicate = any(
                comment.file_path == prior.file_path
                and abs(comment.line_start - prior.line_start) <= 2
                and self._similarity(comment.message, prior.message) > 0.85
                for prior in seen
            )
            if not is_duplicate:
                seen.append(comment)
        return seen

    def _similarity(self, a: str, b: str) -> float:
        left = set(a.lower().split())
        right = set(b.lower().split())
        return len(left & right) / len(left | right) if left | right else 0.0
