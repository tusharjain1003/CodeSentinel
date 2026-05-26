from __future__ import annotations

import asyncio
from dataclasses import dataclass

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


@dataclass
class CoordinationResult:
    comments: list[ReviewComment]
    agent_reviews: list[AgentReview]
    model_used: str


class ReviewCoordinator:
    def __init__(
        self,
        model_name: str | None = None,
        fallback_model_name: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.fallback_model_name = fallback_model_name
        heuristic_fallback = model_name is None
        self.agents = [
            BugAgent(model_name=model_name, heuristic_fallback=heuristic_fallback),
            SecurityAgent(model_name=model_name, heuristic_fallback=heuristic_fallback),
            StyleAgent(model_name=model_name, heuristic_fallback=heuristic_fallback),
        ]

    async def coordinate(self, diff: str, file_path: str) -> list[ReviewComment]:
        result = await self.coordinate_with_reviews(diff, file_path)
        return result.comments

    async def coordinate_with_reviews(self, diff: str, file_path: str) -> CoordinationResult:
        try:
            agent_reviews = await self._run_agents(diff, file_path)
            return self._build_result(agent_reviews, self.model_name or "heuristic")
        except Exception:
            if not self.fallback_model_name:
                raise
            fallback = ReviewCoordinator(model_name=self.fallback_model_name)
            try:
                return await fallback.coordinate_with_reviews(diff, file_path)
            except Exception:
                return CoordinationResult(
                    comments=[],
                    agent_reviews=[],
                    model_used="unavailable",
                )

    async def _run_agents(self, diff: str, file_path: str) -> list[AgentReview]:
        return await asyncio.gather(*[agent.review(diff, file_path) for agent in self.agents])

    def _build_result(
        self,
        agent_reviews: list[AgentReview],
        model_used: str,
    ) -> CoordinationResult:
        all_comments = [comment for review in agent_reviews for comment in review.comments]
        deduped = self._deduplicate(all_comments)
        ranked = sorted(
            deduped,
            key=lambda comment: (SEVERITY_ORDER[comment.severity], -comment.confidence),
        )
        return CoordinationResult(
            comments=ranked[:20],
            agent_reviews=agent_reviews,
            model_used=model_used,
        )

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
