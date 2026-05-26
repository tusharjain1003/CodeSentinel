from __future__ import annotations

import logging
from typing import Any, TypedDict

try:
    from langgraph.graph import END, StateGraph
except Exception:  # pragma: no cover - optional dependency in local smoke tests
    END = "__end__"
    StateGraph = None

from agents.base import ReviewComment
from agents.coordinator import ReviewCoordinator
from config import settings
from pipeline.parse_pr import diff_hash as hash_diff
from pipeline.parse_pr import parse_diff
from pipeline.post_review import post_pr_review

logger = logging.getLogger(__name__)


class ReviewState(TypedDict):
    pr_url: str
    repo: str
    pr_number: int
    diff: str
    parsed_hunks: list[dict[str, str]]
    final_comments: list[dict[str, Any]]
    agent_reviews: list[dict[str, Any]]
    timing_ms: dict[str, int]
    token_cost: dict[str, Any]
    model_used: str
    posted_to_github: bool
    session_id: str


async def node_parse_pr(state: ReviewState) -> ReviewState:
    hunks = parse_diff(state["diff"])
    state["parsed_hunks"] = [
        {"file_path": hunk.file_path, "hunk": hunk.hunk}
        for hunk in hunks
    ]
    return state


async def node_run_agents(state: ReviewState) -> ReviewState:
    coordinator = ReviewCoordinator(
        model_name=None,  # Use heuristic-only mode (no LLM calls needed)
        fallback_model_name=settings.gpt4o_model_name,
    )
    comments: list[ReviewComment] = []
    agent_reviews = []
    for hunk in state["parsed_hunks"]:
        result = await coordinator.coordinate_with_reviews(hunk["hunk"], hunk["file_path"])
        comments.extend(result.comments)
        agent_reviews.extend(result.agent_reviews)
        state["model_used"] = result.model_used
    state["final_comments"] = [comment.model_dump(mode="json") for comment in comments[:20]]
    state["agent_reviews"] = [review.model_dump(mode="json") for review in agent_reviews]
    timing_ms: dict[str, int] = {}
    token_cost: dict[str, Any] = {}
    for review in agent_reviews:
        timing_ms[review.agent_name] = timing_ms.get(review.agent_name, 0) + review.timing_ms
        if review.token_usage:
            token_cost[review.agent_name] = review.token_usage
    state["timing_ms"] = timing_ms
    state["token_cost"] = token_cost
    return state


async def node_post_review(state: ReviewState) -> ReviewState:
    comments = [ReviewComment(**comment) for comment in state["final_comments"]]
    try:
        state["posted_to_github"] = await post_pr_review(state["repo"], state["pr_number"], comments)
    except Exception as exc:
        logger.info("Failed to post review to GitHub: %s", exc)
        state["posted_to_github"] = False
    return state


async def node_persist(state: ReviewState) -> ReviewState:
    try:
        from db.access import insert_review

        await insert_review(
            review_id=state["session_id"],
            pr_url=state["pr_url"],
            repo=state["repo"],
            pr_number=state["pr_number"],
            diff_hash=hash_diff(state["diff"]),
            model_used=state.get("model_used") or settings.finetuned_model_name,
            comments=state["final_comments"],
            timing_ms=state.get("timing_ms", {}),
            token_cost=state.get("token_cost", {}),
        )
    except Exception as exc:
        logger.info("Review pipeline completed but persistence failed: %s", exc)
    return state


async def run_pipeline(state: ReviewState) -> ReviewState:
    state = await node_parse_pr(state)
    state = await node_run_agents(state)
    state = await node_post_review(state)
    return await node_persist(state)


def build_graph():
    if StateGraph is None:
        return None
    graph = StateGraph(ReviewState)
    graph.add_node("parse_pr", node_parse_pr)
    graph.add_node("run_agents", node_run_agents)
    graph.add_node("post_review", node_post_review)
    graph.add_node("persist", node_persist)
    graph.set_entry_point("parse_pr")
    graph.add_edge("parse_pr", "run_agents")
    graph.add_edge("run_agents", "post_review")
    graph.add_edge("post_review", "persist")
    graph.add_edge("persist", END)
    return graph.compile()


pipeline = build_graph()
