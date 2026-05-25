from __future__ import annotations

from typing import Any, TypedDict

try:
    from langgraph.graph import END, StateGraph
except Exception:  # pragma: no cover - optional dependency in local smoke tests
    END = "__end__"
    StateGraph = None

from agents.base import ReviewComment
from agents.coordinator import ReviewCoordinator
from pipeline.parse_pr import parse_diff
from pipeline.post_review import post_pr_review


class ReviewState(TypedDict):
    pr_url: str
    repo: str
    pr_number: int
    diff: str
    parsed_hunks: list[dict[str, str]]
    final_comments: list[dict[str, Any]]
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
    coordinator = ReviewCoordinator()
    comments: list[ReviewComment] = []
    for hunk in state["parsed_hunks"]:
        comments.extend(await coordinator.coordinate(hunk["hunk"], hunk["file_path"]))
    state["final_comments"] = [comment.model_dump(mode="json") for comment in comments[:20]]
    return state


async def node_post_review(state: ReviewState) -> ReviewState:
    comments = [ReviewComment(**comment) for comment in state["final_comments"]]
    await post_pr_review(state["repo"], state["pr_number"], comments)
    state["posted_to_github"] = True
    return state


async def node_persist(state: ReviewState) -> ReviewState:
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
