from __future__ import annotations

import time
from typing import Any


_reviews: dict[str, dict[str, Any]] = {}
_feedback: list[dict[str, Any]] = []


async def list_reviews(limit: int = 20, offset: int = 0) -> list[dict[str, Any]]:
    items = sorted(_reviews.values(), key=lambda r: r.get("created_at", ""), reverse=True)
    return items[offset : offset + limit]


async def get_review(review_id: str) -> dict[str, Any] | None:
    return _reviews.get(review_id)


async def get_review_by_diff_hash(diff_hash: str) -> dict[str, Any] | None:
    for review in _reviews.values():
        if review.get("diff_hash") == diff_hash:
            return review
    return None


async def insert_review(
    review_id: str,
    pr_url: str,
    repo: str,
    pr_number: int,
    diff_hash: str,
    model_used: str,
    comments: list[dict[str, Any]],
    timing_ms: dict[str, Any] | None = None,
    token_cost: dict[str, Any] | None = None,
) -> None:
    _reviews[review_id] = {
        "id": review_id,
        "pr_url": pr_url,
        "repo": repo,
        "pr_number": pr_number,
        "diff_hash": diff_hash,
        "model_used": model_used,
        "comments": comments,
        "timing_ms": timing_ms or {},
        "token_cost": token_cost or {},
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


async def insert_feedback(
    review_id: str,
    comment_idx: int,
    rating: int,
    correction: str | None = None,
) -> None:
    _feedback.append({
        "review_id": review_id,
        "comment_idx": comment_idx,
        "rating": rating,
        "correction": correction,
    })
