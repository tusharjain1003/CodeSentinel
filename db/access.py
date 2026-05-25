from __future__ import annotations

from typing import Any

import asyncpg

from config import settings


async def get_pool() -> asyncpg.Pool:
    return await asyncpg.create_pool(settings.database_url)


async def list_reviews(limit: int = 20, offset: int = 0) -> list[dict[str, Any]]:
    pool = await get_pool()
    async with pool:
        rows = await pool.fetch(
            """
            SELECT id, pr_url, repo, pr_number, model_used, comments, created_at
            FROM reviews
            ORDER BY created_at DESC
            LIMIT $1 OFFSET $2
            """,
            limit,
            offset,
        )
        return [dict(row) for row in rows]


async def get_review(review_id: str) -> dict[str, Any] | None:
    pool = await get_pool()
    async with pool:
        row = await pool.fetchrow("SELECT * FROM reviews WHERE id = $1", review_id)
        return dict(row) if row else None


async def insert_feedback(
    review_id: str,
    comment_idx: int,
    rating: int,
    correction: str | None = None,
) -> None:
    pool = await get_pool()
    async with pool:
        await pool.execute(
            """
            INSERT INTO review_feedback (review_id, comment_idx, rating, correction)
            VALUES ($1, $2, $3, $4)
            """,
            review_id,
            comment_idx,
            rating,
            correction,
        )
