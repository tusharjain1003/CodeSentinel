from __future__ import annotations

import json
from typing import Any

import asyncpg

from config import settings


def _normalize_row(row: asyncpg.Record) -> dict[str, Any]:
    data = dict(row)
    for key in ("comments", "timing_ms", "token_cost"):
        if isinstance(data.get(key), str):
            data[key] = json.loads(data[key])
    return data


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
        return [_normalize_row(row) for row in rows]


async def get_review(review_id: str) -> dict[str, Any] | None:
    pool = await get_pool()
    async with pool:
        row = await pool.fetchrow("SELECT * FROM reviews WHERE id = $1", review_id)
        return _normalize_row(row) if row else None


async def get_review_by_diff_hash(diff_hash: str) -> dict[str, Any] | None:
    pool = await get_pool()
    async with pool:
        row = await pool.fetchrow(
            """
            SELECT *
            FROM reviews
            WHERE diff_hash = $1
            ORDER BY created_at DESC
            LIMIT 1
            """,
            diff_hash,
        )
        return _normalize_row(row) if row else None


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
    pool = await get_pool()
    async with pool:
        await pool.execute(
            """
            INSERT INTO reviews (
                id, pr_url, repo, pr_number, diff_hash, model_used, comments, timing_ms, token_cost
            )
            VALUES ($1::uuid, $2, $3, $4, $5, $6, $7::jsonb, $8::jsonb, $9::jsonb)
            ON CONFLICT (id) DO UPDATE SET
                comments = EXCLUDED.comments,
                timing_ms = EXCLUDED.timing_ms,
                token_cost = EXCLUDED.token_cost
            """,
            review_id,
            pr_url,
            repo,
            pr_number,
            diff_hash,
            model_used,
            json.dumps(comments),
            json.dumps(timing_ms or {}),
            json.dumps(token_cost or {}),
        )


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
            VALUES ($1::uuid, $2, $3, $4)
            """,
            review_id,
            comment_idx,
            rating,
            correction,
        )
