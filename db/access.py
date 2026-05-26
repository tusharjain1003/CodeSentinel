from __future__ import annotations

import json
import logging
from typing import Any

import asyncpg

from config import settings

logger = logging.getLogger(__name__)
_pool: asyncpg.Pool | None = None
_use_memory = False
_memory_store = None


def _normalize_row(row: asyncpg.Record) -> dict[str, Any]:
    data = dict(row)
    for key in ("comments", "timing_ms", "token_cost"):
        if isinstance(data.get(key), str):
            data[key] = json.loads(data[key])
    return data


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(settings.database_url)
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


async def _get_store():
    global _use_memory, _memory_store
    if _use_memory:
        if _memory_store is None:
            from db.memory import (  # noqa: PLC0415
                get_review as mem_get_review,
                get_review_by_diff_hash as mem_get_review_by_diff_hash,
                insert_feedback as mem_insert_feedback,
                insert_review as mem_insert_review,
                list_reviews as mem_list_reviews,
            )

            _memory_store = {
                "list_reviews": mem_list_reviews,
                "get_review": mem_get_review,
                "get_review_by_diff_hash": mem_get_review_by_diff_hash,
                "insert_review": mem_insert_review,
                "insert_feedback": mem_insert_feedback,
            }
        return _memory_store
    return None


def _try_memory_fallback() -> bool:
    global _use_memory
    if not settings.allow_memory_db_fallback:
        return False
    if not _use_memory:
        _use_memory = True
        logger.info("Switched to in-memory store (Postgres unavailable)")
    return True


async def _query(sql: str, *args: Any) -> list[asyncpg.Record]:
    pool = await get_pool()
    return await pool.fetch(sql, *args)


async def _query_row(sql: str, *args: Any) -> asyncpg.Record | None:
    pool = await get_pool()
    return await pool.fetchrow(sql, *args)


async def _execute(sql: str, *args: Any) -> None:
    pool = await get_pool()
    await pool.execute(sql, *args)


async def list_reviews(limit: int = 20, offset: int = 0) -> list[dict[str, Any]]:
    store = await _get_store()
    if store:
        return await store["list_reviews"](limit=limit, offset=offset)
    try:
        rows = await _query(
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
    except Exception as exc:
        logger.info("Postgres unavailable for list_reviews: %s", exc)
        if not _try_memory_fallback():
            raise
        return await list_reviews(limit=limit, offset=offset)


async def get_review(review_id: str) -> dict[str, Any] | None:
    store = await _get_store()
    if store:
        return await store["get_review"](review_id)
    try:
        row = await _query_row(
            "SELECT * FROM reviews WHERE id = $1::uuid", review_id,
        )
        return _normalize_row(row) if row else None
    except Exception as exc:
        logger.info("Postgres unavailable for get_review: %s", exc)
        if not _try_memory_fallback():
            raise
        return await get_review(review_id)


async def get_review_by_diff_hash(diff_hash: str) -> dict[str, Any] | None:
    store = await _get_store()
    if store:
        return await store["get_review_by_diff_hash"](diff_hash)
    try:
        row = await _query_row(
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
    except Exception as exc:
        logger.info("Postgres unavailable for get_review_by_diff_hash: %s", exc)
        if not _try_memory_fallback():
            raise
        return await get_review_by_diff_hash(diff_hash)


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
    store = await _get_store()
    if store:
        return await store["insert_review"](
            review_id, pr_url, repo, pr_number, diff_hash, model_used,
            comments, timing_ms, token_cost,
        )
    try:
        await _execute(
            """
            INSERT INTO reviews (
                id, pr_url, repo, pr_number, diff_hash, model_used,
                comments, timing_ms, token_cost
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
    except Exception as exc:
        logger.info("Postgres unavailable for insert_review: %s", exc)
        if not _try_memory_fallback():
            raise
        return await insert_review(
            review_id, pr_url, repo, pr_number, diff_hash, model_used,
            comments, timing_ms, token_cost,
        )


async def insert_feedback(
    review_id: str,
    comment_idx: int,
    rating: int,
    correction: str | None = None,
) -> None:
    store = await _get_store()
    if store:
        return await store["insert_feedback"](review_id, comment_idx, rating, correction)
    try:
        await _execute(
            """
            INSERT INTO review_feedback (review_id, comment_idx, rating, correction)
            VALUES ($1::uuid, $2, $3, $4)
            """,
            review_id,
            comment_idx,
            rating,
            correction,
        )
    except Exception as exc:
        logger.info("Postgres unavailable for insert_feedback: %s", exc)
        if not _try_memory_fallback():
            raise
        return await insert_feedback(review_id, comment_idx, rating, correction)
