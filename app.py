from __future__ import annotations

import hashlib
import hmac
import logging
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, Optional
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import settings
from evals.results import load_latest_result, load_model_results
from pipeline.graph import ReviewState, run_pipeline
from pipeline.parse_pr import diff_hash, fetch_pr_diff

logger = logging.getLogger(__name__)
rate_limit_buckets: dict[str, Deque[float]] = defaultdict(deque)

app = FastAPI(title="CodeSentinel", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def seed_demo_reviews():
    try:
        from db.memory import insert_review as mem_insert  # noqa: PLC0415

        await mem_insert(
            review_id="00000000-0000-0000-0000-000000000001",
            pr_url="https://github.com/pallets/flask/pull/4300",
            repo="pallets/flask",
            pr_number=4300,
            diff_hash="abc123",
            model_used="heuristic",
            comments=[
                {
                    "category": "bug",
                    "severity": "critical",
                    "file_path": "src/flask/app.py",
                    "line_start": 245,
                    "line_end": 245,
                    "message": (
                        "Bare except blocks can hide real failures and make review "
                        "results unreliable."
                    ),
                    "suggestion": (
                        "Catch the specific exception type and log or re-raise "
                        "unexpected failures."
                    ),
                    "confidence": 0.72,
                },
                {
                    "category": "security",
                    "severity": "critical",
                    "file_path": "src/flask/config.py",
                    "line_start": 89,
                    "line_end": 89,
                    "message": "This change appears to introduce a hardcoded secret.",
                    "suggestion": (
                        "Move the value into a secret manager or environment variable "
                        "and rotate the exposed credential."
                    ),
                    "confidence": 0.82,
                },
                {
                    "category": "bug",
                    "severity": "minor",
                    "file_path": "src/flask/helpers.py",
                    "line_start": 156,
                    "line_end": 156,
                    "message": "Dictionary access via get may return None if the key is missing.",
                    "suggestion": (
                        "Provide an explicit default or validate the value before "
                        "using it."
                    ),
                    "confidence": 0.55,
                },
                {
                    "category": "security",
                    "severity": "major",
                    "file_path": "src/flask/db.py",
                    "line_start": 42,
                    "line_end": 42,
                    "message": (
                        "SQL built through string interpolation or concatenation can "
                        "allow injection."
                    ),
                    "suggestion": (
                        "Use parameterized queries instead of constructing SQL "
                        "strings directly."
                    ),
                    "confidence": 0.78,
                },
                {
                    "category": "maintainability",
                    "severity": "nit",
                    "file_path": "src/flask/views.py",
                    "line_start": 312,
                    "line_end": 312,
                    "message": (
                        "This added line is long enough to be difficult to scan "
                        "during reviews."
                    ),
                    "suggestion": "Break it into named intermediate values or multiple lines.",
                    "confidence": 0.61,
                },
            ],
            timing_ms={
                "bug_detector": 150,
                "security_scanner": 200,
                "style_reviewer": 100,
            },
        )
        logger.info("Seeded demo review for dashboard screenshots")
    except Exception as exc:
        logger.info("Could not seed demo review (non-critical): %s", exc)


class ManualReviewRequest(BaseModel):
    pr_url: str
    repo: str
    pr_number: int = Field(gt=0)
    diff_url: str


class FeedbackRequest(BaseModel):
    review_id: str
    comment_idx: int = Field(ge=0)
    rating: int = Field(ge=-1, le=1)
    correction: Optional[str] = None  # noqa: UP045


def verify_webhook_signature(request: Request, body: bytes) -> None:
    if settings.github_webhook_secret == "change-me":
        return
    signature = request.headers.get("X-Hub-Signature-256", "")
    expected = "sha256=" + hmac.new(
        settings.github_webhook_secret.encode("utf-8"),
        body,
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(signature, expected):
        raise HTTPException(status_code=401, detail="Invalid signature")


def enforce_rate_limit(request: Request) -> None:
    if settings.rate_limit_per_minute <= 0:
        return
    client_host = request.client.host if request.client else "unknown"
    now = time.monotonic()
    bucket = rate_limit_buckets[client_host]
    while bucket and now - bucket[0] > 60:
        bucket.popleft()
    if len(bucket) >= settings.rate_limit_per_minute:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    bucket.append(now)


async def run_review_pipeline(
    pr_url: str,
    repo: str,
    pr_number: int,
    diff_url: str,
    session_id: str | None = None,
) -> ReviewState:
    diff = await fetch_pr_diff(diff_url, token=settings.github_token)
    review_id = session_id or str(uuid4())
    diff_digest = diff_hash(diff)
    try:
        from db.access import get_review_by_diff_hash

        existing_review = await get_review_by_diff_hash(diff_digest)
    except Exception as exc:
        logger.info("Could not check cached review for %s: %s", pr_url, exc)
        existing_review = None
    if existing_review:
        return {
            "pr_url": pr_url,
            "repo": repo,
            "pr_number": pr_number,
            "diff": diff,
            "parsed_hunks": [],
            "final_comments": existing_review.get("comments", []),
            "agent_reviews": [],
            "timing_ms": existing_review.get("timing_ms") or {},
            "token_cost": existing_review.get("token_cost") or {},
            "model_used": existing_review.get("model_used", settings.finetuned_model_name),
            "posted_to_github": False,
            "session_id": str(existing_review.get("id", review_id)),
        }

    state: ReviewState = {
        "pr_url": pr_url,
        "repo": repo,
        "pr_number": pr_number,
        "diff": diff,
        "parsed_hunks": [],
        "final_comments": [],
        "agent_reviews": [],
        "timing_ms": {},
        "token_cost": {},
        "model_used": settings.finetuned_model_name,
        "posted_to_github": False,
        "session_id": review_id,
    }
    return await run_pipeline(state)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/webhook/github")
async def github_webhook(request: Request, background_tasks: BackgroundTasks) -> JSONResponse:
    body = await request.body()
    verify_webhook_signature(request, body)

    event = request.headers.get("X-GitHub-Event")
    payload = await request.json()

    if event == "pull_request" and payload.get("action") in {"opened", "synchronize"}:
        pr = payload["pull_request"]
        review_id = str(uuid4())
        background_tasks.add_task(
            run_review_pipeline,
            pr_url=pr["html_url"],
            repo=payload["repository"]["full_name"],
            pr_number=pr["number"],
            diff_url=pr["diff_url"],
            session_id=review_id,
        )

    return JSONResponse({"status": "accepted"})


@app.post("/api/review")
async def manual_review(
    request: Request,
    payload: ManualReviewRequest,
    background_tasks: BackgroundTasks,
) -> dict[str, str]:
    enforce_rate_limit(request)
    review_id = str(uuid4())
    background_tasks.add_task(
        run_review_pipeline,
        pr_url=payload.pr_url,
        repo=payload.repo,
        pr_number=payload.pr_number,
        diff_url=payload.diff_url,
        session_id=review_id,
    )
    return {"status": "accepted", "review_id": review_id}


@app.get("/api/reviews")
async def list_reviews(limit: int = 20, offset: int = 0) -> dict:
    try:
        from db.access import list_reviews as db_list_reviews

        reviews = await db_list_reviews(limit=limit, offset=offset)
    except Exception as exc:
        logger.info("Returning demo reviews because DB is unavailable: %s", exc)
        reviews = []
    return {"reviews": reviews}


@app.get("/api/reviews/{review_id}")
async def get_review(review_id: str) -> dict:
    try:
        from db.access import get_review as db_get_review

        review = await db_get_review(review_id)
    except Exception as exc:
        logger.info("DB unavailable while loading review %s: %s", review_id, exc)
        review = None
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")
    return review


@app.get("/api/eval/metrics")
async def get_eval_metrics() -> dict:
    return {
        "models": load_model_results(),
        "latest": load_latest_result(),
    }


@app.post("/api/feedback")
async def submit_feedback(request: FeedbackRequest) -> dict[str, str]:
    try:
        from db.access import insert_feedback

        await insert_feedback(
            request.review_id,
            request.comment_idx,
            request.rating,
            request.correction,
        )
    except Exception as exc:
        logger.info("Feedback accepted but DB write failed: %s", exc)
    return {"status": "accepted"}


frontend_dist = Path(__file__).parent / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
