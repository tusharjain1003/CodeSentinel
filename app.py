from __future__ import annotations

import hashlib
import hmac
import logging
from typing import Optional
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from config import settings
from evals.results import load_latest_result, load_model_results
from pipeline.graph import ReviewState, run_pipeline
from pipeline.parse_pr import fetch_pr_diff

logger = logging.getLogger(__name__)

app = FastAPI(title="CodeSentinel", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


async def run_review_pipeline(
    pr_url: str,
    repo: str,
    pr_number: int,
    diff_url: str,
    session_id: str | None = None,
) -> ReviewState:
    diff = await fetch_pr_diff(diff_url, token=settings.github_token)
    state: ReviewState = {
        "pr_url": pr_url,
        "repo": repo,
        "pr_number": pr_number,
        "diff": diff,
        "parsed_hunks": [],
        "final_comments": [],
        "posted_to_github": False,
        "session_id": session_id or str(uuid4()),
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
    request: ManualReviewRequest,
    background_tasks: BackgroundTasks,
) -> dict[str, str]:
    review_id = str(uuid4())
    background_tasks.add_task(
        run_review_pipeline,
        pr_url=request.pr_url,
        repo=request.repo,
        pr_number=request.pr_number,
        diff_url=request.diff_url,
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
