from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx

from config import settings

REPOS = [
    "huggingface/transformers",
    "langchain-ai/langchain",
    "fastapi/fastapi",
    "pallets/flask",
    "psf/requests",
    "pytorch/pytorch",
]


async def fetch_json(client: httpx.AsyncClient, url: str, params: dict | None = None) -> list[dict]:
    headers = {"Accept": "application/vnd.github+json"}
    if settings.github_token:
        headers["Authorization"] = f"Bearer {settings.github_token}"
    response = await client.get(url, params=params, headers=headers)
    response.raise_for_status()
    return response.json()


async def collect_repo(repo: str, limit: int = 25) -> list[dict]:
    async with httpx.AsyncClient(timeout=30) as client:
        pulls = await fetch_json(
            client,
            f"https://api.github.com/repos/{repo}/pulls",
            {"state": "closed", "per_page": limit},
        )
        samples = []
        for pr in pulls:
            if not pr.get("merged_at"):
                continue
            comments = await fetch_json(
                client,
                f"https://api.github.com/repos/{repo}/pulls/{pr['number']}/comments",
            )
            if len(comments) < 2:
                continue
            diff_response = await client.get(
                pr["diff_url"],
                headers={"Accept": "application/vnd.github.v3.diff"},
            )
            diff_response.raise_for_status()
            samples.append(
                {
                    "repo": repo,
                    "pr_number": pr["number"],
                    "title": pr["title"],
                    "diff": diff_response.text,
                    "comments": comments,
                }
            )
        return samples


async def main() -> None:
    results = []
    for repo in REPOS:
        results.extend(await collect_repo(repo))
    Path("data/raw/pr_samples.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
