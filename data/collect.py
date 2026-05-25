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


TARGET_PER_REPO = 30
PER_PAGE = 100
MAX_PAGES = 5


async def collect_repo(repo: str) -> list[dict]:
    print(f"  Collecting from {repo} ...")
    async with httpx.AsyncClient(timeout=30) as client:
        samples: list[dict] = []
        page = 1
        while len(samples) < TARGET_PER_REPO and page <= MAX_PAGES:
            pulls = await fetch_json(
                client,
                f"https://api.github.com/repos/{repo}/pulls",
                {"state": "closed", "per_page": PER_PAGE, "page": page, "sort": "updated", "direction": "desc"},
            )
            if not pulls:
                break
            print(f"    Page {page}: {len(pulls)} PRs (total samples so far: {len(samples)})")
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
                    follow_redirects=True,
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
                if len(samples) >= TARGET_PER_REPO:
                    break
            page += 1
        print(f"    Collected {len(samples)} samples from {repo}")
        return samples


async def main() -> None:
    results = []
    for repo in REPOS:
        repo_samples = await collect_repo(repo)
        results.extend(repo_samples)
        print(f"  Running total: {len(results)} samples")
    out_path = "data/raw/pr_samples.json"
    Path(out_path).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {len(results)} samples to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
