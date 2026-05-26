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
    "django/django",
    "kubernetes/kubernetes",
    "microsoft/vscode",
    "nodejs/node",
    "rust-lang/rust",
    "rails/rails",
    "homebrew/brew",
    "ansible/ansible",
    "spring-projects/spring-framework",
    "scikit-learn/scikit-learn",
    "numpy/numpy",
    "apache/airflow",
]


async def fetch_json(client: httpx.AsyncClient, url: str, params: dict | None = None) -> list[dict]:
    headers = {"Accept": "application/vnd.github+json"}
    if settings.github_token:
        headers["Authorization"] = f"Bearer {settings.github_token}"
    response = await client.get(url, params=params, headers=headers)
    if response.status_code == 429:
        import time
        retry_after = int(response.headers.get("retry-after", 60))
        print(f"    Rate limited, waiting {retry_after}s...")
        time.sleep(retry_after)
        response = await client.get(url, params=params, headers=headers)
    response.raise_for_status()
    return response.json()


TARGET_PER_REPO = 50
PER_PAGE = 100
MAX_PAGES = 10


async def collect_repo(repo: str) -> list[dict]:
    print(f"  Collecting from {repo} ...")
    async with httpx.AsyncClient(timeout=60) as client:
        samples: list[dict] = []
        page = 1
        while len(samples) < TARGET_PER_REPO and page <= MAX_PAGES:
            pulls = await fetch_json(
                client,
                f"https://api.github.com/repos/{repo}/pulls",
                {
                    "state": "closed",
                    "per_page": PER_PAGE,
                    "page": page,
                    "sort": "updated",
                    "direction": "desc",
                },
            )
            if not pulls:
                break
            print(f"    Page {page}: {len(pulls)} PRs (total samples so far: {len(samples)})")
            for pr in pulls:
                if not pr.get("merged_at"):
                    continue
                try:
                    comments = await fetch_json(
                        client,
                        f"https://api.github.com/repos/{repo}/pulls/{pr['number']}/comments",
                    )
                except Exception as exc:
                    print(f"      Skipping PR #{pr['number']}: {exc}")
                    continue
                if len(comments) < 1:
                    continue
                try:
                    diff_response = await client.get(
                        pr["diff_url"],
                        headers={"Accept": "application/vnd.github.v3.diff"},
                        follow_redirects=True,
                    )
                    diff_response.raise_for_status()
                except Exception as exc:
                    print(f"      Skipping diff for PR #{pr['number']}: {exc}")
                    continue
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
    sem = asyncio.Semaphore(4)

    async def limited(repo: str) -> list[dict]:
        async with sem:
            return await collect_repo(repo)

    results = await asyncio.gather(*[limited(repo) for repo in REPOS])
    all_samples = []
    for repo_samples in results:
        all_samples.extend(repo_samples)
    out_path = "data/raw/pr_samples.json"
    Path(out_path).write_text(json.dumps(all_samples, indent=2), encoding="utf-8")
    print(f"\nWrote {len(all_samples)} samples to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
