from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import httpx

MAX_DIFF_BYTES = 500_000
MAX_FILES = 50


@dataclass(frozen=True)
class DiffHunk:
    file_path: str
    hunk: str


def diff_hash(diff: str) -> str:
    return hashlib.sha256(diff.encode("utf-8")).hexdigest()


def validate_diff(diff: str) -> None:
    if len(diff.encode("utf-8")) > MAX_DIFF_BYTES:
        raise ValueError(f"Diff too large: {len(diff.encode('utf-8'))} bytes")
    files_changed = diff.count("diff --git")
    if files_changed > MAX_FILES:
        raise ValueError(f"Too many files: {files_changed}")


def parse_diff(diff: str) -> list[DiffHunk]:
    validate_diff(diff)
    hunks: list[DiffHunk] = []
    current_file = "unknown"
    current_lines: list[str] = []

    for line in diff.splitlines():
        if line.startswith("diff --git"):
            if current_lines:
                hunks.append(DiffHunk(file_path=current_file, hunk="\n".join(current_lines)))
                current_lines = []
            match = re.search(r" b/(.+)$", line)
            current_file = match.group(1) if match else current_file
            continue
        if line.startswith("+++ b/"):
            current_file = line.removeprefix("+++ b/")
            continue
        if line.startswith("index ") or line.startswith("Binary files"):
            continue
        current_lines.append(line)

    if current_lines:
        hunks.append(DiffHunk(file_path=current_file, hunk="\n".join(current_lines)))
    return [hunk for hunk in hunks if hunk.hunk.strip()]


async def fetch_pr_diff(diff_url: str, token: str = "") -> str:
    headers = {"Accept": "application/vnd.github.v3.diff"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        response = await client.get(diff_url, headers=headers)
        content_length = response.headers.get("content-length")
        if content_length and int(content_length) > MAX_DIFF_BYTES:
            raise ValueError(
                f"Diff too large: {int(content_length)} bytes "
                f"(max {MAX_DIFF_BYTES})",
            )
        response.raise_for_status()
        return response.text
