import pytest

from pipeline.graph import ReviewState, node_persist, node_post_review


def make_state() -> ReviewState:
    return {
        "pr_url": "https://github.com/example/repo/pull/1",
        "repo": "example/repo",
        "pr_number": 1,
        "diff": "diff --git a/app.py b/app.py\n@@ -1 +1 @@\n-old\n+new\n",
        "parsed_hunks": [],
        "final_comments": [],
        "posted_to_github": False,
        "session_id": "11111111-1111-1111-1111-111111111111",
    }


@pytest.mark.asyncio
async def test_node_post_review_uses_actual_post_result(monkeypatch) -> None:
    async def fake_post(repo, pr_number, comments):
        return False

    monkeypatch.setattr("pipeline.graph.post_pr_review", fake_post)

    state = await node_post_review(make_state())

    assert state["posted_to_github"] is False


@pytest.mark.asyncio
async def test_node_persist_inserts_completed_review(monkeypatch) -> None:
    captured = {}

    async def fake_insert_review(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("db.access.insert_review", fake_insert_review)

    state = await node_persist(make_state())

    assert state["session_id"] == captured["review_id"]
    assert captured["repo"] == "example/repo"
    assert captured["diff_hash"]
    assert captured["comments"] == []
