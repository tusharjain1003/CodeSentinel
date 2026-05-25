from pipeline.parse_pr import diff_hash, parse_diff, validate_diff


def test_parse_diff_groups_by_file() -> None:
    diff = """diff --git a/app.py b/app.py
index abc..def 100644
--- a/app.py
+++ b/app.py
@@ -1 +1 @@
-old
+new
diff --git a/readme.md b/readme.md
--- a/readme.md
+++ b/readme.md
@@ -1 +1 @@
+hello
"""
    hunks = parse_diff(diff)
    assert len(hunks) == 2
    assert hunks[0].file_path == "app.py"
    assert hunks[1].file_path == "readme.md"


def test_diff_hash_is_stable() -> None:
    assert diff_hash("abc") == diff_hash("abc")


def test_validate_diff_rejects_large_file_count() -> None:
    diff = "\n".join(f"diff --git a/{i}.py b/{i}.py" for i in range(51))
    try:
        validate_diff(diff)
    except ValueError as exc:
        assert "Too many files" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
