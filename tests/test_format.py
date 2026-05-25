from data.format import extract_line_range, format_sample


def test_extract_line_range_uses_github_review_comment_fields() -> None:
    comment = {
        "body": "This can fail when the key is missing.",
        "path": "src/app.py",
        "start_line": 12,
        "line": 14,
    }

    assert extract_line_range(comment) == (12, 14)


def test_format_sample_preserves_comment_path_and_line_numbers() -> None:
    sample = format_sample(
        diff_hunk="@@ -12,2 +12,3 @@\n+value = payload['key']",
        file_path="fallback.py",
        comment={
            "body": "This direct lookup can raise KeyError.",
            "path": "src/app.py",
            "line": 13,
        },
        category="bug",
    )

    payload = sample["messages"][2]["content"]
    assert '"file_path": "src/app.py"' in payload
    assert '"line_start": 13' in payload
    assert '"line_end": 13' in payload
