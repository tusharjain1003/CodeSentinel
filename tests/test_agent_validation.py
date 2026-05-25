import pytest
from pydantic import ValidationError

from agents.base import validate_review_result


def test_validate_review_result_applies_fallback_file_path() -> None:
    comments = validate_review_result(
        {
            "comments": [
                {
                    "category": "bug",
                    "severity": "major",
                    "line_start": 4,
                    "line_end": 4,
                    "message": "This can crash.",
                    "confidence": 0.8,
                }
            ]
        },
        fallback_file_path="src/app.py",
    )

    assert comments[0].file_path == "src/app.py"


def test_validate_review_result_rejects_invalid_line_range() -> None:
    with pytest.raises(ValidationError):
        validate_review_result(
            {
                "comments": [
                    {
                        "category": "bug",
                        "severity": "major",
                        "file_path": "src/app.py",
                        "line_start": 10,
                        "line_end": 2,
                        "message": "Line range is wrong.",
                        "confidence": 0.8,
                    }
                ]
            },
            fallback_file_path="src/app.py",
        )
