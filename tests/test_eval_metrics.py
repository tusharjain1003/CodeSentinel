from agents.base import Category, ReviewComment, Severity
from evals.metrics import match_comments_detailed, safe_divide


def test_match_comments_detailed_reports_severity_accuracy() -> None:
    predicted = [
        ReviewComment(
            category=Category.security,
            severity=Severity.critical,
            file_path="src/auth.py",
            line_start=12,
            line_end=12,
            message="SQL injection risk.",
            confidence=0.9,
        )
    ]
    ground_truth = [
        {
            "category": "security",
            "severity": "critical",
            "line_start": 13,
            "message": "Interpolated SQL can allow injection.",
        }
    ]

    detailed = match_comments_detailed(predicted, ground_truth)

    assert detailed["tp"] == 1
    assert detailed["fp"] == 0
    assert detailed["fn"] == 0
    assert detailed["severity_correct"] == 1
    assert detailed["matches"][0].predicted == predicted[0]


def test_safe_divide_handles_zero_denominator() -> None:
    assert safe_divide(1, 0) == 0.0
    assert safe_divide(3, 2) == 1.5
