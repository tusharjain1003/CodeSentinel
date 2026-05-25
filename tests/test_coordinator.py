from agents.base import Category, ReviewComment, Severity
from agents.coordinator import ReviewCoordinator


def test_deduplicate_similar_comments() -> None:
    coordinator = ReviewCoordinator()
    comments = [
        ReviewComment(
            category=Category.bug,
            severity=Severity.major,
            file_path="app.py",
            line_start=10,
            line_end=10,
            message="This value may be None and can crash",
            confidence=0.8,
        ),
        ReviewComment(
            category=Category.bug,
            severity=Severity.major,
            file_path="app.py",
            line_start=11,
            line_end=11,
            message="This value may be None and can crash",
            confidence=0.7,
        ),
    ]
    assert len(coordinator._deduplicate(comments)) == 1
