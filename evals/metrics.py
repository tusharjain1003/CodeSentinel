from __future__ import annotations

from dataclasses import dataclass

from agents.base import ReviewComment


@dataclass(frozen=True)
class CommentMatch:
    predicted: ReviewComment
    truth: dict


def match_comments(
    predicted: list[ReviewComment],
    ground_truth: list[dict],
    line_tolerance: int = 3,
) -> tuple[int, int, int]:
    detailed = match_comments_detailed(predicted, ground_truth, line_tolerance)
    return detailed["tp"], detailed["fp"], detailed["fn"]


def match_comments_detailed(
    predicted: list[ReviewComment],
    ground_truth: list[dict],
    line_tolerance: int = 3,
) -> dict:
    matched: set[int] = set()
    matches: list[CommentMatch] = []

    for pred in predicted:
        for idx, truth in enumerate(ground_truth):
            if idx in matched:
                continue
            same_category = pred.category.value == truth.get("category")
            close_line = abs(pred.line_start - int(truth.get("line_start", 0))) <= line_tolerance
            if same_category and close_line:
                matched.add(idx)
                matches.append(CommentMatch(predicted=pred, truth=truth))
                break

    true_positive = len(matches)
    false_positive = len(predicted) - true_positive
    false_negative = len(ground_truth) - true_positive
    severity_correct = sum(
        1
        for match in matches
        if match.predicted.severity.value == match.truth.get("severity")
    )
    return {
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "severity_correct": severity_correct,
        "matches": matches,
    }


def precision_recall_f1(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def safe_divide(numerator: int | float, denominator: int | float) -> float:
    return numerator / denominator if denominator else 0.0
