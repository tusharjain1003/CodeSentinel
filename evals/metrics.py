from __future__ import annotations

from agents.base import ReviewComment


def match_comments(
    predicted: list[ReviewComment],
    ground_truth: list[dict],
    line_tolerance: int = 3,
) -> tuple[int, int, int]:
    matched: set[int] = set()
    true_positive = 0

    for pred in predicted:
        for idx, truth in enumerate(ground_truth):
            if idx in matched:
                continue
            same_category = pred.category.value == truth.get("category")
            close_line = abs(pred.line_start - int(truth.get("line_start", 0))) <= line_tolerance
            if same_category and close_line:
                matched.add(idx)
                true_positive += 1
                break

    false_positive = len(predicted) - true_positive
    false_negative = len(ground_truth) - true_positive
    return true_positive, false_positive, false_negative


def precision_recall_f1(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}
