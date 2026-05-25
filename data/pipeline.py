from __future__ import annotations

import json
import random
from pathlib import Path

from data.clean import clean_diff_hunk, is_valid_comment
from data.format import format_sample, infer_category

RAW_PATH = "data/raw/pr_samples.json"
TRAIN_PATH = "data/train.jsonl"
VAL_PATH = "data/val.jsonl"
VAL_SPLIT = 0.1
SEED = 42


def run(raw_path: str = RAW_PATH, train_path: str = TRAIN_PATH, val_path: str = VAL_PATH) -> dict:
    random.seed(SEED)

    with open(raw_path, encoding="utf-8") as f:
        pr_samples = json.load(f)

    formatted = []
    skipped_no_comments = 0
    skipped_invalid = 0
    skipped_no_diff = 0
    category_counts: dict[str, int] = {}

    for pr in pr_samples:
        comments = pr.get("comments", [])
        if not comments:
            skipped_no_comments += 1
            continue

        for comment in comments:
            diff_hunk = comment.get("diff_hunk", "")
            if not diff_hunk or len(diff_hunk.strip()) < 20:
                skipped_no_diff += 1
                continue

            if not is_valid_comment(comment):
                skipped_invalid += 1
                continue

            cleaned_hunk = clean_diff_hunk(diff_hunk)
            sample = format_sample(
                diff_hunk=cleaned_hunk,
                file_path=pr.get("file_path", ""),
                comment=comment,
            )
            body = (comment.get("body") or "").strip().lower()
            cat = infer_category(body)
            category_counts[cat] = category_counts.get(cat, 0) + 1
            formatted.append(sample)

    random.shuffle(formatted)
    split_idx = max(1, int(len(formatted) * (1 - VAL_SPLIT)))
    train = formatted[:split_idx]
    val = formatted[split_idx:]

    Path(train_path).parent.mkdir(parents=True, exist_ok=True)
    with open(train_path, "w", encoding="utf-8") as f:
        for sample in train:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    with open(val_path, "w", encoding="utf-8") as f:
        for sample in val:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    stats = {
        "pr_samples_loaded": len(pr_samples),
        "total_formatted": len(formatted),
        "train_samples": len(train),
        "val_samples": len(val),
        "skipped_no_comments": skipped_no_comments,
        "skipped_invalid": skipped_invalid,
        "skipped_no_diff": skipped_no_diff,
        "category_counts": category_counts,
    }
    return stats


if __name__ == "__main__":
    stats = run()
    print(json.dumps(stats, indent=2))
