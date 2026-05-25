from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

from agents.coordinator import ReviewCoordinator
from evals.benchmark import load_benchmark
from evals.metrics import match_comments, precision_recall_f1


async def run_benchmark(smoke: bool = False) -> dict[str, float]:
    samples = load_benchmark()
    if smoke:
        samples = samples[:5]
    coordinator = ReviewCoordinator()

    totals = {"tp": 0, "fp": 0, "fn": 0}
    for sample in samples:
        predicted = await coordinator.coordinate(sample["diff"], sample["file_path"])
        tp, fp, fn = match_comments(predicted, sample.get("ground_truth_comments", []))
        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn

    metrics = precision_recall_f1(totals["tp"], totals["fp"], totals["fn"])
    output = {
        "created_at": datetime.now(UTC).isoformat(),
        "samples": len(samples),
        **metrics,
    }
    Path("evals/results").mkdir(parents=True, exist_ok=True)
    Path("evals/results/latest.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    print(asyncio.run(run_benchmark(smoke=True)))
