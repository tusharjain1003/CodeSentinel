from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from agents.coordinator import ReviewCoordinator
from evals.benchmark import load_benchmark
from evals.metrics import match_comments, precision_recall_f1


async def run_benchmark(model: str = "heuristic", smoke: bool = False) -> dict[str, float]:
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
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "samples": len(samples),
        "totals": totals,
        **metrics,
    }
    Path("evals/results").mkdir(parents=True, exist_ok=True)
    Path(f"evals/results/{model}.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    Path("evals/results/latest.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="heuristic")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    print(asyncio.run(run_benchmark(model=args.model, smoke=args.smoke)))
