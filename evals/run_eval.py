from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from agents.coordinator import ReviewCoordinator
from config import settings
from evals.benchmark import load_benchmark
from evals.metrics import match_comments_detailed, precision_recall_f1, safe_divide

MODEL_ALIASES = {
    "base": settings.base_model_name,
    "finetuned": settings.finetuned_model_name,
    "gpt4o": settings.gpt4o_model_name,
    "groq": settings.groq_model_name,
    "heuristic": None,
}


async def run_benchmark(
    model: str = "heuristic",
    smoke: bool = False,
    judge_quality: bool = False,
) -> dict[str, float]:
    samples = load_benchmark()
    if smoke:
        samples = samples[:5]
    model_name = MODEL_ALIASES.get(model, model)
    coordinator = ReviewCoordinator(model_name=model_name)

    totals = {"tp": 0, "fp": 0, "fn": 0, "severity_correct": 0}
    quality_scores: list[int] = []
    parse_success = 0
    latencies_ms: list[int] = []
    for sample in samples:
        started = time.monotonic()
        try:
            predicted = await coordinator.coordinate(sample["diff"], sample["file_path"])
            parse_success += 1
        finally:
            latencies_ms.append(int((time.monotonic() - started) * 1000))
        detailed = match_comments_detailed(predicted, sample.get("ground_truth_comments", []))
        totals["tp"] += detailed["tp"]
        totals["fp"] += detailed["fp"]
        totals["fn"] += detailed["fn"]
        totals["severity_correct"] += detailed["severity_correct"]

        if judge_quality and detailed["matches"]:
            from evals.judge import judge_comment_quality

            for match in detailed["matches"]:
                score = await judge_comment_quality(
                    sample["diff"],
                    match.predicted,
                    str(match.truth.get("message", "")),
                )
                quality_scores.append(score)

    metrics = precision_recall_f1(totals["tp"], totals["fp"], totals["fn"])
    avg_latency_ms = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else None
    output = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "model_name": model_name or "heuristic-fallback",
        "samples": len(samples),
        "totals": totals,
        "severity_accuracy": safe_divide(totals["severity_correct"], totals["tp"]),
        "json_parse_rate": safe_divide(parse_success, len(samples)),
        "avg_latency_ms": avg_latency_ms,
        "quality": avg_quality,
        **metrics,
    }
    Path("evals/results").mkdir(parents=True, exist_ok=True)
    Path(f"evals/results/{model}.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    Path("evals/results/latest.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    return {
        **metrics,
        "severity_accuracy": output["severity_accuracy"],
        "json_parse_rate": output["json_parse_rate"],
        "avg_latency_ms": avg_latency_ms,
        "quality": avg_quality,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="heuristic")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--judge-quality", action="store_true")
    args = parser.parse_args()
    print(
        asyncio.run(
            run_benchmark(
                model=args.model,
                smoke=args.smoke,
                judge_quality=args.judge_quality,
            )
        )
    )
