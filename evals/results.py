from __future__ import annotations

import json
from pathlib import Path
from typing import Any

RESULTS_DIR = Path("evals/results")
MODEL_KEYS = ("base", "finetuned", "gpt4o", "groq", "heuristic")


def load_result_file(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def missing_result(model: str) -> dict[str, Any]:
    return {
        "model": model,
        "status": "missing",
        "samples": 0,
        "precision": None,
        "recall": None,
        "f1": None,
        "quality": None,
    }


def load_model_results(results_dir: Path = RESULTS_DIR) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for model in MODEL_KEYS:
        result = load_result_file(results_dir / f"{model}.json")
        results[model] = result if result else missing_result(model)
    return results


def load_latest_result(results_dir: Path = RESULTS_DIR) -> dict[str, Any]:
    result = load_result_file(results_dir / "latest.json")
    if result:
        return result
    return {
        "status": "not_run",
        "message": "Run python -m evals.run_eval --model <name> after creating benchmark data.",
    }
