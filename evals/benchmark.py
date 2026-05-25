from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_benchmark(path: str = "data/eval_benchmark.jsonl") -> list[dict[str, Any]]:
    benchmark_path = Path(path)
    if not benchmark_path.exists():
        return []
    return [
        json.loads(line)
        for line in benchmark_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
