import asyncio
import json
import re
from pathlib import Path

from openai import AsyncOpenAI

from config import settings

client = AsyncOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=settings.groq_api_key,
)


def build_prompt(diff: str, file_path: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "You are an expert code reviewer. Analyze the provided code diff and "
                "identify issues. Return a JSON object with keys: category (one of: "
                "bug, security, style, maintainability), severity (one of: critical, "
                "major, minor, nit), file_path, line_start, line_end, message, "
                "suggestion (or null), confidence (0.0 to 1.0). If the code has no "
                "issues, return an empty JSON object {}."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Review this code change in `{file_path}`:\n\n"
                f"```diff\n{diff}\n```\n\n"
                "Return ONLY valid JSON, no markdown or backticks."
            ),
        },
    ]


async def run_eval():
    benchmark_path = "data/eval_benchmark.jsonl"
    with open(benchmark_path) as f:
        samples = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(samples)} samples\n")

    tp, fp, fn = 0, 0, 0

    for i, sample in enumerate(samples):
        messages = build_prompt(sample["diff"], sample["file_path"])
        ground_truth = sample.get("ground_truth_comments", [])

        response = await client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.1,
            max_tokens=2048,
        )

        text = response.choices[0].message.content or ""
        print(f"--- Sample {i+1}/{len(samples)} ---")
        print(f"  File: {sample['file_path']}")

        def _parse(text: str) -> list[dict]:
            """Parse JSON from response, handling markdown fences and empty objects."""
            raw = text.strip()
            # Strip markdown code fences
            m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
            if m:
                raw = m.group(1).strip()
            if not raw:
                return []
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                print(f"  PARSE FAIL, raw 200 chars: {raw[:200]}")
                return []
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                # empty dict means "no issues"
                if not parsed:
                    return []
                return [parsed]
            return []

        predicted_list = _parse(text)

        sample_tp, sample_fn, sample_fp = 0, len(ground_truth), 0

        for pred in predicted_list:
            matched = False
            for truth in ground_truth:
                same_cat = pred.get("category") == truth.get("category")
                close_line = abs(pred.get("line_start", 0) - int(truth.get("line_start", 0))) <= 3
                if same_cat and close_line:
                    matched = True
                    sample_fn -= 1
                    break
            if matched:
                sample_tp += 1
            else:
                sample_fp += 1

        tp += sample_tp
        fp += sample_fp
        fn += sample_fn
        print(f"  tp={sample_tp} fp={sample_fp} fn={sample_fn}")

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    results = {
        "model": "groq-llama3.3-70b",
        "samples": len(samples),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
    }

    print(f"\n{'='*50}")
    print(json.dumps(results, indent=2))

    Path("evals/results/groq.json").parent.mkdir(parents=True, exist_ok=True)
    Path("evals/results/groq.json").write_text(json.dumps(results, indent=2))

    return results


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    results = loop.run_until_complete(run_eval())
    print(f"\nDone: {json.dumps(results)}")
