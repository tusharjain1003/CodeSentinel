import json

import modal

app = modal.App("codesentinel-eval")

eval_image = (
    modal.Image.debian_slim()
    .pip_install(
        "transformers>=4.43",
        "accelerate>=0.33",
        "torch>=2.3",
    )
    .add_local_dir(
        ".",
        remote_path="/repo",
        ignore=["*.pyc", "__pycache__", ".git", ".venv", ".env", ".pytest_cache", ".ruff_cache", "node_modules"],
    )
)

checkpoints_vol = modal.Volume.from_name("codesentinel-checkpoints", create_if_missing=True)


def build_prompt(diff: str, file_path: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": "You are an expert code reviewer. Analyze the provided code diff and identify issues. Respond only with a valid JSON object matching the ReviewComment schema.",
        },
        {
            "role": "user",
            "content": f"Review this code change in `{file_path}`:\n\n```diff\n{diff}\n```\n\nIdentify any issues. Respond with a JSON ReviewComment object.",
        },
    ]


@app.function(
    image=eval_image,
    gpu="A10G",
    timeout=600,
    volumes={"/checkpoints": checkpoints_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def eval_benchmark() -> str:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = "/checkpoints/codesentinel-merged"
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        fix_mistral_regex=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
        device_map="auto",
    )

    benchmark_path = "/repo/data/eval_benchmark.jsonl"
    with open(benchmark_path) as f:
        samples = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(samples)} benchmark samples")

    tp, fp, fn = 0, 0, 0

    for i, sample in enumerate(samples):
        messages = build_prompt(sample["diff"], sample["file_path"])
        ground_truth = sample.get("ground_truth_comments", [])

        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)

        outputs = model.generate(
            inputs,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

        response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        print(f"\n--- Sample {i+1}/{len(samples)} ---")

        try:
            predicted_json = json.loads(response)
            predicted_list = [predicted_json] if isinstance(predicted_json, dict) else predicted_json
        except json.JSONDecodeError:
            predicted_list = []

        sample_tp = 0
        sample_fn = len(ground_truth)
        sample_fp = 0

        for pred in predicted_list:
            matched = False
            for j, truth in enumerate(ground_truth):
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
        print(f"  Sample: tp={sample_tp}, fp={sample_fp}, fn={sample_fn}, total_truth={len(ground_truth)}")

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    results = {
        "model": "finetuned",
        "samples": len(samples),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
    }

    print(f"\n{'='*40}")
    print(f"RESULTS: {json.dumps(results, indent=2)}")

    print(f"\nAll {len(samples)} samples processed.")
    return json.dumps(results)


if __name__ == "__main__":
    result = eval_benchmark.remote()
    print(f"\nFinal result: {result}")
