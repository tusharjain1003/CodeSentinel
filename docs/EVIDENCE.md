# Evidence Checklist

CodeSentinel is currently positioned as a structured multi-agent PR review system with QLoRA/vLLM scaffolding. Use this checklist before claiming fine-tuned model improvements on a resume.

## Benchmark

- Create `data/eval_benchmark.jsonl` with 30-50 manually verified PR diff samples.
- Include `diff`, `file_path`, `ground_truth_comments`, and `source` for every sample.
- Cover bugs, security, style, and maintainability comments.
- Include several languages if possible: Python, TypeScript/JavaScript, Go, and Java.

## Model Runs

Run the same benchmark for each model:

```bash
python -m evals.run_eval --model base
python -m evals.run_eval --model finetuned
python -m evals.run_eval --model gpt4o
```

Expected tracked artifacts after a real run:

- `evals/results/base.json`
- `evals/results/finetuned.json`
- `evals/results/gpt4o.json`
- `evals/results/latest.json`

## Training Proof

Record:

- Dataset size and source repositories.
- Train/validation/eval split.
- Base model and LoRA configuration.
- Epochs, final train loss, final eval loss, and JSON parse rate.
- W&B run link or screenshot.
- Hugging Face adapter/model link, or a note that the adapter is private.

## Demo Proof

- Add a screenshot or GIF showing CodeSentinel posting inline comments on a real GitHub PR.
- Include the PR link if it is public.

## Resume-Safe Wording

Before proof:

> AI-powered PR review system with concurrent specialist agents, structured function-call outputs, GitHub PR integration, and QLoRA/vLLM fine-tuning scaffolding.

After proof:

> Fine-tuned Qwen2.5-Coder reviewer evaluated against base and GPT-4o on a held-out PR benchmark, served through vLLM, and integrated with GitHub inline review comments.
