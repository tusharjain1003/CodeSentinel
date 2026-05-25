# CodeSentinel

CodeSentinel is an AI-powered pull request review system. It combines specialist review agents, structured outputs, a FastAPI webhook/API service, a Vite dashboard, and scaffolding for QLoRA fine-tuning plus vLLM serving.

Current status: the multi-agent review pipeline, schema contracts, GitHub integration, dashboard, and training/eval scaffolds are implemented. Fine-tuned model quality claims are intentionally not made until a real benchmark, training run, and model-comparison table are checked in.

## What Is Included

- FastAPI app with GitHub webhook, manual review trigger, metrics, reviews, feedback, and health endpoints.
- Multi-agent review pipeline for bugs, security, and maintainability.
- Pydantic `ReviewComment` schema used across agents, evals, and GitHub posting.
- Diff parsing, validation, and deduplication helpers.
- Dataset collection, cleaning, and ChatML formatting scripts that preserve GitHub review line metadata when available.
- QLoRA training config and scripts for training, merging adapters, and pushing to Hugging Face.
- vLLM serving helpers and Modal deployment placeholder.
- Evaluation framework with precision, recall, F1, model-specific result files, and GPT-4o judge hook.
- Vite/React dashboard for model comparison, recent reviews, and manual PR review.
- Docker, Railway config, GitHub Actions CI, and tests.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python -m pytest -q
uvicorn app:app --reload --port 8765
```

In another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

## Configuration

Copy `.env.example` to `.env` and fill in values as needed.

- `GITHUB_TOKEN` is required to post reviews.
- `GITHUB_WEBHOOK_SECRET` should match the secret configured in your GitHub webhook.
- `VLLM_BASE_URL` points to the vLLM OpenAI-compatible endpoint.
- `OPENAI_API_KEY` is used for judge/evaluation flows and can also serve as a fallback key.

## API

- `GET /health`
- `POST /webhook/github`
- `POST /api/review`
- `GET /api/reviews`
- `GET /api/reviews/{review_id}`
- `GET /api/eval/metrics`
- `POST /api/feedback`

## Training Flow

```bash
pip install -r requirements-training.txt
python -m data.collect
python -m training.train
```

Training samples should preserve real review metadata from GitHub review comments. `data.format.format_sample` reads `path`, `line`, `original_line`, `start_line`, and `original_start_line` fields when present instead of defaulting every label to line 1.

After training:

```python
from training.merge import merge_and_save

merge_and_save(
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    "checkpoints",
    "models/codesentinel-merged",
)
```

## Serving Flow

```bash
pip install -r requirements-serving.txt
python -m serving.server models/codesentinel-merged
```

The API will call the model through `VLLM_BASE_URL`.

## Evaluation

Create `data/eval_benchmark.jsonl` with held-out samples. A tiny schema/example seed is available at `data/eval_benchmark.example.jsonl`; it is not a substitute for the real 30-50+ manually verified benchmark.

```bash
python -m evals.run_eval --model base --smoke
python -m evals.run_eval --model finetuned
python -m evals.run_eval --model gpt4o
```

Results are written to `evals/results/{model}.json` and `evals/results/latest.json`. The tracked `evals/results/example.schema.json` shows the expected result shape.

## Evidence Needed Before Resume Claims

- Add a manually checked benchmark with 30-50 real PR diff samples and ground-truth comments.
- Run the same benchmark for base Qwen2.5-Coder, the fine-tuned model, and GPT-4o.
- Check in the result JSONs and update the README with a real comparison table.
- Add a screenshot or GIF showing CodeSentinel posting inline GitHub PR comments.
- Add training evidence: dataset size, train/val split, epochs, final loss, LoRA config, and observed eval improvement.

Until those artifacts exist, describe this repo as:

> AI-powered PR review system with multi-agent structured review pipeline, GitHub integration, and QLoRA/vLLM fine-tuning scaffold.
