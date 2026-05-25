# CodeSentinel

CodeSentinel is an AI-powered pull request review system. It combines specialist review agents, structured outputs, a FastAPI webhook/API service, a Vite dashboard, and scaffolding for QLoRA fine-tuning plus vLLM serving.

## What Is Included

- FastAPI app with GitHub webhook, manual review trigger, metrics, reviews, feedback, and health endpoints.
- Multi-agent review pipeline for bugs, security, and maintainability.
- Pydantic `ReviewComment` schema used across agents, evals, and GitHub posting.
- Diff parsing, validation, and deduplication helpers.
- Dataset collection, cleaning, and ChatML formatting scripts.
- QLoRA training config and scripts for training, merging adapters, and pushing to Hugging Face.
- vLLM serving helpers and Modal deployment placeholder.
- Evaluation framework with precision, recall, F1, and GPT-4o judge hook.
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

Create `data/eval_benchmark.jsonl` with held-out samples, then run:

```bash
python -m evals.run_eval
```

Results are written to `evals/results/latest.json`.
