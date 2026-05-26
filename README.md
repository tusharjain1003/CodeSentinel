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

See `docs/EVIDENCE.md` for the exact benchmark, training, and demo artifacts needed before making fine-tuned model improvement claims.

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
python -m data.collect                          # scrape PR comments from GitHub
python -m data.pipeline                         # clean, format, train/val split
python -m training.train                        # QLoRA fine-tune (requires GPU)
python -m training.merge                        # merge adapters into base model
```

### Training on Modal (cloud GPU, no local GPU needed)

```bash
pip install modal
modal setup  # authenticate
modal run training/modal_app.py
```

Requires two Modal secrets:
- `huggingface-secret` with `HF_TOKEN` to download Qwen2.5-Coder-7B-Instruct
- `wandb-secret` with `WANDB_API_KEY` for experiment logging (optional)

Runs on A10G, persists checkpoints to a Modal Volume, and auto-merges adapters after training.

### Data Pipeline

`python -m data.collect` scrapes merged PR review comments from 6 major open-source repos and writes to `data/raw/pr_samples.json`.

`python -m data.pipeline` chains:
1. **Clean** — filters bot comments, boilerplate, short/low-signal comments via `data/clean.py`
2. **Format** — converts each comment into ChatML (system/user/assistant) format via `data/format.py`
3. **Split** — 90/10 train/val shuffle with fixed seed

Outputs `data/train.jsonl` and `data/val.jsonl` ready for `SFTTrainer`.

`data.format.format_sample` preserves real review metadata from GitHub — reading `path`, `line`, `original_line`, `start_line`, and `original_start_line` fields when present instead of defaulting every label to line 1.

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

A 30-sample manually verified benchmark is at `data/eval_benchmark.jsonl` with 8 security, 8 bug, 7 style, and 7 maintainability samples drawn from real PR patterns. A schema/example seed is also at `data/eval_benchmark.example.jsonl`.

```bash
python -m evals.run_eval --model heuristic    # rule-based baseline
python -m evals.run_eval --model base --smoke  # Qwen base model (requires vLLM)
python -m evals.run_eval --model finetuned     # fine-tuned model
python -m evals.run_eval --model gpt4o         # GPT-4o baseline
```

Results are written to `evals/results/{model}.json` and `evals/results/latest.json`. The tracked `evals/results/latest.json` records the most recent run.

### Current Baseline (Heuristic Rules)

| Metric | Value |
|---|---|
| Samples | 30 |
| Precision | 20.0% |
| Recall | 3.3% |
| F1 | 5.7% |

Rule-based heuristic only catches bare `except:`, hardcoded secret patterns, and lines > 140 characters — intentionally minimal. Real model results will replace this table once API keys and inference infrastructure are available.

## Evidence Status

| Item | Status |
|---|---|
| ✅ Manually checked benchmark (30 samples) | Checked in |
| ✅ Heuristic baseline eval | Done (precision 20%, recall 3.3%) |
| ❌ Base Qwen2.5-Coder eval | Blocked — needs vLLM inference |
| ❌ Fine-tuned model eval | Blocked — needs training run |
| ❌ GPT-4o eval | Blocked — needs API key with active quota |
| ❌ Screenshot of live PR review | Not yet |
| ❌ Training evidence (dataset, loss, config) | Not yet |

Until those artifacts exist, describe this repo as:

> AI-powered PR review system with multi-agent structured review pipeline, GitHub integration, and QLoRA/vLLM fine-tuning scaffold.
