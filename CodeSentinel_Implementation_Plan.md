# CodeSentinel — Detailed Implementation Plan

> AI-powered PR review system with QLoRA fine-tuning, vLLM serving, multi-agent coordination, and structured outputs.

**Target roles:** AI Engineer, ML Engineer, LLM Engineer  
**Estimated timeline:** 6–8 weeks (solo, part-time)  
**Prerequisites:** Python 3.11+, basic PyTorch knowledge, GitHub account, access to a GPU (Modal/Colab/Vast.ai)

---

## Table of Contents

- [Project Structure](#project-structure)
- [Phase 1 — Foundation](#phase-1--foundation-week-1)
- [Phase 2 — Dataset Pipeline](#phase-2--dataset-pipeline-week-2)
- [Phase 3 — Fine-Tuning with QLoRA](#phase-3--fine-tuning-with-qlora-weeks-3-4)
- [Phase 4 — Model Serving with vLLM](#phase-4--model-serving-with-vllm-week-4)
- [Phase 5 — Agent Pipeline](#phase-5--agent-pipeline-week-5)
- [Phase 6 — GitHub Integration](#phase-6--github-integration-week-5)
- [Phase 7 — Evaluation Framework](#phase-7--evaluation-framework-week-6)
- [Phase 8 — Frontend Dashboard](#phase-8--frontend-dashboard-week-7)
- [Phase 9 — Production Hardening & Deployment](#phase-9--production-hardening--deployment-week-8)
- [Design Decisions Reference](#design-decisions-reference)
- [Failure Modes & Fallbacks](#failure-modes--fallbacks)
- [Resume Bullets Unlocked Per Phase](#resume-bullets-unlocked-per-phase)

---

## Project Structure

```
codesentinel/
├── .github/
│   └── workflows/
│       ├── ci.yml                  # lint + unit tests on push
│       ├── eval.yml                # eval smoke test on push to main
│       └── deploy.yml              # deploy to Railway on main merge
├── data/
│   ├── raw/                        # downloaded GitHub PR data (gitignored)
│   ├── processed/                  # cleaned, formatted JSONL
│   ├── train.jsonl                 # training split
│   ├── val.jsonl                   # validation split
│   └── eval_benchmark.jsonl        # held-out eval set (200 samples)
├── training/
│   ├── dataset.py                  # dataset loader + tokenizer
│   ├── train.py                    # QLoRA training script
│   ├── config.yaml                 # training hyperparameters
│   ├── merge.py                    # merge LoRA adapters into base model
│   └── push_to_hub.py              # upload merged model to HF Hub
├── serving/
│   ├── server.py                   # vLLM OpenAI-compatible server launcher
│   ├── client.py                   # thin async client wrapping vLLM endpoint
│   └── modal_app.py                # Modal deployment for GPU inference
├── agents/
│   ├── base.py                     # BaseAgent protocol (structured output schema)
│   ├── bug_agent.py                # bug detection specialist
│   ├── security_agent.py           # vulnerability detection specialist
│   ├── style_agent.py              # style/maintainability specialist
│   └── coordinator.py              # merges agent outputs, deduplicates, ranks
├── pipeline/
│   ├── graph.py                    # LangGraph StateGraph orchestration
│   ├── runtime.py                  # SSE queue + contextvars (same pattern as AgentLens)
│   ├── parse_pr.py                 # parse GitHub PR diff into reviewable hunks
│   └── post_review.py              # post structured comments to GitHub PR API
├── evals/
│   ├── run_eval.py                 # main eval runner
│   ├── metrics.py                  # precision, recall, F1, comment quality scorer
│   ├── judge.py                    # GPT-4o-as-judge for comment quality
│   ├── benchmark.py                # loads eval_benchmark.jsonl, runs pipeline
│   └── results/                    # eval result JSONs + W&B run links
├── db/
│   ├── schema.sql                  # reviews, feedback, sessions tables
│   ├── setup.py                    # initialise DB
│   └── access.py                   # asyncpg query functions
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── state/                  # Zustand store
│   │   └── pages/
│   │       ├── Dashboard.tsx       # eval results + model comparison
│   │       └── ReviewHistory.tsx   # past reviews with comment breakdown
│   └── vite.config.ts
├── app.py                          # FastAPI entrypoint (webhook + REST API)
├── config.py                       # pydantic settings from env
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── requirements.txt
├── requirements.lock
└── README.md
```

---

## Phase 1 — Foundation (Week 1)

### 1.1 Repo Setup

```bash
git init codesentinel && cd codesentinel
python3 -m venv .venv && source .venv/bin/activate

# Core dependencies
pip install fastapi uvicorn asyncpg pydantic-settings python-dotenv \
            langgraph langchain-core pygithub httpx

# Training dependencies (install separately in training environment)
pip install transformers peft trl bitsandbytes accelerate datasets wandb

# Serving
pip install vllm openai

# Dev
pip install ruff pytest pytest-asyncio
```

Create `.env.example`:
```
# LLM
OPENAI_API_KEY=           # fallback + GPT-4o judge
VLLM_BASE_URL=            # e.g. http://localhost:8000/v1
VLLM_MODEL_NAME=          # your fine-tuned model name

# GitHub
GITHUB_TOKEN=             # for posting PR comments
GITHUB_WEBHOOK_SECRET=    # webhook HMAC secret

# Database
DATABASE_URL=             # postgresql+asyncpg://...

# Training
HF_TOKEN=                 # Hugging Face Hub
WANDB_API_KEY=            # Weights & Biases

# App
API_KEY=                  # optional auth
RATE_LIMIT_PER_MINUTE=30
```

### 1.2 Config Layer (`config.py`)

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_model_name: str = "codesentinel-codeqwen-7b"
    github_token: str
    github_webhook_secret: str
    database_url: str
    hf_token: str = ""
    wandb_api_key: str = ""
    api_key: str = ""
    rate_limit_per_minute: int = 30

    class Config:
        env_file = ".env"

settings = Settings()
```

### 1.3 Database Schema (`db/schema.sql`)

```sql
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE reviews (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    pr_url      TEXT NOT NULL,
    repo        TEXT NOT NULL,
    pr_number   INTEGER NOT NULL,
    diff_hash   TEXT NOT NULL,           -- SHA256 of the diff for dedup
    model_used  TEXT NOT NULL,           -- "finetuned" | "base" | "gpt4o"
    comments    JSONB NOT NULL,          -- array of ReviewComment objects
    timing_ms   JSONB,                   -- per-agent latency breakdown
    token_cost  JSONB,                   -- input/output tokens per agent
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE review_feedback (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    review_id   UUID REFERENCES reviews(id),
    comment_idx INTEGER NOT NULL,        -- index into reviews.comments array
    rating      SMALLINT CHECK (rating IN (-1, 0, 1)),  -- -1 wrong, 0 ok, 1 helpful
    correction  TEXT,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- Index for dedup check on re-review
CREATE INDEX idx_reviews_diff_hash ON reviews(diff_hash);
CREATE INDEX idx_reviews_repo_pr ON reviews(repo, pr_number);
```

### 1.4 CI Setup (`.github/workflows/ci.yml`)

```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: pip install -r requirements.txt
      - run: python -m ruff check agents/ pipeline/ app.py config.py
      - run: python -m pytest tests/ -q
```

### 1.5 FastAPI Skeleton (`app.py`)

```python
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import JSONResponse
import hashlib, hmac
from config import settings

app = FastAPI(title="CodeSentinel")

def verify_webhook_signature(request: Request, body: bytes):
    sig = request.headers.get("X-Hub-Signature-256", "")
    expected = "sha256=" + hmac.new(
        settings.github_webhook_secret.encode(),
        body, hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(sig, expected):
        raise HTTPException(status_code=401, detail="Invalid signature")

@app.post("/webhook/github")
async def github_webhook(request: Request):
    body = await request.body()
    verify_webhook_signature(request, body)
    event = request.headers.get("X-GitHub-Event")
    payload = await request.json()
    if event == "pull_request" and payload["action"] in ("opened", "synchronize"):
        # trigger review pipeline (background task)
        ...
    return JSONResponse({"status": "accepted"})

@app.post("/api/review")
async def manual_review(pr_url: str):
    # manual trigger for demo/testing
    ...
```

---

## Phase 2 — Dataset Pipeline (Week 2)

This is the most critical phase. Dataset quality determines fine-tuning quality.

### 2.1 Data Sources

**Primary: GitHub PR comments via REST API**

```python
# data/collect.py
import asyncio, httpx, json
from pathlib import Path

REPOS = [
    "huggingface/transformers",
    "langchain-ai/langchain",
    "fastapi/fastapi",
    "pallets/flask",
    "psf/requests",
    "python/cpython",
    "pytorch/pytorch",
    # add 20-30 high-quality open source repos
]

async def fetch_pr_comments(repo: str, client: httpx.AsyncClient) -> list[dict]:
    """Fetch PRs with review comments. Filter for merged PRs with ≥2 review comments."""
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    pulls_url = f"https://api.github.com/repos/{repo}/pulls"
    params = {"state": "closed", "per_page": 100}
    
    samples = []
    async for pr in paginate(client, pulls_url, params, headers):
        if not pr.get("merged_at"):
            continue
        comments = await fetch_review_comments(repo, pr["number"], client, headers)
        if len(comments) < 2:
            continue
        diff = await fetch_diff(repo, pr["number"], client, headers)
        samples.append({
            "repo": repo,
            "pr_number": pr["number"],
            "title": pr["title"],
            "diff": diff,
            "comments": comments,
        })
    return samples
```

**Target volume:** 50,000–80,000 (diff hunk, review comment) pairs after filtering.

**Secondary: CodeSearchNet** — for general code understanding context (used in training mix at 20% weight).

### 2.2 Data Cleaning (`data/clean.py`)

Critical filters — apply all of them:

```python
def is_valid_comment(comment: dict) -> bool:
    body = comment.get("body", "")
    
    # Minimum quality filters
    if len(body) < 30:              return False   # too short to be useful
    if len(body) > 2000:            return False   # probably a wall-of-text paste
    if body.startswith("LGTM"):     return False   # non-substantive approval
    if body.lower() in BOILERPLATE: return False   # "thanks!", "nit:", alone, etc.
    
    # Must have actionable content
    actionable_signals = [
        "should", "could", "consider", "instead", "bug", "issue",
        "error", "null", "race condition", "overflow", "injection",
        "vulnerability", "refactor", "rename", "extract", "missing"
    ]
    if not any(s in body.lower() for s in actionable_signals):
        return False
    
    # Filter bot comments
    if comment.get("user", {}).get("type") == "Bot":
        return False
    
    return True

def clean_diff_hunk(hunk: str) -> str:
    """Keep only the changed lines + 3 lines context. Remove metadata headers."""
    lines = hunk.split("\n")
    return "\n".join(
        l for l in lines
        if not l.startswith("diff --git")
        and not l.startswith("index ")
        and not l.startswith("Binary")
    )[:3000]  # hard cap to fit in context window
```

### 2.3 Training Format

Use the ChatML format that CodeQwen-7B expects:

```python
def format_sample(diff_hunk: str, file_path: str, comment: str, category: str) -> dict:
    """Format a single (hunk, comment) pair into the ChatML training format."""
    system = """You are an expert code reviewer. Analyze the provided code diff and 
identify issues. Respond ONLY with a valid JSON object matching the ReviewComment schema. 
Do not add any text outside the JSON."""

    user = f"""Review this code change in {file_path}:

```diff
{diff_hunk}
```

Identify any issues (bugs, security vulnerabilities, style problems, or maintainability concerns).
Respond with a JSON ReviewComment object."""

    # The comment body is already human-written — we classify it
    assistant = json.dumps({
        "category": category,          # "bug" | "security" | "style" | "maintainability"
        "severity": infer_severity(comment),  # "critical" | "major" | "minor" | "nit"
        "line_reference": extract_line_ref(comment),
        "message": comment,
        "suggestion": extract_suggestion(comment),
        "confidence": 0.9
    })

    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }
```

### 2.4 Category Classification

You need to classify each raw comment into bug/security/style/maintainability. Use GPT-4o-mini for this (cheap, fast):

```python
# data/classify.py
async def classify_comment(comment: str, diff: str) -> str:
    """Use GPT-4o-mini to classify the comment category. Costs ~$0.001 per 1000 comments."""
    response = await openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""Classify this code review comment into exactly one category.
Comment: {comment}
Categories: bug, security, style, maintainability
Respond with just the category word."""
        }],
        max_tokens=5,
    )
    return response.choices[0].message.content.strip().lower()
```

**Cost estimate:** 60,000 samples × ~200 tokens = ~$1.20 with GPT-4o-mini. Worth it.

### 2.5 Train/Val/Eval Split

```python
# Stratified split to ensure category balance
train: 45,000 samples (75%)
val:    8,000 samples (13%)
eval:   7,000 samples (12%)  # held out, NEVER seen during training

# From eval, manually curate 200 high-quality samples into eval_benchmark.jsonl
# These 200 should cover: all 4 categories, various languages (Python/JS/Go/Java),
# trivial bugs, subtle bugs, real security vulns, ambiguous cases
```

---

## Phase 3 — Fine-Tuning with QLoRA (Weeks 3–4)

### 3.1 Why QLoRA

- **Q** (4-bit NF4 quantization): loads the 7B base model in ~4GB VRAM instead of ~14GB
- **Lo** (Low-Rank): adds small trainable adapter matrices (rank 64) instead of updating all 7B params
- **RA** (Rank Adaptation): adapters are added to attention layers (Q, K, V, O projections)
- Net effect: fine-tune a 7B model on a single T4 (16GB) or A10G (24GB) GPU

### 3.2 Training Config (`training/config.yaml`)

```yaml
model:
  base: "Qwen/Qwen2.5-Coder-7B-Instruct"  # or "Qwen/CodeQwen1.5-7B-Chat"
  
lora:
  r: 64                        # rank — higher = more capacity, more memory
  alpha: 128                   # scaling factor (usually 2×r)
  dropout: 0.05
  target_modules:              # which attention layers to adapt
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj

quantization:
  load_in_4bit: true
  bnb_4bit_compute_dtype: bfloat16
  bnb_4bit_quant_type: nf4    # NormalFloat4 — better than fp4 for LLMs
  bnb_4bit_use_double_quant: true  # quantize the quantization constants too

training:
  output_dir: "./checkpoints"
  num_train_epochs: 3
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 8   # effective batch = 4×8 = 32
  learning_rate: 2.0e-4
  warmup_ratio: 0.03
  lr_scheduler_type: cosine
  max_seq_length: 2048
  fp16: false
  bf16: true                       # T4 supports fp16; A10G/A100 prefer bf16
  logging_steps: 10
  eval_steps: 200
  save_steps: 200
  load_best_model_at_end: true
  metric_for_best_model: eval_loss

wandb:
  project: "codesentinel"
  run_name: "qwen2.5-coder-7b-qlora-v1"
```

### 3.3 Training Script (`training/train.py`)

```python
import wandb
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig
import yaml, torch

def train(config_path: str):
    cfg = yaml.safe_load(open(config_path))
    
    wandb.init(
        project=cfg["wandb"]["project"],
        name=cfg["wandb"]["run_name"],
        config=cfg,
    )
    
    # 1. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["base"])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"   # important for causal LMs
    
    # 2. Load base model in 4-bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["base"],
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False          # required for gradient checkpointing
    model.config.pretraining_tp = 1
    
    # 3. Apply LoRA
    lora_cfg = cfg["lora"]
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    # Expected output: trainable params: ~40M / 7B total = ~0.6%
    
    # 4. Load datasets
    train_dataset = load_dataset("json", data_files="data/train.jsonl", split="train")
    val_dataset   = load_dataset("json", data_files="data/val.jsonl",   split="train")
    
    # 5. Train
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=SFTConfig(
            output_dir=cfg["training"]["output_dir"],
            num_train_epochs=cfg["training"]["num_train_epochs"],
            per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
            gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
            learning_rate=cfg["training"]["learning_rate"],
            warmup_ratio=cfg["training"]["warmup_ratio"],
            lr_scheduler_type=cfg["training"]["lr_scheduler_type"],
            max_seq_length=cfg["training"]["max_seq_length"],
            bf16=cfg["training"]["bf16"],
            logging_steps=cfg["training"]["logging_steps"],
            eval_steps=cfg["training"]["eval_steps"],
            save_steps=cfg["training"]["save_steps"],
            load_best_model_at_end=True,
            report_to="wandb",
        ),
    )
    trainer.train()
    trainer.save_model()
    wandb.finish()

if __name__ == "__main__":
    train("training/config.yaml")
```

### 3.4 W&B Metrics to Watch

During training, monitor these in the W&B dashboard:

| Metric | Healthy Range | Action if Unhealthy |
|---|---|---|
| `train/loss` | Decreasing smoothly | If plateau after epoch 1: lower LR |
| `eval/loss` | Decreasing, close to train loss | If diverging: lower LR or reduce epochs |
| `train/grad_norm` | < 1.0 generally | If spiky: clip gradients (`max_grad_norm=1.0`) |
| GPU memory | < 90% of VRAM | If OOM: reduce batch size, increase grad_accum |
| JSON parse rate (custom) | > 95% | Log % of outputs that parse as valid JSON |

**Custom metric — JSON parse rate:** After each eval step, run 50 inference samples through the model and measure how many outputs are valid JSON. Add this as a custom W&B metric:

```python
# In a custom eval callback
def on_evaluate(self, args, state, control, **kwargs):
    parse_rate = evaluate_json_compliance(model, tokenizer, val_samples[:50])
    wandb.log({"eval/json_parse_rate": parse_rate}, step=state.global_step)
```

### 3.5 Merge LoRA Adapters (`training/merge.py`)

Before serving with vLLM, merge the LoRA adapters back into the base model weights:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

def merge_and_save(base_model_id: str, adapter_path: str, output_path: str):
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    
    # Load base in fp16 (NOT 4-bit) for merging
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    # Load and merge adapters
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()   # merges LoRA into base weights, removes adapters
    
    # Save merged model
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"Merged model saved to {output_path}")
```

### 3.6 Push to Hugging Face Hub (`training/push_to_hub.py`)

```python
from huggingface_hub import HfApi

def push(local_path: str, repo_id: str):
    api = HfApi()
    api.create_repo(repo_id=repo_id, private=True, exist_ok=True)
    api.upload_folder(
        folder_path=local_path,
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"Pushed to https://huggingface.co/{repo_id}")
```

**Where to run training:** Modal (cheapest GPU access, pay-per-second), or Vast.ai (cheapest raw GPU rental). On an A10G, 3 epochs over 45K samples at batch_size=32 takes ~6–8 hours. Cost: ~$12-18 on Modal.

---

## Phase 4 — Model Serving with vLLM (Week 4)

### 4.1 Why vLLM

vLLM implements **PagedAttention** — it manages KV-cache memory like virtual memory pages, allowing up to 24× higher throughput than naive HuggingFace inference. It also implements **continuous batching**: instead of waiting for a batch to fill before processing, it adds new requests to in-flight batches as decode slots free up.

### 4.2 Local vLLM Server (`serving/server.py`)

```python
# serving/server.py — launch the vLLM OpenAI-compatible server
import subprocess, sys

def start_server(
    model: str,             # HF model path or local path to merged model
    port: int = 8000,
    gpu_memory_utilization: float = 0.90,
    max_model_len: int = 4096,
    quantization: str = "awq",  # apply AWQ post-training quantization for serving
):
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--port", str(port),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--max-model-len", str(max_model_len),
        "--served-model-name", "codesentinel",
        "--enable-prefix-caching",   # cache common prefixes (system prompt)
    ]
    if quantization:
        cmd.extend(["--quantization", quantization])
    
    subprocess.run(cmd)
```

### 4.3 vLLM Client (`serving/client.py`)

```python
# Thin async wrapper around the vLLM OpenAI-compatible endpoint
from openai import AsyncOpenAI
from config import settings

_client = AsyncOpenAI(
    base_url=settings.vllm_base_url,
    api_key="not-needed",           # vLLM doesn't require auth locally
)

async def complete(
    messages: list[dict],
    model: str = None,
    temperature: float = 0.1,       # low temp for structured output consistency
    max_tokens: int = 512,
    response_format: dict = None,   # {"type": "json_object"} for JSON mode
) -> str:
    kwargs = dict(
        model=model or settings.vllm_model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if response_format:
        kwargs["response_format"] = response_format
    
    response = await _client.chat.completions.create(**kwargs)
    return response.choices[0].message.content

async def complete_with_function_call(
    messages: list[dict],
    tools: list[dict],
    model: str = None,
) -> dict:
    """For structured output via function calling."""
    response = await _client.chat.completions.create(
        model=model or settings.vllm_model_name,
        messages=messages,
        tools=tools,
        tool_choice="required",     # force the model to call the tool
        temperature=0.1,
    )
    tool_call = response.choices[0].message.tool_calls[0]
    return json.loads(tool_call.function.arguments)
```

### 4.4 AWQ Quantization for Serving

After merging LoRA, apply AWQ (Activation-aware Weight Quantization) to the merged model for faster serving:

```python
# serving/quantize_awq.py
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

def quantize(model_path: str, output_path: str):
    quant_config = {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM"
    }
    model = AutoAWQForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.quantize(tokenizer, quant_config=quant_config)
    model.save_quantized(output_path)
    tokenizer.save_pretrained(output_path)
```

AWQ reduces serving memory from ~14GB (fp16 7B) to ~4.5GB (4-bit 7B), enabling a single T4 to serve the model.

### 4.5 Modal Deployment (`serving/modal_app.py`)

```python
import modal

app = modal.App("codesentinel-serving")

vllm_image = (
    modal.Image.debian_slim()
    .pip_install("vllm", "autoawq", "transformers")
)

@app.function(
    image=vllm_image,
    gpu=modal.gpu.A10G(),
    timeout=600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
@modal.web_endpoint(method="POST")
async def serve(request: dict):
    # Lazy-load vLLM engine on first request
    ...
```

---

## Phase 5 — Agent Pipeline (Week 5)

### 5.1 ReviewComment Schema

This is the structured output contract all agents must return:

```python
# agents/base.py
from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional

class Category(str, Enum):
    bug = "bug"
    security = "security"
    style = "style"
    maintainability = "maintainability"

class Severity(str, Enum):
    critical = "critical"   # must fix before merge
    major = "major"         # should fix
    minor = "minor"         # consider fixing
    nit = "nit"             # optional polish

class ReviewComment(BaseModel):
    category: Category
    severity: Severity
    file_path: str
    line_start: int
    line_end: int
    message: str            # what's wrong and why
    suggestion: Optional[str] = None   # concrete fix
    confidence: float = Field(ge=0.0, le=1.0)

class AgentReview(BaseModel):
    agent_name: str
    comments: list[ReviewComment]
    timing_ms: int
    token_usage: dict       # {"input": N, "output": M}
```

### 5.2 Base Agent Pattern

```python
# agents/base.py
import json, time
from abc import ABC, abstractmethod
from serving.client import complete_with_function_call

REVIEW_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_review",
        "description": "Submit code review comments",
        "parameters": {
            "type": "object",
            "properties": {
                "comments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "category":    {"type": "string", "enum": ["bug","security","style","maintainability"]},
                            "severity":    {"type": "string", "enum": ["critical","major","minor","nit"]},
                            "file_path":   {"type": "string"},
                            "line_start":  {"type": "integer"},
                            "line_end":    {"type": "integer"},
                            "message":     {"type": "string"},
                            "suggestion":  {"type": "string"},
                            "confidence":  {"type": "number"},
                        },
                        "required": ["category","severity","file_path","line_start","line_end","message","confidence"]
                    }
                }
            },
            "required": ["comments"]
        }
    }
}

class BaseAgent(ABC):
    name: str
    focus: str              # what this agent specifically looks for
    system_prompt: str

    async def review(self, diff: str, file_path: str) -> AgentReview:
        t0 = time.monotonic()
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": self._build_prompt(diff, file_path)},
        ]
        result = await complete_with_function_call(messages, tools=[REVIEW_TOOL])
        comments = [ReviewComment(**c) for c in result.get("comments", [])]
        return AgentReview(
            agent_name=self.name,
            comments=comments,
            timing_ms=int((time.monotonic() - t0) * 1000),
            token_usage={},   # populated from vLLM response headers
        )

    def _build_prompt(self, diff: str, file_path: str) -> str:
        return f"""Review this code change in `{file_path}`. Focus ONLY on {self.focus}.

```diff
{diff}
```

Use the submit_review function to return your findings. Return an empty comments array if no issues found."""
```

### 5.3 Specialist Agents

```python
# agents/bug_agent.py
class BugAgent(BaseAgent):
    name = "bug_detector"
    focus = "logical errors, null pointer dereferences, off-by-one errors, race conditions, incorrect error handling, resource leaks, and incorrect API usage"
    system_prompt = """You are a senior software engineer specialising in bug detection.
You have deep knowledge of common programming mistakes across Python, JavaScript, Go, Java, and TypeScript.
Identify only genuine bugs — not style issues or subjective preferences.
For each bug, explain clearly why it is wrong and what the correct behaviour should be."""

# agents/security_agent.py
class SecurityAgent(BaseAgent):
    name = "security_scanner"
    focus = "security vulnerabilities including SQL injection, XSS, CSRF, path traversal, insecure deserialization, hardcoded secrets, insufficient input validation, broken authentication, and insecure dependencies"
    system_prompt = """You are a security engineer specialising in application security.
You follow OWASP Top 10 and CWE/SANS Top 25.
Only flag real vulnerabilities — not theoretical ones that require unlikely exploit chains.
Rate critical only when exploitation is straightforward."""

# agents/style_agent.py
class StyleAgent(BaseAgent):
    name = "style_reviewer"
    focus = "code maintainability: overly complex functions (cyclomatic complexity), poor naming, missing type hints, unclear abstractions, duplicated logic, violated SOLID principles, and missing error handling patterns"
    system_prompt = """You are a senior engineer focused on code quality and maintainability.
Focus on changes that would meaningfully improve readability or reduce future bugs.
Do not flag trivial stylistic preferences (tabs vs spaces, trailing commas) unless they violate the project's existing style."""
```

### 5.4 Coordinator (`agents/coordinator.py`)

```python
class ReviewCoordinator:
    def __init__(self):
        self.agents = [BugAgent(), SecurityAgent(), StyleAgent()]

    async def coordinate(self, diff: str, file_path: str) -> list[ReviewComment]:
        # Run all agents concurrently
        agent_reviews: list[AgentReview] = await asyncio.gather(
            *[agent.review(diff, file_path) for agent in self.agents]
        )

        all_comments = [c for review in agent_reviews for c in review.comments]

        # Deduplicate: two comments are duplicates if they reference the same
        # line range AND their messages have cosine similarity > 0.85
        deduped = self._deduplicate(all_comments)

        # Rank by severity then confidence
        ranked = sorted(
            deduped,
            key=lambda c: (SEVERITY_ORDER[c.severity], -c.confidence)
        )

        # Cap at 20 comments — more than this is overwhelming in a PR review
        return ranked[:20]

    def _deduplicate(self, comments: list[ReviewComment]) -> list[ReviewComment]:
        """Remove comments that point to the same line with similar messages."""
        seen = []
        for c in comments:
            is_dup = any(
                abs(c.line_start - s.line_start) <= 2
                and self._similarity(c.message, s.message) > 0.85
                for s in seen
            )
            if not is_dup:
                seen.append(c)
        return seen

    def _similarity(self, a: str, b: str) -> float:
        """Simple Jaccard similarity on word sets — fast, good enough for dedup."""
        sa, sb = set(a.lower().split()), set(b.lower().split())
        return len(sa & sb) / len(sa | sb) if sa | sb else 0.0
```

### 5.5 LangGraph Orchestration (`pipeline/graph.py`)

```python
from langgraph.graph import StateGraph
from typing import TypedDict

class ReviewState(TypedDict):
    pr_url: str
    repo: str
    pr_number: int
    diff: str
    parsed_hunks: list[dict]       # list of {file_path, hunk}
    agent_reviews: list[AgentReview]
    final_comments: list[ReviewComment]
    posted_to_github: bool
    session_id: str

graph = StateGraph(ReviewState)

graph.add_node("parse_pr",    node_parse_pr)     # parse diff into hunks
graph.add_node("run_agents",  node_run_agents)   # coordinator runs all agents
graph.add_node("post_review", node_post_review)  # post to GitHub PR API
graph.add_node("persist",     node_persist)      # save to DB

graph.set_entry_point("parse_pr")
graph.add_edge("parse_pr",    "run_agents")
graph.add_edge("run_agents",  "post_review")
graph.add_edge("post_review", "persist")
graph.add_edge("persist",     END)

pipeline = graph.compile()
```

---

## Phase 6 — GitHub Integration (Week 5)

### 6.1 Webhook Handler

```python
# In app.py — full webhook handler
from fastapi import BackgroundTasks

@app.post("/webhook/github")
async def github_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
):
    body = await request.body()
    verify_webhook_signature(request, body)
    
    event = request.headers.get("X-GitHub-Event")
    payload = await request.json()
    
    if event == "pull_request" and payload["action"] in ("opened", "synchronize"):
        pr = payload["pull_request"]
        background_tasks.add_task(
            run_review_pipeline,
            pr_url=pr["html_url"],
            repo=payload["repository"]["full_name"],
            pr_number=pr["number"],
            diff_url=pr["url"] + "/files",
        )
    
    return JSONResponse({"status": "accepted"})

async def run_review_pipeline(pr_url, repo, pr_number, diff_url):
    try:
        diff = await fetch_pr_diff(diff_url)
        state = ReviewState(
            pr_url=pr_url, repo=repo, pr_number=pr_number, diff=diff,
            parsed_hunks=[], agent_reviews=[], final_comments=[],
            posted_to_github=False, session_id=str(uuid4()),
        )
        await pipeline.ainvoke(state)
    except Exception as e:
        logger.error(f"Pipeline failed for {pr_url}: {e}")
```

### 6.2 Posting Reviews (`pipeline/post_review.py`)

```python
from github import Github
from config import settings

gh = Github(settings.github_token)

async def post_pr_review(
    repo: str,
    pr_number: int,
    comments: list[ReviewComment],
):
    """Post all comments as a single GitHub PR review (not individual comments)."""
    repo_obj = gh.get_repo(repo)
    pr = repo_obj.get_pull(pr_number)
    head_sha = pr.head.sha

    # Build review body — summary table
    critical = [c for c in comments if c.severity == "critical"]
    major    = [c for c in comments if c.severity == "major"]

    body = f"""## 🤖 CodeSentinel Review

| Severity | Count |
|---|---|
| 🔴 Critical | {len(critical)} |
| 🟠 Major | {len(major)} |
| 🟡 Minor | {len(comments) - len(critical) - len(major)} |

*Reviewed by fine-tuned CodeQwen-7B + GPT-4o fallback*
"""

    # Build inline comments for each ReviewComment
    pr_comments = []
    for c in comments:
        severity_emoji = {"critical": "🔴", "major": "🟠", "minor": "🟡", "nit": "⚪"}
        comment_body = f"""{severity_emoji[c.severity]} **{c.category.upper()} — {c.severity}**

{c.message}"""
        if c.suggestion:
            comment_body += f"\n\n**Suggestion:** {c.suggestion}"
        comment_body += f"\n\n*Confidence: {c.confidence:.0%}*"

        pr_comments.append({
            "path": c.file_path,
            "line": c.line_end,
            "body": comment_body,
        })

    pr.create_review(
        commit=pr.get_commits()[0],   # head commit
        body=body,
        event="COMMENT",
        comments=pr_comments,
    )
```

### 6.3 Setting Up the Webhook on GitHub

1. Go to your repo → Settings → Webhooks → Add webhook
2. Payload URL: `https://your-railway-url.railway.app/webhook/github`
3. Content type: `application/json`
4. Secret: value of `GITHUB_WEBHOOK_SECRET`
5. Events: "Pull requests" only

---

## Phase 7 — Evaluation Framework (Week 6)

### 7.1 What You're Measuring

| Metric | Definition | How to Compute |
|---|---|---|
| **Bug Recall** | % of real bugs in eval set caught by the model | TP / (TP + FN) |
| **Comment Precision** | % of model comments that identify real issues | TP / (TP + FP) |
| **F1 Score** | Harmonic mean of precision and recall | 2 × P × R / (P + R) |
| **Comment Quality** | Human-like quality of message + suggestion | GPT-4o-as-judge (0–3 scale) |
| **JSON Parse Rate** | % of model outputs that parse as valid ReviewComment | string parsing |
| **Severity Accuracy** | % of severity labels that match ground truth | classification accuracy |

### 7.2 Ground-Truth Construction

The `eval_benchmark.jsonl` (200 samples) has this format:

```json
{
  "diff": "...",
  "file_path": "src/auth.py",
  "ground_truth_comments": [
    {
      "category": "security",
      "severity": "critical",
      "line_start": 47,
      "line_end": 47,
      "message": "SQL query constructed with string concatenation allows injection",
      "is_real_issue": true
    }
  ],
  "source": "github.com/pallets/flask/pull/4821"
}
```

The `is_real_issue` flag was manually verified — this is 2-3 hours of human annotation work, but it's essential for meaningful eval.

### 7.3 GPT-4o-as-Judge (`evals/judge.py`)

```python
async def judge_comment_quality(
    diff: str,
    comment: ReviewComment,
    ground_truth: str,
) -> int:
    """Score 0–3: 0=wrong/irrelevant, 1=identifies issue poorly, 2=good, 3=excellent"""
    response = await openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": f"""You are evaluating an AI code review comment.

Code diff:
```diff
{diff}
```

AI comment: {comment.message}
AI suggestion: {comment.suggestion or 'None'}
Ground truth issue: {ground_truth}

Score 0–3:
0 = Wrong or irrelevant (not a real issue)
1 = Identifies a real issue but message is unclear or misleading
2 = Correctly identifies the issue with clear explanation
3 = Excellent: clear, actionable, correct, includes a good suggestion

Respond with just the integer score."""
        }],
        max_tokens=3,
    )
    return int(response.choices[0].message.content.strip())
```

### 7.4 Eval Runner (`evals/run_eval.py`)

```python
async def run_benchmark(
    model: str = "finetuned",   # "finetuned" | "base" | "gpt4o"
    smoke: bool = False,
):
    samples = load_benchmark()[:5] if smoke else load_benchmark()
    coordinator = ReviewCoordinator(model=model)
    
    results = []
    for sample in tqdm(samples):
        pred_review = await coordinator.coordinate(sample["diff"], sample["file_path"])
        gt_comments  = sample["ground_truth_comments"]
        
        # Match predicted comments to ground truth
        tp, fp, fn = match_comments(pred_review, gt_comments)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1        = 2*precision*recall / (precision+recall) if (precision+recall) > 0 else 0
        
        # Judge quality of matched TPs
        quality_scores = [
            await judge_comment_quality(sample["diff"], c, gt)
            for c, gt in zip(pred_review, gt_comments[:len(pred_review)])
        ]
        
        results.append({
            "model": model,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "avg_quality": sum(quality_scores)/len(quality_scores) if quality_scores else 0,
        })
    
    # Aggregate and log to W&B
    agg = aggregate_metrics(results)
    wandb.log({f"eval/{k}": v for k, v in agg.items()})
    
    # Save to evals/results/
    save_results(results, model)
    print_comparison_table(agg)
```

### 7.5 The Comparison Table (for README)

Run eval for all three model variants and populate this table in your README:

| Metric | Base Model | Fine-tuned | GPT-4o |
|---|---|---|---|
| Bug Recall | X% | **Y%** | Z% |
| Comment Precision | X% | **Y%** | Z% |
| F1 Score | X% | **Y%** | Z% |
| Avg Comment Quality | X/3 | **Y/3** | Z/3 |
| JSON Parse Rate | X% | **Y%** | 100% |
| Avg Latency | Xs | **Ys** | Zs |
| Cost / Review | $X | **$Y** | $Z |

This table is the single most powerful thing in your resume — "fine-tuned model achieved Y% F1 vs X% for the base, at Z× lower cost than GPT-4o."

---

## Phase 8 — Frontend Dashboard (Week 7)

### 8.1 Pages

**Dashboard (`/`):**
- Model comparison table (eval metrics, live-updated)
- Recent reviews feed with agent breakdown
- Cost tracker (tokens used per model, per day)
- Link to GitHub PR for each review

**Review Detail (`/review/:id`):**
- Side-by-side diff view with inline comment chips
- Per-agent breakdown (which agent caught which comment)
- Feedback buttons (👍 / 👎 / ✏️ correct) per comment
- Timing breakdown (parse → agent A → agent B → agent C → coordinator → post)

### 8.2 State Shape (Zustand)

```typescript
interface ReviewStore {
  reviews: Review[]
  evalMetrics: {
    finetuned: Metrics
    base: Metrics
    gpt4o: Metrics
  }
  selectedReview: Review | null
  
  fetchReviews: () => Promise<void>
  fetchEvalMetrics: () => Promise<void>
  submitFeedback: (reviewId: string, commentIdx: number, rating: -1|0|1, correction?: string) => Promise<void>
}
```

### 8.3 API Endpoints for Frontend

```python
@app.get("/api/reviews")
async def list_reviews(limit: int = 20, offset: int = 0):
    ...

@app.get("/api/reviews/{review_id}")
async def get_review(review_id: str):
    ...

@app.get("/api/eval/metrics")
async def get_eval_metrics():
    """Return latest eval results for all three model variants."""
    ...

@app.post("/api/feedback")
async def submit_feedback(
    review_id: str,
    comment_idx: int,
    rating: int,
    correction: str = None,
):
    ...
```

---

## Phase 9 — Production Hardening & Deployment (Week 8)

### 9.1 Input Validation

Before any diff enters the pipeline:

```python
MAX_DIFF_BYTES = 500_000   # 500KB — larger diffs are impractical to review
MAX_FILES = 50             # skip PRs touching >50 files

def validate_diff(diff: str) -> None:
    if len(diff.encode()) > MAX_DIFF_BYTES:
        raise ValueError(f"Diff too large: {len(diff.encode())} bytes")
    files_changed = diff.count("diff --git")
    if files_changed > MAX_FILES:
        raise ValueError(f"Too many files: {files_changed}")
```

### 9.2 Diff Deduplication

Avoid re-reviewing the same diff on repeated webhook calls (GitHub sends "synchronize" on every push):

```python
import hashlib

async def get_or_create_review(repo, pr_number, diff) -> tuple[Review | None, bool]:
    """Returns (existing_review, is_cached)."""
    diff_hash = hashlib.sha256(diff.encode()).hexdigest()
    existing = await db.fetch_review_by_hash(diff_hash)
    if existing:
        return existing, True
    return None, False
```

### 9.3 Graceful Degradation

```python
async def run_agents_with_fallback(diff: str, file_path: str) -> list[ReviewComment]:
    try:
        # Try fine-tuned model first
        return await coordinator.coordinate(diff, file_path)
    except (httpx.TimeoutException, VLLMUnavailableError) as e:
        logger.warning(f"vLLM unavailable, falling back to GPT-4o: {e}")
        # Fall back to GPT-4o with same structured output schema
        return await coordinator.coordinate(diff, file_path, model="gpt-4o")
    except Exception as e:
        logger.error(f"Both models failed: {e}")
        # Post a minimal "review unavailable" comment to the PR
        return []
```

### 9.4 Rate Limiting

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/api/review")
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def manual_review(request: Request, pr_url: str):
    ...
```

### 9.5 Dockerfile

```dockerfile
# Stage 1: Frontend build
FROM node:20-alpine AS frontend-build
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ .
RUN npm run build

# Stage 2: Python build
FROM python:3.11-slim AS python-build
WORKDIR /app
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock

# Stage 3: Runtime (no build tools)
FROM python:3.11-slim AS runtime
WORKDIR /app
COPY --from=python-build /opt/venv /opt/venv
COPY --from=frontend-build /app/frontend/dist /app/frontend/dist
ENV PATH="/opt/venv/bin:$PATH"
COPY . .
EXPOSE 8765
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8765"]
```

### 9.6 Railway Deployment

```toml
# railway.toml
[build]
builder = "dockerfile"

[deploy]
startCommand = "uvicorn app:app --host 0.0.0.0 --port $PORT"
healthcheckPath = "/health"
restartPolicyType = "on-failure"
```

Note: vLLM serving runs on Modal (GPU), not Railway (CPU-only). Railway hosts the FastAPI app and frontend only. The `VLLM_BASE_URL` env var points to the Modal endpoint.

---

## Design Decisions Reference

**"Why three separate agents instead of one prompt with all instructions?"**  
Specialization improves precision. A single prompt asked to find bugs AND security issues AND style problems produces lower-quality results because attention is split. Three focused agents, each with a domain-specific system prompt and few-shot examples, consistently outperform a general reviewer on each dimension. The coordinator then handles deduplication.

**"Why function calling over JSON mode for structured output?"**  
Both work, but function calling provides a schema the model is explicitly trained to follow. JSON mode just says "output valid JSON" — the structure is in the prompt. Function calling puts the schema in the API payload, which vLLM can constrain output to. The `tool_choice: "required"` parameter guarantees a tool call — you never get a freetext response when you need JSON.

**"Why QLoRA over full fine-tuning or prompt engineering alone?"**  
Full fine-tuning: requires 4-8× more VRAM, much longer training time, higher risk of catastrophic forgetting. Prompt engineering alone: you're limited by what the base model knows about code review; it can't learn your specific schema or the statistical patterns of high-quality reviews. QLoRA hits the middle: meaningful learned adaptation (~40M trainable params out of 7B) at a fraction of the cost.

**"Why CodeQwen over CodeLlama or StarCoder?"**  
As of 2025, Qwen2.5-Coder-7B leads MBPP, HumanEval, and CRUXEval benchmarks at the 7B class. It supports 128K context natively (important for large diffs), has strong instruction-following, and the ChatML format is widely compatible with vLLM and the OpenAI client library.

**"Why Modal for GPU serving instead of keeping it in Railway?"**  
Railway and Render don't offer GPU instances in their standard plans. Modal provides on-demand GPU compute (pay per second of use, not per month), which is significantly cheaper for a project with bursty request patterns. Cold start on Modal is ~8s for a loaded vLLM model — acceptable for webhook-triggered reviews where a few seconds don't matter.

---

## Failure Modes & Fallbacks

| Failure | Detection | Fallback |
|---|---|---|
| vLLM server down | `httpx.ConnectError` on client call | Fall back to GPT-4o with same schema |
| Model outputs invalid JSON | `json.JSONDecodeError` on parse | Retry once with `temperature=0`; if fails again, skip that agent's output |
| GitHub API rate limit | `RateLimitExceededException` | Retry with exponential backoff (60s, 120s, 240s); if still failing, queue for later |
| Diff too large | Size check pre-pipeline | Post a comment: "Diff too large (>500KB). Please break into smaller PRs." |
| All agents return empty | `len(final_comments) == 0` | Post a comment: "No issues found. Review may be incomplete for binary or generated files." |
| Modal cold start timeout | `asyncio.TimeoutError` after 15s | Fall back to GPT-4o; log cold start to W&B |
| DB write failure | `asyncpg.PostgresError` | Fire-and-forget: log error, never raise into pipeline |

---

## Resume Bullets Unlocked Per Phase

**After Phase 3 (Fine-tuning only — minimum viable):**
> Fine-tuned Qwen2.5-Coder-7B with QLoRA (NF4 4-bit, rank-64 LoRA adapters) on 50K curated GitHub PR comments; tracked training with W&B; improved bug-detection F1 from X% (base) to Y% (fine-tuned) on a 200-sample held-out benchmark

**After Phase 4 (+ serving):**
> Served fine-tuned model with vLLM (PagedAttention + AWQ INT4 quantization); reduced inference latency from Xs → Ys at 10 concurrent requests; deployed on Modal GPU with OpenAI-compatible endpoint and GPT-4o fallback

**After Phase 5 (+ multi-agent):**
> Designed a three-specialist multi-agent coordinator (bug, security, style) with structured JSON outputs via OpenAI function calling; coordinator deduplicates overlapping comments by line reference + Jaccard similarity and ranks by severity

**After Phase 7 (+ eval):**
> Built a custom eval framework with GPT-4o-as-judge scoring; fine-tuned model achieved Y% F1 vs X% base at Z× lower cost than GPT-4o on the same 200-sample benchmark

**Full project (all phases):**
> Built CodeSentinel: QLoRA fine-tuned Qwen2.5-Coder-7B code reviewer with three-agent LangGraph pipeline, vLLM serving, GitHub webhook integration, and GPT-4o-as-judge eval framework — fine-tuned model achieved Y% F1 vs X% base at Z× lower cost than GPT-4o
