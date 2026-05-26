import os

import torch
import wandb
import yaml
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    msg = "No GPU detected. Training a 7B model requires CUDA or MPS (Apple Silicon)."
    raise RuntimeError(msg)


def train(config_path: str = "training/config.yaml"):
    with open(config_path, encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    device = _detect_device()
    is_cuda = device == "cuda"
    print(f"Device: {device}")

    if not is_cuda:
        cfg["model"]["base"] = os.environ.get(
            "CODESENTINEL_LOCAL_TRAINING_MODEL",
            "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        )
        cfg["training"]["output_dir"] = os.environ.get(
            "CODESENTINEL_LOCAL_OUTPUT_DIR",
            "./checkpoints-v2",
        )
        cfg["training"]["per_device_train_batch_size"] = 1
        cfg["training"]["gradient_accumulation_steps"] = 16
        cfg["training"]["bf16"] = False
        cfg["training"]["fp16"] = True
        cfg["lora"]["r"] = 16
        cfg["lora"]["alpha"] = 32

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["base"])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_kwargs: dict = {
        "torch_dtype": torch.float16,
        "low_cpu_mem_usage": True,
    }
    if is_cuda:
        model_kwargs["device_map"] = "auto"
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["torch_dtype"] = torch.bfloat16
    else:
        model_kwargs["device_map"] = None

    model = AutoModelForCausalLM.from_pretrained(cfg["model"]["base"], **model_kwargs)
    model.config.use_cache = False

    if not is_cuda:
        model = model.to(device)
        model.gradient_checkpointing_enable()

    lora_cfg = cfg["lora"]
    model = get_peft_model(
        model,
        LoraConfig(
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["alpha"],
            lora_dropout=lora_cfg["dropout"],
            target_modules=lora_cfg["target_modules"],
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        ),
    )

    train_dataset = load_dataset("json", data_files="data/train.jsonl", split="train")
    val_dataset = load_dataset("json", data_files="data/val.jsonl", split="train")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    effective_bs = (
        cfg["training"]["per_device_train_batch_size"]
        * cfg["training"]["gradient_accumulation_steps"]
    )
    steps_per_epoch = max(1, len(train_dataset) // effective_bs)
    total_steps = steps_per_epoch * cfg["training"]["num_train_epochs"]

    eval_steps = max(1, steps_per_epoch)
    save_steps = eval_steps * 2 if eval_steps * 2 <= total_steps else total_steps

    args = SFTConfig(
        output_dir=cfg["training"]["output_dir"],
        num_train_epochs=cfg["training"]["num_train_epochs"],
        per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        learning_rate=cfg["training"]["learning_rate"],
        warmup_ratio=cfg["training"]["warmup_ratio"],
        lr_scheduler_type=cfg["training"]["lr_scheduler_type"],
        bf16=cfg.get("training", {}).get("bf16", False) and is_cuda,
        fp16=not is_cuda,
        max_length=1024,
        logging_steps=min(5, steps_per_epoch),
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_steps=save_steps,
        load_best_model_at_end=True,
        report_to="none" if not os.environ.get("WANDB_API_KEY") else "wandb",
    )

    if os.environ.get("WANDB_API_KEY"):
        wandb.init(project=cfg["wandb"]["project"], name=cfg["wandb"]["run_name"], config=cfg)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=args,
    )
    trainer.train()
    trainer.save_model()
    print(f"Training complete. Model saved to {cfg['training']['output_dir']}")

    if os.environ.get("WANDB_API_KEY"):
        wandb.finish()

    return model
