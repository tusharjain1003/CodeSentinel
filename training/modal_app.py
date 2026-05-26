import sys

import modal

app = modal.App("codesentinel-training")

training_image = (
    modal.Image.debian_slim()
    .pip_install(
        "transformers>=4.43",
        "peft>=0.12",
        "trl>=0.9",
        "bitsandbytes>=0.43",
        "accelerate>=0.33",
        "datasets>=2.20",
        "wandb>=0.17",
        "torch>=2.3",
        "huggingface-hub>=0.24",
        "pyyaml>=6.0",
        "scipy",
    )
    .add_local_dir(
        ".",
        remote_path="/repo",
        ignore=[
            "*.pyc", "__pycache__", ".git", ".venv", ".env",
            ".pytest_cache", ".ruff_cache", "node_modules",
        ],
    )
)

checkpoints_vol = modal.Volume.from_name("codesentinel-checkpoints", create_if_missing=True)


@app.function(
    image=training_image,
    gpu="A10G",
    timeout=7200,
    volumes={"/checkpoints": checkpoints_vol},
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret"),
    ],
)
def train(config_path: str = "/repo/training/config.yaml") -> None:
    import gc
    import os
    from pathlib import Path

    import torch
    import yaml

    os.chdir("/repo")
    sys.path.insert(0, "/repo")

    from training.merge import merge_and_save
    from training.train import train as run_train

    model = run_train(config_path=config_path)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("Cleared training memory, starting merge...")

    with open(config_path, encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    output_dir = Path(cfg["training"]["output_dir"])
    adapter_path = output_dir if output_dir.is_absolute() else Path("/repo") / output_dir

    merge_and_save(
        cfg["model"]["base"],
        str(adapter_path),
        "/checkpoints/codesentinel-merged",
    )
    print("Training complete. Merged model saved to /checkpoints/codesentinel-merged")
