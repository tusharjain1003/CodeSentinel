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
)

checkpoints_vol = modal.Volume.from_name("codesentinel-checkpoints", create_if_missing=True)


@app.function(
    image=training_image,
    gpu=modal.gpu.A10G(),
    timeout=7200,
    volumes={"/checkpoints": checkpoints_vol},
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret"),
    ],
    mounts=[modal.Mount.from_local_dir(".", remote_path="/repo")],
)
def train(config_path: str = "/repo/training/config.yaml") -> None:
    sys.path.insert(0, "/repo")

    from training.merge import merge_and_save
    from training.train import train as run_train

    run_train(config_path=config_path)

    merge_and_save(
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        "/repo/checkpoints",
        "/checkpoints/codesentinel-merged",
    )
    print("Training complete. Merged model saved to /checkpoints/codesentinel-merged")
