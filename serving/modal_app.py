import modal

app = modal.App("codesentinel-serving")

vllm_image = (
    modal.Image.debian_slim()
    .pip_install("vllm")
)

checkpoints_vol = modal.Volume.from_name("codesentinel-checkpoints", create_if_missing=True)


@app.function(
    gpu="A10G",
    image=vllm_image,
    timeout=600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/checkpoints": checkpoints_vol},
    allow_concurrent_inputs=2,
    container_idle_timeout=300,
)
@modal.asgi_app()
def openai_server():
    from vllm.entrypoints.openai.api_server import build_app

    app = build_app(model="/checkpoints/codesentinel-merged")
    return app
