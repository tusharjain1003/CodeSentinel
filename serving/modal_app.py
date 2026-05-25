import modal

app = modal.App("codesentinel-serving")

vllm_image = modal.Image.debian_slim().pip_install("vllm", "autoawq", "transformers")


@app.function(
    image=vllm_image,
    gpu=modal.gpu.A10G(),
    timeout=600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
@modal.web_endpoint(method="POST")
async def serve(request: dict) -> dict:
    return {
        "status": "not_implemented",
        "message": "Deploy a merged or AWQ model here, then forward requests to vLLM.",
        "request": request,
    }
