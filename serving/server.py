from __future__ import annotations

import subprocess
import sys


def start_server(
    model: str,
    port: int = 8000,
    gpu_memory_utilization: float = 0.90,
    max_model_len: int = 4096,
    quantization: str | None = "awq",
) -> None:
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model,
        "--port",
        str(port),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--max-model-len",
        str(max_model_len),
        "--served-model-name",
        "codesentinel",
        "--enable-prefix-caching",
    ]
    if quantization:
        cmd.extend(["--quantization", quantization])
    subprocess.run(cmd, check=False)


if __name__ == "__main__":
    start_server(sys.argv[1])
