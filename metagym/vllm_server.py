"""vLLM server lifecycle: start a server subprocess and wait until ready."""

import atexit
import os
import socket
import subprocess
import sys
import time
import urllib.request
from typing import Optional


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def _wait_for_server(base_url: str, process: subprocess.Popen, timeout: int) -> None:
    root = base_url.rstrip("/").rsplit("/v1", 1)[0]
    health_url = f"{root}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"vLLM server process exited early (code {process.returncode})"
            )
        try:
            urllib.request.urlopen(health_url, timeout=2)
            return
        except Exception:
            time.sleep(2)
    raise RuntimeError(f"vLLM server at {base_url} did not become ready within {timeout}s")


def launch_vllm_server(
    model: str,
    port: Optional[int] = None,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 32768,
    tensor_parallel_size: int = 1,
    dtype: str = "bfloat16",
    enable_prefix_caching: bool = True,
    timeout: int = 600,
    cuda_visible_devices: Optional[str] = None,
) -> str:
    """Start a vLLM OpenAI-compatible server and return its base URL.

    The server process is automatically terminated when the Python process exits.

    Returns:
        The server base URL, e.g. ``"http://localhost:8123/v1"``.
    """
    if port is None:
        port = _find_free_port()

    base_url = f"http://localhost:{port}/v1"

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--port", str(port),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--max-model-len", str(max_model_len),
        "--dtype", dtype,
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--trust-remote-code",
        "--uvicorn-log-level", "warning",
        "--disable-log-stats",
    ]
    if enable_prefix_caching:
        cmd.append("--enable-prefix-caching")

    env = {**os.environ, "SETUPTOOLS_USE_DISTUTILS": "stdlib"}
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    process = subprocess.Popen(cmd, env=env)
    atexit.register(lambda p=process: p.terminate() if p.poll() is None else None)

    print(f"Starting vLLM server for '{model}' on port {port}...")
    _wait_for_server(base_url, process, timeout)
    print(f"vLLM server ready at {base_url}")

    return base_url
