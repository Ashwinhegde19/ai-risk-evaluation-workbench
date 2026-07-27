"""Self-deployed open-source target: Qwen3-8B on Modal.com (NVIDIA L4, 24 GB).

This module deploys the Qwen3-8B open-weight model as an OpenAI-compatible
inference endpoint on Modal, served by vLLM. It is the open-source lane of the
frontier-vs-open safety comparison: frontier models (GPT-5, Claude Opus 4.1,
Gemini 2.5 Pro) are reached through the Kilo gateway, while this self-hosted
Qwen3-8B endpoint is reached directly via ``OPEN_MODEL_BASE_URL``.

GPU sizing (read before changing the model):
    * NVIDIA L4 ........ 24 GB VRAM
    * Qwen3-8B (FP16) .. ~16 GB of weights
    * Headroom ......... ~8 GB left for the KV cache at ``--max-model-len 4096``
                         with ``--gpu-memory-utilization 0.90``.

    This fits comfortably. DO NOT attempt 32B+ (or even 14B at long context) on
    the L4 -- the weights plus KV cache will not fit in 24 GB and the container
    will OOM at startup. Qwen3-8B is the largest sensible choice here.

Serving pattern:
    vLLM's OpenAI-compatible server is launched as a subprocess inside the
    container on ``0.0.0.0:8000``. Modal's ``@modal.web_server(port=8000)``
    exposes that port directly to the internet -- no reverse proxy is needed.
    The endpoint serves ``/v1/chat/completions``, ``/v1/models``, etc. natively.

Model weights:
    Qwen/Qwen3-8B is downloaded into a Modal ``Volume`` on first run so that
    subsequent cold starts do not re-download ~16 GB from the Hub.

Deploy / serve:
    modal deploy modal_deploy.py     # create a persistent HTTPS endpoint
    modal serve modal_deploy.py      # ephemeral endpoint for local testing

The deployed endpoint URL is printed on success; set ``OPEN_MODEL_BASE_URL`` to
``<that-url>/v1`` for the workbench backends.
"""

from __future__ import annotations

import subprocess
import time

import modal

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_REPO = "Qwen/Qwen3-8B"          # Hugging Face repo id
SERVED_MODEL_NAME = "qwen3-8b"        # name clients pass as ``model=``
VLLM_PORT = 8000                      # vLLM listens on this port inside the container
MAX_MODEL_LEN = 4096                  # context window (fits L4 with headroom)
GPU_MEMORY_UTILIZATION = 0.90
SCALEDOWN_WINDOW = 300                # keep warm 5 min between requests

# ---------------------------------------------------------------------------
# Image + Volume
# ---------------------------------------------------------------------------
# HF cache volume persists model weights across cold starts so the ~16 GB
# download only happens once.
hf_cache_vol = modal.Volume.from_name("qwen3-8b-hf-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("qwen3-8b-vllm-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm",
        "torch",
        "transformers",
        "huggingface_hub",
    )
)

app = modal.App("qwen3-8b-inference")


@app.cls(
    image=image,
    gpu="L4",  # 24 GB VRAM -- see sizing note in the module docstring.
    scaledown_window=SCALEDOWN_WINDOW,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    timeout=600,
)
class Qwen3Inference:
    """vLLM-backed OpenAI-compatible inference service for Qwen3-8B."""

    @modal.enter()
    def start(self) -> None:
        """Launch vLLM's OpenAI-compatible server as a subprocess."""
        cmd = [
            "vllm", "serve", MODEL_REPO,
            "--served-model-name", SERVED_MODEL_NAME,
            "--dtype", "float16",
            "--max-model-len", str(MAX_MODEL_LEN),
            "--gpu-memory-utilization", str(GPU_MEMORY_UTILIZATION),
            "--host", "0.0.0.0",
            "--port", str(VLLM_PORT),
        ]
        print(f"[modal_deploy] Starting vLLM: {' '.join(cmd)}", flush=True)
        self._proc = subprocess.Popen(cmd)

    @modal.exit()
    def stop(self) -> None:
        """Terminate the vLLM subprocess on container shutdown."""
        if hasattr(self, "_proc") and self._proc.poll() is None:
            print("[modal_deploy] Stopping vLLM...", flush=True)
            self._proc.terminate()
            self._proc.wait(timeout=30)

    @modal.web_server(port=VLLM_PORT)
    def serve(self) -> None:
        """Block until vLLM is healthy, then let Modal route traffic to it.

        Modal exposes the container's ``VLLM_PORT`` directly to the internet.
        This method just needs to wait for vLLM to finish loading before
        Modal starts forwarding requests.
        """
        import httpx

        deadline = time.time() + 480  # 8 min max for cold start + model load
        url = f"http://127.0.0.1:{VLLM_PORT}/health"
        while time.time() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"vLLM exited early with code {self._proc.returncode}"
                )
            try:
                resp = httpx.get(url, timeout=5.0)
                if resp.status_code == 200:
                    print(f"[modal_deploy] vLLM healthy on port {VLLM_PORT}", flush=True)
                    return
            except httpx.HTTPError:
                pass
            time.sleep(3.0)
        raise RuntimeError("vLLM did not become healthy within 480s")


@app.local_entrypoint()
def main() -> None:
    """Print the deployed endpoint URL after (re)deploying the service."""
    inference = Qwen3Inference()
    url = None
    try:
        url = inference.serve.web_url  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - URL discovery is best-effort only
        url = None
    if url:
        print(f"[modal_deploy] Qwen3-8B endpoint ready: {url}")
        print(f"[modal_deploy] Set OPEN_MODEL_BASE_URL={url.rstrip('/')}/v1")
    else:
        print("[modal_deploy] Deployed. Find the endpoint URL in the Modal dashboard")
        print("             under Apps -> qwen3-8b-inference -> serve.")


if __name__ == "__main__":
    print("Deploy with:  modal deploy modal_deploy.py")
    print("Test locally: modal serve modal_deploy.py")
