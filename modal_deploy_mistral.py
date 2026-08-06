"""Self-deployed Mistral Shieldstral safety classifier on Modal.com (NVIDIA L4).

This module deploys Shieldstral (Mistral AI's 3B open-weights policy-adaptive
safety classifier) as an OpenAI-compatible inference endpoint on Modal, served
by vLLM. It exposes a ``classify_safety()`` interface over the standard
``/v1/chat/completions`` endpoint.

GPU sizing:
    * NVIDIA L4 ........ 24 GB VRAM
    * Shieldstral (3B) . ~6 GB of weights (bf16)
    * Headroom ......... ~18 GB left for the KV cache

Deploy / serve:
    modal deploy modal_deploy_mistral.py     # persistent HTTPS endpoint
    modal serve modal_deploy_mistral.py      # ephemeral endpoint for testing

Set ``MISTRAL_MODEL_BASE_URL=<endpoint-url>/v1`` for the workbench backends.
"""

from __future__ import annotations

import subprocess
import time

import modal

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_REPO = "mistralai/Shieldstral-1.0-3B"  # Hugging Face repo id
SERVED_MODEL_NAME = "mistral-shieldstral"  # name clients pass as ``model=``
VLLM_PORT = 8000
MAX_MODEL_LEN = 4096
GPU_MEMORY_UTILIZATION = 0.85
SCALEDOWN_WINDOW = 300

# ---------------------------------------------------------------------------
# Image + Volume
# ---------------------------------------------------------------------------
hf_cache_vol = modal.Volume.from_name("mistral-shieldstral-hf-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("mistral-shieldstral-vllm-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm",
        "torch",
        "transformers",
        "huggingface_hub",
    )
)

app = modal.App("mistral-shieldstral-inference")


@app.cls(
    image=image,
    gpu="L4",
    scaledown_window=SCALEDOWN_WINDOW,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    timeout=600,
)
class ShieldstralInference:
    """vLLM-backed OpenAI-compatible inference service for Shieldstral."""

    @modal.enter()
    def start(self) -> None:
        """Launch vLLM's OpenAI-compatible server as a subprocess."""
        cmd = [
            "vllm", "serve", MODEL_REPO,
            "--served-model-name", SERVED_MODEL_NAME,
            "--dtype", "bfloat16",
            "--max-model-len", str(MAX_MODEL_LEN),
            "--gpu-memory-utilization", str(GPU_MEMORY_UTILIZATION),
            "--host", "0.0.0.0",
            "--port", str(VLLM_PORT),
        ]
        print(f"[modal_deploy_mistral] Starting vLLM: {' '.join(cmd)}", flush=True)
        self._proc = subprocess.Popen(cmd)

    @modal.exit()
    def stop(self) -> None:
        """Terminate the vLLM subprocess on container shutdown."""
        if hasattr(self, "_proc") and self._proc.poll() is None:
            print("[modal_deploy_mistral] Stopping vLLM...", flush=True)
            self._proc.terminate()
            self._proc.wait(timeout=30)

    @modal.web_server(port=VLLM_PORT)
    def serve(self) -> None:
        """Block until vLLM is healthy, then let Modal route traffic."""
        import httpx

        deadline = time.time() + 480
        url = f"http://127.0.0.1:{VLLM_PORT}/health"
        while time.time() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"vLLM exited early with code {self._proc.returncode}"
                )
            try:
                resp = httpx.get(url, timeout=5.0)
                if resp.status_code == 200:
                    print(f"[modal_deploy_mistral] vLLM healthy on port {VLLM_PORT}", flush=True)
                    return
            except httpx.HTTPError:
                pass
            time.sleep(3.0)
        raise RuntimeError("vLLM did not become healthy within 480s")


@app.local_entrypoint()
def main() -> None:
    """Print the deployed endpoint URL."""
    inference = ShieldstralInference()
    url = None
    try:
        url = inference.serve.web_url  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        url = None
    if url:
        print(f"[modal_deploy_mistral] Shieldstral endpoint ready: {url}")
        print(f"[modal_deploy_mistral] Set MISTRAL_MODEL_BASE_URL={url.rstrip('/')}/v1")
    else:
        print("[modal_deploy_mistral] Deployed. Find the endpoint URL in the Modal dashboard")
        print("                      under Apps -> mistral-shieldstral-inference -> serve.")


if __name__ == "__main__":
    print("Deploy with:  modal deploy modal_deploy_mistral.py")
    print("Test locally: modal serve modal_deploy_mistral.py")
