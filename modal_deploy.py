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
    container on ``127.0.0.1:8000``. A small FastAPI reverse proxy is exposed to
    the internet via ``@modal.asgi_app()`` and forwards every request (including
    ``/v1/chat/completions`` and ``/v1/models``) to the local vLLM process. This
    is Modal's recommended pattern for long-running OpenAI-compatible servers.

Model weights:
    Qwen/Qwen3-8B is downloaded into a Modal ``Volume`` at image build / first
    run so that cold starts do not re-download ~16 GB from the Hub.

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
MODEL_DIR = "/models"                 # Volume mount point for weights
MODEL_LOCAL_DIR = f"{MODEL_DIR}/Qwen3-8B"
VLLM_PORT = 8000                      # vLLM listens on loopback inside the container
VLLM_BASE = f"http://127.0.0.1:{VLLM_PORT}"
MAX_MODEL_LEN = 4096                  # context window (fits L4 with headroom)
GPU_MEMORY_UTILIZATION = 0.90
CONTAINER_IDLE_TIMEOUT = 300          # keep warm 5 min between requests

# ---------------------------------------------------------------------------
# Image: debian_slim + vLLM stack, with weights pre-downloaded into a Volume.
# ---------------------------------------------------------------------------
# A Modal Volume persists the ~16 GB of weights across image rebuilds and cold
# starts; the download command is idempotent (skips files already present), so
# the weights are fetched once and reused thereafter.
model_cache = modal.Volume.from_name("qwen3-8b-weights", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm",
        "torch",
        "transformers",
        "huggingface_hub",
        "fastapi",
        "httpx",
    )
    .run_commands(
        # Pre-download the weights at build time so cold starts don't re-fetch.
        f"huggingface-cli download {MODEL_REPO} --local-dir {MODEL_LOCAL_DIR}"
    )
)

app = modal.App("qwen3-8b-inference")


@app.cls(
    image=image,
    gpu="L4",  # 24 GB VRAM -- see sizing note in the module docstring.
    container_idle_timeout=CONTAINER_IDLE_TIMEOUT,
    volumes={MODEL_DIR: model_cache},
    # Allow enough time for vLLM to load weights and warm up on cold start.
    timeout=600,
)
class Qwen3Inference:
    """vLLM-backed OpenAI-compatible inference service for Qwen3-8B."""

    @modal.enter()
    def start_vllm(self) -> None:
        """Launch vLLM's OpenAI server as a subprocess and wait until ready."""
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", MODEL_LOCAL_DIR,
            "--served-model-name", SERVED_MODEL_NAME,
            "--dtype", "float16",
            "--max-model-len", str(MAX_MODEL_LEN),
            "--gpu-memory-utilization", str(GPU_MEMORY_UTILIZATION),
            "--host", "0.0.0.0",
            "--port", str(VLLM_PORT),
        ]
        # Start vLLM detached so it keeps serving while the proxy handles traffic.
        self._proc = subprocess.Popen(cmd)  # noqa: SIM101 (store for keepalive/teardown)
        self._wait_until_ready()

    def _wait_until_ready(self, timeout: float = 480.0) -> None:
        """Block until vLLM answers ``/v1/models`` or ``timeout`` seconds elapse.

        Args:
            timeout: Maximum seconds to wait for the server to become healthy.

        Raises:
            RuntimeError: If vLLM does not become healthy within ``timeout``.
        """
        import httpx

        deadline = time.time() + timeout
        url = f"{VLLM_BASE}/v1/models"
        while time.time() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"vLLM process exited early with code {self._proc.returncode}"
                )
            try:
                resp = httpx.get(url, timeout=5.0)
                if resp.status_code == 200:
                    print(f"[modal_deploy] vLLM ready at {VLLM_BASE}", flush=True)
                    return
            except httpx.HTTPError:
                pass
            time.sleep(3.0)
        raise RuntimeError(f"vLLM did not become ready within {timeout:.0f}s")

    @modal.asgi_app()
    def serve(self):
        """Expose a FastAPI reverse proxy that forwards all routes to vLLM.

        Returns:
            A FastAPI app proxying ``/v1/chat/completions``, ``/v1/models`` and
            every other route to the local vLLM OpenAI-compatible server.
        """
        import httpx
        from fastapi import FastAPI, Request
        from fastapi.responses import Response, StreamingResponse

        web = FastAPI(title="Qwen3-8B on Modal (L4)")
        # Generous read timeout: long generations at 4K context can take a while.
        client = httpx.AsyncClient(base_url=VLLM_BASE, timeout=httpx.Timeout(600.0))

        @web.get("/")
        async def root() -> dict:
            """Return basic deployment metadata."""
            return {
                "model": SERVED_MODEL_NAME,
                "repo": MODEL_REPO,
                "gpu": "L4",
                "max_model_len": MAX_MODEL_LEN,
                "vllm_base": VLLM_BASE,
            }

        @web.api_route(
            "/{path:path}",
            methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
        )
        async def proxy(path: str, request: Request):
            """Forward any request to vLLM, streaming the response back."""
            target_url = f"{VLLM_BASE}/{path}"
            body = await request.body()
            headers = {
                k: v for k, v in request.headers.items() if k.lower() != "host"
            }
            upstream = client.build_request(
                request.method,
                target_url,
                headers=headers,
                content=body,
                params=request.query_params,
            )
            resp = await client.send(upstream, stream=True)

            async def stream():
                try:
                    async for chunk in resp.aiter_raw():
                        yield chunk
                finally:
                    await resp.aclose()

            excluded = {"content-encoding", "content-length", "transfer-encoding"}
            out_headers = {
                k: v for k, v in resp.headers.items() if k.lower() not in excluded
            }
            return StreamingResponse(
                stream(), status_code=resp.status_code, headers=out_headers
            )

        return web


@app.local_entrypoint()
def main() -> None:
    """Print the deployed endpoint URL after (re)deploying the service."""
    inference = Qwen3Inference()
    # Touch the class so Modal materializes the function and reports its URL.
    url = None
    try:
        # ``serve`` is the asgi_app; its public URL is stable per environment.
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
    # ``modal deploy modal_deploy.py`` / ``modal serve modal_deploy.py`` invoke
    # the local_entrypoint above; running directly prints usage guidance.
    print("Deploy with:  modal deploy modal_deploy.py")
    print("Test locally: modal serve modal_deploy.py")
