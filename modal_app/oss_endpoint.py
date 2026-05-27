"""Modal endpoint for hosted OSS assistant inference.

Deploy with:

    modal deploy modal_app/oss_endpoint.py

The endpoint accepts:

    {"messages": [...], "temperature": 0.2, "max_tokens": 512}

and returns:

    {"response": "...", "model": "..."}
"""

from __future__ import annotations

import os
from typing import Any

import modal

APP_NAME = "ai-risk-workbench-oss"
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "accelerate>=0.33.0",
        "torch>=2.3.0",
        "transformers>=4.44.0",
    )
)


@app.cls(
    image=image,
    gpu=os.getenv("MODAL_GPU", "A10G"),
    timeout=600,
    scaledown_window=300,
)
class OSSAssistantModel:
    """Lazy-loaded chat model hosted on Modal GPU."""

    @modal.enter()
    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        self.model_id = os.getenv("MODAL_MODEL_ID", DEFAULT_MODEL_ID)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            device_map="auto",
        )
        self.pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
        )

    @modal.method()
    def generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        messages = payload.get("messages") or []
        temperature = float(payload.get("temperature", 0.2))
        max_tokens = int(payload.get("max_tokens", 512))
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        outputs = self.pipeline(
            prompt,
            max_new_tokens=max_tokens,
            temperature=max(temperature, 0.01),
            do_sample=temperature > 0,
            return_full_text=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return {
            "response": outputs[0]["generated_text"].strip(),
            "model": self.model_id,
        }


@app.function(image=image, timeout=600)
@modal.fastapi_endpoint(method="POST")
def chat(payload: dict[str, Any]) -> dict[str, Any]:
    """HTTP endpoint called by models.modal_endpoint.ModalEndpointClient."""

    return OSSAssistantModel().generate.remote(payload)
